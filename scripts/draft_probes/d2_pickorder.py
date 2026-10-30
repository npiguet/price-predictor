"""D2 — the pick order: what each generation ranks first, and where that came from.

At (pack 1, pick 1) the POOL, PASSED and TAKEN blocks are empty *by
construction*, so the policy at that state is a pure card ranking with nothing
approximated away: 15 cards in, 15 logits out, no context. That makes P1P1 the
one place where "the agent's pick order" is a measurable object rather than an
inferred one. Four parts, in order of what they let the next part say:

1. **The order itself.** Every opening booster in the corpus (one per seat —
   the state depends only on the booster, not on who is sitting there), run
   through all four checkpoints. Only within-state contrasts are behavioural
   (the policy head is invariant to a per-state constant), so each state's
   logits are centred before a card's values are averaged across the boosters
   it appears in. That per-card mean is the **P1P1 scalar**.

2. **Is the colour lean unconditional?** (H6) The prior study found gen-4
   leaning W/G/B and away from U/R by ~4 pp at *off-lane* picks, which is
   consistent with either a standing colour prior or a read of what is open.
   P1P1 discriminates: nothing is open yet. Reported as both a mean centred
   logit per colour and a taken-minus-available argmax lean, gen1 alongside —
   gen-1 is distilled Forge, so a lean already in gen-1 is Forge's, not the RL's.

3. **Where does the order come from?** (H7) The P1P1 scalar is regressed on
   three sources that are mutually independent and all exogenous to the RL:
   Forge's bundled human draft rankings (itself a pick order, the best
   yardstick available), the scorer's own marginal card value ``v_swap``, and
   the encoder's leading text axes (PC1 = played-rate, PC2 = winnability) plus
   the 32 deterministic dims. ``shrunk_score_play`` rides along as a column but
   is the encoder's *training label* and therefore not an independent grader.
   Headline: does gen-4 sit closer to or further from the human order than
   gen-1? Then the gen4-minus-gen1 shift is cut by card category, because the
   reward demotes removal below generic bodies while gen-1 inherits Forge's
   partly-human order — the prediction under test is a negative removal shift.

4. **How much of the policy IS that fixed order?** (H2) Over every pick of the
   gen-4-labelled seats, how often each model's real argmax equals the argmax
   of its own P1P1 scalar restricted to the cards in that pack. The complement
   is everything context does.

Inference only: frozen checkpoints, recorded states, no training, nothing under
``src/`` or ``models/`` touched.

Usage
-----
    python scripts/draft_probes/d2_pickorder.py
    python scripts/draft_probes/d2_pickorder.py --limit-drafts 100 --part4-drafts 50
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _lib.corpora import card_column
from _lib.stats import ols, spearman
from draft_corpus_common import WUBRG, ColourResolver
from draft_probes.probe_lib import (
    CARD_VALUES,
    CARDS,
    CORPUS,
    HINTS,
    MODELS,
    OUT,
    SCORER_OUT,
    CardTable,
    PickSample,
    argmaxes,
    centred,
    colour_weights,
    iter_corpus_states,
    iter_runners,
    replay_fidelity,
)
from scorer_probes.probe_lib import card_features

from sealed.domain.card_embedding_layout import FEATURE_COUNT
from sealed.infrastructure.converted_card_locator import ConvertedCardLocator

GENS = list(MODELS)
TEXT_PCA = SCORER_OUT / "text_pca_512.npz"
COLOURS = list(WUBRG) + ["C"]


# --------------------------------------------------------------------------
# Part 1 — the P1P1 states and the per-card scalar
# --------------------------------------------------------------------------

def p1p1_samples(table: CardTable, limit_drafts: int | None) -> list[PickSample]:
    """One sample per distinct opening booster, from every seat of every draft.

    Boosters are deduped on their (sorted) card multiset: the P1P1 state holds
    nothing but the pack, so two seats opening the same 15 cards are the same
    state and would double-count in the per-card average.
    """
    seen: set[tuple[str, ...]] = set()
    out: list[PickSample] = []
    for sample in iter_corpus_states(CORPUS, table, limit_drafts=limit_drafts,
                                     clocks=[(1, 1)]):
        key = tuple(sorted(sample.pack_names))
        if key not in seen:
            seen.add(key)
            out.append(sample)
    return out


def card_scalars(samples, cent) -> tuple[dict[str, float], Counter]:
    """Per-card mean centred P1P1 logit, plus how many boosters it appeared in."""
    total: defaultdict[str, float] = defaultdict(float)
    n: Counter = Counter()
    for s, c in zip(samples, cent):
        for name, v in zip(s.pack_names, c):
            total[name] += float(v)
            n[name] += 1
    return {k: total[k] / n[k] for k in total}, n


def run_models(table, p1, allpicks, batch, seat_label) -> dict:
    """Every model over both state sets: centred P1P1 logits, per-card scalars,
    and the all-picks argmax."""
    runs = {"p1_cent": {}, "scalar": {}, "appear": Counter(), "ap_arg": {}, "replay": {}}
    for gen, runner in iter_runners(table, batch):
        runs["p1_cent"][gen] = centred(runner.logits(p1))
        runs["scalar"][gen], runs["appear"] = card_scalars(p1, runs["p1_cent"][gen])
        lg = runner.logits(allpicks)
        runs["ap_arg"][gen] = argmaxes(lg)
        runs["replay"][gen] = replay_fidelity(allpicks, lg)
        print(f"  {gen}: P1P1 done, all-picks done "
              f"(self-replay on {seat_label} seats {runs['replay'][gen]:.4f})")
    return runs


def write_logits_csv(p1, p1_cent, gens) -> None:
    with (OUT / "d2_p1p1_logits.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["booster", "set_code", "card", "taken_in_corpus"]
                   + [f"centred_logit_{g}" for g in gens])
        for i, s in enumerate(p1):
            for j, name in enumerate(s.pack_names):
                w.writerow([i, s.set_code, name, int(j == s.target)]
                           + [f"{p1_cent[g][i][j]:.6f}" for g in gens])


# --------------------------------------------------------------------------
# Part 3 inputs — the exogenous card-value sources
# --------------------------------------------------------------------------

def read_forge_hints() -> dict[str, dict]:
    rows = {}
    with HINTS.open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            rank = r.get("draft_rank") or ""
            rows[r["name"]] = {
                # .rnk files are normalized rank in (0,1], lower = better, so
                # the sign is flipped to make "higher = better" everywhere.
                "human_value": -float(rank) if rank else None,
                "ai_remove_deck": int(r.get("ai_remove_deck") or 0),
            }
    return rows


def card_category(feats: dict) -> str:
    if feats.get("is_creature"):
        return "creature"
    if feats.get("is_removal"):
        return "noncreature removal"
    if feats.get("draws_cards"):
        return "card draw"
    return "other"


def card_meta(cards, table: CardTable, colour_of: ColourResolver) -> dict[str, dict]:
    """Per card: colours, category, the exogenous card values, and its leading
    text axes plus deterministic features read off its embedding."""
    locator = ConvertedCardLocator(CARDS)
    hints = read_forge_hints()
    v_swap = card_column(CARD_VALUES, "v_swap")
    score_play = card_column(CARD_VALUES, "shrunk_score_play")
    pca = np.load(TEXT_PCA)
    pc_mean, pc_comp = pca["mean"], pca["components"]

    mat = table.matrix()
    meta: dict[str, dict] = {}
    for name in cards:
        row = table.index(name)
        pcs, det = (None, None), None
        if row is not None:
            proj = (mat[row][:-FEATURE_COUNT] - pc_mean) @ pc_comp[:2].T
            pcs = (float(proj[0]), float(proj[1]))
            det = mat[row][-FEATURE_COUNT:].astype(np.float64)
        text = locator.load_text(name)
        h = hints.get(name, {})
        meta[name] = {
            "colours": sorted(colour_of(name)),
            "category": card_category(card_features(text.text) if text else {}),
            "human_value": h.get("human_value"),
            "ai_remove_deck": h.get("ai_remove_deck"),
            "v_swap": v_swap.get(name),
            "shrunk_score_play": score_play.get(name),
            "pc1": pcs[0], "pc2": pcs[1], "det": det,
        }
    return meta


def write_values_csv(cards, meta, appear, scalar, gens) -> None:
    def cell(v, fmt):
        return "" if v is None else format(v, fmt)

    with (OUT / "d2_p1p1_values.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["card", "n_boosters", "colours", "category",
                    "human_value_neg_draft_rank", "ai_remove_deck", "v_swap",
                    "shrunk_score_play_NOT_INDEPENDENT", "text_pc1", "text_pc2"]
                   + [f"p1p1_{g}" for g in gens])
        for name in cards:
            m = meta[name]
            w.writerow([
                name, appear[name], "".join(m["colours"]) or "C", m["category"],
                cell(m["human_value"], ".6f"),
                "" if m["ai_remove_deck"] is None else m["ai_remove_deck"],
                cell(m["v_swap"], ".6f"), cell(m["shrunk_score_play"], ".6f"),
                cell(m["pc1"], ".4f"), cell(m["pc2"], ".4f"),
            ] + [f"{scalar[g][name]:.6f}" for g in gens])


# --------------------------------------------------------------------------
# Part 2 — the unconditional colour prior
# --------------------------------------------------------------------------

def colour_prior(p1, p1_cent, gens, colour_of) -> dict:
    supply = dict.fromkeys(COLOURS, 0.0)
    for s in p1:
        for name in s.pack_names:
            for c, w in colour_weights(colour_of(name)):
                supply[c] += w
    supply_tot = sum(supply.values())
    supply_share = {c: 100.0 * supply[c] / supply_tot for c in COLOURS}

    colour_logit, colour_lean = {}, {}
    for g in gens:
        wsum, wn, taken = (dict.fromkeys(COLOURS, 0.0) for _ in range(3))
        for s, cvec in zip(p1, p1_cent[g]):
            for name, v in zip(s.pack_names, cvec):
                for c, w in colour_weights(colour_of(name)):
                    wsum[c] += float(v) * w
                    wn[c] += w
            for c, w in colour_weights(colour_of(s.pack_names[int(np.argmax(cvec))])):
                taken[c] += w
        tot = sum(taken.values())
        colour_logit[g] = {c: wsum[c] / wn[c] if wn[c] else float("nan") for c in COLOURS}
        colour_lean[g] = {c: 100.0 * taken[c] / tot - supply_share[c] for c in COLOURS}
    return {"supply_share_pct": supply_share, "mean_centred_logit": colour_logit,
            "argmax_lean_pp": colour_lean}


# --------------------------------------------------------------------------
# Part 3 — attribution
# --------------------------------------------------------------------------

def r2_of(y: np.ndarray, X: np.ndarray | None) -> float:
    """OLS R² of ``y`` on ``X`` with an intercept (0.0 for the empty design)."""
    if X is None or X.shape[1] == 0:
        return 0.0
    return ols(np.column_stack([np.ones(len(y)), X]), y).r2


def attribution(cards, meta, scalar, gens, n_boot) -> dict:
    """Spearman per source, the R² decomposition, the bootstrap headline against
    the human order, and the gen-minus-gen1 shift by card category."""
    def col(key):
        return np.array([np.nan if meta[c][key] is None else meta[c][key] for c in cards])

    y = {g: np.array([scalar[g][c] for c in cards]) for g in gens}
    human, vswap, pc1, pc2 = col("human_value"), col("v_swap"), col("pc1"), col("pc2")
    det = np.array([meta[c]["det"] if meta[c]["det"] is not None
                    else np.full(FEATURE_COUNT, np.nan) for c in cards])

    sources = {"human_draft_rank": human, "scorer_v_swap": vswap,
               "text_pc1": pc1, "text_pc2": pc2,
               "shrunk_score_play (NOT independent)": col("shrunk_score_play")}
    pair = {g: {} for g in gens}
    for g in gens:
        for sname, sv in sources.items():
            ok = np.isfinite(y[g]) & np.isfinite(sv)
            pair[g][sname] = {"n": int(ok.sum()), "spearman": spearman(y[g][ok], sv[ok])}
        for h in gens:
            if h != g:
                pair[g][f"model:{h}"] = {"n": len(cards), "spearman": spearman(y[g], y[h])}

    # R² decomposition on the subset where every source exists.
    ok = (np.isfinite(human) & np.isfinite(vswap) & np.isfinite(pc1)
          & np.isfinite(det).all(1))
    for g in gens:
        ok &= np.isfinite(y[g])
    idx = np.flatnonzero(ok)
    det_keep = det[idx][:, det[idx].std(0) > 1e-9]
    blocks = {
        "human": human[idx][:, None],
        "scorer": vswap[idx][:, None],
        "encoder": np.column_stack([pc1[idx], pc2[idx], det_keep]),
    }
    decomp = {}
    for g in gens:
        yy = y[g][idx]
        full = r2_of(yy, np.column_stack(list(blocks.values())))
        d = {"n_cards": int(len(idx)), "R2_full": full, "R2_unexplained": 1.0 - full}
        for bname in blocks:
            d[f"R2_{bname}_alone"] = r2_of(yy, blocks[bname])
            rest = [v for k, v in blocks.items() if k != bname]
            d[f"R2_unique_{bname}"] = full - r2_of(yy, np.column_stack(rest))
        d["R2_human+scorer"] = r2_of(yy, np.column_stack([blocks["human"], blocks["scorer"]]))
        decomp[g] = d

    # Headline: paired bootstrap over cards of the gen-vs-gen1 difference in
    # Spearman against the human order. Paired = the same card set both sides.
    ok_h = np.isfinite(human)
    for g in gens:
        ok_h &= np.isfinite(y[g])
    hi = np.flatnonzero(ok_h)
    rng = np.random.default_rng(7)
    human_rho = {g: spearman(y[g][hi], human[hi]) for g in gens}
    head = {"n_cards": int(len(hi)), "spearman_vs_human": human_rho, "vs_gen1": {}}
    if "gen1" in gens:
        boot = {g: np.empty(n_boot) for g in gens if g != "gen1"}
        for b in range(n_boot):
            take = rng.integers(0, len(hi), len(hi))
            hh = human[hi][take]
            base = spearman(y["gen1"][hi][take], hh)
            for g in boot:
                boot[g][b] = spearman(y[g][hi][take], hh) - base
        for g, arr in boot.items():
            head["vs_gen1"][g] = {
                "delta_spearman": human_rho[g] - human_rho["gen1"],
                "ci95": [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))],
                "p_delta_gt_0": float((arr > 0).mean()),
            }

    # gen4 - gen1 shift by card category, on z-scored scalars (different models
    # have different logit scales, so only the standardized order is comparable)
    cat_shift = {}
    if "gen1" in gens:
        zs = {g: (y[g] - y[g].mean()) / y[g].std() for g in gens}
        cats = np.array([meta[c]["category"] for c in cards])
        for g in gens:
            if g == "gen1":
                continue
            d = zs[g] - zs["gen1"]
            cat_shift[g] = {
                cat: {"n": int((cats == cat).sum()),
                      "mean_z_shift": float(d[cats == cat].mean()),
                      "se": float(d[cats == cat].std(ddof=1)
                                  / max(1, (cats == cat).sum()) ** 0.5)}
                for cat in sorted(set(cats))
            }
    attribution = {"spearman": pair, "r2_decomposition": decomp,
                   "headline_vs_human": head, "gen_minus_gen1_by_category_z": cat_shift}
    return attribution, y, list(sources)


# --------------------------------------------------------------------------
# Part 4 — how much of the policy is the fixed order
# --------------------------------------------------------------------------

def static_order(allpicks, scalar, ap_arg, gens) -> dict:
    pack_no = np.array([s.state.pack_number for s in allpicks])
    pick_no = np.array([s.state.pick_number for s in allpicks])
    covered = np.array([all(n in scalar[gens[0]] for n in s.pack_names) for s in allpicks])
    static = {}
    for g in gens:
        agree = np.full(len(allpicks), np.nan)
        for i, s in enumerate(allpicks):
            if covered[i]:
                vals = np.array([scalar[g][n] for n in s.pack_names])
                agree[i] = float(int(np.argmax(vals)) == ap_arg[g][i])
        m = np.isfinite(agree)
        static[g] = {
            "n_states": int(m.sum()),
            "coverage": float(covered.mean()),
            "agree_overall": float(agree[m].mean()),
            "agree_by_pack": {int(p): float(agree[m & (pack_no == p)].mean())
                              for p in sorted(set(pack_no.tolist()))},
            "agree_by_pick": {int(k): float(agree[m & (pick_no == k)].mean())
                              for k in sorted(set(pick_no.tolist()))},
        }
    return static


# --------------------------------------------------------------------------

def report(args, p1, cards, gens, runs, payload, y, source_names) -> None:
    scalar, p1_cent = runs["scalar"], runs["p1_cent"]
    colour, static = payload["part2_colour"], payload["part4_static_order"]
    attr = payload["part3_attribution"]
    W = 22
    print("\n" + "=" * 72)
    print(f"D2 PICK ORDER — {len(p1)} distinct opening boosters, "
          f"{payload['meta']['n_sets']} sets, {len(cards)} cards with "
          f">= {args.min_appearances} appearances")
    print("=" * 72)

    print("\n--- PART 1: scale of the P1P1 order (centred logits) ---")
    print(f"{'':<{W}}" + "".join(f"{g:>10}" for g in gens))
    print(f"{'sd of card scalar':<{W}}" + "".join(f"{y[g].std():>10.3f}" for g in gens))
    print(f"{'sd within a booster':<{W}}"
          + "".join(f"{np.mean([c.std() for c in p1_cent[g]]):>10.3f}" for g in gens))
    for g in gens:
        top = sorted(cards, key=lambda c: -scalar[g][c])[:5]
        bot = sorted(cards, key=lambda c: scalar[g][c])[:3]
        print(f"  {g:>5} top: {', '.join(top)}")
        print(f"  {g:>5} bot: {', '.join(bot)}")

    supply_share = colour["supply_share_pct"]
    print("\n--- PART 2 (H6): the colour prior with nothing open yet ---")
    print(f"{'supply share %':<{W}}" + "".join(f"{c:>8}" for c in COLOURS))
    print(f"{'':<{W}}" + "".join(f"{supply_share[c]:>8.1f}" for c in COLOURS))
    print("\nmean centred P1P1 logit by colour identity (gold split):")
    print(f"{'model':<{W}}" + "".join(f"{c:>8}" for c in COLOURS))
    for g in gens:
        print(f"{g:<{W}}"
              + "".join(f"{colour['mean_centred_logit'][g][c]:>8.3f}" for c in COLOURS))
    print("\nargmax lean, taken% - available% (percentage points):")
    print(f"{'model':<{W}}" + "".join(f"{c:>8}" for c in COLOURS))
    for g in gens:
        print(f"{g:<{W}}"
              + "".join(f"{colour['argmax_lean_pp'][g][c]:>+8.2f}" for c in COLOURS))

    pair, decomp = attr["spearman"], attr["r2_decomposition"]
    head, cat_shift = attr["headline_vs_human"], attr["gen_minus_gen1_by_category_z"]
    print("\n--- PART 3 (H7): where the order comes from ---")
    print("Spearman of each model's P1P1 scalar against each source:")
    print(f"{'source':<38}" + "".join(f"{g:>10}" for g in gens) + f"{'n':>8}")
    for sname in source_names + [f"model:{h}" for h in gens]:
        cells = []
        n = 0
        for g in gens:
            e = pair[g].get(sname)
            cells.append("       ." if e is None else f"{e['spearman']:>10.3f}")
            if e:
                n = e["n"]
        print(f"{sname:<38}" + "".join(cells) + f"{n:>8}")

    print(f"\nR2 decomposition (n={decomp[gens[0]]['n_cards']} cards with all "
          "sources present):")
    keys = ["R2_human_alone", "R2_scorer_alone", "R2_encoder_alone",
            "R2_human+scorer", "R2_full", "R2_unique_human",
            "R2_unique_scorer", "R2_unique_encoder", "R2_unexplained"]
    print(f"{'quantity':<22}" + "".join(f"{g:>10}" for g in gens))
    for k in keys:
        print(f"{k:<22}" + "".join(f"{decomp[g][k]:>10.3f}" for g in gens))

    print(f"\nHEADLINE - distance to the human pick order (n={head['n_cards']} "
          "ranked cards, paired):")
    for g in gens:
        line = f"  {g:<6} spearman vs human = {head['spearman_vs_human'][g]:+.4f}"
        if g in head["vs_gen1"]:
            d = head["vs_gen1"][g]
            line += (f"   delta vs gen1 = {d['delta_spearman']:+.4f} "
                     f"[{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}] "
                     f"P(>0)={d['p_delta_gt_0']:.3f}")
        print(line)

    if cat_shift:
        print("\ngen - gen1 shift in z-scored P1P1 scalar, by card category:")
        cats = sorted(next(iter(cat_shift.values())))
        print(f"{'category':<22}{'n':>6}" + "".join(f"{g:>18}" for g in cat_shift))
        for cat in cats:
            n = next(iter(cat_shift.values()))[cat]["n"]
            print(f"{cat:<22}{n:>6}"
                  + "".join(f"{cat_shift[g][cat]['mean_z_shift']:>+11.3f}"
                            f" +-{cat_shift[g][cat]['se']:.3f}" for g in cat_shift))

    print("\n--- PART 4 (H2): share of picks the context-free order already "
          "explains ---")
    print(f"states: {static[gens[0]]['n_states']} of {payload['meta']['part4_states']} "
          f"({args.seat_label} seats, {args.part4_drafts} drafts; "
          f"coverage {static[gens[0]]['coverage']:.3f})")
    print(f"{'model':<10}{'overall':>9}" + "".join(f"{'pack' + str(p):>8}" for p in (1, 2, 3)))
    for g in gens:
        print(f"{g:<10}{static[g]['agree_overall']:>9.3f}"
              + "".join(f"{static[g]['agree_by_pack'][p]:>8.3f}" for p in (1, 2, 3)))
    print("\nby pick number:")
    picks = sorted(static[gens[0]]["agree_by_pick"])
    print(f"{'model':<10}" + "".join(f"{k:>6}" for k in picks))
    for g in gens:
        print(f"{g:<10}" + "".join(f"{static[g]['agree_by_pick'][k]:>6.2f}" for k in picks))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-drafts", type=int, default=None,
                    help="drafts to read for the P1P1 sweep (default: all)")
    ap.add_argument("--part4-drafts", type=int, default=150,
                    help="drafts for the all-picks static-order check")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--min-appearances", type=int, default=5,
                    help="boosters a card must appear in to get a scalar")
    ap.add_argument("--seat-label", default="gen4")
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    table = CardTable()

    print("building P1P1 states ...")
    p1 = p1p1_samples(table, args.limit_drafts)
    print(f"  {len(p1)} distinct opening boosters, "
          f"{len({s.set_code for s in p1})} sets")

    print(f"building all-picks states for '{args.seat_label}' seats "
          f"({args.part4_drafts} drafts) ...")
    allpicks = list(iter_corpus_states(
        CORPUS, table, labels=[args.seat_label], limit_drafts=args.part4_drafts))
    print(f"  {len(allpicks)} states")

    runs = run_models(table, p1, allpicks, args.batch, args.seat_label)
    gens = [g for g in GENS if g in runs["scalar"]]
    appear, scalar = runs["appear"], runs["scalar"]

    cards = sorted(c for c in appear if appear[c] >= args.min_appearances)
    print(f"\n{len(cards)} cards with >= {args.min_appearances} appearances "
          f"(of {len(appear)} seen)")
    write_logits_csv(p1, runs["p1_cent"], gens)

    colour_of = ColourResolver(CARDS)
    meta = card_meta(cards, table, colour_of)
    write_values_csv(cards, meta, appear, scalar, gens)
    attr, y, source_names = attribution(cards, meta, scalar, gens, args.bootstrap)

    payload = {
        "meta": {
            "corpus": str(CORPUS),
            "n_boosters_p1p1": len(p1),
            "n_sets": len({s.set_code for s in p1}),
            "n_cards_scalar": len(cards),
            "min_appearances": args.min_appearances,
            "part4_seat_label": args.seat_label,
            "part4_drafts": args.part4_drafts,
            "part4_states": len(allpicks),
            "self_replay_fidelity": runs["replay"],
            "checkpoints": {g: str(MODELS[g]) for g in gens},
            "note": "shrunk_score_play is the encoder's own training label and "
                    "is NOT an independent grader; human_value = -draft_rank "
                    "(Forge .rnk, lower rank = better, sign flipped).",
        },
        "part2_colour": colour_prior(p1, runs["p1_cent"], gens, colour_of),
        "part3_attribution": attr,
        "part4_static_order": static_order(allpicks, scalar, runs["ap_arg"], gens),
    }
    (OUT / "d2_pickorder.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    report(args, p1, cards, gens, runs, payload, y, source_names)
    print(f"\nwrote {OUT / 'd2_pickorder.json'}")
    print(f"wrote {OUT / 'd2_p1p1_values.csv'}")
    print(f"wrote {OUT / 'd2_p1p1_logits.csv'}")


if __name__ == "__main__":
    main()

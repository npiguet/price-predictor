"""T5: input-block ablation of the gen-4 sealed scorer on held-out matches.

Per-card input is (text || det): the leading ``d_model - 32`` dims are the
sealed-trained encoder's pooled text vector, the trailing 32 are deterministic
script features (pips, MV, color flags, P/T, is_land, ...).

Question (experiments/2026-05-02-deterministic-feature-reliance.md): how much
of the scorer's held-out ranking ability comes from each block?

Five inference-time conditions, each a transformation applied to the RAW 544-dim
vector of EVERY card of EVERY deck (the model normalizes the trailing det block
itself, so all edits happen pre-normalization, in raw space):

    full          unmodified
    text_mean     text block <- corpus-mean text vector (per-card text info gone)
    text_shuffle  text block permuted within each deck (identity gone, bag kept)
    det_mean      det block  <- corpus-mean det vector
    both_mean     both        (floor: what does deck size alone predict?)

Corpus means are taken over the *distinct cards* of the eval decks, one vote per
distinct card.

Eval corpus: matches-b07/match-outcomes-gen5-vs-gen4-forge.txt, which postdates
the gen-4 scorer's training cutoff, so it is genuinely held out.

``--groups`` asks instead which of the 32 deterministic features the scorer
reads: each condition erases one feature group's slots to their corpus means
and keeps everything else. Two variance reductions make the small per-group
effects resolvable. Both post-cutoff corpora are scored (gen5-vs-gen4-forge
and gen4-vs-forge-best-gen3, ~21.5K matches -- the second was generated
2026-05-19..21, after the scorer's 2026-05-18 training run), and significance
comes from the paired per-match difference against the full model
(condition-correct minus full-correct), whose standard error is set by the
prediction flip rate rather than the base accuracy. Writes ``t5_groups.json``
(consumed by ``make_figures.py``).

Read-only w.r.t. the repo and the Y: drive.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _lib.jsonio import write_json
from _lib.markdown import banner, print_table
from _lib.markdown import num as fmt
from _lib.stats import coin_flip_band, pairwise_correct
from scorer_probes import probe_lib as pl
from scorer_probes.probe_lib import DeckBank

from sealed.domain import card_embedding_layout as layout

MATCH_FILE = pl.YDATA / "matches-b07" / "match-outcomes-gen5-vs-gen4-forge.txt"
POST_CUTOFF_FILES = [
    MATCH_FILE,
    pl.YDATA / "matches-b07" / "match-outcomes-gen4-vs-forge-best-gen3.txt",
]
DEFAULT_OUT = pl.SCRATCH / "t5_ablation.json"
GROUPS_OUT = pl.SCRATCH / "t5_groups.json"
SEED = 42
BATCH = 1024

CONDITIONS = ["full", "text_mean", "text_shuffle", "det_mean", "both_mean"]
PAIR_TABLE_CONDITIONS = ["full", "text_mean", "det_mean"]

# --groups conditions: label -> indices within the trailing deterministic block
GROUPS = {
    "is_land": [layout.IS_LAND],
    "mana cost (pips+generic+X+MV)": list(range(layout.COLOR_PIPS.start, layout.MANA_VALUE + 1)),
    "color pips only": list(range(layout.COLOR_PIPS.start, layout.COLORLESS_PIP + 1)),
    "mana value only": [layout.MANA_VALUE],
    "color flags": list(range(layout.COLOR_FLAGS.start, layout.IS_COLORLESS + 1)),
    "mana production": list(range(layout.PRODUCES_COLORS.start, layout.MANA_PRODUCED + 1)),
    "power/toughness/loyalty": [layout.POWER, layout.TOUGHNESS, layout.LOYALTY],
    f"all {layout.FEATURE_COUNT}": list(range(layout.FEATURE_COUNT)),
}


def corpus_means(reg: DeckBank, used: set[int], text_dim: int):
    """Raw-space mean vector over distinct cards of the used decks."""
    names = {c for i in used for c in reg.keys[i]}
    mean = np.stack([reg.emb[n] for n in sorted(names)]).mean(axis=0)
    return mean[:text_dim].copy(), mean[text_dim:].copy(), len(names)


def apply_condition(mat: np.ndarray, cond: str, idx: int, text_dim: int,
                    mean_text: np.ndarray, mean_det: np.ndarray) -> np.ndarray:
    """Transform one deck's raw card matrix in place (mat is already a copy)."""
    if cond == "full":
        return mat
    if cond in GROUPS:
        idx = GROUPS[cond]
        mat[:, [text_dim + i for i in idx]] = mean_det[idx]
        return mat
    if cond in ("text_mean", "both_mean"):
        mat[:, :text_dim] = mean_text
    if cond in ("det_mean", "both_mean"):
        mat[:, text_dim:] = mean_det
    if cond == "text_shuffle":
        rng = np.random.default_rng([SEED, idx])
        perm = rng.permutation(mat.shape[0])
        mat[:, :text_dim] = mat[perm, :text_dim]
    return mat


def score_condition(probe: pl.Probe, reg: DeckBank, order: list[int],
                    cond: str, text_dim: int, mean_text, mean_det,
                    batch: int = BATCH) -> np.ndarray:
    """Score every distinct deck under one condition (built batch by batch)."""
    return reg.score_all(probe, order, batch, transform=lambda mat, i: apply_condition(
        mat, cond, i, text_dim, mean_text, mean_det))


def condition_metrics(scores: np.ndarray, full: np.ndarray,
                      win_idx: np.ndarray, lose_idx: np.ndarray,
                      pairs: list[str]) -> dict:
    diff = scores[win_idx] - scores[lose_idx]
    correct = pairwise_correct(diff)
    full_diff = full[win_idx] - full[lose_idx]
    agree = pairwise_correct(diff * full_diff)  # same sign as full's ranking
    # paired against the full model: its SE is set by the flip rate alone
    paired = correct - pairwise_correct(full_diff)
    paired_se = float(paired.std(ddof=1) / np.sqrt(len(paired))) if len(paired) > 1 else 0.0

    per_pair = {}
    by_pair: dict[str, list[int]] = defaultdict(list)
    for i, p in enumerate(pairs):
        by_pair[p].append(i)
    for p, rows in sorted(by_pair.items()):
        sel = correct[rows]
        per_pair[p] = {"n": len(rows), "acc": float(sel.mean())}

    m = {
        "n_matches": int(len(diff)),
        "acc": float(correct.mean()),
        "acc_strict": float((diff > 0).mean()),
        "tie_rate": float((diff == 0).mean()),
        "rank_agree_with_full": float(agree.mean()),
        "mean_abs_score_shift": float(np.abs(scores - full).mean()),
        "score_mean": float(scores.mean()),
        "score_std": float(scores.std()),
        "d_acc": float(paired.mean()),
        "paired_se": paired_se,
        "z": float(paired.mean() / paired_se) if paired_se > 0 else 0.0,
        "flip_rate": float((paired != 0).mean()),
        "per_pair": per_pair,
    }
    if np.std(scores) > 0 and np.std(full) > 0:
        m["pearson_vs_full"] = float(pearsonr(scores, full)[0])
        m["spearman_vs_full"] = float(spearmanr(scores, full)[0])
    else:  # degenerate (constant) condition scores
        m["pearson_vs_full"] = float("nan")
        m["spearman_vs_full"] = float("nan")
    return m


def print_meta(meta: dict) -> None:
    print(f"eval files       : {', '.join(meta['match_files'])}")
    print(f"matches parsed   : {meta['matches_parsed']} "
          f"(undecided dropped {meta['matches_undecided']}, "
          f"no-embedding dropped {meta['matches_dropped_missing']})")
    print(f"matches scored   : {meta['matches_scored']}")
    print(f"distinct decks   : {meta['distinct_decks']} "
          f"of {meta['deck_instances']} instances")
    print(f"distinct cards   : {meta['distinct_cards']} "
          f"(missing embeddings: {meta['missing_cards']})")
    print(f"d_model={meta['d_model']}  text_dim={meta['text_dim']}  "
          f"det_dim={meta['det_dim']}  device={meta['device']}")


def print_groups_report(res: dict) -> None:
    banner("T5 DETERMINISTIC-GROUP ERASURES")
    print_meta(res["meta"])
    cond = res["conditions"]
    print(f"\nfull model acc = {cond['full']['acc']:.4f}  "
          f"(n={cond['full']['n_matches']})\n")
    print_table(["group erased", "features", "d-acc", "paired SE", "z", "flip%",
                 "spearman vs full"],
                [[g, len(GROUPS[g]), f"{m['d_acc']:+.4f}", f"{m['paired_se']:.4f}",
                  f"{m['z']:+.1f}", f"{100 * m['flip_rate']:.2f}", fmt(m["spearman_vs_full"], 3)]
                 for g, m in ((g, cond[g]) for g in GROUPS)])


def print_report(res: dict) -> None:
    meta, cond = res["meta"], res["conditions"]

    banner("T5 ABLATION SUMMARY")
    print_meta(meta)

    print("\n### Condition summary\n")
    print_table(["condition", "acc", "acc(strict)", "tie%", "pearson vs full",
                 "spearman vs full", "rank-agree w/ full", "mean abs shift"],
                [[c, fmt(m["acc"]), fmt(m["acc_strict"]), f"{100 * m['tie_rate']:.1f}%",
                  fmt(m["pearson_vs_full"]), fmt(m["spearman_vs_full"]),
                  fmt(m["rank_agree_with_full"]), fmt(m["mean_abs_score_shift"], 3)]
                 for c, m in ((c, cond[c]) for c in CONDITIONS)])

    print("\n### Accuracy by method pair\n")
    full_pairs = cond["full"]["per_pair"]
    print_table(["pair", "n"] + PAIR_TABLE_CONDITIONS,
                [[p, full_pairs[p]["n"]]
                 + [fmt(cond[c]["per_pair"][p]["acc"], 3) for c in PAIR_TABLE_CONDITIONS]
                 for p in sorted(full_pairs, key=lambda p: -full_pairs[p]["n"])])

    a = {c: cond[c]["acc"] for c in CONDITIONS}
    print("\n### Interpretation\n")
    print(f"acc(full)                          = {fmt(a['full'])}")
    print(f"acc(full) - acc(text_mean)         = {fmt(a['full'] - a['text_mean'])}"
          "   <- contribution of per-card TEXT identity")
    print(f"acc(full) - acc(det_mean)          = {fmt(a['full'] - a['det_mean'])}"
          "   <- contribution of DETERMINISTIC features")
    print(f"acc(full) - acc(text_shuffle)      = {fmt(a['full'] - a['text_shuffle'])}"
          "   <- value of WHICH card carries WHICH text")
    print(f"acc(text_shuffle) - acc(text_mean) = "
          f"{fmt(a['text_shuffle'] - a['text_mean'])}"
          "   <- value of the deck-level BAG of text vectors")
    print(f"acc(both_mean)                     = {fmt(a['both_mean'])}"
          "   <- deck-size-only floor")

    n = cond["full"]["n_matches"]
    band = coin_flip_band(n)
    print(f"\n(noise band: +/-{band:.4f} = 2 SE at n={n})")

    ts, fu = a["text_shuffle"], a["full"]
    if abs(ts - fu) <= band:
        print("\n- text_shuffle ~= full (within noise): the scorer reads the "
              "deck's BAG of text vectors, not which card each vector is "
              "attached to -- the per-card text/det pairing carries little "
              "ranking signal.")
    else:
        print("\n- text_shuffle < full (beyond noise): the scorer does use the "
              "per-card pairing of text with deterministic features, not just "
              "the deck-level bag of text vectors.")

    bm = a["both_mean"]
    if abs(bm - 0.5) <= band:
        print(f"- both_mean = {fmt(bm)} ~ 0.5 (within noise): with all card "
              "content erased, deck size alone carries no held-out signal, so "
              "0.5 is the true floor and the gains above are real.")
    elif bm > 0.5:
        print(f"- both_mean = {fmt(bm)} is ABOVE 0.5 (beyond noise): nonbasic "
              "deck SIZE alone predicts the winner -- size leaks builder "
              "identity (builders differ systematically in card count, and the "
              "size the scorer prefers is the one that wins). Treat this as the "
              "real floor, not 0.5.")
    else:
        print(f"- both_mean = {fmt(bm)} is BELOW 0.5 (beyond noise): deck size "
              "alone ANTI-predicts the winner -- the scorer's size preference "
              "points the wrong way on held-out data. Also a size leak, with "
              "inverted sign; the honest floor is |acc-0.5| either way.")
    print(f"  (tie rate under both_mean = "
          f"{100 * cond['both_mean']['tie_rate']:.1f}%, scored 0.5 each; "
          "equal-size decks are exactly tied by construction, so `acc` is the "
          "meaningful column here and `acc_strict` is not.)")

    print("\n- Score fidelity: mean |dScore| vs full is "
          + ", ".join(f"{c}={fmt(cond[c]['mean_abs_score_shift'], 3)}"
                      for c in CONDITIONS[1:])
          + f"; full score std = {fmt(cond['full']['score_std'], 3)}.")


def load_matches(bank: DeckBank, files: list[Path], limit: int | None):
    """``(matches, counts)``: per kept match ``(win_idx, lose_idx, pair_label)``
    into ``bank``, and the parse/drop tallies over ``files``."""
    counts: Counter = Counter()
    pair_counts: Counter = Counter()
    matches = []
    for path in files:
        rows, undecided, malformed = pl.decided_matches(path, limit)
        counts.update(parsed=len(rows), undecided=undecided, malformed=malformed)
        print(f"{path.name}: {len(rows)} decided matches "
              f"({undecided} undecided, {malformed} malformed)")
        for m, winner in rows:
            ia, ib = bank.add(m.deck_a.cards), bank.add(m.deck_b.cards)
            if ia is None or ib is None:
                counts["dropped"] += 1
                continue
            pair = " vs ".join(sorted((m.method_a, m.method_b)))
            pair_counts[pair] += 1
            win, lose = (ia, ib) if winner == "A" else (ib, ia)
            matches.append((win, lose, pair))
    return matches, counts, pair_counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None,
                    help="max matches to parse (default: all)")
    ap.add_argument("--out", type=Path, default=None,
                    help=f"JSON output path (default {DEFAULT_OUT.name}, "
                         f"{GROUPS_OUT.name} with --groups)")
    ap.add_argument("--groups", action="store_true",
                    help="erase deterministic-feature groups over both post-cutoff corpora")
    ap.add_argument("--smoke", action="store_true",
                    help="CPU, 200 matches (unless --limit given)")
    ap.add_argument("--batch", type=int, default=BATCH)
    args = ap.parse_args()

    device = "cpu" if args.smoke else None
    limit = args.limit if args.limit is not None else (200 if args.smoke else None)
    files = POST_CUTOFF_FILES if args.groups else [MATCH_FILE]
    conditions = ["full", *GROUPS] if args.groups else CONDITIONS
    out = args.out or (GROUPS_OUT if args.groups else DEFAULT_OUT)

    print(f"loading scorer (device={device or 'auto'}) ...")
    probe = pl.Probe(device=device)
    text_dim = probe.d_model - pl.layout.FEATURE_COUNT
    print(f"d_model={probe.d_model}  text[0:{text_dim}]  "
          f"det[{text_dim}:{probe.d_model}]  device={probe.device}")

    # Decks are canonicalized to sorted order, which both dedups more and makes
    # the text_shuffle permutation reproducible.
    reg = DeckBank(probe.embedding)
    matches, counts, pair_counts = load_matches(reg, files, limit)
    print(f"kept {len(matches)} matches ({counts['dropped']} dropped for missing "
          f"embeddings); {len(reg.keys)} distinct decks; "
          f"{len(reg.emb)} distinct cards; {len(reg.missing)} cards unembeddable")
    if not matches:
        raise SystemExit("no usable matches")

    print("\nmatches per method pair:")
    for p, n in pair_counts.most_common():
        print(f"  {p:<28} {n}")

    used = {i for w, ln, _ in matches for i in (w, ln)}
    order = sorted(used)
    pos = {d: k for k, d in enumerate(order)}
    win_idx = np.array([pos[w] for w, _, _ in matches])
    lose_idx = np.array([pos[ln] for _, ln, _ in matches])
    pairs = [p for _, _, p in matches]

    mean_text, mean_det, n_mean_cards = corpus_means(reg, used, text_dim)
    print(f"\ncorpus mean over {n_mean_cards} distinct cards "
          f"(|text|={np.linalg.norm(mean_text):.3f}, "
          f"|det|={np.linalg.norm(mean_det):.3f})")

    scores: dict[str, np.ndarray] = {}
    for cond in conditions:
        print(f"scoring {len(order)} decks under condition {cond!r} ...")
        scores[cond] = score_condition(probe, reg, order, cond, text_dim,
                                       mean_text, mean_det, args.batch)

    res = {
        "meta": {
            "match_files": [str(p) for p in files],
            "matches_parsed": counts["parsed"],
            "matches_undecided": counts["undecided"],
            "matches_malformed": counts["malformed"],
            "matches_dropped_missing": counts["dropped"],
            "matches_scored": len(matches),
            "deck_instances": 2 * len(matches),
            "distinct_decks": len(order),
            "distinct_cards": len(reg.emb),
            "missing_cards": sorted(reg.missing)[:50],
            "n_missing_cards": len(reg.missing),
            "mean_over_cards": n_mean_cards,
            "d_model": probe.d_model,
            "text_dim": text_dim,
            "det_dim": pl.layout.FEATURE_COUNT,
            "device": probe.device,
            "seed": SEED,
            "limit": limit,
            "checkpoint": str(pl.SCORER_CKPT),
            "pair_counts": dict(pair_counts),
            "group_features": {g: len(GROUPS[g]) for g in conditions if g in GROUPS},
        },
        "conditions": {
            c: condition_metrics(scores[c], scores["full"],
                                 win_idx, lose_idx, pairs)
            for c in conditions
        },
    }
    write_json(out, res)
    print(f"\nwrote {out}")

    (print_groups_report if args.groups else print_report)(res)


if __name__ == "__main__":
    main()

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

Read-only w.r.t. the repo and the Y: drive.
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

import probe_lib as pl

MATCH_FILE = pl.YDATA / "matches-b07" / "match-outcomes-gen5-vs-gen4-forge.txt"
DEFAULT_OUT = pl.SCRATCH / "t5_ablation.json"
SEED = 42
BATCH = 1024

CONDITIONS = ["full", "text_mean", "text_shuffle", "det_mean", "both_mean"]
PAIR_TABLE_CONDITIONS = ["full", "text_mean", "det_mean"]


# ---------------------------------------------------------------- match corpus

def _open(path: Path):
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" \
        else open(path, "rt", encoding="utf-8")


def parse_matches(path: Path, limit: int | None = None):
    """Yield (method_A, deck_A, method_B, deck_B, winner) per match line.

    Winner = the side with more game wins in the 'games' field (index 7, a
    string like "ABAAA"). Lines with an equal count (or an empty field) carry
    no decidable winner and are dropped.
    """
    rows, undecided, malformed = [], 0, 0
    with _open(path) as f:
        for line in f:
            p = line.rstrip("\n").split(";")
            if len(p) < 10:
                malformed += 1
                continue
            a, b = p[7].count("A"), p[7].count("B")
            if a == b:
                undecided += 1
                continue
            rows.append((p[3], p[5].split("|"), p[4], p[6].split("|"),
                         "A" if a > b else "B"))
            if limit and len(rows) >= limit:
                break
    return rows, undecided, malformed


# ------------------------------------------------------------- deck collection

class DeckRegistry:
    """Distinct decks (by nonbasic-card multiset), with embedding validity.

    The scorer is permutation-invariant over the card set, so decks are
    canonicalized to sorted order before hashing; that both dedups more and
    makes the text_shuffle permutation reproducible.
    """

    def __init__(self, probe: pl.Probe):
        self.probe = probe
        self.index: dict[tuple[str, ...], int] = {}
        self.decks: list[list[str]] = []
        self.emb: dict[str, np.ndarray] = {}
        self.missing: set[str] = set()

    def _card_ok(self, name: str) -> bool:
        if name in self.emb:
            return True
        if name in self.missing:
            return False
        e = self.probe.embedding(name)
        if e is None:
            self.missing.add(name)
            return False
        self.emb[name] = e.astype(np.float32)
        return True

    def add(self, deck: list[str]) -> int | None:
        """Register a deck; None if it has an unembeddable nonbasic card."""
        nonbasic = tuple(sorted(c for c in deck
                                if c.lower() not in pl.BASIC_LAND_NAMES))
        if not nonbasic:
            return None
        hit = self.index.get(nonbasic)
        if hit is not None:
            return hit
        if not all(self._card_ok(c) for c in nonbasic):
            return None
        idx = len(self.decks)
        self.index[nonbasic] = idx
        self.decks.append(list(nonbasic))
        return idx

    def matrix(self, idx: int) -> np.ndarray:
        """Fresh (n_cards, d_model) float32 matrix for a distinct deck."""
        return np.stack([self.emb[c] for c in self.decks[idx]])


# ------------------------------------------------------------------ conditions

def corpus_means(reg: DeckRegistry, used: set[int], text_dim: int):
    """Raw-space mean vector over distinct cards of the used decks."""
    names = {c for i in used for c in reg.decks[i]}
    mean = np.stack([reg.emb[n] for n in sorted(names)]).mean(axis=0)
    return mean[:text_dim].copy(), mean[text_dim:].copy(), len(names)


def apply_condition(mat: np.ndarray, cond: str, idx: int, text_dim: int,
                    mean_text: np.ndarray, mean_det: np.ndarray) -> np.ndarray:
    """Transform one deck's raw card matrix in place (mat is already a copy)."""
    if cond == "full":
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


def score_condition(probe: pl.Probe, reg: DeckRegistry, order: list[int],
                    cond: str, text_dim: int, mean_text, mean_det,
                    batch: int = BATCH) -> np.ndarray:
    """Score every distinct deck under one condition (built batch by batch)."""
    out = np.empty(len(order), dtype=np.float64)
    for lo in range(0, len(order), batch):
        chunk = order[lo:lo + batch]
        mats = [apply_condition(reg.matrix(i), cond, i, text_dim,
                                mean_text, mean_det) for i in chunk]
        out[lo:lo + len(chunk)] = probe.score_matrices(mats, batch_size=batch)
    return out


# --------------------------------------------------------------------- metrics

def _cmp(diff: np.ndarray) -> np.ndarray:
    """Tie-aware correctness: 1.0 for >0, 0.0 for <0, 0.5 for exact ties."""
    return np.where(diff > 0, 1.0, np.where(diff < 0, 0.0, 0.5))


def condition_metrics(scores: np.ndarray, full: np.ndarray,
                      win_idx: np.ndarray, lose_idx: np.ndarray,
                      pairs: list[str]) -> dict:
    diff = scores[win_idx] - scores[lose_idx]
    correct = _cmp(diff)
    full_diff = full[win_idx] - full[lose_idx]
    agree = _cmp(diff * full_diff)  # same sign as full's ranking

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
        "per_pair": per_pair,
    }
    if np.std(scores) > 0 and np.std(full) > 0:
        m["pearson_vs_full"] = float(pearsonr(scores, full)[0])
        m["spearman_vs_full"] = float(spearmanr(scores, full)[0])
    else:  # degenerate (constant) condition scores
        m["pearson_vs_full"] = float("nan")
        m["spearman_vs_full"] = float("nan")
    return m


# ----------------------------------------------------------------- report

def fmt(x: float, nd: int = 4) -> str:
    return "nan" if x != x else f"{x:.{nd}f}"


def print_report(res: dict) -> None:
    meta, cond = res["meta"], res["conditions"]

    print("\n" + "=" * 78)
    print("T5 ABLATION SUMMARY")
    print("=" * 78)
    print(f"eval file        : {meta['match_file']}")
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

    print("\n### Condition summary\n")
    hdr = ("| condition | acc | acc(strict) | tie% | pearson vs full | "
           "spearman vs full | rank-agree w/ full | mean abs shift |")
    print(hdr)
    print("|---|---|---|---|---|---|---|---|")
    for c in CONDITIONS:
        m = cond[c]
        print(f"| {c} | {fmt(m['acc'])} | {fmt(m['acc_strict'])} | "
              f"{100 * m['tie_rate']:.1f}% | {fmt(m['pearson_vs_full'])} | "
              f"{fmt(m['spearman_vs_full'])} | {fmt(m['rank_agree_with_full'])} | "
              f"{fmt(m['mean_abs_score_shift'], 3)} |")

    print("\n### Accuracy by method pair\n")
    pairs = sorted(cond["full"]["per_pair"],
                   key=lambda p: -cond["full"]["per_pair"][p]["n"])
    print("| pair | n | " + " | ".join(PAIR_TABLE_CONDITIONS) + " |")
    print("|---|---|" + "---|" * len(PAIR_TABLE_CONDITIONS))
    for p in pairs:
        n = cond["full"]["per_pair"][p]["n"]
        cells = " | ".join(fmt(cond[c]["per_pair"][p]["acc"], 3)
                           for c in PAIR_TABLE_CONDITIONS)
        print(f"| {p} | {n} | {cells} |")

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

    # 2 standard errors of a coin-flip accuracy at this sample size: differences
    # smaller than this are not distinguishable from noise.
    n = cond["full"]["n_matches"]
    band = 2 * (0.25 / max(n, 1)) ** 0.5
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


# ------------------------------------------------------------------------ main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None,
                    help="max matches to parse (default: all)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help="JSON output path")
    ap.add_argument("--smoke", action="store_true",
                    help="CPU, 200 matches (unless --limit given)")
    ap.add_argument("--batch", type=int, default=BATCH)
    args = ap.parse_args()

    device = "cpu" if args.smoke else None
    limit = args.limit if args.limit is not None else (200 if args.smoke else None)

    print(f"loading scorer (device={device or 'auto'}) ...")
    probe = pl.Probe(device=device)
    text_dim = probe.d_model - pl.layout.FEATURE_COUNT
    print(f"d_model={probe.d_model}  text[0:{text_dim}]  "
          f"det[{text_dim}:{probe.d_model}]  device={probe.device}")

    rows, undecided, malformed = parse_matches(MATCH_FILE, limit)
    print(f"parsed {len(rows)} decided matches "
          f"({undecided} undecided, {malformed} malformed)")

    reg = DeckRegistry(probe)
    matches = []           # (win_idx, lose_idx, pair_label)
    dropped = 0
    pair_counts: Counter = Counter()
    for m_a, deck_a, m_b, deck_b, winner in rows:
        ia, ib = reg.add(deck_a), reg.add(deck_b)
        if ia is None or ib is None:
            dropped += 1
            continue
        pair = " vs ".join(sorted((m_a, m_b)))
        pair_counts[pair] += 1
        win, lose = (ia, ib) if winner == "A" else (ib, ia)
        matches.append((win, lose, pair))
    print(f"kept {len(matches)} matches ({dropped} dropped for missing "
          f"embeddings); {len(reg.decks)} distinct decks; "
          f"{len(reg.emb)} distinct cards; {len(reg.missing)} cards unembeddable")
    if not matches:
        raise SystemExit("no usable matches")

    print("\nmatches per method pair:")
    for p, n in pair_counts.most_common():
        print(f"  {p:<28} {n}")

    used = {i for w, l, _ in matches for i in (w, l)}
    order = sorted(used)
    pos = {d: k for k, d in enumerate(order)}
    win_idx = np.array([pos[w] for w, _, _ in matches])
    lose_idx = np.array([pos[l] for _, l, _ in matches])
    pairs = [p for _, _, p in matches]

    mean_text, mean_det, n_mean_cards = corpus_means(reg, used, text_dim)
    print(f"\ncorpus mean over {n_mean_cards} distinct cards "
          f"(|text|={np.linalg.norm(mean_text):.3f}, "
          f"|det|={np.linalg.norm(mean_det):.3f})")

    scores: dict[str, np.ndarray] = {}
    for cond in CONDITIONS:
        print(f"scoring {len(order)} decks under condition {cond!r} ...")
        scores[cond] = score_condition(probe, reg, order, cond, text_dim,
                                       mean_text, mean_det, args.batch)

    res = {
        "meta": {
            "match_file": str(MATCH_FILE),
            "matches_parsed": len(rows),
            "matches_undecided": undecided,
            "matches_malformed": malformed,
            "matches_dropped_missing": dropped,
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
        },
        "conditions": {
            c: condition_metrics(scores[c], scores["full"],
                                 win_idx, lose_idx, pairs)
            for c in CONDITIONS
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"\nwrote {args.out}")

    print_report(res)


if __name__ == "__main__":
    main()

"""T5b: which of the 32 deterministic features does the scorer actually read?

Extends t5_ablation.py's block-level ``det_mean`` condition to feature groups:
each condition replaces one group's slots with their corpus means (raw space,
one vote per distinct card), keeps the other deterministic features and the
text block intact, and rescores every held-out match.

Two variance reductions make the small per-group effects resolvable. Both
post-cutoff corpora are used (gen5-vs-gen4-forge and gen4-vs-forge-best-gen3,
~21.5K matches — the second was generated 2026-05-19..21, after the scorer's
2026-05-18 training run). Significance comes from the paired per-match
difference against the full model (condition-correct minus full-correct), whose
standard error is set by the prediction flip rate rather than the base
accuracy.

Writes ``t5b_results.json`` into ``output/scorer-probes/`` (consumed by
``make_figures.py``) and prints the summary table.
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_lib as pl  # noqa: E402

MATCH_FILES = [
    pl.YDATA / "matches-b07" / "match-outcomes-gen5-vs-gen4-forge.txt",
    pl.YDATA / "matches-b07" / "match-outcomes-gen4-vs-forge-best-gen3.txt",
]
TEXT_DIM = 512
OUT_JSON = pl.SCRATCH / "t5b_results.json"

# label -> indices within the trailing 32-dim deterministic block
# (see sealed.domain.card_embedding_layout)
GROUPS = {
    "is_land": [0],
    "mana cost (pips+generic+X+MV)": list(range(1, 10)),
    "color pips only": list(range(1, 7)),
    "mana value only": [9],
    "color flags": list(range(10, 16)),
    "mana production": list(range(16, 23)),
    "power/toughness/loyalty": [23, 24, 25],
    "all 32": list(range(32)),
}


def main():
    probe = pl.Probe()

    matches, decks = [], {}
    for path in MATCH_FILES:
        n0 = len(matches)
        with open(path, encoding="utf-8") as f:
            for line in f:
                p = line.rstrip("\n").split(";")
                if len(p) < 10:
                    continue
                a, b = p[7].count("A"), p[7].count("B")
                if a == b:
                    continue
                pair, ok = [], True
                for di in (5, 6):
                    names = [c for c in p[di].split("|")
                             if c.lower() not in pl.BASIC_LAND_NAMES]
                    key = "|".join(sorted(names))
                    if key not in decks:
                        try:
                            probe.deck_matrix(names)
                        except KeyError:
                            ok = False
                            break
                        decks[key] = names
                    pair.append(key)
                if ok:
                    matches.append((pair[0], pair[1], a > b))
        print(f"{path.name}: +{len(matches) - n0} matches", flush=True)
    keys = list(decks)
    n = len(matches)
    print(f"total {n} matches, {len(decks)} distinct decks", flush=True)

    mats = {k: probe.deck_matrix(decks[k]) for k in keys}
    card_rows = {}
    for k in keys:
        nonbasic = [c for c in decks[k] if c.lower() not in pl.BASIC_LAND_NAMES]
        for name, row in zip(nonbasic, mats[k]):
            card_rows.setdefault(name, row)
    det_mean = np.stack(list(card_rows.values()))[:, TEXT_DIM:].mean(axis=0)

    def run(feat_idx):
        edited = []
        for k in keys:
            m = mats[k].copy()
            if feat_idx:
                cols = [TEXT_DIM + i for i in feat_idx]
                m[:, cols] = det_mean[feat_idx]
            edited.append(m)
        return dict(zip(keys, probe.score_matrices(edited)))

    def per_match_correct(sc):
        out = np.empty(n)
        for i, (ka, kb, a_wins) in enumerate(matches):
            d = sc[ka] - sc[kb]
            out[i] = 0.5 if d == 0 else float((d > 0) == a_wins)
        return out

    full_sc = run([])
    full_c = per_match_correct(full_sc)
    full_arr = np.array([full_sc[k] for k in keys])
    print(f"\nfull model acc = {full_c.mean():.4f}  (n={n})\n", flush=True)

    rows = []
    print(f"{'group erased':32s} {'d-acc':>8s} {'paired SE':>9s} {'z':>6s} {'flip%':>6s} {'rho':>6s}")
    for name, idx in GROUPS.items():
        sc = run(idx)
        d = per_match_correct(sc) - full_c
        se = d.std(ddof=1) / np.sqrt(n)
        z = d.mean() / se if se > 0 else 0.0
        flip = float((d != 0).mean())
        rho = float(stats.spearmanr(full_arr, np.array([sc[k] for k in keys]))[0])
        rows.append({"group": name, "n_features": len(idx), "d_acc": float(d.mean()),
                     "paired_se": float(se), "z": float(z), "flip_rate": flip,
                     "spearman_vs_full": rho})
        print(f"{name:32s} {d.mean():+8.4f} {se:9.4f} {z:+6.1f} {100 * flip:6.2f} {rho:6.3f}",
              flush=True)

    OUT_JSON.write_text(json.dumps({
        "match_files": [p.name for p in MATCH_FILES],
        "n_matches": n, "n_distinct_decks": len(decks),
        "full_acc": float(full_c.mean()), "groups": rows,
    }, indent=1), encoding="utf-8")
    print("\nwrote", OUT_JSON)


if __name__ == "__main__":
    main()

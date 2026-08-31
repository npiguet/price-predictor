"""D1 report — each token block against the placebo of its own size.

`d1_channels.py` blanks a block and separately blanks ``k`` uniformly-chosen
non-``PACK`` tokens for a ladder of ``k``. Blanking a big block moves the policy
more than blanking a small one whatever the block contains, so the ladder is the
magnitude law and a block's honest effect is its position against it.

The comparison is made inside each pack, because the blocks and the state grow at
very different rates across the draft: by pack 3 a seat holds about 37 cards it
drafted, has passed about 32, and knows of about 143 taken by others.

Reads `output/draft-probes/d1_channels.json` plus the per-pack block sizes
measured here, and writes `d1_placebo_table.csv`.

Usage
-----
    python scripts/draft_probes/d1_report.py
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from probe_lib import (  # noqa: E402
    TYPE_PASSED,
    TYPE_POOL,
    TYPE_TAKEN,
    CardTable,
    iter_corpus_states,
)

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "output" / "draft-probes"
CORPUS = (REPO / "models/draft/agent/gen4"
          / "lr1e-5_t2all_decay0.3-yardstick-v-forge-drafts.jsonl")
BLOCKS = {"POOL": TYPE_POOL, "PASSED": TYPE_PASSED, "TAKEN": TYPE_TAKEN}


def main() -> None:
    data = json.loads((OUT / "d1_channels.json").read_text(encoding="utf-8"))
    n_drafts = 40  # the run this report reads

    table = CardTable()
    sizes: dict[int, dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list))
    for s in iter_corpus_states(CORPUS, table, labels=["gen4"],
                                limit_drafts=n_drafts):
        for name, tt in BLOCKS.items():
            sizes[s.state.pack_number][name].append(
                int((s.state.type_idx == tt).sum()))
    mean_size = {p: {b: float(np.mean(v)) for b, v in d.items()}
                 for p, d in sizes.items()}

    rows = []
    for gen, res in data["models"].items():
        arms = {a["arm"]: a for a in res["arms"]}
        ks = sorted(int(a.split("=")[1]) for a in arms if a.startswith("placebo"))
        for pack in (1, 2, 3):
            curve_x = np.array([0.0] + [float(k) for k in ks])
            curve_y = np.array(
                [0.0] + [arms[f"placebo k={k}"]["flip_by_pack"][str(pack)]
                         for k in ks])
            for block in BLOCKS:
                k = mean_size[pack][block]
                expected = float(np.interp(k, curve_x, curve_y))
                observed = arms[f"{block} identity"]["flip_by_pack"][str(pack)]
                rows.append({
                    "model": gen, "pack": pack, "block": block,
                    "mean_size": round(k, 1),
                    "observed_flip": round(observed, 4),
                    "placebo_flip": round(expected, 4),
                    "ratio": round(observed / expected, 2) if expected > 0
                    else float("nan"),
                })

    with (OUT / "d1_placebo_table.csv").open("w", encoding="utf-8",
                                             newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("Blanking a block, against blanking the same number of random tokens.")
    print("ratio > 1: the block carries more than its size predicts.\n")
    for gen in data["models"]:
        print(f"=== {gen} ===")
        print(f"{'pack':>5s} {'block':>7s} {'size':>6s} {'observed':>9s} "
              f"{'placebo':>8s} {'ratio':>6s}")
        for r in rows:
            if r["model"] != gen:
                continue
            print(f"{r['pack']:5d} {r['block']:>7s} {r['mean_size']:6.1f} "
                  f"{r['observed_flip']:9.3f} {r['placebo_flip']:8.3f} "
                  f"{r['ratio']:6.2f}")
        print()
    print(f"wrote {OUT / 'd1_placebo_table.csv'}")


if __name__ == "__main__":
    main()

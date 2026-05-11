"""Estimate the split-half reliability of the per-card winnability labels.

``train-encoder``'s ``val corr (pred vs target)`` numbers only mean
something relative to a ceiling. The labels in ``cards-played.txt`` are
aggregated from a finite pile of games and then shrunk toward neutral,
so even a perfect model cannot correlate with the noise in the labels
themselves. This script splits the *games* randomly into two halves,
recomputes the nine per-card shrunk labels on each half independently
(the same aggregation + ``--shrinkage-k`` the trainer uses), and reports
per head:

  r_half  Pearson r between half A's and half B's shrunk labels, over the
          cards present in both halves with a non-empty cell on that head
          (unweighted, one row per card — mirrors the trainer's val-corr).
  r_full  Spearman-Brown projection ``2 r / (1 + r)``. Each half holds
          only ~half the games, so its labels are noisier than the
          full-data labels the model actually trains against; this is the
          standard correction that projects the reliability up to the
          full game count.

Compare your latest ``val corr`` line against ``r_full``: a model corr
already near ``r_full`` means there is essentially no signal left to
chase, and a low ``r_full`` on a colour head explains a low model corr
without anything being wrong.

    python scripts/split_half_reliability.py \
        [--cards-played output/sealed/cards-played.txt] \
        [--cards-folder output/cardsfolder/] \
        [--shrinkage-k 20] [--seed 42] [--keep-missing]
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sealed.application.train_encoder import (  # noqa: E402
    _ALL_HEAD_NAMES,
    _COLOR_LIFT_OFFSET,
    _aggregate,
    _build_label_map,
    _label_value,
)
from sealed.infrastructure.cards_played_reader import (  # noqa: E402
    CardsPlayedRow,
    iter_rows,
)
from sealed.infrastructure.converted_card_locator import (  # noqa: E402
    ConvertedCardLocator,
)

DEFAULT_CARDS_PLAYED = Path("output/sealed/cards-played.txt")
DEFAULT_CARDS_FOLDER = Path("output/cardsfolder/")


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if x.std() == 0.0 or y.std() == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _spearman_brown(r_half: float | None) -> float | None:
    if r_half is None or 1.0 + r_half == 0.0:
        return None
    return 2.0 * r_half / (1.0 + r_half)


def _labels_for(
    rows: list[CardsPlayedRow],
    locator: ConvertedCardLocator,
    shrinkage_k: float,
    drop_missing: bool,
):
    counters = _aggregate(rows, locator)
    if drop_missing:
        for name in [n for n in counters if locator.text_path(n) is None]:
            del counters[name]
    return _build_label_map(counters, shrinkage_k)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--cards-played", type=Path, default=DEFAULT_CARDS_PLAYED)
    parser.add_argument("--cards-folder", type=Path, default=DEFAULT_CARDS_FOLDER)
    parser.add_argument("--shrinkage-k", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--keep-missing",
        action="store_true",
        help="keep cards with no converted .txt (default: drop them, "
        "matching train-encoder)",
    )
    args = parser.parse_args()

    if not args.cards_played.exists():
        print(f"error: {args.cards_played} not found", file=sys.stderr)
        return 1

    rng = random.Random(args.seed)
    rows_a: list[CardsPlayedRow] = []
    rows_b: list[CardsPlayedRow] = []
    for row in iter_rows(args.cards_played):
        (rows_a if rng.random() < 0.5 else rows_b).append(row)
    print(
        f"Split {len(rows_a) + len(rows_b)} games randomly (seed {args.seed}) "
        f"-> {len(rows_a)} / {len(rows_b)}"
    )

    locator = ConvertedCardLocator(args.cards_folder)
    drop_missing = not args.keep_missing
    labels_a = _labels_for(rows_a, locator, args.shrinkage_k, drop_missing)
    labels_b = _labels_for(rows_b, locator, args.shrinkage_k, drop_missing)
    common = sorted(set(labels_a) & set(labels_b))
    print(
        f"Cards with labels in both halves: {len(common)} "
        f"(half A: {len(labels_a)}, half B: {len(labels_b)}); "
        f"shrinkage-k={args.shrinkage_k:g}\n"
    )

    header = f"{'head':<16}{'n_cards':>10}{'r_half':>10}{'r_full (SB)':>14}"
    print(header)
    print("-" * len(header))
    for h, name in enumerate(_ALL_HEAD_NAMES):
        if h == _COLOR_LIFT_OFFSET:
            print()
        xs: list[float] = []
        ys: list[float] = []
        for cn in common:
            va = _label_value(labels_a[cn], h)
            vb = _label_value(labels_b[cn], h)
            if va is None or vb is None:
                continue
            xs.append(float(va))
            ys.append(float(vb))
        r = _pearson(xs, ys)
        rf = _spearman_brown(r)
        r_s = f"{r:+.3f}" if r is not None else "--"
        rf_s = f"{rf:+.3f}" if rf is not None else "--"
        print(f"{name:<16}{len(xs):>10}{r_s:>10}{rf_s:>14}")

    print(
        "\nCompare 'r_full (SB)' against the latest "
        "'val corr (pred vs target)' line from train-encoder:\n"
        "  model corr near r_full  -> at the ceiling; little signal left.\n"
        "  model corr well below   -> headroom remains.\n"
        "  low r_full (esp. colour heads) -> the label itself is noisy; "
        "a low model corr there is expected."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

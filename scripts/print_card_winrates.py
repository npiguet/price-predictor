"""Human-readable per-card score table from ``cards-played.txt``.

The counters and labels are ``train-encoder``'s own (its ``_aggregate`` and
``_build_label_map``), so this table reads the same numbers the encoder trains
against. For every card name observed in the cards-played log, count:
  * wins_when_played   — winning side played at least one copy
  * wins_when_in_deck  — card was in the winning side's deck
  * losses_when_played — losing side played at least one copy
  * losses_when_in_deck — card was in the losing side's deck

The per-card raw score is

    (wins_played - losses_played) / (wins_in_deck + losses_in_deck)

bounded between -1 and +1. The numerator captures whether playing the
card tends to coincide with winning or losing (positive vs negative
influence); the denominator down-weights cards that are rarely actually
played, so a card's magnitude reflects both its effect and how often it
shows up.

The adjusted score applies Bayesian shrinkage with a pseudocount ``k``:

    adj = (wins_played - losses_played) / (wins_in_deck + losses_in_deck + k)

Low-n cards get pulled toward 0; cards with many observations stay
nearly identical to their raw score. ``k`` is configurable via
``--shrinkage-k`` (default 20). Adj Play / Adj Draw / Lift X are the shrunk
``score_play`` / ``score_draw`` / ``color_lift_X`` labels and Lift the shrunk
``cast_lift``; a cell is blank where its slice is empty.

The quantile column reports each card's rank-quantile of the adjusted
score, with 100.0% = best, 0.0% = worst. Tied cards share the average
rank, so the column is suitable as a regression target.

Output is sorted by adjusted score descending, with raw wins-played
desc, losses-played asc, and unplayed count asc as tiebreakers. The
final tiebreaker surfaces cards that were drafted into decks but never
actually played (e.g. ``Shackles`` had 637 in-deck observations and 0
plays) — within the score=0 tier they sort to the bottom, separating
them from score=0 cards that do see occasional play.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from collections.abc import Iterator
from pathlib import Path

from scipy.stats import rankdata

from sealed.application.train_encoder import CardLabels, _aggregate, _build_label_map
from sealed.domain.encoder_model import COLOR_ORDER
from sealed.infrastructure.cards_played_reader import CardsPlayedRow, iter_rows
from sealed.infrastructure.converted_card_locator import ConvertedCardLocator

DEFAULT_CARDS_PLAYED = Path("output/sealed/cards-played.txt")
DEFAULT_CARDS_FOLDER = Path("output/cardsfolder")


def _allowed_sides(rows: Iterator[CardsPlayedRow], allowed: set[str] | None):
    """The rows with each side built by a method outside ``allowed`` emptied, so
    the aggregation counts only the allowed sides of each game."""
    for row in rows:
        if allowed is None:
            yield row
            continue
        keep_a, keep_b = row.method_a in allowed, row.method_b in allowed
        if not (keep_a or keep_b):
            continue
        if not keep_a:
            row = dataclasses.replace(row, cards_played_a=[], cards_not_played_a=[])
        if not keep_b:
            row = dataclasses.replace(row, cards_played_b=[], cards_not_played_b=[])
        yield row


def _pct(value: float | None, sign: str = "+") -> str:
    return "" if value is None else f"{value * 100:{sign}.2f}%"


def _format_table(labels: dict[str, CardLabels], mana_costs: dict[str, str],
                  shrinkage_k: float) -> str:
    rows = []
    for name, lbl in labels.items():
        c = lbl.counters
        in_deck = c.wins_when_in_deck + c.losses_when_in_deck
        net = c.wins_when_played - c.losses_when_played
        unplayed = in_deck - c.wins_when_played - c.losses_when_played
        rows.append((name, lbl, net / in_deck, net / (in_deck + shrinkage_k), unplayed))
    if len(rows) == 1:
        quantiles = [1.0]
    else:
        quantiles = list((rankdata([r[3] for r in rows]) - 1) / (len(rows) - 1))
    quantile = {r[0]: q for r, q in zip(rows, quantiles)}
    rows.sort(key=lambda r: (-r[3], -r[1].counters.wins_when_played,
                             r[1].counters.losses_when_played, r[4]))

    data_rows = [
        (
            name,
            mana_costs.get(name, ""),
            str(lbl.counters.wins_when_played),
            str(lbl.counters.wins_when_in_deck),
            str(lbl.counters.losses_when_played),
            str(lbl.counters.losses_when_in_deck),
            str(unplayed),
            f"{score:+.4f}",
            f"{adjusted:+.4f}",
            _pct(lbl.shrunk_score_play),
            _pct(lbl.shrunk_score_draw),
            *(_pct(lbl.shrunk_color_lift[color]) for color in COLOR_ORDER),
            f"{quantile[name] * 100:.1f}%",
            _pct(lbl.shrunk_cast_lift),
        )
        for name, lbl, score, adjusted, unplayed in rows
    ]
    headers = (
        "Card", "Mana Cost", "Wins Played", "Wins In Deck", "Losses Played",
        "Losses In Deck", "Unplayed", "Score", f"Adj Score (k={shrinkage_k:g})",
        "Adj Play", "Adj Draw", *(f"Lift {color}" for color in COLOR_ORDER),
        "Quantile", "Lift",
    )
    widths = [max(len(cell) for cell in column) for column in zip(headers, *data_rows)]
    fmt = "  ".join(f"{{:<{w}}}" if i < 2 else f"{{:>{w}}}" for i, w in enumerate(widths))
    lines = [fmt.format(*headers), "  ".join("-" * w for w in widths)]
    lines.extend(fmt.format(*row) for row in data_rows)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    # Card names carry macrons and accents that cp1252, Windows' default
    # codec for a redirected stream, cannot encode.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description=(
            "Print a per-card score table from cards-played.txt. Sorted by "
            "the Bayesian-shrunk adjusted score descending; also reports the "
            "raw score and the rank quantile of the adjusted score."
        ),
    )
    parser.add_argument(
        "cards_played", nargs="?", type=Path, default=DEFAULT_CARDS_PLAYED,
        help=f"Path to cards-played.txt (default: {DEFAULT_CARDS_PLAYED})",
    )
    parser.add_argument(
        "--cards-folder", type=Path, default=DEFAULT_CARDS_FOLDER,
        help=(
            "Converted card corpus root for mana-cost lookup "
            f"(default: {DEFAULT_CARDS_FOLDER})"
        ),
    )
    parser.add_argument(
        "--shrinkage-k", type=float, default=20.0,
        help=(
            "Pseudocount added to the adjusted-score denominator "
            "(default: 20). Higher values pull low-observation cards "
            "more aggressively toward 0."
        ),
    )
    parser.add_argument(
        "--methods", type=str, default=None,
        help=(
            "Comma-separated list of deck-generation method labels to "
            "restrict the analysis to (e.g. 'gen3-256,forge-best'). When "
            "present, only count card observations from decks built by "
            "the listed methods; per-side, so a game between an allowed "
            "method and a non-allowed method still contributes the "
            "allowed side. Default: include all decks."
        ),
    )
    args = parser.parse_args(argv)

    if not args.cards_played.exists():
        print(f"Error: {args.cards_played} does not exist", file=sys.stderr)
        return 1

    allowed_methods: set[str] | None = None
    if args.methods is not None:
        allowed_methods = {m.strip() for m in args.methods.split(",") if m.strip()}
        if not allowed_methods:
            print(
                "Error: --methods was given but parsed to an empty set",
                file=sys.stderr,
            )
            return 1

    locator = ConvertedCardLocator(args.cards_folder)
    counters = _aggregate(_allowed_sides(iter_rows(args.cards_played), allowed_methods),
                          locator)
    if not counters:
        suffix = (f" matching --methods {sorted(allowed_methods)}"
                  if allowed_methods is not None else "")
        print(f"No card-play data found in {args.cards_played}{suffix}", file=sys.stderr)
        return 0

    mana_costs: dict[str, str] = {}
    missing = 0
    for name in counters:
        text = locator.load_text(name)
        if text is None:
            missing += 1
        else:
            mana_costs[name] = text.mana_cost_line() or ""
    if missing:
        print(
            f"Note: {missing}/{len(counters)} card(s) had no converted .txt "
            f"under {args.cards_folder}; mana cost left blank for those.",
            file=sys.stderr,
        )

    print(_format_table(_build_label_map(counters, args.shrinkage_k), mana_costs,
                        args.shrinkage_k))
    return 0


if __name__ == "__main__":
    sys.exit(main())

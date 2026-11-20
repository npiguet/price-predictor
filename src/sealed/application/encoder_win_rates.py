"""The ``cards-win-rates.txt`` writer: the per-card label snapshot
``train-encoder`` writes for inspection (FR-013a).
"""

from __future__ import annotations

from pathlib import Path

from sealed.application.encoder_labels import CardLabelMap, CardLabels
from sealed.domain.encoder_model import COLOR_ORDER

_WIN_RATES_FILENAME: str = "cards-win-rates.txt"


# ── cards-win-rates.txt writer (FR-013a, Decision D-15) ────────────────


def _format_cell(value: float | None) -> str:
    return "" if value is None else f"{value:.5f}"


_WIN_RATES_HEADER_FIELDS: tuple[str, ...] = (
    "card_name",
    "wins_when_played", "wins_when_in_deck",
    "losses_when_played", "losses_when_in_deck",
    "raw_score_play", "shrunk_score_play",
    "raw_score_draw", "shrunk_score_draw",
    "raw_played_rate", "shrunk_played_rate",
    "raw_cast_lift", "shrunk_cast_lift",
    "raw_color_lift_W", "shrunk_color_lift_W",
    "raw_color_lift_U", "shrunk_color_lift_U",
    "raw_color_lift_B", "shrunk_color_lift_B",
    "raw_color_lift_R", "shrunk_color_lift_R",
    "raw_color_lift_G", "shrunk_color_lift_G",
)
_WIN_RATES_HEADER: str = ";".join(_WIN_RATES_HEADER_FIELDS)


def _sort_key(lbl: CardLabels) -> tuple[int, float, str]:
    """Sort by ``shrunk_score_play`` descending, empty cells at the end.

    Returned tuple is consumed with ``reverse=False``: the leading
    integer puts non-empty values first; the negated float reverses
    descending order; the trailing card name breaks ties stably.
    """
    if lbl.shrunk_score_play is None:
        return (1, 0.0, lbl.card_name)
    return (0, -lbl.shrunk_score_play, lbl.card_name)


def _write_win_rates(labels: CardLabelMap, path: Path) -> None:
    """Write the per-card label snapshot in the FR-013a 24-column schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(labels.values(), key=_sort_key)
    lines = [_WIN_RATES_HEADER]
    for lbl in rows:
        c = lbl.counters
        fields_out = [
            lbl.card_name,
            str(c.wins_when_played), str(c.wins_when_in_deck),
            str(c.losses_when_played), str(c.losses_when_in_deck),
            _format_cell(lbl.raw_score_play),
            _format_cell(lbl.shrunk_score_play),
            _format_cell(lbl.raw_score_draw),
            _format_cell(lbl.shrunk_score_draw),
            _format_cell(lbl.raw_played_rate),
            _format_cell(lbl.shrunk_played_rate),
            _format_cell(lbl.raw_cast_lift),
            _format_cell(lbl.shrunk_cast_lift),
        ]
        for color in COLOR_ORDER:
            fields_out.append(_format_cell(lbl.raw_color_lift[color]))
            fields_out.append(_format_cell(lbl.shrunk_color_lift[color]))
        lines.append(";".join(fields_out))
    path.write_text("\n".join(lines), encoding="utf-8")

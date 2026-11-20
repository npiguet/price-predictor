"""The nine raw and nine shrunk per-card labels ``train-encoder`` computes
from the aggregated counters (FR-011, FR-012).
"""

from __future__ import annotations

from dataclasses import dataclass

from sealed.application.encoder_counters import CardCounters
from sealed.domain.encoder_model import COLOR_ORDER


@dataclass(frozen=True)
class CardLabels:
    """Nine raw + nine shrunk labels per card.

    ``None`` cells signal that the head's slice denominator was zero
    (FR-012). Such cells are written as the empty string in
    ``cards-win-rates.txt`` and contribute zero loss at training time
    via the per-card head_mask. ``raw_color_lift`` and
    ``shrunk_color_lift`` are dicts keyed by WUBRG letter so the table
    layout stays explicit; mapping to tensor positions happens in
    ``_WinnabilityDataset``.
    """

    card_name: str
    counters: CardCounters
    raw_score_play: float | None
    shrunk_score_play: float | None
    raw_score_draw: float | None
    shrunk_score_draw: float | None
    raw_played_rate: float | None
    shrunk_played_rate: float | None
    raw_cast_lift: float | None
    shrunk_cast_lift: float | None
    raw_color_lift: dict[str, float | None]
    shrunk_color_lift: dict[str, float | None]


CardLabelMap = dict[str, CardLabels]


# ── Label arithmetic (FR-011 / FR-012) ──────────────────────────────────


def _safe_div(num: float, denom: float) -> float | None:
    return num / denom if denom > 0 else None


def _score_play(c: CardCounters, k: float) -> tuple[float | None, float | None]:
    """``score_play``: the signed played score over the ``@play`` games."""
    in_deck_play = c.wins_when_in_deck_at_play + c.losses_when_in_deck_at_play
    score_play_num = c.wins_when_played_at_play - c.losses_when_played_at_play
    raw_score_play = _safe_div(score_play_num, in_deck_play)
    shrunk_score_play = (
        None if in_deck_play == 0
        else score_play_num / (in_deck_play + k)
    )
    return raw_score_play, shrunk_score_play


def _score_draw(c: CardCounters, k: float) -> tuple[float | None, float | None]:
    """``score_draw``: the same over the ``@draw`` games (= primary - ``@play``)."""
    wins_in_deck_draw = c.wins_when_in_deck - c.wins_when_in_deck_at_play
    losses_in_deck_draw = c.losses_when_in_deck - c.losses_when_in_deck_at_play
    wins_played_draw = c.wins_when_played - c.wins_when_played_at_play
    losses_played_draw = c.losses_when_played - c.losses_when_played_at_play
    in_deck_draw = wins_in_deck_draw + losses_in_deck_draw
    score_draw_num = wins_played_draw - losses_played_draw
    raw_score_draw = _safe_div(score_draw_num, in_deck_draw)
    shrunk_score_draw = (
        None if in_deck_draw == 0
        else score_draw_num / (in_deck_draw + k)
    )
    return raw_score_draw, shrunk_score_draw


def _played_rate(c: CardCounters, k: float) -> tuple[float, float]:
    """``played_rate``: the share of in-deck games the card was played in."""
    in_deck_total = c.wins_when_in_deck + c.losses_when_in_deck
    played_total = c.wins_when_played + c.losses_when_played
    raw_played_rate = played_total / in_deck_total
    shrunk_played_rate = (played_total + k / 2.0) / (in_deck_total + k)
    return raw_played_rate, shrunk_played_rate


def _cast_lift(c: CardCounters, k: float) -> tuple[float | None, float | None]:
    """``cast_lift``: win rate when played minus win rate when not played."""
    half_k = k / 2.0
    played_total = c.wins_when_played + c.losses_when_played
    wins_not_played = c.wins_when_in_deck - c.wins_when_played
    losses_not_played = c.losses_when_in_deck - c.losses_when_played
    not_played_total = wins_not_played + losses_not_played
    if min(played_total, not_played_total) == 0:
        return None, None
    played_rate = c.wins_when_played / played_total
    not_played_rate = wins_not_played / not_played_total
    raw_cast_lift = played_rate - not_played_rate
    shrunk_played = (
        (c.wins_when_played + half_k) / (played_total + k)
    )
    shrunk_not_played = (
        (wins_not_played + half_k) / (not_played_total + k)
    )
    return raw_cast_lift, shrunk_played - shrunk_not_played


def _color_lifts(
    c: CardCounters, k: float,
) -> tuple[dict[str, float | None], dict[str, float | None]]:
    """``color_lift_X`` per WUBRG color: the signed played score in decks
    of that color minus the card's overall (raw and shrunk) signed score.
    """
    in_deck_total = c.wins_when_in_deck + c.losses_when_in_deck
    overall_num = c.wins_when_played - c.losses_when_played
    raw_overall = overall_num / in_deck_total
    shrunk_overall = overall_num / (in_deck_total + k)

    raw_color_lift: dict[str, float | None] = {}
    shrunk_color_lift: dict[str, float | None] = {}
    for idx, color in enumerate(COLOR_ORDER):
        in_deck_color = (
            c.wins_when_in_deck_with[idx]
            + c.losses_when_in_deck_with[idx]
        )
        played_with_color_num = (
            c.wins_when_played_with[idx]
            - c.losses_when_played_with[idx]
        )
        if in_deck_color == 0:
            raw_color_lift[color] = None
            shrunk_color_lift[color] = None
            continue
        raw_color_lift[color] = (
            played_with_color_num / in_deck_color - raw_overall
        )
        shrunk_color_lift[color] = (
            played_with_color_num / (in_deck_color + k) - shrunk_overall
        )
    return raw_color_lift, shrunk_color_lift


def _card_labels(name: str, c: CardCounters, k: float) -> CardLabels:
    """The nine raw and nine shrunk labels of one card seen in a deck."""
    raw_score_play, shrunk_score_play = _score_play(c, k)
    raw_score_draw, shrunk_score_draw = _score_draw(c, k)
    raw_played_rate, shrunk_played_rate = _played_rate(c, k)
    raw_cast_lift, shrunk_cast_lift = _cast_lift(c, k)
    raw_color_lift, shrunk_color_lift = _color_lifts(c, k)
    return CardLabels(
        card_name=name,
        counters=c,
        raw_score_play=raw_score_play,
        shrunk_score_play=shrunk_score_play,
        raw_score_draw=raw_score_draw,
        shrunk_score_draw=shrunk_score_draw,
        raw_played_rate=raw_played_rate,
        shrunk_played_rate=shrunk_played_rate,
        raw_cast_lift=raw_cast_lift,
        shrunk_cast_lift=shrunk_cast_lift,
        raw_color_lift=raw_color_lift,
        shrunk_color_lift=shrunk_color_lift,
    )


def _build_label_map(
    counters: dict[str, CardCounters], shrinkage_k: float,
) -> CardLabelMap:
    """Compute 9 raw + 9 shrunk labels per card (FR-011, FR-012).

    Cards with zero total in-deck observations are excluded entirely.
    Cells whose slice denominator is zero are stored as ``None`` in
    both raw and shrunk columns; downstream code (the writer and the
    dataset) handles the empty cell as "no signal" and zeroes its
    loss contribution.
    """
    k = float(shrinkage_k)
    return {
        name: _card_labels(name, c, k)
        for name, c in counters.items()
        if c.wins_when_in_deck + c.losses_when_in_deck != 0
    }

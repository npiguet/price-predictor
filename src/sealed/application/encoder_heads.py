"""The nine regression heads of ``train-encoder``: their order (tensor index
to head name), and per card the value each head is trained toward and the
FR-017a sample weight it carries.
"""

from __future__ import annotations

from sealed.application.encoder_labels import CardLabels
from sealed.domain.encoder_model import COLOR_ORDER

# Head ordering — tensor index → head name. Matches the column order in
# ``cards-win-rates.txt`` and the FR-017 / FR-017a head list.
_SIGNED_HEAD_NAMES: tuple[str, ...] = (
    "score_play", "score_draw", "played_rate", "cast_lift",
)
_COLOR_LIFT_HEAD_NAMES: tuple[str, ...] = tuple(
    f"color_lift_{c}" for c in COLOR_ORDER
)
_ALL_HEAD_NAMES: tuple[str, ...] = _SIGNED_HEAD_NAMES + _COLOR_LIFT_HEAD_NAMES
N_HEADS: int = len(_ALL_HEAD_NAMES)  # 9
_COLOR_LIFT_OFFSET: int = len(_SIGNED_HEAD_NAMES)  # 4


# ── Per-card label / weight / mask tensors ──────────────────────────────


def _label_value(lbl: CardLabels, head_idx: int) -> float | None:
    if head_idx == 0:
        return lbl.shrunk_score_play
    if head_idx == 1:
        return lbl.shrunk_score_draw
    if head_idx == 2:
        return lbl.shrunk_played_rate
    if head_idx == 3:
        return lbl.shrunk_cast_lift
    color = COLOR_ORDER[head_idx - _COLOR_LIFT_OFFSET]
    return lbl.shrunk_color_lift[color]


def _per_head_weight(lbl: CardLabels, head_idx: int, k: float) -> float:
    """FR-017a per-head sample weight for one card.

    Returns 0.0 when the head's slice denominator is zero (which is
    exactly the condition that makes the cell empty per FR-012).
    """
    c = lbl.counters
    if head_idx == 0:
        n = c.wins_when_in_deck_at_play + c.losses_when_in_deck_at_play
    elif head_idx == 1:
        n = (
            (c.wins_when_in_deck - c.wins_when_in_deck_at_play)
            + (c.losses_when_in_deck - c.losses_when_in_deck_at_play)
        )
    elif head_idx == 2:
        n = c.wins_when_in_deck + c.losses_when_in_deck
    elif head_idx == 3:
        played_total = c.wins_when_played + c.losses_when_played
        in_deck_total = c.wins_when_in_deck + c.losses_when_in_deck
        n = min(played_total, in_deck_total - played_total)
    else:
        idx = head_idx - _COLOR_LIFT_OFFSET  # 0..4 == _COLOR_INDEX value
        n = c.wins_when_in_deck_with[idx] + c.losses_when_in_deck_with[idx]
    if n <= 0:
        return 0.0
    return n / (n + k)

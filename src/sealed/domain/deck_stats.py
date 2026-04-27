"""Compute hand-computed deck-level statistics for the scorer.

Returns a fixed-length 23-feature vector summarising a deck of nonland cards.
Concatenated to the pooled deck representation in ``SetTransformerScorer.forward``
to give the scoring MLP crisp threshold/bucket information that the transformer's
pooled output cannot easily express.

Layout (see ``specs/sealed-deck-picker.md`` § "Hand-computed deck statistics"):

    [0:8]   mana value histogram (buckets MV 0, 1, 2, 3, 4, 5, 6, 7+)
    [8]     color count (number of color identities the deck is committed to)
    [9:15]  cards per color (W, U, B, R, G, C)
    [15:21] total pips per color (W, U, B, R, G, C)
    [21]    creature count
    [22]    noncreature count

All slots are scaled by analytical divisors before return: 23 (deck size in nonland
cards) for everything except color count, which is scaled by 6 (max color identities
WUBRGC). Scaling is fixed at definition time rather than corpus-derived because deck
distributions shift across self-play generations.
"""

from __future__ import annotations

import numpy as np
import torch

from price_predictor.domain.entities import Card
from price_predictor.domain.value_objects import WUBRG

DECK_STATS_DIM = 23

_DECK_SIZE_NONLAND = 23  # used as the scaling divisor for all count features
_COLOR_COUNT_MAX = 6     # WUBRGC — used as the scaling divisor for color count

_MV_BUCKETS = 8  # MV 0, 1, 2, 3, 4, 5, 6, 7+

# Slice offsets within the final 23-feature vector
_MV_HISTOGRAM = slice(0, 8)
_COLOR_COUNT = 8
_CARDS_PER_COLOR = slice(9, 15)
_PIPS_PER_COLOR = slice(15, 21)
_CREATURE_COUNT = 21
_NONCREATURE_COUNT = 22

# Per-color slot positions within the cards-per-color and pips-per-color blocks
_W_OFFSET, _U_OFFSET, _B_OFFSET, _R_OFFSET, _G_OFFSET, _C_OFFSET = range(6)

# Layout of the per-card "additive contribution" vector used by the vectorized
# inference path. Color count is omitted (it is non-additive — derived from the
# cards-per-color sums via threshold-and-count) and recomputed in
# ``aggregate_contributions``.
CONTRIBUTION_DIM = 22

_C_MV_HISTOGRAM = slice(0, 8)
_C_CARDS_PER_COLOR = slice(8, 14)
_C_PIPS_PER_COLOR = slice(14, 20)
_C_CREATURE = 20
_C_NONCREATURE = 21


def compute_deck_stats(cards: list[Card]) -> np.ndarray:
    """Compute a 23-feature deck-stats vector from a list of nonland cards.

    Returns a ``(DECK_STATS_DIM,)`` float32 array with all slots already scaled
    by the analytical divisors. Cards with parse-failure or no-mana-cost are
    treated leniently: they still contribute to the colorless cards-per-color
    slot (matching ``Card.is_colorless()`` semantics) and to the noncreature
    count, but contribute nothing to the MV histogram or pip totals.
    """
    feats = np.zeros(DECK_STATS_DIM, dtype=np.float32)

    color_w_active = False
    color_u_active = False
    color_b_active = False
    color_r_active = False
    color_g_active = False
    color_c_active = False

    for card in cards:
        # Mana value histogram — only contributed by cards with a parsed cost.
        if card.mana_cost is not None:
            mv = int(card.mana_cost.total_mana_value)
            bucket = mv if mv < _MV_BUCKETS - 1 else _MV_BUCKETS - 1
            feats[_MV_HISTOGRAM.start + bucket] += 1.0

        # Cards per color. Colored cards and devoid/no-cost cards are mutually
        # exclusive: is_colorless() returns True for devoid cards and cards with
        # no mana cost regardless of their printed pips.
        if card.is_colorless():
            feats[_CARDS_PER_COLOR.start + _C_OFFSET] += 1.0
            color_c_active = True  # tentatively; refined by {C}-pip check below
        else:
            mc = card.mana_cost
            if mc.w > 0:
                feats[_CARDS_PER_COLOR.start + _W_OFFSET] += 1.0
                color_w_active = True
            if mc.u > 0:
                feats[_CARDS_PER_COLOR.start + _U_OFFSET] += 1.0
                color_u_active = True
            if mc.b > 0:
                feats[_CARDS_PER_COLOR.start + _B_OFFSET] += 1.0
                color_b_active = True
            if mc.r > 0:
                feats[_CARDS_PER_COLOR.start + _R_OFFSET] += 1.0
                color_r_active = True
            if mc.g > 0:
                feats[_CARDS_PER_COLOR.start + _G_OFFSET] += 1.0
                color_g_active = True

        # Pips per color. {C} pips count for the C slot; generic mana never does.
        # Devoid cards and no-cost cards still contribute their printed pips
        # (devoid changes color identity, not the pip composition of the cost).
        if card.mana_cost is not None:
            mc = card.mana_cost
            feats[_PIPS_PER_COLOR.start + _W_OFFSET] += float(mc.w)
            feats[_PIPS_PER_COLOR.start + _U_OFFSET] += float(mc.u)
            feats[_PIPS_PER_COLOR.start + _B_OFFSET] += float(mc.b)
            feats[_PIPS_PER_COLOR.start + _R_OFFSET] += float(mc.r)
            feats[_PIPS_PER_COLOR.start + _G_OFFSET] += float(mc.g)
            feats[_PIPS_PER_COLOR.start + _C_OFFSET] += float(mc.colorless_mana)

        # Creature vs noncreature.
        if any(t == "Creature" for t in card.types):
            feats[_CREATURE_COUNT] += 1.0
        else:
            feats[_NONCREATURE_COUNT] += 1.0

    # Color count: a colorless card alone (no {C} pips) does NOT activate the C
    # slot — only an actual {C} pip does. Recompute the C activation here from
    # the pip total, overriding the tentative flag above.
    color_c_active = feats[_PIPS_PER_COLOR.start + _C_OFFSET] > 0.0
    feats[_COLOR_COUNT] = float(
        sum(
            (
                color_w_active,
                color_u_active,
                color_b_active,
                color_r_active,
                color_g_active,
                color_c_active,
            ),
        ),
    )

    # Apply analytical scaling.
    feats[_MV_HISTOGRAM] /= _DECK_SIZE_NONLAND
    feats[_COLOR_COUNT] /= _COLOR_COUNT_MAX
    feats[_CARDS_PER_COLOR] /= _DECK_SIZE_NONLAND
    feats[_PIPS_PER_COLOR] /= _DECK_SIZE_NONLAND
    feats[_CREATURE_COUNT] /= _DECK_SIZE_NONLAND
    feats[_NONCREATURE_COUNT] /= _DECK_SIZE_NONLAND

    return feats


def compute_per_card_contributions(cards: list[Card]) -> np.ndarray:
    """Per-card additive contribution vectors for vectorized deck-stats aggregation.

    Returns a ``(len(cards), CONTRIBUTION_DIM)`` float32 array. Each row encodes
    the per-card contribution to every additive deck-stats feature (everything
    except color count, which is non-additive and computed in
    ``aggregate_contributions``).

    Used by inference paths (``GreedyDeckBuilder``, ``score_decks``) that need
    to compute deck stats for many candidate decks drawn from a fixed pool.
    Pre-compute this once for the pool, then sum-and-aggregate per candidate.
    """
    contribs = np.zeros((len(cards), CONTRIBUTION_DIM), dtype=np.float32)
    for i, card in enumerate(cards):
        # MV histogram one-hot (cards with no parsed cost contribute nothing).
        if card.mana_cost is not None:
            mv = int(card.mana_cost.total_mana_value)
            bucket = mv if mv < _MV_BUCKETS - 1 else _MV_BUCKETS - 1
            contribs[i, _C_MV_HISTOGRAM.start + bucket] = 1.0

        # Cards-per-color one-hot.
        if card.is_colorless():
            contribs[i, _C_CARDS_PER_COLOR.start + _C_OFFSET] = 1.0
        else:
            mc = card.mana_cost
            if mc.w > 0:
                contribs[i, _C_CARDS_PER_COLOR.start + _W_OFFSET] = 1.0
            if mc.u > 0:
                contribs[i, _C_CARDS_PER_COLOR.start + _U_OFFSET] = 1.0
            if mc.b > 0:
                contribs[i, _C_CARDS_PER_COLOR.start + _B_OFFSET] = 1.0
            if mc.r > 0:
                contribs[i, _C_CARDS_PER_COLOR.start + _R_OFFSET] = 1.0
            if mc.g > 0:
                contribs[i, _C_CARDS_PER_COLOR.start + _G_OFFSET] = 1.0

        # Pips-per-color (devoid changes color identity, not pip composition).
        if card.mana_cost is not None:
            mc = card.mana_cost
            contribs[i, _C_PIPS_PER_COLOR.start + _W_OFFSET] = float(mc.w)
            contribs[i, _C_PIPS_PER_COLOR.start + _U_OFFSET] = float(mc.u)
            contribs[i, _C_PIPS_PER_COLOR.start + _B_OFFSET] = float(mc.b)
            contribs[i, _C_PIPS_PER_COLOR.start + _R_OFFSET] = float(mc.r)
            contribs[i, _C_PIPS_PER_COLOR.start + _G_OFFSET] = float(mc.g)
            contribs[i, _C_PIPS_PER_COLOR.start + _C_OFFSET] = float(mc.colorless_mana)

        # Creature vs noncreature.
        if any(t == "Creature" for t in card.types):
            contribs[i, _C_CREATURE] = 1.0
        else:
            contribs[i, _C_NONCREATURE] = 1.0
    return contribs


def aggregate_contributions(summed: torch.Tensor) -> torch.Tensor:
    """Convert summed per-card contributions into the final scaled deck-stats vector.

    Args:
        summed: shape ``(..., CONTRIBUTION_DIM)`` — sum of per-card contributions
            across the cards in each deck. Any leading batch dims are preserved.

    Returns:
        shape ``(..., DECK_STATS_DIM)`` — fully scaled deck stats with color
        count derived per-color from the appropriate signals (non-additive).
    """
    # Color activation rule: WUBRG slots are active if any card uses that color
    # (cards-per-color > 0). The colorless slot (C) is active only if any card
    # requires a {C} pip — generic mana never activates it. This matches the rule
    # in ``compute_deck_stats`` and the spec's description.
    cards_w_to_g = summed[..., _C_CARDS_PER_COLOR.start : _C_CARDS_PER_COLOR.start + 5]
    pips_c = summed[..., _C_PIPS_PER_COLOR.start + _C_OFFSET]
    wubrg_active = (cards_w_to_g > 0).to(summed.dtype).sum(dim=-1)  # (...)
    c_active = (pips_c > 0).to(summed.dtype)                          # (...)
    color_count = wubrg_active + c_active

    out_shape = summed.shape[:-1] + (DECK_STATS_DIM,)
    out = torch.zeros(out_shape, dtype=summed.dtype, device=summed.device)

    out[..., _MV_HISTOGRAM] = summed[..., _C_MV_HISTOGRAM] / _DECK_SIZE_NONLAND
    out[..., _COLOR_COUNT] = color_count / _COLOR_COUNT_MAX
    out[..., _CARDS_PER_COLOR] = summed[..., _C_CARDS_PER_COLOR] / _DECK_SIZE_NONLAND
    out[..., _PIPS_PER_COLOR] = summed[..., _C_PIPS_PER_COLOR] / _DECK_SIZE_NONLAND
    out[..., _CREATURE_COUNT] = summed[..., _C_CREATURE] / _DECK_SIZE_NONLAND
    out[..., _NONCREATURE_COUNT] = summed[..., _C_NONCREATURE] / _DECK_SIZE_NONLAND

    return out


# Re-export WUBRG so callers can iterate colors without reaching into price_predictor
# from sealed-side code.
__all__ = [
    "CONTRIBUTION_DIM",
    "DECK_STATS_DIM",
    "WUBRG",
    "aggregate_contributions",
    "compute_deck_stats",
    "compute_per_card_contributions",
]

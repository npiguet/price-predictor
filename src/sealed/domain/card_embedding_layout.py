"""Card embedding layout: shape constants for the (text || deterministic) vector.

Each card embedding the sealed pipeline produces is the concatenation of:

- the encoder's pooled **text** vector, then
- a ``FEATURE_COUNT`` slice of deterministic game features parsed from the
  card script (mana cost, types, P/T, etc.) — always the *trailing* block.

The text-vector width depends on the encoder: ``2 * encoder_d_model`` for the
price-predictor encoder and the sealed encoder's dual pool, ``encoder_d_model``
for the sealed encoder's attention-only pool (``--pool-mode attn``). The
``text_dim`` / ``total_dim`` helpers below give the *dual-pool* arithmetic and
serve as the legacy default for ``ScorerConfig.d_model``; the **live** total
width is always read from the loaded ``.npz`` cache (or the encoder config),
never assumed from these helpers.

The deterministic feature vector has a fixed named layout (see constants
below). Color-indexed slots follow WUBRG order throughout. The accessors
``is_land_embedding`` / ``card_colors`` index that block from the *end*, so
they work for an embedding of any total width.
"""

from __future__ import annotations

from price_predictor.domain.value_objects import WUBRG

FEATURE_COUNT = 32
DET_FEATURE_DIM = FEATURE_COUNT  # legacy alias; kept so existing .npz files load unchanged

IS_LAND: int = 0
COLOR_PIPS: slice = slice(1, 6)          # W, U, B, R, G pip counts
COLORLESS_PIP: int = 6
GENERIC: int = 7
X_COUNT: int = 8
MANA_VALUE: int = 9
COLOR_FLAGS: slice = slice(10, 15)       # is_W .. is_G (binary, zeroed when devoid)
IS_COLORLESS: int = 15
PRODUCES_COLORS: slice = slice(16, 21)   # produces W .. produces G (binary)
PRODUCES_COLORLESS: int = 21
MANA_PRODUCED: int = 22
POWER: int = 23
TOUGHNESS: int = 24
LOYALTY: int = 25
PADDING: slice = slice(26, 32)           # reserved zero padding


def text_dim(encoder_d_model: int) -> int:
    """Pooled-text width for a *dual-pool* encoder (``2 * d_model``).

    Not valid for ``--pool-mode attn`` encoders (those produce ``d_model``);
    the live width is read from the ``.npz`` cache, not from this function.
    """
    return 2 * encoder_d_model


def total_dim(encoder_d_model: int) -> int:
    """Card-embedding width for a *dual-pool* encoder (= ``text_dim`` + features).

    Used as the legacy default for ``ScorerConfig.d_model``; ``train-scorer``
    derives the actual value from the embedding cache.
    """
    return text_dim(encoder_d_model) + FEATURE_COUNT


def is_land_embedding(embedding) -> bool:
    """Return True if the embedding's IS_LAND deterministic-feature flag is set.

    Reads the leading slot of the trailing ``FEATURE_COUNT``-wide deterministic
    block. Works for any embedding shape; uses a 0.5 threshold so float noise
    on the boolean flag doesn't matter in practice (legit embeddings carry
    exactly 0.0 or 1.0 here).
    """
    return float(embedding[-FEATURE_COUNT + IS_LAND]) > 0.5


def card_colors(embedding) -> frozenset[str]:
    """Return the set of WUBRG colors flagged by ``COLOR_FLAGS``.

    Empty for colorless and Devoid cards; ``COLOR_FLAGS`` is an approximation
    of color identity derived from cast-cost pips with the Devoid override
    zeroing the slots — see ``deterministic_features._fill_color_flags``.
    Hybrid mana sets multiple flags. Rules-text-only colors are not captured.
    """
    block_start = -FEATURE_COUNT + COLOR_FLAGS.start
    block_stop = -FEATURE_COUNT + COLOR_FLAGS.stop
    flags = embedding[block_start:block_stop]
    return frozenset(
        color for color, flag in zip(WUBRG, flags) if float(flag) > 0.5
    )

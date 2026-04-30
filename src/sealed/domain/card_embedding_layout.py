"""Card embedding layout: shape constants for the (text || deterministic) vector.

Each card embedding the sealed pipeline produces is the concatenation of:

- a ``text_dim`` slice from the price-predictor encoder (``2 * encoder_d_model``,
  one max-pool plus one mean-pool over the encoder's token outputs), and
- a ``FEATURE_COUNT`` slice of deterministic game features parsed from the
  card script (mana cost, types, P/T, etc.).

Centralising the shape arithmetic here means changing the encoder's
``d_model`` only requires retraining and re-encoding — the slice indices
that depend on it are derived, not literal.

The deterministic feature vector has a fixed named layout (see constants
below). Color-indexed slots follow WUBRG order throughout.
"""

from __future__ import annotations

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
    return 2 * encoder_d_model


def total_dim(encoder_d_model: int) -> int:
    return text_dim(encoder_d_model) + FEATURE_COUNT


def is_land_embedding(embedding) -> bool:
    """Return True if the embedding's IS_LAND deterministic-feature flag is set.

    Reads the leading slot of the trailing ``FEATURE_COUNT``-wide deterministic
    block. Works for any embedding shape; uses a 0.5 threshold so float noise
    on the boolean flag doesn't matter in practice (legit embeddings carry
    exactly 0.0 or 1.0 here).
    """
    return float(embedding[-FEATURE_COUNT + IS_LAND]) > 0.5

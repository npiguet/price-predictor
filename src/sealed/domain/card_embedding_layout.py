"""Card embedding layout: shape constants for the (text || deterministic) vector.

Each card embedding the sealed pipeline produces is the concatenation of:

- a ``text_dim`` slice from the price-predictor encoder (``2 * encoder_d_model``,
  one max-pool plus one mean-pool over the encoder's token outputs), and
- a ``DET_FEATURE_DIM`` slice of deterministic game features parsed from the
  card script (mana cost, types, P/T, etc.).

Centralising the shape arithmetic here means changing the encoder's
``d_model`` only requires retraining and re-encoding — the slice indices
that depend on it are derived, not literal.
"""

from __future__ import annotations

DET_FEATURE_DIM = 32


def text_dim(encoder_d_model: int) -> int:
    return 2 * encoder_d_model


def total_dim(encoder_d_model: int) -> int:
    return text_dim(encoder_d_model) + DET_FEATURE_DIM

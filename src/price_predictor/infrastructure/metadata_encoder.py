"""Encode PrintingData into a compact numeric tensor for the transformer side-channel."""

from __future__ import annotations

import math

import torch

from price_predictor.domain.value_objects import RECOGNIZED_FORMATS, PrintingData

_RARITY_ORDINAL: dict[str, float] = {
    "common": 0.0,
    "uncommon": 0.33,
    "rare": 0.67,
    "mythic": 1.0,
}

# Normalization constants for release year: (year - 1992) / 34
# 1993 → 1/34 ≈ 0.029, 2026 → 34/34 = 1.0
_YEAR_BASE = 1992
_YEAR_RANGE = 34.0


def encode_metadata(printing_data: PrintingData) -> torch.Tensor:
    """Encode PrintingData into a float32 tensor of shape (15,).

    Layout:
        [0]    is_reserved         (0.0 / 1.0)
        [1]    rarity ordinal      (common=0.0, uncommon=0.33, rare=0.67, mythic=1.0)
        [2]    log-printings       (log(count) / log(50), clamped 0–1)
        [3]    release_year norm   ((year - 1992) / 34, clamped 0–1)
        [4:14] legalities multi-hot (10 RECOGNIZED_FORMATS slots)
        [14]   is_abu              (0.0 / 1.0)
    """
    feats: list[float] = []

    # is_reserved (1)
    feats.append(1.0 if printing_data.is_reserved else 0.0)

    # rarity ordinal (1)
    feats.append(_RARITY_ORDINAL.get(printing_data.rarity, 0.67))

    # log-printings (1)
    log_print = math.log(max(printing_data.printings_count, 1)) / math.log(50)
    feats.append(min(max(log_print, 0.0), 1.0))

    # release_year normalized (1)
    year_norm = (printing_data.release_year - _YEAR_BASE) / _YEAR_RANGE
    feats.append(min(max(year_norm, 0.0), 1.0))

    # legalities multi-hot (10)
    legal_set = set(printing_data.legalities)
    for fmt in RECOGNIZED_FORMATS:
        feats.append(1.0 if fmt in legal_set else 0.0)

    # is_abu (1)
    feats.append(1.0 if printing_data.is_abu else 0.0)

    return torch.tensor(feats, dtype=torch.float32)

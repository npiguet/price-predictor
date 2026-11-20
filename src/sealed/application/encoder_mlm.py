"""The masked-language-model auxiliary objective's mask draw for
``train-encoder`` (FR-014b): the reserved tokens and which positions a
batch masks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

_MASK_TOKEN: str = "[MASK]"
_PAD_TOKEN: str = "[PAD]"
_UNK_TOKEN: str = "[UNK]"
_CARDNAME_TOKEN: str = "cardname"
_MASKED_SPECIAL_TOKENS: tuple[str, ...] = (
    _PAD_TOKEN, _UNK_TOKEN, _CARDNAME_TOKEN, _MASK_TOKEN,
)


# ── MLM mask draw (FR-014b, Decision D-10) ──────────────────────────────


@dataclass(frozen=True)
class MlmMaskSpec:
    """What an MLM mask draw masks: each eligible position's selection
    probability, the id a selected position is overwritten with, and the
    token ids that are never selected.
    """

    mask_prob: float
    mask_token_id: int
    special_token_ids: Iterable[int]


def _draw_mlm_mask(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    spec: MlmMaskSpec,
    *,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw an MLM mask and produce the masked-input tensor.

    Returns ``(masked_ids, mask_positions)`` where ``mask_positions`` is
    a ``(B, T)`` boolean tensor marking the positions selected for
    cross-entropy loss. Eligible positions are real (``attention_mask
    == 1``) AND carry a token id not in ``spec.special_token_ids``;
    selected positions are overwritten with ``spec.mask_token_id`` in
    ``masked_ids``. The mask is drawn fresh on every call so the same
    card sees different masks across epochs (Decision D-10).
    """
    eligible = attention_mask.bool()
    for special_id in spec.special_token_ids:
        eligible = eligible & (input_ids != special_id)
    probs = torch.full_like(input_ids, fill_value=0, dtype=torch.float32)
    probs[eligible] = spec.mask_prob
    mask_positions = torch.bernoulli(probs, generator=generator).bool()
    masked_ids = input_ids.clone()
    masked_ids[mask_positions] = spec.mask_token_id
    return masked_ids, mask_positions


def _resolve_special_token_ids(
    vocab: dict[str, int],
) -> tuple[int, tuple[int, ...]]:
    """Return ``(mask_id, special_ids)`` for the MLM mask draw."""
    mask_id = vocab[_MASK_TOKEN]
    special_ids = tuple(
        vocab[t] for t in _MASKED_SPECIAL_TOKENS if t in vocab
    )
    return mask_id, special_ids

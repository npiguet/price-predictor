"""CardEncoder: strip name: line, tokenize, and produce a float32 embedding."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import torch

from price_predictor.domain.card_text import ConvertedCardText
from price_predictor.domain.tokenizer import MtgTokenizer
from sealed.domain.deterministic_features import parse_deterministic_features


class _PooledEncoder(Protocol):
    """Minimal interface CardEncoder needs from an encoder model.

    Satisfied by both ``CardPriceTransformerModel`` (pooled text width
    ``2 * d_model``) and ``sealed.domain.encoder_model.SealedEncoderModel``
    (``d_model`` for ``--pool-mode attn``, ``2 * d_model`` for dual pool).
    """

    def encode(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
    ) -> torch.Tensor: ...

    def _encode_and_pool(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
    ) -> torch.Tensor: ...


class CardEncoder:
    """Encodes a card script text into a pooled float32 embedding vector.

    Strips the ``name:`` line before encoding so the embedding captures
    game mechanics, not card identity. The output is the encoder's pooled
    text vector (the width of ``encoder.encode(...)``) concatenated with the
    deterministic-feature block — see ``card_embedding_layout``.
    """

    def __init__(
        self,
        model: _PooledEncoder,
        tokenizer: MtgTokenizer,
        max_seq_len: int,
        device: str = "cpu",
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._max_seq_len = max_seq_len
        self._device = device

    def encode(self, converted: ConvertedCardText) -> np.ndarray:
        """Encode card text, returning a ``(encoder_pooled_dim + FEATURE_COUNT,)``
        float32 array."""
        text = converted.without_name_line()

        input_ids, attention_mask = self._tokenizer.encode(text, self._max_seq_len)

        ids_tensor = torch.tensor([input_ids], dtype=torch.long, device=self._device)
        mask_tensor = torch.tensor([attention_mask], dtype=torch.long, device=self._device)

        embedding = self._model.encode(ids_tensor, mask_tensor)
        text_vec = embedding.squeeze(0).cpu().numpy().astype(np.float32)

        det_feats = parse_deterministic_features(converted)
        return np.concatenate([text_vec, det_feats])

    def encode_batch_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        with_grad: bool,
    ) -> torch.Tensor:
        """Run the encoder on a pre-tokenized batch and return the
        ``(B, encoder_pooled_dim)`` text-vector slice (no deterministic-feature
        concat). When ``with_grad=True``, gradients flow into the encoder
        parameters — this is the Phase B hot path."""
        if with_grad:
            return self._model._encode_and_pool(input_ids, attention_mask)
        with torch.no_grad():
            return self._model._encode_and_pool(input_ids, attention_mask)

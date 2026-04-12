"""CardEncoder: strip name: line, tokenize, and produce a float32 embedding."""

from __future__ import annotations

import numpy as np
import torch

from price_predictor.domain.tokenizer import MtgTokenizer
from price_predictor.infrastructure.transformer_model import CardPriceTransformerModel
from sealed.domain.deterministic_features import parse_deterministic_features


class CardEncoder:
    """Encodes a card script text into a pooled float32 embedding vector.

    Strips the ``name:`` line before encoding so the embedding captures
    game mechanics, not card identity. The output shape is
    ``(total_dim(encoder.d_model),)`` — see ``card_embedding_layout``.
    """

    def __init__(
        self,
        model: CardPriceTransformerModel,
        tokenizer: MtgTokenizer,
        max_seq_len: int,
        device: str = "cpu",
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._max_seq_len = max_seq_len
        self._device = device

    def encode(self, card_text: str) -> np.ndarray:
        """Encode card text, returning a ``(total_dim(d_model),)`` float32 array."""
        lines = [line for line in card_text.splitlines() if not line.startswith("name:")]
        text = "\n".join(lines)

        input_ids, attention_mask = self._tokenizer.encode(text, self._max_seq_len)

        ids_tensor = torch.tensor([input_ids], dtype=torch.long, device=self._device)
        mask_tensor = torch.tensor([attention_mask], dtype=torch.long, device=self._device)

        embedding = self._model.encode(ids_tensor, mask_tensor)
        text_vec = embedding.squeeze(0).cpu().numpy().astype(np.float32)

        det_feats = parse_deterministic_features(card_text)
        return np.concatenate([text_vec, det_feats])

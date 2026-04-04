"""CardEncoder: strip name: line, tokenize, and produce a float32 embedding."""

from __future__ import annotations

import numpy as np
import torch

from price_predictor.domain.tokenizer import MtgTokenizer
from price_predictor.infrastructure.transformer_model import CardPriceTransformerModel
from sealed.domain.mana_scorer import extract_mana_features, normalize_mana_features


class CardEncoder:
    """Encodes a card script text into a pooled float32 embedding vector.

    Strips the ``name:`` line before encoding so the embedding captures
    game mechanics, not card identity.
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
        """Encode card text, returning a (2*d_model + 15,) float32 numpy array.

        The first 2*d_model dimensions are the transformer pooled embedding
        (max+mean pooling). The final 15 dimensions are normalized mana features
        parsed directly from the card text, bypassing the transformer:
          [0–5]  pip counts W/U/B/R/G/C
          [6]    generic mana count
          [7]    X count
          [8]    mana value
          [9–14] mana produced W/U/B/R/G/C
        """
        lines = [line for line in card_text.splitlines() if not line.startswith("name:")]
        text = "\n".join(lines)

        input_ids, attention_mask = self._tokenizer.encode(text, self._max_seq_len)

        ids_tensor = torch.tensor([input_ids], dtype=torch.long, device=self._device)
        mask_tensor = torch.tensor([attention_mask], dtype=torch.long, device=self._device)

        transformer_embed = self._model.encode(ids_tensor, mask_tensor)  # (1, 2*d_model)
        transformer_np = transformer_embed.squeeze(0).cpu().numpy().astype(np.float32)

        mana_feats = np.array(
            normalize_mana_features(extract_mana_features(card_text)),
            dtype=np.float32,
        )
        return np.concatenate([transformer_np, mana_feats])

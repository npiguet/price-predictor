"""Greedy deck builder: build a 23-card non-land deck from a sealed pool."""

from __future__ import annotations

import random

import numpy as np
import torch

from sealed.domain.scorer_model import SetTransformerScorer

NONLAND_DECK_SIZE = 23


class GreedyDeckBuilder:
    """Build the best 23-card non-land deck from a pool via greedy single-card swaps.

    Starts from a random 23-card subset, then iteratively scores all
    (position, candidate) swaps in a single batched forward pass and applies
    the best swap if it improves the score. Stops when no swap improves.
    """

    def __init__(
        self,
        model: SetTransformerScorer,
        pool_embeddings: dict[str, np.ndarray],
    ) -> None:
        self._model = model
        self._pool_embeddings = pool_embeddings

    def build(self, pool_names: list[str]) -> list[str]:
        if len(pool_names) < NONLAND_DECK_SIZE:
            return list(pool_names)

        device = next(self._model.parameters()).device
        pool_arr = torch.from_numpy(
            np.stack([self._pool_embeddings[c] for c in pool_names])
        ).to(device)

        perm = list(range(len(pool_names)))
        random.shuffle(perm)
        n = NONLAND_DECK_SIZE
        r = len(pool_names) - n
        deck_t = torch.tensor(perm[:n], device=device)
        rem_t = torch.tensor(perm[n:], device=device)

        # These are constant across iterations of the greedy loop (r stays
        # constant because each swap exchanges one deck slot for one rem slot).
        positions = torch.arange(n, device=device).repeat_interleave(r)
        rows = torch.arange(n * r, device=device)
        mask_batch = torch.ones(n * r, n, dtype=torch.bool, device=device)

        current_score = self._score_current_deck(pool_arr, deck_t)

        while r > 0:
            batch = deck_t.unsqueeze(0).expand(n * r, -1).clone()
            replacements = rem_t.repeat(n)
            batch[rows, positions] = replacements
            cards_batch = pool_arr[batch]

            with torch.no_grad(), torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda"),
            ):
                scores = self._model(cards_batch, mask_batch).squeeze(-1)

            best_local_t = scores.argmax()
            best_score = float(scores[best_local_t].item())
            if best_score <= current_score:
                break
            best_local = int(best_local_t.item())

            i_pos, j_rem = best_local // r, best_local % r
            old_deck_card = deck_t[i_pos].clone()
            deck_t[i_pos] = rem_t[j_rem]
            rem_t[j_rem] = old_deck_card
            current_score = best_score

        return [pool_names[i] for i in deck_t.tolist()]

    def _score_current_deck(
        self,
        pool_arr: torch.Tensor,
        deck_t: torch.Tensor,
    ) -> float:
        cards_t = pool_arr[deck_t].unsqueeze(0)
        mask_t = torch.ones(1, deck_t.size(0), dtype=torch.bool, device=deck_t.device)
        with torch.no_grad(), torch.autocast(
            device_type=deck_t.device.type,
            dtype=torch.float16,
            enabled=(deck_t.device.type == "cuda"),
        ):
            return self._model(cards_t, mask_t).item()

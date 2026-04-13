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
        self._model.eval()
        if len(pool_names) < NONLAND_DECK_SIZE:
            return list(pool_names)

        device = next(self._model.parameters()).device
        pool_arr = torch.from_numpy(
            np.stack([self._pool_embeddings[c] for c in pool_names])
        ).to(device)

        perm = list(range(len(pool_names)))
        random.shuffle(perm)
        deck_idx = perm[:NONLAND_DECK_SIZE]
        rem_idx = perm[NONLAND_DECK_SIZE:]

        current_score = self._score_indices(pool_arr, deck_idx, device)

        while rem_idx:
            cards_batch, mask_batch = self._build_swap_batch(
                pool_arr, deck_idx, rem_idx, device,
            )
            with torch.no_grad():
                scores = self._model(cards_batch, mask_batch).squeeze(-1)

            best_local = int(scores.argmax().item())
            best_score = float(scores[best_local].item())
            if best_score <= current_score:
                break

            r = len(rem_idx)
            i_pos, j_rem = best_local // r, best_local % r
            deck_idx[i_pos], rem_idx[j_rem] = rem_idx[j_rem], deck_idx[i_pos]
            current_score = best_score

        return [pool_names[i] for i in deck_idx]

    def _score_indices(
        self,
        pool_arr: torch.Tensor,
        idx_list: list[int],
        device: torch.device,
    ) -> float:
        cards_t = pool_arr[torch.tensor(idx_list, device=device)].unsqueeze(0)
        mask_t = torch.ones(1, len(idx_list), dtype=torch.bool, device=device)
        with torch.no_grad():
            return self._model(cards_t, mask_t).item()

    def _build_swap_batch(
        self,
        pool_arr: torch.Tensor,
        deck_idx: list[int],
        rem_idx: list[int],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the (n_nonland*R, n_nonland) candidate matrix for one swap step.

        Row ``k`` corresponds to position ``k // R`` of the current deck being
        replaced by remaining card ``k % R``.
        """
        n = NONLAND_DECK_SIZE
        r = len(rem_idx)
        deck_t = torch.tensor(deck_idx, device=device)
        rem_t = torch.tensor(rem_idx, device=device)

        batch = deck_t.unsqueeze(0).expand(n * r, -1).clone()
        positions = torch.arange(n, device=device).repeat_interleave(r)
        replacements = rem_t.repeat(n)
        rows = torch.arange(n * r, device=device)
        batch[rows, positions] = replacements

        cards_batch = pool_arr[batch]
        mask_batch = torch.ones(n * r, n, dtype=torch.bool, device=device)
        return cards_batch, mask_batch

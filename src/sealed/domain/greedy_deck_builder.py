"""Greedy deck builder: build a 23-card non-land deck from a sealed pool."""

from __future__ import annotations

import random

import numpy as np
import torch

from sealed.domain.scorer_model import SetTransformerScorer

NONLAND_DECK_SIZE = 23


class GreedyDeckBuilder:
    """Build the best 23-card non-land deck from a pool via single-card swaps.

    Starts from a random 23-card subset, then iteratively scores all
    (position, candidate) swaps in a single batched forward pass.

    With ``temperature == 0`` (default), this is pure greedy hill-climbing:
    take the best swap if it improves the score, stop when no swap improves.

    With ``temperature > 0``, this is simulated annealing using softmax-
    temperature sampling: pick a swap with probability proportional to
    ``exp(score / T)``, with T cooling each iteration. SA can escape local
    optima by occasionally taking worse swaps. The best deck seen across
    all iterations is returned (not the final one), so SA is at least as
    good as greedy in expectation.
    """

    def __init__(
        self,
        model: SetTransformerScorer,
        pool_embeddings: dict[str, np.ndarray],
        temperature: float = 0.0,
        cooling: float = 0.95,
        max_iterations: int = 200,
    ) -> None:
        self._model = model
        self._pool_embeddings = pool_embeddings
        self._temperature = temperature
        self._cooling = cooling
        self._max_iterations = max_iterations

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

        # These are constant across iterations of the swap loop (r stays
        # constant because each swap exchanges one deck slot for one rem slot).
        positions = torch.arange(n, device=device).repeat_interleave(r)
        rows = torch.arange(n * r, device=device)
        mask_batch = torch.ones(n * r, n, dtype=torch.bool, device=device)

        current_score = self._score_current_deck(pool_arr, deck_t)
        best_score = current_score
        best_deck = deck_t.clone()

        for t in range(self._max_iterations):
            T = self._temperature * (self._cooling ** t) if self._temperature > 0 else 0.0

            batch = deck_t.unsqueeze(0).expand(n * r, -1).clone()
            replacements = rem_t.repeat(n)
            batch[rows, positions] = replacements
            cards_batch = pool_arr[batch]

            with torch.no_grad(), torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda"),
            ):
                scores = self._model(cards_batch, mask_batch).squeeze(-1)

            chosen_idx = self._select_swap(scores, T)
            candidate_score = float(scores[chosen_idx].item())

            if T == 0.0 and candidate_score <= current_score:
                break  # Pure greedy: no improving swap exists, stop.

            i_pos, j_rem = chosen_idx // r, chosen_idx % r
            old_deck_card = deck_t[i_pos].clone()
            deck_t[i_pos] = rem_t[j_rem]
            rem_t[j_rem] = old_deck_card
            current_score = candidate_score

            if current_score > best_score:
                best_score = current_score
                best_deck = deck_t.clone()

        return [pool_names[i] for i in best_deck.tolist()]

    @staticmethod
    def _select_swap(scores: torch.Tensor, T: float) -> int:
        """Return index of the swap to apply.

        At ``T == 0`` returns argmax (pure greedy). At ``T > 0`` samples
        from a softmax distribution with temperature T (Boltzmann sampling).
        """
        if T <= 0.0:
            return int(scores.argmax().item())
        logits = (scores.float() / T)
        logits = logits - logits.max()
        probs = torch.exp(logits)
        probs = probs / probs.sum()
        return int(torch.multinomial(probs, 1).item())

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

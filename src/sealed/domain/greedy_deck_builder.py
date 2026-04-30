"""Greedy deck builder: build a 23-spell + N-land deck from a sealed pool.

The greedy holds the invariant "exactly 23 non-land cards in the deck" but
lets the land count vary from 0 up to the basic-land budget. Each iteration
considers four operations in a single batched scoring pass:

- swap a deck spell for a pool spell (deck size unchanged);
- swap a deck land for a pool land (deck size unchanged);
- add a pool land to the deck (deck size grows by one, basics shrink by one);
- remove a deck land (deck size shrinks by one, basics grow by one).

This shape lets the scorer make joint decisions like "splash a third color
because the pool has fixers" — adding a dual land and changing a spell to a
splash spell happen in separate iterations, but the greedy walks toward
configurations where both are present whenever the score rewards them.
"""

from __future__ import annotations

import random

import numpy as np
import torch

from sealed.domain.card_embedding_layout import is_land_embedding
from sealed.domain.scorer_model import SetTransformerScorer

NONLAND_DECK_SIZE = 23
DECK_SIZE = 40
MAX_LANDS_IN_DECK = DECK_SIZE - NONLAND_DECK_SIZE  # 17 — basics shrink to 0 at this cap


class GreedyDeckBuilder:
    """Build the best (23-spell + N-land) deck from a pool via single-card moves.

    Starts from a random 23-spell, 0-land subset, then iteratively scores
    every legal one-step move (spell-spell swap, land-land swap, land
    addition, land removal) in batched forward passes and applies the best.

    With ``temperature == 0`` (default), this is pure greedy hill-climbing:
    apply the best move if it improves the score, stop when no move improves.

    With ``temperature > 0``, this is simulated annealing using softmax-
    temperature sampling. SA can escape local optima — useful here because
    "splash a color + add a dual" requires two coordinated moves and the
    cold greedy can't make them both at once. The best deck seen across all
    iterations is returned (not the final one), so SA is at least as good
    as greedy in expectation.
    """

    def __init__(
        self,
        model: SetTransformerScorer,
        pool_embeddings: dict[str, np.ndarray],
        temperature: float = 0.0,
        cooling: float = 0.95,
        max_iterations: int = 200,
        restarts: int = 1,
    ) -> None:
        self._model = model
        self._pool_embeddings = pool_embeddings
        self._temperature = temperature
        self._cooling = cooling
        self._max_iterations = max_iterations
        self._restarts = max(1, restarts)

    def build(self, pool_names: list[str]) -> list[str]:
        if len(pool_names) < NONLAND_DECK_SIZE:
            return list(pool_names)

        device = next(self._model.parameters()).device
        pool_arr = torch.from_numpy(
            np.stack([self._pool_embeddings[c] for c in pool_names])
        ).to(device)

        spell_pool, land_pool = self._partition_pool(pool_names)
        if len(spell_pool) < NONLAND_DECK_SIZE:
            # Not enough spells to satisfy the invariant; fall back to the
            # whole pool, scorer-be-damned. This is a degenerate case in
            # practice (sealed pools of ~80+ cards always have enough spells).
            return list(pool_names)

        best_score = float("-inf")
        best_spells: list[int] = []
        best_lands: list[int] = []
        for _ in range(self._restarts):
            score, spells, lands = self._single_run(pool_arr, spell_pool, land_pool)
            if score > best_score:
                best_score = score
                best_spells = spells
                best_lands = lands

        return [pool_names[i] for i in best_spells + best_lands]

    def _single_run(
        self,
        pool_arr: torch.Tensor,
        spell_pool: list[int],
        land_pool: list[int],
    ) -> tuple[float, list[int], list[int]]:
        """One independent greedy/SA run from a fresh random init. Returns
        the (best score seen, best spells, best lands) tuple for this run."""
        shuffled = list(spell_pool)
        random.shuffle(shuffled)
        deck_spells = shuffled[:NONLAND_DECK_SIZE]
        spells_remaining = shuffled[NONLAND_DECK_SIZE:]
        deck_lands: list[int] = []
        lands_remaining = list(land_pool)

        current_score = self._score_one(pool_arr, deck_spells + deck_lands)
        best_score = current_score
        best_spells = list(deck_spells)
        best_lands = list(deck_lands)

        for t in range(self._max_iterations):
            T = self._temperature * (self._cooling ** t) if self._temperature > 0 else 0.0

            decks, ops = self._enum_candidates(
                deck_spells, deck_lands, spells_remaining, lands_remaining,
            )
            if not decks:
                break

            scores = self._score_many(pool_arr, decks)
            chosen_idx = self._select_swap(scores, T)
            candidate_score = float(scores[chosen_idx].item())

            if T == 0.0 and candidate_score <= current_score:
                break

            deck_spells, deck_lands, spells_remaining, lands_remaining = self._apply(
                ops[chosen_idx],
                deck_spells, deck_lands, spells_remaining, lands_remaining,
            )
            current_score = candidate_score

            if current_score > best_score:
                best_score = current_score
                best_spells = list(deck_spells)
                best_lands = list(deck_lands)

        return best_score, best_spells, best_lands

    def _partition_pool(self, pool_names: list[str]) -> tuple[list[int], list[int]]:
        spells: list[int] = []
        lands: list[int] = []
        for i, name in enumerate(pool_names):
            if is_land_embedding(self._pool_embeddings[name]):
                lands.append(i)
            else:
                spells.append(i)
        return spells, lands

    @staticmethod
    def _enum_candidates(
        deck_spells: list[int],
        deck_lands: list[int],
        spells_remaining: list[int],
        lands_remaining: list[int],
    ) -> tuple[list[list[int]], list[tuple]]:
        """Enumerate every legal one-step move.

        Each move is paired with the resulting full deck (spells + lands)
        used for scoring, and an op tuple used to apply the move on accept.
        """
        decks: list[list[int]] = []
        ops: list[tuple] = []

        for i, old in enumerate(deck_spells):
            for new in spells_remaining:
                new_spells = deck_spells.copy()
                new_spells[i] = new
                decks.append(new_spells + deck_lands)
                ops.append(("swap_spell", i, new, old))

        for i, old in enumerate(deck_lands):
            for new in lands_remaining:
                new_lands = deck_lands.copy()
                new_lands[i] = new
                decks.append(deck_spells + new_lands)
                ops.append(("swap_land", i, new, old))

        if len(deck_lands) < MAX_LANDS_IN_DECK:
            for new in lands_remaining:
                decks.append(deck_spells + deck_lands + [new])
                ops.append(("add_land", new))

        for i, old in enumerate(deck_lands):
            new_lands = deck_lands.copy()
            new_lands.pop(i)
            decks.append(deck_spells + new_lands)
            ops.append(("remove_land", i, old))

        return decks, ops

    @staticmethod
    def _apply(
        op: tuple,
        deck_spells: list[int],
        deck_lands: list[int],
        spells_remaining: list[int],
        lands_remaining: list[int],
    ) -> tuple[list[int], list[int], list[int], list[int]]:
        kind = op[0]
        if kind == "swap_spell":
            _, i, new, old = op
            deck_spells = deck_spells.copy()
            deck_spells[i] = new
            spells_remaining = [s for s in spells_remaining if s != new] + [old]
        elif kind == "swap_land":
            _, i, new, old = op
            deck_lands = deck_lands.copy()
            deck_lands[i] = new
            lands_remaining = [l for l in lands_remaining if l != new] + [old]
        elif kind == "add_land":
            _, new = op
            deck_lands = deck_lands + [new]
            lands_remaining = [l for l in lands_remaining if l != new]
        elif kind == "remove_land":
            _, i, old = op
            deck_lands = deck_lands.copy()
            deck_lands.pop(i)
            lands_remaining = lands_remaining + [old]
        else:
            raise AssertionError(f"unknown op kind: {kind!r}")
        return deck_spells, deck_lands, spells_remaining, lands_remaining

    def _score_many(
        self, pool_arr: torch.Tensor, decks: list[list[int]],
    ) -> torch.Tensor:
        """Score every candidate deck in a single batched forward.

        Decks have variable length (spell swaps keep the size, adds grow by
        one, removes shrink by one), so the batch is padded to the longest
        deck and the mask marks the real positions.
        """
        device = pool_arr.device
        B = len(decks)
        max_len = max(len(d) for d in decks)

        indices = torch.zeros(B, max_len, dtype=torch.long, device=device)
        mask = torch.zeros(B, max_len, dtype=torch.bool, device=device)
        for i, d in enumerate(decks):
            n = len(d)
            indices[i, :n] = torch.tensor(d, device=device, dtype=torch.long)
            mask[i, :n] = True

        cards = pool_arr[indices]
        with torch.no_grad(), torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda"),
        ):
            return self._model(cards, mask).squeeze(-1)

    @staticmethod
    def _select_swap(scores: torch.Tensor, T: float) -> int:
        """Return the index of the move to apply.

        At ``T == 0`` returns argmax (pure greedy). At ``T > 0`` samples
        from a softmax distribution with temperature T (Boltzmann sampling).
        """
        if T <= 0.0:
            return int(scores.argmax().item())
        logits = scores.float() / T
        logits = logits - logits.max()
        probs = torch.exp(logits)
        probs = probs / probs.sum()
        return int(torch.multinomial(probs, 1).item())

    def _score_one(self, pool_arr: torch.Tensor, deck: list[int]) -> float:
        device = pool_arr.device
        idx = torch.tensor(deck, device=device, dtype=torch.long).unsqueeze(0)
        mask = torch.ones(1, len(deck), dtype=torch.bool, device=device)
        cards = pool_arr[idx]
        with torch.no_grad(), torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda"),
        ):
            return self._model(cards, mask).item()

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
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np
import torch

from sealed.domain.card_embedding_layout import card_colors, is_land_embedding
from sealed.domain.scorer_model import SetTransformerScorer

NONLAND_DECK_SIZE = 23
DECK_SIZE = 40
MAX_LANDS_IN_DECK = DECK_SIZE - NONLAND_DECK_SIZE  # 17 — basics shrink to 0 at this cap

# When the SA temperature has cooled below this, softmax sampling collapses to
# argmax for any realistic score scale, so we apply the greedy stop condition
# instead of grinding through iterations that no longer change the deck.
_EFFECTIVELY_GREEDY_TEMPERATURE = 1e-3

# Sentinel value for the ``restarts`` parameter selecting color-pair seeded
# init: one restart per MTG two-color pair, each restart's initial 23 spells
# filtered to cards whose colors are a subset of the pair. The full pool
# remains available as ``spells_remaining``/``lands_remaining``, so the
# search is unconstrained after init.
COLOR_PAIRS_STRATEGY = "color-pairs"

COLOR_PAIRS: tuple[frozenset[str], ...] = (
    frozenset({"W", "U"}), frozenset({"W", "B"}),
    frozenset({"W", "R"}), frozenset({"W", "G"}),
    frozenset({"U", "B"}), frozenset({"U", "R"}),
    frozenset({"U", "G"}), frozenset({"B", "R"}),
    frozenset({"B", "G"}), frozenset({"R", "G"}),
)


class MoveKind(IntEnum):
    """The four single-card local-search operators."""
    SWAP_SPELL = 0
    SWAP_LAND = 1
    ADD_LAND = 2
    REMOVE_LAND = 3


@dataclass(frozen=True, slots=True)
class Move:
    """One candidate modification of the working deck.

    Field meaning per ``kind``:

    - ``SWAP_SPELL``: ``position`` indexes into ``deck_spells``;
      ``new_card`` and ``old_card`` are pool indices.
    - ``SWAP_LAND``: ``position`` indexes into ``deck_lands``;
      ``new_card`` and ``old_card`` are pool indices.
    - ``ADD_LAND``: ``new_card`` is the pool index to append.
    - ``REMOVE_LAND``: ``position`` indexes into ``deck_lands``;
      ``old_card`` is the pool index being returned to ``lands_remaining``.
    """
    kind: MoveKind
    position: int = -1
    new_card: int = -1
    old_card: int = -1


@dataclass
class _RestartState:
    """Mutable per-restart state for lockstep batched search.

    All restarts share one ``_score_batch`` call per iteration. Each restart
    tracks its own working deck, scoring history, and ``finished`` flag (set
    when the SA loop has converged at effectively-greedy temperature or when
    no legal moves remain).
    """
    deck_spells: list[int]
    deck_lands: list[int]
    spells_remaining: list[int]
    lands_remaining: list[int]
    current_score: float = 0.0
    best_score: float = float("-inf")
    best_spells: list[int] = field(default_factory=list)
    best_lands: list[int] = field(default_factory=list)
    finished: bool = False


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
        restarts: int | str = 1,
    ) -> None:
        self._model = model
        self._pool_embeddings = pool_embeddings
        self._temperature = temperature
        self._cooling = cooling
        self._max_iterations = max_iterations
        self._restarts = self._validate_restarts(restarts)

    @staticmethod
    def _validate_restarts(restarts: int | str) -> int | str:
        if isinstance(restarts, str):
            if restarts != COLOR_PAIRS_STRATEGY:
                raise ValueError(
                    f"restarts string must be {COLOR_PAIRS_STRATEGY!r}, "
                    f"got {restarts!r}"
                )
            return restarts
        return max(1, restarts)

    def build(self, pool_names: list[str]) -> list[str]:
        if len(pool_names) < NONLAND_DECK_SIZE:
            return list(pool_names)

        device = next(self._model.parameters()).device
        pool_arr = torch.from_numpy(
            np.stack([self._pool_embeddings[c] for c in pool_names])
        ).to(device)
        # Normalize once for the whole build; every per-iteration scoring call
        # then uses forward_prenormalized and skips a (B, L, d_model) clone.
        pool_arr = self._model.normalize_features(pool_arr)

        spell_pool, land_pool = self._partition_pool(pool_names)
        if len(spell_pool) < NONLAND_DECK_SIZE:
            # Not enough spells to satisfy the invariant; fall back to the
            # whole pool, scorer-be-damned. This is a degenerate case in
            # practice (sealed pools of ~80+ cards always have enough spells).
            return list(pool_names)

        states = self._init_restart_states(pool_names, spell_pool, land_pool)
        self._run_search(pool_arr, states)

        best = max(states, key=lambda s: s.best_score)
        return [pool_names[i] for i in best.best_spells + best.best_lands]

    def _init_restart_states(
        self,
        pool_names: list[str],
        spell_pool: list[int],
        land_pool: list[int],
    ) -> list[_RestartState]:
        """Initialize one ``_RestartState`` per restart.

        Two strategies, dispatched on ``self._restarts``:

        - integer ``N`` → ``N`` random-shuffle inits (legacy behavior).
        - ``COLOR_PAIRS_STRATEGY`` → one init per MTG two-color pair,
          each restart's initial 23 spells filtered to on-color cards.
          The full pool remains as ``spells_remaining``/``lands_remaining``,
          so the search is unconstrained after init.
        """
        if self._restarts == COLOR_PAIRS_STRATEGY:
            return self._color_pair_states(pool_names, spell_pool, land_pool)
        assert isinstance(self._restarts, int)
        return self._random_restart_states(spell_pool, land_pool, self._restarts)

    @staticmethod
    def _random_restart_states(
        spell_pool: list[int], land_pool: list[int], count: int,
    ) -> list[_RestartState]:
        states: list[_RestartState] = []
        for _ in range(count):
            shuffled = list(spell_pool)
            random.shuffle(shuffled)
            states.append(_RestartState(
                deck_spells=shuffled[:NONLAND_DECK_SIZE],
                deck_lands=[],
                spells_remaining=shuffled[NONLAND_DECK_SIZE:],
                lands_remaining=list(land_pool),
            ))
        return states

    def _color_pair_states(
        self,
        pool_names: list[str],
        spell_pool: list[int],
        land_pool: list[int],
    ) -> list[_RestartState]:
        """Build up to 10 restart states, one per color pair.

        For each pair, the initial 23 spells are drawn from cards whose
        colors are a subset of the pair (colorless cards always qualify).
        ``spells_remaining`` and ``lands_remaining`` are the full pool minus
        the chosen 23 — the color filter applies only to the seed deck.

        Pairs without 23 eligible on-color spells are skipped. If every pair
        is skipped (degenerate pool), falls back to one random restart so
        the build always returns a deck.
        """
        states: list[_RestartState] = []
        for pair in COLOR_PAIRS:
            on_color = [
                i for i in spell_pool
                if card_colors(self._pool_embeddings[pool_names[i]]).issubset(pair)
            ]
            if len(on_color) < NONLAND_DECK_SIZE:
                continue
            random.shuffle(on_color)
            deck_spells = on_color[:NONLAND_DECK_SIZE]
            chosen = set(deck_spells)
            spells_remaining = [i for i in spell_pool if i not in chosen]
            states.append(_RestartState(
                deck_spells=deck_spells,
                deck_lands=[],
                spells_remaining=spells_remaining,
                lands_remaining=list(land_pool),
            ))

        if not states:
            return self._random_restart_states(spell_pool, land_pool, count=1)
        return states

    def _run_search(
        self, pool_arr: torch.Tensor, states: list[_RestartState],
    ) -> None:
        """Drive every restart in lockstep, sharing one forward per iteration.

        Each iteration concatenates the candidate batches of all still-active
        restarts into a single ``(sum_B, max_len)`` tensor scored in one
        forward pass. With the default ``--restarts 4``, this collapses four
        sequential 200-iteration searches into one 200-iteration search whose
        per-iteration batch is up to 4× larger, amortizing GPU launch
        overhead and giving the scorer a much higher arithmetic intensity.
        """
        self._bootstrap_states(pool_arr, states)

        for t in range(self._max_iterations):
            T = self._temperature * (self._cooling ** t) if self._temperature > 0 else 0.0
            effectively_greedy = T < _EFFECTIVELY_GREEDY_TEMPERATURE

            active_batches = self._gather_active_batches(states)
            if not active_batches:
                break

            big_indices, big_mask, slice_lengths = self._concat_batches(active_batches)
            scores = self._score_batch(pool_arr, big_indices, big_mask)
            self._dispatch_scores(
                active_batches, slice_lengths, scores, T, effectively_greedy,
            )

    def _bootstrap_states(
        self, pool_arr: torch.Tensor, states: list[_RestartState],
    ) -> None:
        """Score every restart's initial deck in a single batched forward."""
        base_lengths = [len(s.deck_spells) + len(s.deck_lands) for s in states]
        max_len = max(base_lengths)
        n = len(states)

        indices_np = np.zeros((n, max_len), dtype=np.int64)
        mask_np = np.zeros((n, max_len), dtype=bool)
        for i, s in enumerate(states):
            length = base_lengths[i]
            indices_np[i, :length] = s.deck_spells + s.deck_lands
            mask_np[i, :length] = True

        scores = self._score_batch(pool_arr, indices_np, mask_np).cpu().tolist()
        for s, score in zip(states, scores):
            s.current_score = float(score)
            s.best_score = float(score)
            s.best_spells = list(s.deck_spells)
            s.best_lands = list(s.deck_lands)

    def _gather_active_batches(
        self, states: list[_RestartState],
    ) -> list[tuple[_RestartState, np.ndarray, np.ndarray, list[Move]]]:
        """Build the candidate batch for every still-active restart.

        Restarts whose enumeration produced no legal moves are marked
        finished and excluded from the returned list.
        """
        active: list[tuple[_RestartState, np.ndarray, np.ndarray, list[Move]]] = []
        for s in states:
            if s.finished:
                continue
            indices_np, mask_np, moves = self._build_candidate_batch(
                s.deck_spells, s.deck_lands, s.spells_remaining, s.lands_remaining,
            )
            if not moves:
                s.finished = True
                continue
            active.append((s, indices_np, mask_np, moves))
        return active

    @staticmethod
    def _concat_batches(
        active_batches: list[tuple[_RestartState, np.ndarray, np.ndarray, list[Move]]],
    ) -> tuple[np.ndarray, np.ndarray, list[int]]:
        """Pad and concatenate per-restart batches into one ``(sum_B, max_len)`` pair.

        Restarts may produce different ``max_len`` values (a restart with
        room to add lands has one extra slot); the global ``max_len`` is
        their max, with shorter rows zero-padded and mask-padded.
        """
        max_len = max(indices_np.shape[1] for _, indices_np, _, _ in active_batches)
        slice_lengths = [indices_np.shape[0] for _, indices_np, _, _ in active_batches]
        total = sum(slice_lengths)

        big_indices = np.zeros((total, max_len), dtype=np.int64)
        big_mask = np.zeros((total, max_len), dtype=bool)
        offset = 0
        for _, indices_np, mask_np, _ in active_batches:
            B, L = indices_np.shape
            big_indices[offset:offset + B, :L] = indices_np
            big_mask[offset:offset + B, :L] = mask_np
            offset += B
        return big_indices, big_mask, slice_lengths

    def _dispatch_scores(
        self,
        active_batches: list[tuple[_RestartState, np.ndarray, np.ndarray, list[Move]]],
        slice_lengths: list[int],
        scores: torch.Tensor,
        T: float,
        effectively_greedy: bool,
    ) -> None:
        """For each active restart, pick its best move from its score slice
        and either apply it or mark the restart finished."""
        offset = 0
        for (s, _, _, moves), B in zip(active_batches, slice_lengths):
            slice_scores = scores[offset:offset + B]
            offset += B

            chosen_idx, candidate_score = self._select_swap(slice_scores, T)
            if effectively_greedy and candidate_score <= s.current_score:
                s.finished = True
                continue

            (
                s.deck_spells, s.deck_lands, s.spells_remaining, s.lands_remaining,
            ) = self._apply(
                moves[chosen_idx],
                s.deck_spells, s.deck_lands, s.spells_remaining, s.lands_remaining,
            )
            s.current_score = candidate_score

            if candidate_score > s.best_score:
                s.best_score = candidate_score
                s.best_spells = list(s.deck_spells)
                s.best_lands = list(s.deck_lands)

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
    def _build_candidate_batch(
        deck_spells: list[int],
        deck_lands: list[int],
        spells_remaining: list[int],
        lands_remaining: list[int],
    ) -> tuple[np.ndarray, np.ndarray, list[Move]]:
        """Enumerate every legal one-step move and produce the score batch.

        Returns ``(indices, mask, moves)`` where ``indices`` and ``mask`` are
        ``(B, max_len)`` numpy arrays directly consumable by ``_score_batch``,
        and ``moves[i]`` is the ``Move`` corresponding to row ``i``.

        The batch is built via vectorized numpy slice assignments on a
        pre-filled "base deck" template — never materializing per-candidate
        Python lists. With B ≈ 1100 candidates per iteration this avoids the
        ~2200 list allocations the per-candidate copy/concat pattern would
        otherwise incur.
        """
        n_spells = len(deck_spells)
        n_lands = len(deck_lands)
        n_sr = len(spells_remaining)
        n_lr = len(lands_remaining)
        base_len = n_spells + n_lands
        can_add = n_lands < MAX_LANDS_IN_DECK

        n_swap_spell = n_spells * n_sr
        n_swap_land = n_lands * n_lr
        n_add_land = n_lr if can_add else 0
        n_remove_land = n_lands
        batch_size = n_swap_spell + n_swap_land + n_add_land + n_remove_land
        max_len = base_len + (1 if can_add else 0)

        if batch_size == 0:
            return (
                np.zeros((0, max_len), dtype=np.int64),
                np.zeros((0, max_len), dtype=bool),
                [],
            )

        base_indices = np.asarray(deck_spells + deck_lands, dtype=np.int64)
        spells_remaining_arr = np.asarray(spells_remaining, dtype=np.int64)
        lands_remaining_arr = np.asarray(lands_remaining, dtype=np.int64)

        indices_np = np.zeros((batch_size, max_len), dtype=np.int64)
        mask_np = np.zeros((batch_size, max_len), dtype=bool)
        # Pre-fill: every row starts as a copy of the base deck.
        indices_np[:, :base_len] = base_indices
        mask_np[:, :base_len] = True

        moves: list[Move] = []

        # swap_spell: position i in deck_spells cycles through spells_remaining
        for i in range(n_spells):
            old = deck_spells[i]
            row_lo = i * n_sr
            row_hi = row_lo + n_sr
            indices_np[row_lo:row_hi, i] = spells_remaining_arr
            moves.extend(
                Move(MoveKind.SWAP_SPELL, i, int(new), old)
                for new in spells_remaining
            )

        # swap_land: position i in deck_lands cycles through lands_remaining
        swap_land_offset = n_swap_spell
        for i in range(n_lands):
            old = deck_lands[i]
            row_lo = swap_land_offset + i * n_lr
            row_hi = row_lo + n_lr
            indices_np[row_lo:row_hi, n_spells + i] = lands_remaining_arr
            moves.extend(
                Move(MoveKind.SWAP_LAND, i, int(new), old)
                for new in lands_remaining
            )

        # add_land: append at slot base_len (deck length grows by 1)
        if can_add:
            row_lo = n_swap_spell + n_swap_land
            row_hi = row_lo + n_add_land
            indices_np[row_lo:row_hi, base_len] = lands_remaining_arr
            mask_np[row_lo:row_hi, base_len] = True
            moves.extend(
                Move(MoveKind.ADD_LAND, new_card=int(new))
                for new in lands_remaining
            )

        # remove_land: shift trailing lands left and mask the last real slot
        remove_offset = n_swap_spell + n_swap_land + n_add_land
        for i in range(n_lands):
            old = deck_lands[i]
            row = remove_offset + i
            slot = n_spells + i
            if slot < base_len - 1:
                indices_np[row, slot:base_len - 1] = base_indices[slot + 1:base_len]
            mask_np[row, base_len - 1] = False
            moves.append(Move(MoveKind.REMOVE_LAND, position=i, old_card=old))

        return indices_np, mask_np, moves

    @staticmethod
    def _apply(
        move: Move,
        deck_spells: list[int],
        deck_lands: list[int],
        spells_remaining: list[int],
        lands_remaining: list[int],
    ) -> tuple[list[int], list[int], list[int], list[int]]:
        if move.kind is MoveKind.SWAP_SPELL:
            deck_spells = deck_spells.copy()
            deck_spells[move.position] = move.new_card
            spells_remaining = [
                s for s in spells_remaining if s != move.new_card
            ] + [move.old_card]
        elif move.kind is MoveKind.SWAP_LAND:
            deck_lands = deck_lands.copy()
            deck_lands[move.position] = move.new_card
            lands_remaining = [
                idx for idx in lands_remaining if idx != move.new_card
            ] + [move.old_card]
        elif move.kind is MoveKind.ADD_LAND:
            deck_lands = deck_lands + [move.new_card]
            lands_remaining = [idx for idx in lands_remaining if idx != move.new_card]
        else:  # MoveKind.REMOVE_LAND
            deck_lands = deck_lands.copy()
            deck_lands.pop(move.position)
            lands_remaining = lands_remaining + [move.old_card]
        return deck_spells, deck_lands, spells_remaining, lands_remaining

    def _score_batch(
        self,
        pool_arr: torch.Tensor,
        indices_np: np.ndarray,
        mask_np: np.ndarray,
    ) -> torch.Tensor:
        """Score every row of ``(indices_np, mask_np)`` in a single forward.

        The matrices are built on CPU and transferred in one host→device copy.
        Building them directly on-device with ``torch.tensor(d, device=cuda)``
        inside a per-candidate loop launches B small host→device copies and
        dominates the per-iteration wall-clock at B ≈ 1000.
        """
        device = pool_arr.device
        indices = torch.from_numpy(indices_np).to(device, non_blocking=True)
        mask = torch.from_numpy(mask_np).to(device, non_blocking=True)
        cards = pool_arr[indices]
        with torch.no_grad(), torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda"),
        ):
            # pool_arr was normalized once at build start; skip the per-forward clone.
            return self._model.forward_prenormalized(cards, mask).squeeze(-1)

    @staticmethod
    def _select_swap(scores: torch.Tensor, T: float) -> tuple[int, float]:
        """Return ``(index of move to apply, its score)``.

        At ``T == 0`` returns argmax (pure greedy). At ``T > 0`` samples
        from a softmax distribution with temperature T (Boltzmann sampling).
        Index and score are packed into a single host transfer to halve the
        per-iteration GPU↔CPU sync count.
        """
        if T <= 0.0:
            idx_t = scores.argmax()
        else:
            logits = scores.float() / T
            logits = logits - logits.max()
            probs = torch.exp(logits)
            probs = probs / probs.sum()
            idx_t = torch.multinomial(probs, 1).squeeze(0)
        chosen = scores[idx_t]
        idx_val, score_val = torch.stack([idx_t.float(), chosen.float()]).tolist()
        return int(idx_val), float(score_val)

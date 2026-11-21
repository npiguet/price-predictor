"""Shared per-pick typed-token state walk for draft corpora (spec 020 FR-007).

For one seat, walk its booster sightings once and yield the raw typed-token state
at every pick — the four mutually-exclusive token blocks
(``POOL``/``PACK``/``PASSED``/``TAKEN``) with per-card recency, plus the position
of the taken action. The state is identical to building each pick via
:func:`draft.domain.draft_state.build_state` (locked by the gen-1 equivalence
test); recency follows FR-019/FR-021 of the gen-1 spec.

This logic was previously inlined in the gen-1 loader's ``_Loader._emit_seat``
(now in ``imitation_examples``) / ``_emit_example``. It is extracted here so
both the gen-1 imitation loader and the gen-2 RL loader build per-pick states
from one implementation (research §Overlapping vocabulary — extract, don't
duplicate). The caller owns the shared
embedding table via the ``card_index`` callable (name → int row, or ``None`` when
the card has no ``.npz``); each distinct card is resolved/last-seen exactly once
on entry to a block, so per-pick assembly is a list-concat plus vectorized
recency in numpy with only a small Python loop over the PACK/PASSED blocks.

Pure domain-adjacent helper: numpy only, no torch.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

import numpy as np

from draft.domain.draft_geometry import Booster, DraftGeometry
from draft.domain.draft_state import (
    TYPE_PACK,
    TYPE_PASSED,
    TYPE_POOL,
    TYPE_TAKEN,
)


@dataclass(slots=True)
class RawPickState:
    """The typed-token state at one ``(seat, pack, pick)``, embedding-rows form.

    Card embeddings are *not* materialized here: ``card_idx`` holds int32 rows
    into the caller's shared table. The small per-token int arrays are int8.
    ``action_position`` is the absolute index of the taken ``PACK`` token (the
    action a learner/imitation target), or ``-1`` when the taken card has no
    ``.npz`` embedding (then there is no usable action at this pick).
    """

    card_idx: np.ndarray         # (N,) int32, rows into the shared table
    type_idx: np.ndarray         # (N,) int8
    packs_ago: np.ndarray        # (N,) int8
    pick_ago: np.ndarray         # (N,) int8
    pack_number: int
    pick_number: int
    action_position: int


def iter_seat_pick_states(
    geo: DraftGeometry,
    boosters: list[Booster],
    seat: int,
    card_index: Callable[[str], int | None],
) -> Iterator[RawPickState]:
    """Yield the :class:`RawPickState` for every pick ``seat`` made, in order.

    Walks the seat's boosters once. ``passed`` and ``last_seen`` track *all*
    cards (the wheel/flush logic needs them); POOL/TAKEN/PACK/PASSED *emission*
    drops cards with no ``.npz`` (consistent with the missing-embedding policy).
    A pick whose entire PACK is un-embeddable yields nothing (a gap in the
    sequence), matching the gen-1 loader.
    """
    walk = _SeatWalk(geo, card_index)
    for p in range(1, geo.packs + 1):
        for i in range(1, geo.pack_size + 1):
            k, off = geo.booster_for_pick(seat, p, i)
            picks = boosters[k].picks
            vis = picks[off:]
            walk.resolve_wheel(picks, off)
            state = walk.state_at(p, i, vis)
            if state is not None:
                yield state
            walk.advance(p, i, vis)
        walk.flush_passed()


@dataclass(slots=True)
class _Block:
    """One token block as parallel lists: embedding row, last pack, last pick."""

    idx: list[int] = field(default_factory=list)
    lp: list[int] = field(default_factory=list)
    li: list[int] = field(default_factory=list)

    def append(self, idx: int, lp: int, li: int) -> None:
        self.idx.append(idx)
        self.lp.append(lp)
        self.li.append(li)

    def __len__(self) -> int:
        return len(self.idx)


class _SeatWalk:
    """The running state of one seat's walk: POOL / TAKEN blocks, PASSED and last-seen."""

    def __init__(self, geo: DraftGeometry, card_index: Callable[[str], int | None]) -> None:
        self._pod = geo.pod_size
        self._pack_size = geo.pack_size
        self._card_index = card_index
        # POOL / TAKEN as embeddable (idx, last-pack, last-pick); built once on entry.
        self._pool = _Block()
        self._taken = _Block()
        self._passed: dict[str, tuple[int, int]] = {}      # all cards (state)
        self._last_seen: dict[str, tuple[int, int]] = {}   # all cards (state)

    def resolve_wheel(self, picks: list[str], off: int) -> None:
        """Wheel diff: cards passed from this booster last time, gone now,
        were taken by others (persistent; serves emit + advance)."""
        if off < self._pod:
            return
        for c in picks[off - self._pod + 1: off]:
            self._passed.pop(c, None)
            idx = self._card_index(c)
            if idx is not None:
                lp, li = self._last_seen[c]
                self._taken.append(idx, lp, li)

    def flush_passed(self) -> None:
        """Pack boundary: remaining PASSED flush to TAKEN (FR-019b)."""
        for c, (lp, li) in self._passed.items():
            idx = self._card_index(c)
            if idx is not None:
                self._taken.append(idx, lp, li)
        self._passed.clear()

    def advance(self, p: int, i: int, vis: list[str]) -> None:
        """Advance past pick ``(p, i)``: the taken card joins POOL, the rest are PASSED."""
        taken_card = vis[0]
        t_idx = self._card_index(taken_card)
        if t_idx is not None:
            self._pool.append(t_idx, p, i)
        self._passed.pop(taken_card, None)
        for c in vis:
            self._last_seen[c] = (p, i)
        for c in vis[1:]:
            self._passed[c] = (p, i)

    def state_at(self, p: int, i: int, vis: list[str]) -> RawPickState | None:
        """Assemble one state: concat the 4 typed blocks, vectorize recency."""
        pack_actions = list(dict.fromkeys(vis))
        pack, action_position = self._pack_block(p, i, pack_actions)
        if not len(pack):
            return None  # no usable PACK tokens — nothing to learn here
        passed = self._passed_block(set(pack_actions))
        blocks = (self._pool, pack, passed, self._taken)
        packs_ago, pick_ago = self._recency_ago(blocks, p, i)
        return RawPickState(
            card_idx=np.asarray(_joined(blocks, "idx"), dtype=np.int32),
            type_idx=_type_idx([len(b) for b in blocks]),
            packs_ago=packs_ago,
            pick_ago=pick_ago,
            pack_number=p,
            pick_number=i,
            action_position=action_position,
        )

    def _recency_ago(
        self, blocks: tuple[_Block, ...], p: int, i: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Vectorized recency (FR-021), entirely in int8: packs_ago = min(2, p - lp);
        pick_ago is (i - li) within the current pack else the frozen end-of-pack
        value (P - li). Both are provably already in range (lp ≤ p, 1 ≤ li ≤ P),
        so no clip/astype is needed.
        """
        lpli = np.asarray((_joined(blocks, "lp"), _joined(blocks, "li")), dtype=np.int8)
        lp_arr, li_arr = lpli[0], lpli[1]  # (2, n)
        packs_ago = np.minimum(p - lp_arr, 2)
        pick_ago = np.where(packs_ago == 0, i - li_arr, self._pack_size - li_arr)
        return packs_ago, pick_ago

    def _pack_block(self, p: int, i: int, pack_actions: list[str]) -> tuple[_Block, int]:
        """The PACK block (small Python loop) and the action's absolute position.

        The taken card is ``pack_actions[0]``; its position fixes the action
        index (or -1 if it was dropped).
        """
        pack = _Block()
        action_position = -1
        for j, name in enumerate(pack_actions):
            idx = self._card_index(name)
            if idx is None:
                continue
            ls = self._last_seen.get(name)
            lp, li = ls if ls is not None else (p, i)  # never-seen -> recency 0,0
            if j == 0:
                action_position = len(self._pool) + len(pack)
            pack.append(idx, lp, li)
        return pack, action_position

    def _passed_block(self, pack_set: set[str]) -> _Block:
        """The PASSED block (small Python loop); wheel survivors are now in PACK."""
        block = _Block()
        for name, (lp, li) in self._passed.items():
            if name in pack_set:
                continue
            idx = self._card_index(name)
            if idx is not None:
                block.append(idx, lp, li)
        return block


def _joined(blocks: tuple[_Block, ...], column: str) -> list[int]:
    """One column (``idx``, ``lp`` or ``li``) of every block, in block order."""
    return [x for block in blocks for x in getattr(block, column)]


def _type_idx(sizes: list[int]) -> np.ndarray:
    """Token types for blocks of ``sizes`` laid out POOL, PACK, PASSED, TAKEN."""
    type_idx = np.empty(sum(sizes), dtype=np.int8)
    a = sizes[0]
    b = a + sizes[1]
    c = b + sizes[2]
    type_idx[:a] = TYPE_POOL
    type_idx[a:b] = TYPE_PACK
    type_idx[b:c] = TYPE_PASSED
    type_idx[c:] = TYPE_TAKEN
    return type_idx

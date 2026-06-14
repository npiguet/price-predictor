"""Shared per-pick typed-token state walk for draft corpora (spec 020 FR-007).

For one seat, walk its booster sightings once and yield the raw typed-token state
at every pick — the four mutually-exclusive token blocks
(``POOL``/``PACK``/``PASSED``/``TAKEN``) with per-card recency, plus the position
of the taken action. The state is identical to building each pick via
:func:`draft.domain.draft_state.build_state` (locked by the gen-1 equivalence
test); recency follows FR-019/FR-021 of the gen-1 spec.

This logic was previously inlined in ``train_draft_agent._Loader._emit_seat`` /
``_emit_example``. It is extracted here so both the gen-1 imitation loader and the
gen-2 RL loader build per-pick states from one implementation (research
§Overlapping vocabulary — extract, don't duplicate). The caller owns the shared
embedding table via the ``card_index`` callable (name → int row, or ``None`` when
the card has no ``.npz``); each distinct card is resolved/last-seen exactly once
on entry to a block, so per-pick assembly is a list-concat plus vectorized
recency in numpy with only a small Python loop over the PACK/PASSED blocks.

Pure domain-adjacent helper: numpy only, no torch.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass

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
    pod, P, packs = geo.pod_size, geo.pack_size, geo.packs

    # POOL / TAKEN as embeddable (idx, last-pack, last-pick); built once on entry.
    pool_idx: list[int] = []
    pool_lp: list[int] = []
    pool_li: list[int] = []
    taken_idx: list[int] = []
    taken_lp: list[int] = []
    taken_li: list[int] = []
    passed: dict[str, tuple[int, int]] = {}      # all cards (state)
    last_seen: dict[str, tuple[int, int]] = {}   # all cards (state)

    for p in range(1, packs + 1):
        for i in range(1, P + 1):
            k, off = geo.booster_for_pick(seat, p, i)
            picks = boosters[k].picks
            vis = picks[off:]
            # Wheel diff: cards passed from this booster last time, gone now,
            # were taken by others (persistent; serves emit + advance).
            if off >= pod:
                for c in picks[off - pod + 1: off]:
                    passed.pop(c, None)
                    idx = card_index(c)
                    if idx is not None:
                        lp, li = last_seen[c]
                        taken_idx.append(idx)
                        taken_lp.append(lp)
                        taken_li.append(li)
            pack_set: set[str] = set()
            pack_actions: list[str] = []
            for c in vis:
                if c not in pack_set:
                    pack_set.add(c)
                    pack_actions.append(c)

            state = _build_state(
                p, i, P, pack_actions, pack_set,
                pool_idx, pool_lp, pool_li, taken_idx, taken_lp, taken_li,
                passed, last_seen, card_index,
            )
            if state is not None:
                yield state

            # --- advance past pick (p, i) ---
            taken_card = vis[0]
            t_idx = card_index(taken_card)
            if t_idx is not None:
                pool_idx.append(t_idx)
                pool_lp.append(p)
                pool_li.append(i)
            passed.pop(taken_card, None)
            for c in vis:
                last_seen[c] = (p, i)
            for c in vis[1:]:
                passed[c] = (p, i)
        # Pack boundary: remaining PASSED flush to TAKEN (FR-019b).
        for c, (lp, li) in passed.items():
            idx = card_index(c)
            if idx is not None:
                taken_idx.append(idx)
                taken_lp.append(lp)
                taken_li.append(li)
        passed.clear()


def _build_state(
    p: int,
    i: int,
    P: int,
    pack_actions: list[str],
    pack_set: set[str],
    pool_idx: list[int],
    pool_lp: list[int],
    pool_li: list[int],
    taken_idx: list[int],
    taken_lp: list[int],
    taken_li: list[int],
    passed: dict[str, tuple[int, int]],
    last_seen: dict[str, tuple[int, int]],
    card_index: Callable[[str], int | None],
) -> RawPickState | None:
    """Assemble one state: concat the 4 typed blocks, vectorize recency."""
    n_pool = len(pool_idx)

    # PACK block (small Python loop). The taken card is pack_actions[0]; its
    # position fixes the action index (or -1 if it was dropped).
    pack_i: list[int] = []
    pack_lp: list[int] = []
    pack_li: list[int] = []
    action_position = -1
    for j, name in enumerate(pack_actions):
        idx = card_index(name)
        if idx is None:
            continue
        ls = last_seen.get(name)
        lp, li = ls if ls is not None else (p, i)  # never-seen -> recency 0,0
        if j == 0:
            action_position = n_pool + len(pack_i)
        pack_i.append(idx)
        pack_lp.append(lp)
        pack_li.append(li)
    if not pack_i:
        return None  # no usable PACK tokens — nothing to learn here

    # PASSED block (small Python loop); wheel survivors are now in PACK.
    pass_i: list[int] = []
    pass_lp: list[int] = []
    pass_li: list[int] = []
    for name, (lp, li) in passed.items():
        if name in pack_set:
            continue
        idx = card_index(name)
        if idx is None:
            continue
        pass_i.append(idx)
        pass_lp.append(lp)
        pass_li.append(li)

    idx_all = pool_idx + pack_i + pass_i + taken_idx
    lp_all = pool_lp + pack_lp + pass_lp + taken_lp
    li_all = pool_li + pack_li + pass_li + taken_li
    n = len(idx_all)

    # Vectorized recency (FR-021), entirely in int8: packs_ago = min(2, p - lp);
    # pick_ago is (i - li) within the current pack else the frozen end-of-pack
    # value (P - li). Both are provably already in range (lp ≤ p, 1 ≤ li ≤ P),
    # so no clip/astype is needed.
    lpli = np.asarray((lp_all, li_all), dtype=np.int8)  # (2, n)
    lp_arr, li_arr = lpli[0], lpli[1]
    packs_ago = np.minimum(p - lp_arr, 2)
    pick_ago = np.where(packs_ago == 0, i - li_arr, P - li_arr)

    type_idx = np.empty(n, dtype=np.int8)
    a = n_pool
    b = a + len(pack_i)
    c = b + len(pass_i)
    type_idx[:a] = TYPE_POOL
    type_idx[a:b] = TYPE_PACK
    type_idx[b:c] = TYPE_PASSED
    type_idx[c:] = TYPE_TAKEN

    return RawPickState(
        card_idx=np.asarray(idx_all, dtype=np.int32),
        type_idx=type_idx,
        packs_ago=packs_ago,
        pick_ago=pick_ago,
        pack_number=p,
        pick_number=i,
        action_position=action_position,
    )

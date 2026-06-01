"""Typed-token state reconstruction (FR-017 … FR-021)."""

from __future__ import annotations

from draft.domain.draft_geometry import Booster, DraftGeometry, DraftRecord, Seat
from draft.domain.draft_state import (
    TYPE_PACK,
    TYPE_PASSED,
    TYPE_POOL,
    TYPE_TAKEN,
    build_state,
)


def _by_type(state, token_type) -> set[str]:
    return {c.name for c in state.cards if c.token_type == token_type}


def _wheel_record() -> DraftRecord:
    """pod_size=2, packs=1, P=4 — boosters wheel back (P > pod_size)."""
    seats = [Seat("forge-full", [], 0.0), Seat("forge-full", [], 0.0)]
    boosters = [
        Booster("TST", ["c00", "c01", "c02", "c03"]),  # k=0, opened by seat 0
        Booster("TST", ["c10", "c11", "c12", "c13"]),  # k=1, opened by seat 1
    ]
    return DraftRecord("d", "r", "t", seats, boosters)


def test_layout_and_disjoint_types_on_wheel() -> None:
    record = _wheel_record()
    geo = DraftGeometry.from_record(record)
    # Seat 0 at pack 1, pick 3: booster k=0 wheels back; c01 was taken by others.
    state = build_state(record, geo, seat=0, pack=1, pick=3)

    assert _by_type(state, TYPE_POOL) == {"c00", "c11"}
    assert _by_type(state, TYPE_PACK) == {"c02", "c03"}
    assert _by_type(state, TYPE_PASSED) == {"c12", "c13"}
    assert _by_type(state, TYPE_TAKEN) == {"c01"}  # wheel diff: gone on return

    # Every observed instance in exactly one type (FR-018).
    names = [c.name for c in state.cards]
    assert len(names) == len(set(names)) == 7  # c10 never seen by seat 0

    assert state.pack_actions == ["c02", "c03"]
    assert state.target_index == 0  # c02 taken at this pick


def test_recency_freezes_across_packs() -> None:
    """packs_ago capped at {0,1,2}; pick_ago frozen once packs_ago>=1 (FR-021)."""
    # pod_size=2, packs=2, P=2: pack-2 target sees pack-1 picks at packs_ago=1.
    seats = [Seat("forge-full", [], 0.0), Seat("forge-full", [], 0.0)]
    boosters = [
        Booster("TST", ["a0", "a1"]),  # k=0 pack1 seat0
        Booster("TST", ["b0", "b1"]),  # k=1 pack1 seat1
        Booster("TST", ["c0", "c1"]),  # k=2 pack2 seat0
        Booster("TST", ["d0", "d1"]),  # k=3 pack2 seat1
    ]
    record = DraftRecord("d", "r", "t", seats, boosters)
    geo = DraftGeometry.from_record(record)
    # Seat 0 pack 2 pick 1: pool from pack 1 = {a0, b1}; both at packs_ago=1.
    state = build_state(record, geo, seat=0, pack=2, pick=1)
    pool = {c.name: c for c in state.cards if c.token_type == TYPE_POOL}
    assert set(pool) == {"a0", "b1"}
    for inst in pool.values():
        assert inst.packs_ago == 1
        # frozen end-of-pack value P - last_pick (P=2): a0 last_pick 1 -> 1; b1 last_pick 2 -> 0
    assert pool["a0"].pick_ago == 1
    assert pool["b1"].pick_ago == 0


def test_first_pick_has_only_pack_tokens() -> None:
    record = _wheel_record()
    geo = DraftGeometry.from_record(record)
    state = build_state(record, geo, seat=0, pack=1, pick=1)
    assert _by_type(state, TYPE_POOL) == set()
    assert _by_type(state, TYPE_PACK) == {"c00", "c01", "c02", "c03"}
    assert _by_type(state, TYPE_PASSED) == set()
    assert _by_type(state, TYPE_TAKEN) == set()
    for c in state.cards:
        assert c.packs_ago == 0 and c.pick_ago == 0  # never seen before


def test_pack_end_flush_to_taken() -> None:
    """Unresolved PASSED cards flush to TAKEN at a pack boundary (FR-019b)."""
    # pod_size=4, packs=2, P=2: no wheel within a pack; pack-1 passes flush.
    seats = [Seat("forge-full", [], 0.0) for _ in range(4)]
    boosters = [Booster("TST", [f"k{k}c{j}" for j in range(2)]) for k in range(8)]
    record = DraftRecord("d", "r", "t", seats, boosters)
    geo = DraftGeometry.from_record(record)
    # Seat 0 in pack 2 pick 1: everything it passed in pack 1 has flushed to TAKEN.
    state = build_state(record, geo, seat=0, pack=2, pick=1)
    # Seat 0 saw pack-1 packs at picks 1 and 2; passed cards became TAKEN.
    assert _by_type(state, TYPE_PASSED) == set()
    assert _by_type(state, TYPE_TAKEN), "pack-1 passes should have flushed to TAKEN"
    for c in state.cards:
        if c.token_type == TYPE_TAKEN:
            assert c.packs_ago == 1

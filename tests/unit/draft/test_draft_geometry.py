"""Geometry round-trip + hand-worked reconstruction (SC-002, FR-016)."""

from __future__ import annotations

import pytest

from draft.domain.draft_geometry import (
    Booster,
    DraftGeometry,
    DraftRecord,
    Seat,
)


def _make_record(pod_size: int, packs: int, pack_size: int) -> DraftRecord:
    """A record whose every booster pick is a unique 'k:offset' token."""
    seats = [Seat(agent="forge-full", deck=[], deck_score=0.0) for _ in range(pod_size)]
    boosters = [
        Booster(
            set_code="TST",
            picks=[f"k{k}p{j}" for j in range(pack_size)],
        )
        for k in range(pod_size * packs)
    ]
    return DraftRecord("d", "r", "t", seats, boosters)


def test_derived_sizes_from_record() -> None:
    record = _make_record(pod_size=8, packs=3, pack_size=15)
    geo = DraftGeometry.from_record(record)
    assert geo.pod_size == 8
    assert geo.packs == 3
    assert geo.pack_size == 15


def test_direction_is_plus_for_packs_1_and_3_minus_for_pack_2() -> None:
    assert DraftGeometry.direction(1) == 1
    assert DraftGeometry.direction(2) == -1
    assert DraftGeometry.direction(3) == 1


def test_inconsistent_booster_count_raises() -> None:
    seats = [Seat("forge-full", [], 0.0) for _ in range(4)]
    boosters = [Booster("TST", ["a"]) for _ in range(6)]  # 6 not a multiple of 4
    with pytest.raises(ValueError):
        DraftGeometry.from_record(DraftRecord("d", "r", "t", seats, boosters))


@pytest.mark.parametrize(
    ("pod_size", "packs", "pack_size"),
    [(8, 3, 15), (4, 2, 3), (2, 2, 2), (6, 3, 14)],
)
def test_forward_inverse_round_trip_all_picks(
    pod_size: int, packs: int, pack_size: int,
) -> None:
    """Every (seat, pack, pick) maps to a booster offset that maps back to the seat."""
    record = _make_record(pod_size, packs, pack_size)
    geo = DraftGeometry.from_record(record)
    for pack in range(1, packs + 1):
        # Within one pack, the (booster, offset) pairs across all seats and picks
        # must cover every booster position exactly once (a permutation).
        seen: set[tuple[int, int]] = set()
        for seat in range(pod_size):
            for pick in range(1, pack_size + 1):
                k, offset = geo.booster_for_pick(seat, pack, pick)
                assert offset == pick - 1
                assert geo.pack_number(k) == pack
                assert geo.seat_of_pick(k, offset) == seat
                seen.add((k, offset))
        assert len(seen) == pod_size * pack_size


def test_hand_worked_drafted_pools() -> None:
    """A 2x2x2 draft worked out by hand (see module docstring conventions)."""
    seats = [Seat("forge-full", [], 0.0) for _ in range(2)]
    boosters = [
        Booster("TST", ["a0", "a1"]),  # k=0, pack1 open seat0
        Booster("TST", ["b0", "b1"]),  # k=1, pack1 open seat1
        Booster("TST", ["c0", "c1"]),  # k=2, pack2 open seat0
        Booster("TST", ["d0", "d1"]),  # k=3, pack2 open seat1
    ]
    record = DraftRecord("d", "r", "t", seats, boosters)
    geo = DraftGeometry.from_record(record)

    assert geo.drafted_pool(record, 0) == ["a0", "b1", "c0", "d1"]
    assert geo.drafted_pool(record, 1) == ["b0", "a1", "d0", "c1"]


def test_legal_actions_and_taken_card() -> None:
    record = _make_record(pod_size=4, packs=1, pack_size=4)
    geo = DraftGeometry.from_record(record)
    # Seat 0, pack 1, pick 1: opens its own booster k=0, offset 0.
    assert geo.taken_card(record, 0, 1, 1) == "k0p0"
    assert geo.legal_actions(record, 0, 1, 1) == ["k0p0", "k0p1", "k0p2", "k0p3"]
    # Pick 3 of the same seat sees only the cards from offset 2 onward.
    k, offset = geo.booster_for_pick(0, 1, 3)
    assert offset == 2
    assert geo.legal_actions(record, 0, 1, 3) == record.boosters[k].picks[2:]


def test_pick_out_of_range_raises() -> None:
    record = _make_record(pod_size=4, packs=2, pack_size=3)
    geo = DraftGeometry.from_record(record)
    with pytest.raises(ValueError):
        geo.booster_for_pick(0, 3, 1)  # only 2 packs
    with pytest.raises(ValueError):
        geo.booster_for_pick(0, 1, 4)  # only 3 picks

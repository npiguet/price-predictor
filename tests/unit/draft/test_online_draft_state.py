"""Gating equivalence: OnlineDraftStateTracker ↔ build_state (SC-003, research D3).

The single load-bearing correctness property of the live-play feature: the
online tracker, fed each seat's pick-requests reconstructed from a finished
record (and the seat's committed picks), must emit the *identical* typed-token
``DraftState`` that ``build_state`` produces for the same ``(seat, pack, pick)``
— same typed-token multiset, same per-instance ``(packs_ago, pick_ago)``, same
``pack_actions``, and the same revealed target.
"""

from __future__ import annotations

from collections import Counter

from draft.domain.draft_geometry import Booster, DraftGeometry, DraftRecord, Seat
from draft.domain.draft_state import DraftState, build_state
from draft.domain.online_draft_state import OnlineDraftStateTracker


def _multiset(state: DraftState) -> Counter:
    return Counter(
        (c.name, c.token_type, c.packs_ago, c.pick_ago) for c in state.cards
    )


def _record(pod_size: int, packs: int, pack_size: int) -> DraftRecord:
    """A finished draft with all-distinct card names (clean multiset compare).

    Booster ``k``'s drained picks are ``c{k}_{j}`` for ``j`` in pick order; the
    geometry alone assigns each offset to its drafting seat.
    """
    seats = [Seat("forge-full", [], 0.0) for _ in range(pod_size)]
    boosters = [
        Booster("TST", [f"c{k}_{j}" for j in range(pack_size)])
        for k in range(packs * pod_size)
    ]
    return DraftRecord("d", "r", "t", seats, boosters)


def _assert_tracker_matches_build_state(
    record: DraftRecord, geo: DraftGeometry,
) -> None:
    for seat in range(geo.pod_size):
        tracker = OnlineDraftStateTracker()
        for pack in range(1, geo.packs + 1):
            for pick in range(1, geo.pack_size + 1):
                k, offset = geo.booster_for_pick(seat, pack, pick)
                held = list(record.boosters[k].picks[offset:])  # request `pack`
                taken = record.boosters[k].picks[offset]        # draw-order pick

                expected = build_state(record, geo, seat, pack, pick)
                got = tracker.observe(
                    pack_number=pack, pick_number=pick,
                    pack=held, pod_size=geo.pod_size,
                )

                where = f"seat={seat} pack={pack} pick={pick}"
                assert _multiset(got) == _multiset(expected), (
                    f"typed-token multiset mismatch at {where}"
                )
                assert got.pack_actions == expected.pack_actions, (
                    f"pack_actions mismatch at {where}"
                )
                # Target is revealed by the commit; its index into the (identical)
                # pack_actions must match build_state's recorded target_index.
                assert got.pack_actions.index(taken) == expected.target_index, (
                    f"revealed target mismatch at {where}"
                )
                tracker.commit(taken)


def test_equivalence_with_wheel_and_multiple_packs() -> None:
    """P > pod_size, so boosters wheel back within a pack; 3 packs exercise flush."""
    record = _record(pod_size=4, packs=3, pack_size=6)
    _assert_tracker_matches_build_state(record, DraftGeometry.from_record(record))


def test_equivalence_no_wheel_within_pack() -> None:
    """pod_size >= pack_size: no wheel; every pack-1 pass flushes to TAKEN."""
    record = _record(pod_size=8, packs=3, pack_size=5)
    _assert_tracker_matches_build_state(record, DraftGeometry.from_record(record))


def test_equivalence_live_pod_geometry() -> None:
    """The real live geometry: pod 8, 3 packs, a full-size pack that wheels."""
    record = _record(pod_size=8, packs=3, pack_size=15)
    _assert_tracker_matches_build_state(record, DraftGeometry.from_record(record))


def test_commit_before_observe_raises() -> None:
    tracker = OnlineDraftStateTracker()
    try:
        tracker.commit("anything")
    except RuntimeError:
        return
    raise AssertionError("commit() before observe() should raise RuntimeError")

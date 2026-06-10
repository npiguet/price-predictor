"""Mixed-pod scoring is on one scale from shared boosters (US2, SC-006, T016).

gen-1 labeling scores *every* seat regardless of who piloted it, so a mixed pod
(agent + Forge seats) yields a per-draft agent-minus-Forge ``deck_score`` delta
with no cross-pod normalization. This story adds no production code — it is
verified, reusing the gen-1 record-assembly path.
"""

from __future__ import annotations

from draft.application.generate_draft_data import assemble_record


class _PoolSizeLabeler:
    """Scores every seat on one scale (a function of its drafted pool)."""

    def build_and_score(self, pool):
        return ["Plains"] * 40, float(len(pool))


def _mixed_pod_transcript() -> dict:
    # pod_size=4, packs=1, P=2 → 4 boosters; agent + forge seats share them.
    return {
        "draft_id": "d1",
        "boosters": [
            {"set_code": "BLB", "picks": ["A", "B"]},
            {"set_code": "BLB", "picks": ["C", "D"]},
            {"set_code": "BLB", "picks": ["E", "F"]},
            {"set_code": "BLB", "picks": ["G", "H"]},
        ],
        "seats": [
            {"agent": "draft-agent"},
            {"agent": "forge-full"},
            {"agent": "draft-agent"},
            {"agent": "forge-full"},
        ],
    }


def test_every_seat_scored_on_same_scale_from_shared_boosters() -> None:
    record = assemble_record(
        _mixed_pod_transcript(), run_id="r", labeler=_PoolSizeLabeler(),
    )
    # Both agent and Forge seats carry a deck + numeric deck_score.
    assert {s.agent for s in record.seats} == {"draft-agent", "forge-full"}
    for seat in record.seats:
        assert len(seat.deck) == 40
        assert seat.deck_score is not None

    # A per-draft agent-minus-Forge delta is well-defined (same boosters).
    agent = [s.deck_score for s in record.seats if s.agent == "draft-agent"]
    forge = [s.deck_score for s in record.seats if s.agent == "forge-full"]
    delta = sum(agent) / len(agent) - sum(forge) / len(forge)
    assert isinstance(delta, float)

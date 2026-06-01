"""Record assembly from a worker transcript (FR-007, FR-014)."""

from __future__ import annotations

from draft.application.generate_draft_data import assemble_record


class _FixedLabeler:
    """Always returns the same 40-card deck and score."""

    def build_and_score(self, pool):
        return ["Plains"] * 40, 7.5


class _FailingLabeler:
    """Always fails to build (mimics a pool with no legal deck)."""

    def build_and_score(self, pool):
        return [], None


def _transcript() -> dict:
    # pod_size=2, packs=1, P=2 -> 2 boosters.
    return {
        "draft_id": "d1",
        "boosters": [
            {"set_code": "BLB", "picks": ["A", "B"]},
            {"set_code": "BLB", "picks": ["C", "D"]},
        ],
        "seats": [{"agent": "forge-full"}, {"agent": "forge-r30"}],
    }


def test_well_formed_record_fields() -> None:
    record = assemble_record(_transcript(), run_id="run-1", labeler=_FixedLabeler())
    assert record.draft_id == "d1"
    assert record.run_id == "run-1"
    assert record.timestamp  # ISO timestamp stamped
    assert [s.agent for s in record.seats] == ["forge-full", "forge-r30"]
    assert all(len(s.deck) == 40 and s.deck_score == 7.5 for s in record.seats)
    # Boosters preserved verbatim.
    assert [b.picks for b in record.boosters] == [["A", "B"], ["C", "D"]]


def test_failed_build_yields_empty_deck_and_null_score() -> None:
    record = assemble_record(_transcript(), run_id="run-1", labeler=_FailingLabeler())
    for seat in record.seats:
        assert seat.deck == []
        assert seat.deck_score is None

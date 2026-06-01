"""Record IO: round-trip, trailing-partial tolerance, resume count (FR-012, FR-013)."""

from __future__ import annotations

from pathlib import Path

from draft.domain.draft_geometry import Booster, DraftRecord, Seat
from draft.infrastructure.draft_record_io import (
    append_record,
    count_complete_records,
    format_record_line,
    read_records,
    record_from_dict,
    record_to_dict,
)


def _record(draft_id: str) -> DraftRecord:
    return DraftRecord(
        draft_id=draft_id,
        run_id="run-1",
        timestamp="2026-06-01T00:00:00Z",
        seats=[
            Seat(agent="forge-full", deck=["Plains"] * 40, deck_score=12.5),
            Seat(agent="forge-r30", deck=[], deck_score=None),
        ],
        boosters=[
            Booster(set_code="BLB", picks=["A", "B", "C"]),
            Booster(set_code="BLB", picks=["D", "E", "F"]),
        ],
    )


def test_dict_round_trip() -> None:
    rec = _record("d1")
    assert record_from_dict(record_to_dict(rec)) == rec


def test_write_read_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "drafts.jsonl"
    recs = [_record("d1"), _record("d2"), _record("d3")]
    with open(path, "a", encoding="utf-8") as out:
        for rec in recs:
            append_record(out, rec)
    assert list(read_records(path)) == recs


def test_line_is_newline_free() -> None:
    line = format_record_line(_record("d1"))
    assert "\n" not in line


def test_trailing_partial_line_tolerated(tmp_path: Path) -> None:
    path = tmp_path / "drafts.jsonl"
    good = format_record_line(_record("d1"))
    # Two complete lines, then a partial (truncated, no trailing newline).
    path.write_text(good + "\n" + good + "\n" + good[: len(good) // 2], encoding="utf-8")
    read = list(read_records(path))
    assert len(read) == 2
    assert all(r.draft_id == "d1" for r in read)


def test_count_complete_records(tmp_path: Path) -> None:
    path = tmp_path / "drafts.jsonl"
    good = format_record_line(_record("d1"))
    path.write_text(good + "\n" + good + "\n" + "partial-no-newline", encoding="utf-8")
    assert count_complete_records(path) == 2


def test_count_and_read_missing_file(tmp_path: Path) -> None:
    path = tmp_path / "absent.jsonl"
    assert count_complete_records(path) == 0
    assert list(read_records(path)) == []

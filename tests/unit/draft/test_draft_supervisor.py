"""Supervisor stdout sentinel parsing (FR-010, worker-protocol.md)."""

from __future__ import annotations

from draft.application.generate_draft_data import SENTINEL, parse_sentinel_line


def test_keeps_and_parses_sentinel_line() -> None:
    transcript = parse_sentinel_line(
        SENTINEL + '{"draft_id":"d1","boosters":[],"seats":[]}'
    )
    assert transcript == {"draft_id": "d1", "boosters": [], "seats": []}


def test_non_sentinel_line_dropped() -> None:
    assert parse_sentinel_line("Forge: initializing environment...") is None
    assert parse_sentinel_line('{"draft_id":"d1"}') is None  # no sentinel prefix


def test_parse_failure_dropped() -> None:
    assert parse_sentinel_line(SENTINEL + "{not valid json") is None
    assert parse_sentinel_line(SENTINEL + "[1,2,3]") is None  # not a dict


def test_trailing_whitespace_tolerated() -> None:
    transcript = parse_sentinel_line(SENTINEL + '{"draft_id":"d1"}\n')
    assert transcript == {"draft_id": "d1"}

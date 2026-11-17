"""``gen3_review_checks.py``: every check runs on the gen-3 fixture records and sidecars."""

from __future__ import annotations

from pathlib import Path

import gen3_review_checks as grc
from gen3_review_checks_parts.model import Line

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "effects"
SIDECARS = FIXTURES / "gen3-sidecars"
RECORDS = FIXTURES / "gen3-records.jsonl.gz"

CHECK_NAMES = (
    "keys-declared-by-two-lines",
    "acting-keys-ambiguous[spell]",
    "board-lines-ambiguous[spell]",
    "board-lines-undescribed-or-unconverted",
    "equip-halves-without-stat-change",
    "equip-halves-with-stat-change",
    "counter-halves-random-seat-countered",
    "counter-halves-random-seat-not-countered",
    "fizzled-halves-with-zero-targets",
    "forced-halves",
    "forced-halves-stamped-before-turn-action",
    "forced-deaths-without-unattached",
    "protection-forks-attachment-leaves-unnamed",
    "trigger-records-without-subject[DamageDealtOnce]",
    "trigger-records-without-subject[DamageAll]",
    "phase-skipped-with-unknown-phase",
    "marked-games-without-player-lost",
    "rescue-records-holding-more-than-the-outcome",
    "lethal-creatures-without-death[game-ending-combat]",
    "lethal-creatures-without-death[other-combat]",
    "library-cards-without-owns-and-controls",
    "non-token-copy-entities-with-granted-line",
    "mana-record-events-without-attribution[tapped]",
    "mana-record-events-without-attribution[mana_produced]",
    "convoke-candidates",
)


def _run() -> dict[str, tuple[int, int]]:
    lines = grc.run_checks([RECORDS], SIDECARS)
    return {line.name: (line.count, line.total) for line in lines}


def test_every_check_reports_a_count_and_a_total() -> None:
    results = _run()
    for name in CHECK_NAMES:
        assert name in results, name
        count, total = results[name]
        assert 0 <= count <= total


def test_phase_skipped_events_all_carry_the_unknown_phase() -> None:
    count, total = _run()["phase-skipped-with-unknown-phase"]
    assert (count, total) == PHASE_SKIPPED


def test_sidecar_tree_has_keys_declared_by_two_lines() -> None:
    count, total = _run()["keys-declared-by-two-lines"]
    assert (count, total) == TREE_DUPLICATES


def test_report_line_names_the_review_count() -> None:
    line = Line("phase-skipped-with-unknown-phase", 3, 3)
    assert grc.format_line(line).split()[0] == "phase-skipped-with-unknown-phase"
    assert "review=101 of 101" in grc.format_line(line)
    assert "review=n/a" in grc.format_line(Line("no-such-check", 0, 0))


PHASE_SKIPPED = (2, 2)
TREE_DUPLICATES = (8, 947)

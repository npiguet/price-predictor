"""``gen3_review_checks.py``: every check runs on the gen-3 fixture records and sidecars."""

from __future__ import annotations

from pathlib import Path

import gen3_review_checks as grc
from gen3_review_checks_parts.convoke import is_convoke_row, mana_value, reduction
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
    "counter-halves-random-seat-countered[spell-target]",
    "counter-halves-random-seat-not-countered[stack-card-target]",
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
    """The fixture tree holds the current converter's sidecars wherever an
    older one declared a key twice, so it has none."""
    count, total = _run()["keys-declared-by-two-lines"]
    assert (count, total) == TREE_DUPLICATES


def test_a_key_declared_by_two_lines_is_flagged(tmp_path: Path) -> None:
    """Binding the Old Gods' sidecar with chapter I's key copied onto chapter
    II's line, as a sidecar written before the key change declared it."""
    import json

    from gen3_review_checks_parts.sidecars import keys_declared_by_two_lines

    relative = Path("cardsfolder/b/binding_the_old_gods.provenance.json")
    data = json.loads((SIDECARS / relative).read_text(encoding="utf-8"))
    data["lines"][1]["provenance"].append(data["lines"][0]["provenance"][0])
    (tmp_path / relative).parent.mkdir(parents=True)
    (tmp_path / relative).write_text(json.dumps(data), encoding="utf-8")
    line = keys_declared_by_two_lines(tmp_path)
    assert (line.count, line.total) == (1, 1)


def test_report_line_names_the_review_count() -> None:
    line = Line("phase-skipped-with-unknown-phase", 3, 3)
    assert grc.format_line(line).split()[0] == "phase-skipped-with-unknown-phase"
    assert "review=101 of 101" in grc.format_line(line)
    assert "review=n/a" in grc.format_line(Line("no-such-check", 0, 0))


PHASE_SKIPPED = (2, 2)
TREE_DUPLICATES = (0, 947)


STOKE_CONVOKE_ROW = {
    "line_index": 3,
    "line_kind": "cost reduction",
    "prose_offset": 16,
    "provenance": [{"face": 0, "trait_kind": "keyword", "index_within_kind": 0}],
    "sub_ability_links": [],
    "script_api_type": "Keyword",
    "script_param_keys": [],
    "script_text": "Convoke",
    "role_spans": [{"start": 0, "end": 7, "role": "effect"}],
}


def test_convoke_is_read_from_the_sidecar_row_whatever_its_line_kind() -> None:
    assert is_convoke_row(STOKE_CONVOKE_ROW)
    assert not is_convoke_row({**STOKE_CONVOKE_ROW, "script_text": "Flying"})


def test_convoke_cost_reduction_against_the_printed_cost() -> None:
    assert mana_value("{2}{R}{R}") == 4
    assert reduction("{R}{R}", "{2}{R}{R}") == "reduced"
    assert reduction("{2}{R}{R}", "{2}{R}{R}") == "same"
    assert reduction(None, "{2}{R}{R}") == "other"

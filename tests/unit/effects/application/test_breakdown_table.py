"""The breakdowns' lines and the table file beside them (task J05, finding 19.7):
a line prints a group's total, the per-field means go to ``breakdowns.csv``, and
every line's name parses back into its head and qualifier."""

from __future__ import annotations

import csv
from dataclasses import replace

import pytest

from effects.application import breakdowns as bd
from effects.application.breakdown_table import COLUMNS, write_breakdowns_csv
from effects.domain.check_label import CheckLabel
from effects.domain.evaluation_report import CheckName, EvaluationReport
from effects.domain.strata import Stratum
from tests.unit.effects.application import test_breakdowns


@pytest.fixture(scope="module")
def rows():
    """``test_breakdowns``' fixture records, one scored row each."""
    return test_breakdowns.rows._get_wrapped_function()()


def _checks(rows):
    many = [replace(row, losses={**row.losses, "counter.a": 1.0, "counter.b": 2.0})
            for row in rows]
    return bd.breakdown_checks(
        {Stratum.CARD_DISJOINT: [*many, replace(many[0], random_seat=True)],
         Stratum.GAME_DISJOINT: many},
        games_by_text={}, withheld_keyword="flying",
    )


class TestTheLines:
    def test_a_line_prints_the_total_and_leaves_every_field_to_the_table(self, rows):
        overall = next(c for c in _checks(rows) if c.name == Stratum.CARD_DISJOINT)
        assert set(overall.values) == {bd.TOTAL, f"text:{bd.TOTAL}"}
        fields = {row.field: row for row in overall.table}
        assert {"gate", "counter.a", "text:counter.b"} <= set(fields)
        assert fields[bd.TOTAL].records == len(rows) + 1
        assert "counter.a" not in EvaluationReport(checks=[overall]).render()

    def test_a_memorization_gap_prints_its_total_and_tables_every_field(self, rows):
        gap = next(c for c in _checks(rows) if c.name.startswith("memorization-gap "))
        assert set(gap.values) <= {bd.TOTAL}
        assert {row.field for row in gap.table} >= {"gate", "counter.a"}


class TestTheNames:
    def test_every_name_parses_back_to_its_head_and_qualifier(self, rows):
        score = test_breakdowns.TestHalfScoreChecks()._score()
        names = [c.name for c in _checks(rows)]
        names += [c.name for c in bd.half_score_checks(score, half="reporting", games={})]
        assert len(names) > 20
        for name in names:
            label = CheckLabel.parse(name)
            assert str(label) == name
            assert " " not in label.head

    def test_the_heads_are_checks_strata_or_halves(self, rows):
        known = {*map(str, CheckName), *map(str, Stratum), "reporting-half"}
        score = test_breakdowns.TestHalfScoreChecks()._score()
        names = [c.name for c in _checks(rows)]
        names += [c.name for c in bd.half_score_checks(score, half="reporting", games={})]
        assert {CheckLabel.parse(name).head for name in names} <= known

    def test_a_qualifier_keeps_its_spaces(self):
        label = CheckLabel.of(Stratum.CARD_DISJOINT, "mean over families")
        assert CheckLabel.parse(str(label)) == label
        assert label.qualifier == "mean over families"


class TestTheTableFile:
    def test_one_row_per_line_and_field(self, tmp_path, rows):
        checks = _checks(rows)
        path = tmp_path / "out" / "breakdowns.csv"
        count = write_breakdowns_csv(path, checks)
        with path.open(encoding="utf-8", newline="") as handle:
            read = list(csv.DictReader(handle))
        assert count == len(read) == sum(len(c.table) for c in checks)
        assert tuple(read[0]) == COLUMNS
        family = next(r for r in read if r["qualifier"].startswith("family ") and
                      r["field"] == "counter.a")
        assert family["check"] == Stratum.CARD_DISJOINT and float(family["value"]) == 1.0
        assert int(family["records"]) >= 1

    def test_a_memorization_gap_row_has_no_record_count(self, tmp_path, rows):
        path = tmp_path / "b.csv"
        write_breakdowns_csv(path, _checks(rows))
        with path.open(encoding="utf-8", newline="") as handle:
            gap = next(r for r in csv.DictReader(handle) if r["check"] == "memorization-gap")
        assert gap["records"] == "" and gap["qualifier"]


@pytest.mark.parametrize("name", ["gate-1", "card-disjoint", "reporting-half control"])
def test_a_name_with_or_without_a_qualifier_round_trips(name):
    assert str(CheckLabel.parse(name)) == name

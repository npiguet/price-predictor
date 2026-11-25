"""A frozen grid keeps the rule's answer for each of its records in each cell,
and its standard error reads them (findings 12.5, 12.6; research E4)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from effects.application.freeze_selection_set import GridQualification
from effects.domain.selection_set import error_of, item_key
from tests.unit.effects.application.test_freeze_selection_set import (
    GD,
    _balanced_records,
    _freeze,
    _two_halves,
)


def _answer(record) -> float:
    """The second cell's answer for a record of game ``gK``: 1 where ``K // 2``
    is even, so each half's four games answer 1, 0, 1, 0."""
    return 1.0 if int(record.game_id[1:]) // 2 % 2 == 0 else 0.0


@dataclass
class _ByRecord:
    """A grid whose second cell's answer differs between records."""

    name: str = "combat"
    cells: tuple = ("1x1", "1x2")

    def qualify(self, record):
        return GridQualification(("E1", "E2"), (0.0, _answer(record)))


@pytest.fixture(scope="module")
def frozen():
    halves = _two_halves(4)
    records, tags = _balanced_records(halves)
    return _freeze(halves, records, tags, grids=[_ByRecord()])


def test_each_cell_holds_each_records_answer_in_the_grids_record_order(frozen):
    for half in ("selection", "reporting"):
        grid = frozen.grids[GD][half]["combat"]
        assert len(grid.records) == 4
        first, second = grid.cells
        assert first.answers == (0.0,) * 4
        assert second.answers == tuple(1.0 if int(row.game_id[1:]) // 2 % 2 == 0 else 0.0
                                       for row in grid.records)
        assert sorted(second.answers) == [0.0, 0.0, 1.0, 1.0]


def test_the_standard_error_is_over_the_per_record_answers(frozen):
    """The grid's mean answer is 2/8 = 0.25, so predicting it everywhere errs
    by B = (6 x 0.25 + 2 x 0.75) / 8 = 0.375 against the records' answers
    (0.25 against the cell means 0 and 0.5); four records a cell spread
    1 / (2 x 2) = 0.25, a standard error of 0.25 / 0.375 = 2/3."""
    for half in ("selection", "reporting"):
        error = error_of(frozen.standard_errors[item_key(GD, half, "grids", "combat")])
        assert error == pytest.approx(2 / 3)

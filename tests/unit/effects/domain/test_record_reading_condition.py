"""A record's turn or step condition is asked of the sidecars once per reading.

The survey reads it for the record's cell signature and again for its condition
counts. Each read joins the acting keys, and the sidecar cache tallies every
join, so a record read twice counted its join gaps twice.
"""

from __future__ import annotations

from effects.domain.condition_balance import ConditionSide, cell_signature, read_condition
from effects.domain.record_reading import RecordReading
from effects.domain.sampling import SamplingClass
from tests.unit.effects.domain.test_corpus_curation import (
    _KINGS_ASSASSIN,
    _decision,
    _key,
    _Lines,
)


class _CountingLines(_Lines):
    """The lookup, tallying every question the condition asks of it."""

    def __init__(self, scripts):
        super().__init__(scripts)
        self.asked = 0

    def line_for(self, key):
        self.asked += 1
        return super().line_for(key)

    def line_or_none(self, key):
        self.asked += 1
        return super().line_or_none(key)

    def first_line(self, keys):
        self.asked += 1
        return super().first_line(keys)


def _fixture():
    key = _key("kings_assassin")
    return _decision(key, ()), _CountingLines({key: _KINGS_ASSASSIN})


def test_a_record_read_for_its_cell_and_its_counts_joins_once():
    record, lines = _fixture()
    reading = RecordReading(record)
    cell_signature(record, SamplingClass.PLAYABILITY_DECISION, lines, reading)
    after_cell = lines.asked
    assert after_cell > 0
    counted = read_condition(reading, lines)
    assert counted is not None and counted.side is ConditionSide.MET
    assert lines.asked == after_cell


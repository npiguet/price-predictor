"""A value position no forward reads is not a non-finite reading (CPD1 I-1).

A rule-test row is asked at each of its tag's entities; one the record gives
no slot, or where it says nothing about the output (a record that trains no
gate, for the gate), is read nowhere. Such a position is left out of its
record's mean, and a record read at no entity is left out of the rule, as
the freeze counts it (``rule_output_targets.RecordIndicators.of``); a value a
forward wrote and that is not finite still floors its item and is counted.

The records are the real gen-3 fixture's; the rule-test rows are written by
hand, as the control score's own tests write them.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from effects.domain.effect_targets import derive_targets
from effects.domain.selection_set import (
    PROBES,
    RULE_TESTS,
    ProbeRecord,
    SelectionSet,
    TaggedRecord,
)
from effects.domain.surface_builder import has_card_slot
from tests.unit.effects.application.test_control_score import (  # noqa: F401
    STRATA,
    _epoch,
    _prepare,
    run,
)

RULE = "r.ent"
NOWHERE = "E-nowhere"
INF = float("inf")


def _slotted(record, affected: bool) -> str | None:
    targets = derive_targets(record)
    return next((e.id for e in record.state.entities if has_card_slot(e)
                 and _affected(targets, e.id) is affected), None)


def _affected(targets, entity: str) -> bool:
    entry = targets.get(entity)
    return entry is not None and entry.affected


def _gated_rows(records, affected: bool) -> list[TaggedRecord]:
    """Eight gated records' rows at an entity the record did (or did not) affect."""
    rows = []
    for record in records:
        entity = _slotted(record, affected) if derive_targets(record).gated else None
        if entity is not None:
            rows.append(TaggedRecord(record.record_id, record.game_id, (entity,)))
    assert len(rows) >= 8
    return rows[:8]


def _ungated_row(records, taken: set[str]) -> TaggedRecord:
    """A row at a slotted entity of a record that trains no gate."""
    record = next(r for r in records if not derive_targets(r).gated
                  and r.record_id not in taken and any(has_card_slot(e) for e in r.state.entities))
    entity = next(e.id for e in record.state.entities if has_card_slot(e))
    return TaggedRecord(record.record_id, record.game_id, (entity,))


def _selection(records, fires, does_not, probe=()) -> SelectionSet:
    """Every game in the reporting half, one rule test over the rows given."""
    games = tuple(sorted({r.game_id for r in records}))
    tests = {RULE: {"fires": tuple(fires), "does_not_fire": tuple(does_not)}}
    return SelectionSet(
        corpus_digest="fixture", r=4,
        halves={s: {"selection": (), "reporting": games} for s in STRATA},
        probes={s: {"reporting": {"dies": tuple(probe)} if probe else {}} for s in STRATA},
        probe_kinds={"dies": "binary"},
        grids={s: {"reporting": {}} for s in STRATA},
        rule_tests={s: {"reporting": tests} for s in STRATA},
        deviance_samples={s: {"reporting": ()} for s in STRATA},
        standard_errors={}, left_out={}, settings={"max_item_error": 0.1},
    )


def _scored(run, records, selection, components=(RULE_TESTS,)):  # noqa: F811
    return _epoch(run, _scorer(run, records, selection, components)).strata[STRATA[0]]


def _scorer(run, records, selection, components=(RULE_TESTS,)):  # noqa: F811
    scorer, _, _ = _prepare(run, records, selection, half="reporting", strict=False,
                            components=components, rule_outputs={RULE: ("gate",)})
    return scorer


def _spoiled(run, scorer, spoil):  # noqa: F811
    """The epoch's score with its readings edited in place by ``spoil``."""
    real = scorer._values

    def spoiled(cache, forward):
        values = real(cache, forward)
        spoil(cache, values)
        return values

    scorer._values = spoiled
    return _epoch(run, scorer).strata[STRATA[0]]


@pytest.fixture(scope="module")
def rows(records):
    fires, does_not = _gated_rows(records, True), _gated_rows(records, False)
    taken = {row.record_id for row in (*fires, *does_not)}
    return fires, does_not, _ungated_row(records, taken)


@pytest.fixture(scope="module")
def base(run, records, rows):  # noqa: F811
    fires, does_not, _ = rows
    return _scored(run, records, _selection(records, fires, does_not))


def test_the_gated_rows_score_a_finite_rule(base):
    assert -1.0 < base.items[RULE_TESTS][RULE] <= 1.0
    assert base.non_finite == 0


def test_a_row_on_a_record_that_trains_no_gate_changes_nothing(run, records, rows, base):  # noqa: F811
    """The reviewer's case: before the fix the extra row's unread position
    floored the rule at −1 and counted one non-finite reading."""
    fires, does_not, ungated = rows
    score = _scored(run, records, _selection(records, fires, [*does_not, ungated]))
    assert score.items[RULE_TESTS][RULE] == pytest.approx(base.items[RULE_TESTS][RULE])
    assert score.non_finite == 0


def test_a_row_whose_other_entity_is_read_is_read_at_it_alone(run, records, rows, base):  # noqa: F811
    fires, does_not, _ = rows
    first = replace(fires[0], entities=(*fires[0].entities, NOWHERE))
    score = _scored(run, records, _selection(records, [first, *fires[1:]], does_not))
    assert score.items[RULE_TESTS][RULE] == pytest.approx(base.items[RULE_TESTS][RULE])
    assert score.non_finite == 0


def test_a_non_finite_value_a_forward_wrote_still_floors_the_rule(run, records, rows):  # noqa: F811
    fires, does_not, ungated = rows

    def spoil(cache, values):
        values[cache.rules[RULE]["gate"].fires[0][0]] = INF

    scorer = _scorer(run, records, _selection(records, fires, [*does_not, ungated]))
    score = _spoiled(run, scorer, spoil)
    assert score.items[RULE_TESTS][RULE] == -1.0
    assert score.non_finite == 1


def test_a_probe_row_at_an_entity_with_no_slot_is_left_out(run, records, rows):  # noqa: F811
    fires, does_not, _ = rows
    probe = [ProbeRecord(row.record_id, row.game_id, row.entities[0], float(i % 2))
             for i, row in enumerate([*fires[:4], *does_not[:4]])]
    components = (PROBES, RULE_TESTS)
    clean = _scored(run, records, _selection(records, fires, does_not, probe), components)
    extra = ProbeRecord(fires[5].record_id, fires[5].game_id, NOWHERE, 1.0)
    score = _scored(run, records, _selection(records, fires, does_not, [*probe, extra]),
                    components)
    assert score.items[PROBES]["dies"] == pytest.approx(clean.items[PROBES]["dies"])
    assert score.non_finite == 0

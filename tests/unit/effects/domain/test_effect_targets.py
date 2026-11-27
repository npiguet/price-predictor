"""Events to per-entity head targets.

The two rules that shape everything else: an entity is affected when any event
names it (not when some field changed — a spell that targets a creature and
deals 0 damage still affected it), and a field with no event is *absent* rather
than zero, so the loss skips it instead of learning that every unmentioned
counter went to zero.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from effects.domain.effect_targets import derive_targets
from effects.domain.event_schema import Event, EventType
from tests.unit.effects.domain.effect_targets_support import (
    _resolution,
    _spec,
)
from tests.unit.effects.file_builders import fresh_interpreter_env


class TestTheGate:
    def test_an_entity_named_by_any_event_is_affected(self):
        targets = derive_targets(_resolution(
            Event(type=EventType.DAMAGE_DEALT, subjects=("E1",),
                  params={"amount": 3}),
        ))
        assert targets["E1"].affected

    def test_an_entity_no_event_names_is_absent_from_the_targets(self):
        targets = derive_targets(_resolution(
            Event(type=EventType.DAMAGE_DEALT, subjects=("E1",),
                  params={"amount": 3}),
        ))
        assert "E2" not in targets

    def test_a_zero_magnitude_event_still_affects_its_subject(self):
        """A spell that targets and deals 0 damage affected the creature."""
        targets = derive_targets(_resolution(
            Event(type=EventType.DAMAGE_DEALT, subjects=("E1",),
                  params={"amount": 0}),
        ))
        assert targets["E1"].affected
        assert targets["E1"].fields["damage_taken"] == 0.0

    def test_one_event_can_affect_several_subjects(self):
        targets = derive_targets(_resolution(
            Event(type=EventType.DESTROYED, subjects=("E1", "E2", "E3")),
        ))
        assert set(targets) == {"E1", "E2", "E3"}


class TestAbsentFields:
    def test_an_unmentioned_field_is_absent_not_zero(self):
        targets = derive_targets(_resolution(
            Event(type=EventType.DAMAGE_DEALT, subjects=("E1",),
                  params={"amount": 2}),
        ))
        assert "damage_taken" in targets["E1"].fields
        assert "life_delta" not in targets["E1"].fields
        assert "counters_delta_p1p1" not in targets["E1"].fields

    def test_a_record_with_no_events_produces_no_targets(self):
        assert derive_targets(_resolution()) == {}


class TestFieldNames:
    def test_every_produced_field_is_a_real_head_output(self):
        from effects.domain.effect_head_layout import FIELDS_BY_NAME

        record = _resolution(
            Event(type=EventType.DAMAGE_DEALT, subjects=("E1",),
                  params={"amount": 1}),
            Event(type=EventType.COUNTER_CHANGE, subjects=("E1",),
                  params={"counter_type": "P1P1", "delta": 1}),
            Event(type=EventType.LIFE_CHANGE, subjects=("P0",),
                  params={"delta": 1}),
            Event(type=EventType.MANA_PRODUCED, subjects=("P0",),
                  params={"mana_by_color": {"R": 1}}),
        )
        for entry in derive_targets(record).values():
            for name in entry.fields:
                assert name in FIELDS_BY_NAME, name

    @pytest.mark.parametrize("counter", ["P1P1", "M1M1", "LOYALTY"])
    def test_every_tracked_counter_type_names_a_real_field(self, counter):
        from effects.domain.effect_head_layout import FIELDS_BY_NAME

        targets = derive_targets(_resolution(
            Event(type=EventType.COUNTER_CHANGE, subjects=("E1",),
                  params={"counter_type": counter, "delta": 1}),
        ))
        assert set(targets["E1"].fields) <= set(FIELDS_BY_NAME)


def test_deriving_targets_loads_no_torch():
    """Every survey and write worker derives targets record by record, and
    none of them needs torch to do it.

    In a fresh interpreter, because any test before this one may have
    imported torch into this one.
    """
    subprocess.run(
        [sys.executable, "-c",
         "import sys, effects.application.corpus_survey, effects.domain.effect_targets; "
         "assert 'torch' not in sys.modules, 'torch was imported'"],
        check=True, env=fresh_interpreter_env(),
    )


class TestEveryFixtureTargetFitsItsField:
    """Every field's target has the shape its ``FieldSpec`` declares, over every
    curated record of the gen-3 fixture."""

    def test_widths_and_classes(self):
        from tests.unit.effects.builders import curated_fixture_records

        seen = set()
        for record in curated_fixture_records():
            for entry in derive_targets(record).values():
                for name, value in entry.fields.items():
                    _assert_fits_its_field(name, value)
                    seen.add(name)
        assert {"target_legal", "attack_target_legal", "combat_requirement",
                "face_after"} <= seen


def _assert_fits_its_field(name, value):
    """``value``, the target of field ``name``, has the shape its ``FieldSpec``
    declares: its slice is the spec's width, a multi-binary target holds one
    bit per class and a categorical one is a class index."""
    from effects.domain.effect_head_layout import FIELD_SLICES
    from effects.domain.field_types import FieldType

    spec = _spec(name)
    start, end = FIELD_SLICES[name]
    assert end - start == spec.width
    if spec.type is FieldType.MULTI_BINARY:
        assert len(value) == spec.arity, name
    elif spec.type is FieldType.CATEGORICAL:
        assert 0 <= value < spec.arity, name

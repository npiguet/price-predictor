"""What a lethality copy of a paid-X resolution rewrites besides ``refs.x`` (H08).

A paid X is redrawn together with the source's ``x_paid`` and the mana its
``costs_paid`` states; a copy's per-entity labels travel in
``Counterfactual.targets`` and its record keeps only the events the record
heads read (HR1).
"""

from __future__ import annotations

import random
from dataclasses import replace

from effects.domain.aux_head_targets import (
    created_objects_targets,
    global_targets,
    verdict_targets,
)
from effects.domain.effect_targets import derive_targets
from effects.domain.event_schema import Event, EventType
from effects.domain.lethality_counterfactuals import CopyMaterial
from effects.domain.state_snapshot import CostsPaid
from tests.unit.effects.domain.test_lethality_counterfactuals import (
    BOLT,
    Scene,
    _creature,
    _damage,
    _died,
    _line,
    _qualify,
    _record,
    counterfactual,
)

X_BOLT = BOLT.replace("NumDmg$ 3", "NumDmg$ X")


def _paid_x_record(mana: dict[str, int]):
    record = _record([_damage("E1", 4), _died("E1")], [_creature("E1", 4)],
                     scene=Scene(x=4, source_x_paid=4))
    state = record.state
    entities = tuple(
        replace(e, costs_paid=CostsPaid(mana=mana)) if e.id == "S1" else e
        for e in state.entities)
    return replace(record, state=replace(state, entities=entities))


def _copies(record, count=40):
    qualified = _qualify(record, X_BOLT)
    return [counterfactual(CopyMaterial(record, qualified, derive_targets(record),
                                        _line(X_BOLT), X_BOLT), random.Random(seed), 0)
            for seed in range(count)]


def test_a_redrawn_paid_x_rewrites_the_total_mana_the_source_paid():
    record = _paid_x_record({"R": 1, "C": 4})  # {X}{R}, X = 4
    for copy in _copies(record):
        x = copy.record.state.refs.x
        mana = copy.record.state.entity("S1").costs_paid.mana
        assert sum(mana.values()) == x + 1
        assert mana["R"] == 1
        assert all(count >= 0 for count in mana.values())


def _token_created() -> Event:
    return Event(type=EventType.TOKEN_CREATED, subjects=("T1",),
                 params={"token_script_id": "c_a_treasure_sac", "count": 2})


def _bolt_and_treasure():
    record = _paid_x_record({"R": 1, "C": 4})
    effect = Event(type=EventType.CONTINUOUS_EFFECT_CREATED, subjects=("S1",),
                   params={"layers": "pt"})
    payload = replace(record.payload,
                      events=(*record.payload.events, _token_created(), effect))
    return replace(record, payload=payload)


def test_a_copy_keeps_the_events_its_record_heads_read_and_none_of_the_outcome():
    record = _bolt_and_treasure()
    assert derive_targets(record)["E1"].fields
    for copy in _copies(record, 5):
        assert [e.type for e in copy.record.payload.events] == [
            EventType.TOKEN_CREATED, EventType.CONTINUOUS_EFFECT_CREATED]
        assert "E1" not in derive_targets(copy.record)
        assert copy.targets["E1"].fields["damage_taken"] == copy.record.state.refs.x


def test_a_copy_of_a_spell_that_also_creates_a_token_keeps_its_record_head_targets():
    record = _bolt_and_treasure()
    created = created_objects_targets(record)
    assert any(created[0])
    for copy in _copies(record, 5):
        assert created_objects_targets(copy.record) == created
        assert verdict_targets(copy.record) == verdict_targets(record)
        assert global_targets(copy.record) == global_targets(record)

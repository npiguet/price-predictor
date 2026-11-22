"""What a lethality copy of a paid-X resolution rewrites besides ``refs.x`` (H08).

A paid X is redrawn together with the source's ``x_paid`` and the mana its
``costs_paid`` states; a copy's record carries no events, its labels travel
in ``Counterfactual.targets`` alone.
"""

from __future__ import annotations

import random
from dataclasses import replace

from effects.domain.effect_targets import derive_targets
from effects.domain.state_snapshot import CostsPaid
from tests.unit.effects.domain.test_lethality_counterfactuals import (
    BOLT,
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
    record = _record([_damage("E1", 4), _died("E1")], [_creature("E1", 4)], x=4, source_x_paid=4)
    state = record.state
    entities = tuple(
        replace(e, costs_paid=CostsPaid(mana=mana)) if e.id == "S1" else e
        for e in state.entities)
    return replace(record, state=replace(state, entities=entities))


def _copies(record, count=40):
    qualified = _qualify(record, X_BOLT)
    return [counterfactual(record, qualified, derive_targets(record), _line(X_BOLT), X_BOLT,
                           random.Random(seed), 0) for seed in range(count)]


def test_a_redrawn_paid_x_rewrites_the_total_mana_the_source_paid():
    record = _paid_x_record({"R": 1, "C": 4})  # {X}{R}, X = 4
    for copy in _copies(record):
        x = copy.record.state.refs.x
        mana = copy.record.state.entity("S1").costs_paid.mana
        assert sum(mana.values()) == x + 1
        assert mana["R"] == 1
        assert all(count >= 0 for count in mana.values())


def test_a_copy_carries_no_events_so_its_targets_are_not_the_originals():
    record = _paid_x_record({"R": 1, "C": 4})
    assert record.payload.events and derive_targets(record)["E1"].fields
    for copy in _copies(record, 5):
        assert copy.record.payload.events == ()
        assert "E1" not in derive_targets(copy.record)
        assert copy.targets["E1"].fields["damage_taken"] == copy.record.state.refs.x

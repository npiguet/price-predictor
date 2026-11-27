"""A player's mana targets read a ``mana_lost``'s ``mana_by_color``.

The events are spelled as the connector writes them (``LostManaTest`` pins
the Java side on real boards): a drain's effect half holds the mana the pool
lost, and a ``LoseMana`` replacement's rewrite record holds, as its
``incoming``, the mana that would have been lost, with no ``outgoing``,
since the mana became colourless instead. Built by hand: the gen-3 fixture
holds no drain.
"""

from __future__ import annotations

from effects.domain.effect_targets import derive_targets
from effects.domain.event_schema import Event, EventType
from effects.domain.records import RecordKind, ResolutionPayload, RewritePayload
from tests.unit.effects.record_builders import make_record


def _ids() -> dict:
    return {"record_id": "r.0.1", "run_id": "r", "game_id": "r.0.1"}


def test_a_drain_s_loss_is_negative_in_each_colour():
    """Mana Short at a player floating {R}{G}."""
    drained = Event(type=EventType.MANA_LOST, subjects=("P1",),
                    params={"mana_by_color": {"R": 1, "G": 1}, "cause": "E2"})
    record = make_record(RecordKind.RESOLUTION, payload=ResolutionPayload(events=(drained,)),
                         **_ids())

    fields = derive_targets(record)["P1"].fields

    assert (fields["mana_delta_r"], fields["mana_delta_g"]) == (-1.0, -1.0)
    assert "mana_delta_u" not in fields


def test_a_would_be_loss_a_replacement_kept_trains_nothing():
    """Horizon Stone and {U}{U} at a step's end: the proposed loss is no outcome."""
    proposed = Event(type=EventType.MANA_LOST, subjects=("P0",),
                     params={"mode": "LoseMana", "mana_by_color": {"U": 2}})
    payload = RewritePayload(incoming=proposed, outgoing=None)
    record = make_record(RecordKind.REWRITE, payload=payload, moment=None, **_ids())

    targets = derive_targets(record)

    assert "P0" not in targets or "mana_delta_u" not in targets["P0"].fields

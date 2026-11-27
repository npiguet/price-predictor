"""Record codec: each payload kind's members, the payload codecs chosen by
a record's ``(kind, discriminator)``, and a continuous contribution's
members."""

from __future__ import annotations

from dataclasses import replace

import pytest

from effects.domain.event_schema import Event, EventType
from effects.domain.provenance import ProvenanceKey
from effects.domain.records import (
    CombatPayload,
    Contribution,
    EffectRecord,
    Moment,
    PlayabilitySubkind,
    RecordKind,
    ResolutionPayload,
    RewritePayload,
    RewriteResult,
    TriggerPayload,
)
from effects.infrastructure.record_codec import (
    record_from_dict,
    record_to_dict,
)
from tests.unit.effects.infrastructure.record_codec_support import (
    ALL_KINDS,
    _record,
)


class TestTheProbedPerturbation:
    """A damage-step probe records what it stripped, and from whom.

    The keyword alone is ambiguous on a board with two tramplers, and an
    evaluator reproducing the perturbation model-side would strip the wrong
    creature — turning the counterfactual the probe exists for into a
    comparison against a different one.
    """

    def _probe(self, **payload_kwargs) -> EffectRecord:
        return _record(
            kind=RecordKind.COMBAT, moment=None, ability=None,
            fork=True, mirror_of="run.0.7",
            payload=CombatPayload(
                attackers=("E12",),
                events=(Event(type=EventType.DAMAGE_DEALT,
                              params={"amount": 2, "combat": True}),),
                **payload_kwargs,
            ),
        )

    def test_an_observed_combat_record_is_written_without_them_and_reads(self):
        """As the collector writes it: an observed combat names no probe."""
        wire = record_to_dict(
            _record(kind=RecordKind.COMBAT, moment=None, ability=None,
                    payload=CombatPayload(attackers=("E12",)))
        )
        assert not {"probed_keyword", "probed_entity"} & wire["payload"].keys()
        loaded = record_from_dict(wire)
        assert loaded.payload.probed_keyword is None
        assert loaded.payload.attackers == ("E12",)


class TestTheRewriteResult:
    """What the replacement did, and which ability did it.

    ``{incoming, outgoing}`` alone modelled an edit-the-event mechanism Forge
    does not have: it substitutes by running a *different* ability, so the two
    maps came back byte-identical and 87% of the channel was dropped as
    identity rewrites. These four fields are the whole of what those records
    were trying to say, so a serializer bug here is the channel going quiet
    again — and the corpus cannot be recollected.
    """

    def _rewrite(self, **payload_kwargs) -> EffectRecord:
        return _record(
            kind=RecordKind.REWRITE, moment=None,
            payload=RewritePayload(
                incoming=Event(type=EventType.DAMAGE_DEALT,
                               subjects=("P1",), params={"amount": 5}),
                **payload_kwargs,
            ),
        )

    def test_a_substitution_writes_null_rather_than_a_copy(self):
        """The rule the channel's readability rests on: a record whose two
        halves are identical says nothing, and null says "not rewritten"."""
        key = ProvenanceKey(
            script_file="cardsfolder/l/leyline_of_the_void.txt",
            face=0, trait_kind="replacement", index_within_kind=0,
        )
        record = self._rewrite(result=RewriteResult.REPLACED, replaced_by=(key,))
        wire = record_to_dict(record)
        assert wire["payload"]["outgoing"] is None
        assert wire["payload"]["result"] == "replaced"
        loaded = record_from_dict(wire)
        assert loaded.payload.outgoing is None
        assert loaded.payload.replaced_by == (key,)

class TestEveryEventReadsOneCounterSpelling:
    """``+1/+1`` becomes ``P1P1`` on read wherever an event sits, not only in a
    resolution's event list: a rewrite's outgoing event is a target too, and a
    raw spelling there falls through to ``counters_delta_other``."""

    @staticmethod
    def _counter_event() -> Event:
        return Event(type=EventType.COUNTER_CHANGE, subjects=("E1",),
                     params={"counter_type": "+1/+1", "delta": 1})

    def test_a_rewrites_events(self):
        record = _record(
            kind=RecordKind.REWRITE, moment=None,
            payload=RewritePayload(
                incoming=self._counter_event(), outgoing=self._counter_event(),
                result=RewriteResult.UPDATED,
            ),
        )
        loaded = record_from_dict(record_to_dict(record))
        assert loaded.payload.incoming.params["counter_type"] == "P1P1"
        assert loaded.payload.outgoing.params["counter_type"] == "P1P1"

    def test_a_triggers_event(self):
        record = _record(
            kind=RecordKind.TRIGGER, moment=None,
            payload=TriggerPayload(event=self._counter_event(), fired=True),
        )
        loaded = record_from_dict(record_to_dict(record))
        assert loaded.payload.event.params["counter_type"] == "P1P1"


class TestADamageEventReadsItsDamageType:
    """The connector writes ``damage_type`` on damage to a creature (contract
    record-schema-delta: ``normal``, ``m1m1_counters``, ``deathtouch``,
    ``loyalty_loss``); a record carrying it decodes, and one without it too."""

    @pytest.mark.parametrize("damage_type", ["normal", "m1m1_counters", "deathtouch",
                                             "loyalty_loss"])
    def test_it_round_trips(self, damage_type):
        event = Event(type=EventType.DAMAGE_DEALT, subjects=("E1",),
                      params={"amount": 2, "combat": False, "damage_type": damage_type})
        record = _record(payload=ResolutionPayload(events=(event,)))
        loaded = record_from_dict(record_to_dict(record))
        assert loaded.payload.events[0].params["damage_type"] == damage_type

    def test_damage_to_a_player_carries_none(self):
        event = Event(type=EventType.DAMAGE_DEALT, subjects=("P1",), params={"amount": 2})
        loaded = record_from_dict(record_to_dict(_record(payload=ResolutionPayload(
            events=(event,)))))
        assert "damage_type" not in loaded.payload.events[0].params


class TestAContinuousContributionsSetBaseAndControl:
    """``set_pt`` and ``controller`` are present only where a static sets a
    base or gives control (L2)."""

    def test_they_read_and_write_back_where_present(self):
        record = _record(**ALL_KINDS["continuous"])
        record = replace(record, payload=replace(record.payload, contributions=(
            Contribution(entity="E13", set_pt=(1, None), abilities=(), hidden_keywords=(),
                         controller="P1"),)))
        written = record_to_dict(record)["payload"]["contributions"][0]
        assert list(written) == ["entity", "pt_boost", "set_pt", "keywords", "types",
                                 "colors", "abilities", "hidden_keywords", "name",
                                 "controller"]
        assert written["set_pt"] == [1, None]
        assert record_from_dict(record_to_dict(record)) == record

    def test_they_are_absent_where_unset(self):
        written = record_to_dict(_record(**ALL_KINDS["continuous"]))["payload"]
        assert "set_pt" not in written["contributions"][0]
        assert "controller" not in written["contributions"][0]


class TestThePayloadCodecs:
    """One codec per payload type, chosen by the record's ``(kind, discriminator)``."""

    def test_every_payload_type_has_exactly_one_codec(self):
        from effects.domain.records import PAYLOAD_TYPES
        from effects.infrastructure.record_codec import _PAYLOAD_CODECS

        assert set(_PAYLOAD_CODECS) == set(PAYLOAD_TYPES.values())

    @pytest.mark.parametrize("kind, discriminator", [
        (RecordKind.RESOLUTION, None),
        (RecordKind.PLAYABILITY, None),
        (RecordKind.REWRITE, Moment.RESOLUTION),
        (RecordKind.COMBAT, PlayabilitySubkind.BLOCKERS),
    ])
    def test_a_pair_no_payload_is_defined_for_is_refused(self, kind, discriminator):
        # The decoder used to read anything it did not recognise as a
        # blockers payload; an unknown pair is now refused by name.
        from effects.infrastructure.record_codec import _payload_from_json

        with pytest.raises(ValueError, match="no payload is defined"):
            _payload_from_json({"anchor_attacker": "E1"}, kind, discriminator)

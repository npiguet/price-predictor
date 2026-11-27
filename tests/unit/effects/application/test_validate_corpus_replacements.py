"""validate-corpus's replacement checks: a buffered replacement agrees with
its payload, and a rewrite names the ability that ran instead.
"""

from __future__ import annotations

import pytest

from effects.application.corpus_validation.findings import Thresholds
from effects.domain.event_schema import Event, EventType
from effects.domain.provenance import ProvenanceKey
from effects.domain.records import RewriteResult
from tests.unit.effects.application.validate_corpus_ports import validate_corpus
from tests.unit.effects.application.validate_corpus_support import (
    _KEY,
    _healthy,
    _named,
    _substitution,
)


class TestABufferedReplacementAgreesWithItsPayload:
    """A damage replacement that set an ability aside (Swans of Bryn
    Argoll's draw, a ``Prevent$ True`` rider) names it in ``buffered`` and its
    run's events are ``buffered_events``; ``replaced_by`` names it too only
    where the replacement's own ability ran in the event's place and set its
    sub-ability aside (Forcefield's ``ReplaceDamage``)."""

    _DAMAGE = Event(type=EventType.DAMAGE_DEALT, subjects=("E3",), params={"amount": 2})
    _DRAW = Event(type=EventType.CARD_DRAWN, subjects=("P0",), params={"count": 1},
                  attributed_to="root")

    def _buffered(self, record_id: str, result=RewriteResult.REPLACED, **payload_kwargs):
        kwargs = {"buffered": (_KEY,), "buffered_ran": True,
                  "buffered_events": (self._DRAW,), **payload_kwargs}
        return _substitution(record_id, incoming=self._DAMAGE, result=result, **kwargs)

    def test_a_prevention_with_a_rider_passes(self):
        """Gatta and Luzzu: ``prevented``, nothing in ``replaced_by``, the rider buffered."""
        records = _healthy() + [self._buffered("run.0-L1.900", RewriteResult.PREVENTED)]
        assert _named(validate_corpus(records), "outgoing is null on").ok

    #: Forcefield's replacement line: its ``ReplaceDamage`` runs in the
    #: event's place and its sub-ability (exile the effect) is set aside.
    _FORCEFIELD = ProvenanceKey("cardsfolder/f/forcefield.txt", 0, "replacement", 0)

    @pytest.mark.parametrize("ran", [False, True])
    def test_a_replacement_run_in_place_with_its_sub_ability_set_aside_passes(self, ran):
        """Forcefield: one line in both ``replaced_by`` and ``buffered``, its
        sub-ability's run whichever way it went (34 such records in a real
        game, the run's ``buffered_ran`` false)."""
        records = _healthy() + [self._buffered(
            "run.0-L1.900", replaced_by=(self._FORCEFIELD,), buffered=(self._FORCEFIELD,),
            buffered_ran=ran, buffered_events=(self._DRAW,) if ran else ())]
        finding = _named(validate_corpus(records), "outgoing is null on")
        assert finding.ok, finding.detail

    def test_a_buffered_ability_of_another_line_than_the_one_run_instead_fails(self):
        """The ability set aside is the sub-ability of the one that ran, so
        both key to one line: two lines is a record no replacement writes."""
        records = _healthy() + [self._buffered(
            "run.0-L1.900", replaced_by=(self._FORCEFIELD,), buffered=(_KEY,))]
        finding = _named(validate_corpus(records), "outgoing is null on")
        assert not finding.ok
        assert any("run.0-L1.900" in line for line in finding.detail)

    def test_a_keyless_buffered_ability_beside_a_named_one_run_instead_fails(self):
        """One resolver keys both: the sub-ability keys to no line only when
        the ability that ran keys to none either."""
        records = _healthy() + [self._buffered(
            "run.0-L1.900", replaced_by=(self._FORCEFIELD,), buffered=())]
        assert not _named(validate_corpus(records), "outgoing is null on").ok

    def test_a_prevention_naming_an_ability_run_instead_beside_its_rider_fails(self):
        """A ``Prevent$ True`` replacement returns before any ability runs."""
        records = _healthy() + [self._buffered(
            "run.0-L1.900", RewriteResult.PREVENTED,
            replaced_by=(self._FORCEFIELD,), buffered=(self._FORCEFIELD,))]
        assert not _named(validate_corpus(records), "outgoing is null on").ok

    def test_events_of_a_run_that_never_came_fail(self):
        records = _healthy() + [self._buffered("run.0-L1.900", buffered_ran=False)]
        assert not _named(validate_corpus(records), "outgoing is null on").ok

    def test_an_unknown_incoming_event_with_an_outgoing_one_fails(self):
        records = _healthy() + [_substitution(
            "run.0-L1.900", incoming=self._DAMAGE, result=RewriteResult.UPDATED,
            incoming_unknown=True, outgoing=self._DAMAGE)]
        assert not _named(validate_corpus(records), "outgoing is null on").ok

    def test_a_buffered_replacement_names_the_ability_it_set_aside(self):
        """Its ``replaced_by`` is empty by design: ``buffered`` names the ability."""
        records = _healthy() + [self._buffered("run.0-L1.900")]
        assert "3/3" in _named(validate_corpus(records), "ran instead").measured

    def test_a_record_naming_its_ability_both_ways_counts_once(self):
        """Forcefield's ``ReplaceDamage`` runs in place and sets its
        sub-ability aside, so a real record carries both ``replaced_by`` and
        ``buffered``: named once, never a share above one (a real run read
        330/312)."""
        records = _healthy() + [self._buffered("run.0-L1.900", replaced_by=(_KEY,))]
        assert "3/3" in _named(validate_corpus(records), "ran instead").measured

    def test_the_collector_s_goldens_and_the_gen3_fixture_agree_with_their_payloads(self):
        """Every rewrite the collector wrote, the buffered golden's included,
        is a shape the rule accepts."""
        from effects.infrastructure.record_codec import record_from_dict
        from tests.unit.effects.builders import gen3_fixture_records, java_golden_records

        raws = [*java_golden_records(), *gen3_fixture_records()]
        rewrites = [record_from_dict(raw) for raw in raws if raw["kind"] == "rewrite"]
        assert any(r.payload.buffered is not None for r in rewrites)
        finding = _named(validate_corpus(rewrites), "outgoing is null on")
        assert finding.ok, finding.detail


class TestRewritesNameTheAbilityThatRanInstead:
    """Watched, because a ``ReplacementResult$`` script runs no ability at all.

    The ceiling is genuinely below 1 and nobody has measured how far, so this
    reports the number and judges nothing until someone passes a floor.
    """

    def test_it_is_watched_by_default(self):
        finding = _named(validate_corpus(_healthy()), "ran instead")
        assert finding.watched
        assert finding.ok
        assert "watched, no floor" in finding.measured

    def test_a_resolver_that_keys_nothing_reads_as_zero(self):
        records = _healthy() + [_substitution("run.0-L1.900", result=RewriteResult.REPLACED)]
        finding = _named(validate_corpus(records), "ran instead")
        assert "2/3" in finding.measured
        assert finding.ok, "watched findings never fail a run"

    def test_a_floor_turns_the_measurement_into_a_verdict(self):
        records = _healthy() + [_substitution("run.0-L1.900", result=RewriteResult.REPLACED)]
        strict = Thresholds(min_rewrite_replaced_by_rate=0.9)
        finding = _named(validate_corpus(records, strict), "ran instead")
        assert not finding.watched
        assert not finding.ok

    def test_results_that_run_no_ability_are_outside_the_rate(self):
        """``prevented`` never reaches an ability, so counting it would drag
        the rate down and blame the resolver for the handler's control flow."""
        records = _healthy() + [
            _substitution("run.0-L1.900", result=RewriteResult.PREVENTED),
        ]
        assert "2/2" in _named(validate_corpus(records), "ran instead").measured

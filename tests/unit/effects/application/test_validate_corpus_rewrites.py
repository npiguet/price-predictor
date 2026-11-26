"""validate-corpus's rewrite checks: a rewrite says what happened, does not
copy its event, and its outgoing event agrees with its result.
"""

from __future__ import annotations

from effects.domain.event_schema import Event, EventType
from effects.domain.records import RecordKind, RewritePayload, RewriteResult
from tests.unit.effects.application.validate_corpus_ports import validate_corpus
from tests.unit.effects.application.validate_corpus_support import (
    _KEY,
    _healthy,
    _named,
    _rewrite,
    _substitution,
)


class TestRewriteRecordsSayWhatHappened:
    """The field the whole channel hangs off.

    Without ``result`` the payload models an edit-the-event mechanism Forge
    does not have: it substitutes by running a different ability, so the before
    and after maps come back byte-identical, 87% of the hook's calls were
    dropped as identity rewrites, and the channel held 34 records in 1.88M
    against a 7% share of the trainer's mix.
    """

    def test_a_healthy_window_reports_every_record_carrying_a_result(self):
        finding = _named(validate_corpus(_healthy()), "which replacement result")
        assert finding.ok
        assert "2/2" in finding.measured
        assert "updated 2" in finding.measured

    def test_a_record_with_no_result_fails_and_says_which_two_causes(self):
        records = _healthy() + [
            _substitution("run.0-L1.900", outgoing=None),
        ]
        finding = _named(validate_corpus(records), "which replacement result")
        assert not finding.ok
        assert "absent 1" in finding.measured
        assert any("argument slot" in line for line in finding.detail)

    def test_the_histogram_names_every_outcome_the_window_saw(self):
        records = _healthy() + [
            _substitution(f"run.0-L1.{900 + index}", result=result)
            for index, result in enumerate(
                (RewriteResult.REPLACED, RewriteResult.PREVENTED,
                 RewriteResult.SKIPPED, RewriteResult.NOT_REPLACED)
            )
        ]
        measured = _named(
            validate_corpus(records), "which replacement result"
        ).measured
        for name in ("replaced 1", "prevented 1", "skipped 1", "not_replaced 1"):
            assert name in measured

    def test_a_window_with_no_rewrites_says_so_rather_than_failing(self):
        records = [r for r in _healthy() if r.kind is not RecordKind.REWRITE]
        assert _named(validate_corpus(records), "which replacement result").ok


class TestARewriteDoesNotCopyItsEvent:
    """The defect that made the channel unreadable, now measured.

    A record whose two halves are byte-identical cannot be told from a
    replacement that changed a parameter back to its own value, and there is no
    such thing. Null is the honest spelling.
    """

    def _copy(self, record_id: str, result=RewriteResult.REPLACED, **overrides):
        same = Event(type=EventType.DAMAGE_DEALT, params={"amount": 4})
        return _rewrite(
            record_id,
            payload=RewritePayload(incoming=same, outgoing=same, result=result),
            **overrides,
        )

    def test_a_healthy_window_copies_nothing(self):
        finding = _named(validate_corpus(_healthy()), "is not a copy of")
        assert finding.ok
        assert "0/2" in finding.measured

    def test_a_copied_event_fails_and_names_the_record(self):
        records = _healthy() + [self._copy("run.0-L1.900")]
        finding = _named(validate_corpus(records), "is not a copy of")
        assert not finding.ok
        assert "1/3" in finding.measured
        assert any("run.0-L1.900" in line for line in finding.detail)

    def test_a_pre_contract_record_is_not_counted_against_the_writer(self):
        """It had no null to write. Failing on its shards would report a
        defect that was fixed rather than one that is present."""
        same = Event(type=EventType.DAMAGE_DEALT, params={"amount": 4})
        records = _healthy() + [
            _rewrite(
                "run.0-L1.900",
                payload=RewritePayload(incoming=same, outgoing=same),
            ),
        ]
        finding = _named(validate_corpus(records), "is not a copy of")
        assert finding.ok
        assert "0/2" in finding.measured, "the no-result record is not judged"

    def test_a_genuine_edit_back_to_the_same_value_is_still_reported(self):
        """There is no such replacement, so this is the collector, not Magic."""
        records = _healthy() + [
            self._copy("run.0-L1.900", result=RewriteResult.UPDATED),
        ]
        assert not _named(validate_corpus(records), "is not a copy of").ok


class TestOutgoingAgreesWithTheResult:
    """``prevented`` and ``skipped`` return above the ability call.

    Both are bare ``return`` statements in ``executeReplacementInternal``: no
    ability ran and nothing was written to the parameter map. A record of one
    carrying a payload is a reflective listener reading a stale argument slot,
    which nothing compiles against and so nothing else would catch.
    """

    def _with(self, record_id: str, result, **payload_kwargs):
        return _substitution(
            record_id, incoming=Event(type=EventType.DAMAGE_DEALT, params={"amount": 3}),
            result=result, **payload_kwargs,
        )

    def test_a_healthy_window_reports_none(self):
        finding = _named(validate_corpus(_healthy()), "outgoing is null on")
        assert finding.ok
        assert finding.measured.startswith("0 ")

    def test_a_prevented_record_carrying_an_outgoing_fails(self):
        records = _healthy() + [
            self._with(
                "run.0-L1.900", RewriteResult.PREVENTED,
                outgoing=Event(type=EventType.DAMAGE_DEALT,
                               params={"amount": 0}),
            ),
        ]
        finding = _named(validate_corpus(records), "outgoing is null on")
        assert not finding.ok
        assert any("run.0-L1.900" in line for line in finding.detail)

    def test_a_skipped_record_naming_a_substituted_ability_fails(self):
        records = _healthy() + [
            self._with(
                "run.0-L1.900", RewriteResult.SKIPPED, replaced_by=(_KEY,),
            ),
        ]
        assert not _named(validate_corpus(records), "outgoing is null on").ok

    def test_a_prevented_record_with_neither_passes(self):
        records = _healthy() + [
            self._with("run.0-L1.900", RewriteResult.PREVENTED),
        ]
        assert _named(validate_corpus(records), "outgoing is null on").ok

    def test_not_replaced_may_carry_an_outgoing(self):
        """Its prevention branch writes PreventedAmount into the map on the
        way out, so this one is a real shape rather than a stale read."""
        records = _healthy() + [
            self._with(
                "run.0-L1.900", RewriteResult.NOT_REPLACED,
                outgoing=Event(type=EventType.DAMAGE_DEALT,
                               params={"amount": 0}),
            ),
        ]
        assert _named(validate_corpus(records), "outgoing is null on").ok

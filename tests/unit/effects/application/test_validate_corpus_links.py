"""validate-corpus's link and acting-line checks: the halves of a cast pair
up, and every record names the lines that acted.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from effects.application.corpus_validation.findings import Thresholds
from effects.application.validate_corpus import validate_corpus
from effects.domain.event_schema import AttributionKind, Event, EventType
from effects.domain.provenance import ProvenanceKey
from effects.domain.records import ResolutionPayload
from tests.unit.effects.application.validate_corpus_support import (
    _KEY,
    _activation,
    _healthy,
    _named,
    _resolution,
    _rewrite,
    _snapshot,
    _trigger,
)


def _mode(option: int) -> ProvenanceKey:
    """A mode of the Lightning Bolt key's line, as a charm's mode key reads."""
    return ProvenanceKey(
        _KEY.script_file, _KEY.face, _KEY.trait_kind, _KEY.index_within_kind, option,
    )


class TestLinkHalvesPair:
    def test_an_unpaired_half_fails(self):
        records = _healthy() + [
            _resolution(f"run.0-L1.{index}", link_id=f"lonely.{index}")
            for index in range(200, 210)
        ]
        finding = _named(validate_corpus(records), "link_id")
        assert not finding.ok
        assert "unpaired" in finding.measured

    @staticmethod
    def _cast(index: int, source: str = "E7", game: str = "run.0-L1.0"):
        """A cost half: written at the cast, no outcome (FR-042a)."""
        snapshot = replace(_snapshot(), refs=replace(_snapshot().refs, source=source))
        return _activation(f"run.0-L1.{index}", link_id=f"cast.{index}",
                           game_id=game, state=snapshot)

    def test_cost_halves_none_of_which_meets_its_effect_half_fail(self):
        """V5 #18: a lone cost half is a paired link (its item countered, or on
        the stack at the game's end), so 500 of them read 500/500 paired; a
        window where no cost half meets its effect half is the collector."""
        records = [r for r in _healthy() if r.link_id is None] + [
            self._cast(300 + i) for i in range(200)]
        finding = _named(validate_corpus(records), "every cost half")
        assert not finding.ok
        assert "0/200" in finding.measured and "200 alone" in finding.measured

    @pytest.mark.parametrize("event", [
        Event(type=EventType.SPELL_COUNTERED, subjects=("E7",),
              attributed_to=AttributionKind.ROOT),
        Event(type=EventType.ABILITY_COUNTERED,
              params={"ability": {"entity": "E7", "key": _KEY.as_dict()}},
              attributed_to=AttributionKind.ROOT),
    ], ids=["a spell", "an activated ability, by its host"])
    def test_a_lone_cost_half_whose_item_was_countered_is_explained(self, event):
        countered = _resolution("run.0-L1.990", game_id="run.0-L1.0",
                                payload=ResolutionPayload(events=(event,)))
        paired = [half for i in range(5) for half in (
            self._cast(400 + i, source=f"E{40 + i}"),
            _resolution(f"run.0-L1.{500 + i}", link_id=f"cast.{400 + i}"))]
        records = [r for r in _healthy() if r.link_id is None] + paired + [
            self._cast(300), countered]
        finding = _named(validate_corpus(records), "every cost half")
        assert finding.ok and finding.watched
        assert "5/6" in finding.measured and "1 alone, 1 of them countered" in finding.measured

    def test_a_link_id_shared_by_three_halves_on_one_line_fails(self):
        """A link id whose tail is a JVM-static counter collides like an id does.

        Three halves acting through a root key are a collision, not a modal
        resolution: only effect halves acting through mode keys of one charm
        may share a cost half (FR-029d).
        """
        records = _healthy() + [
            _resolution("run.0-L1.910", link_id="link.0"),
        ]
        finding = _named(validate_corpus(records), "link_id")
        assert not finding.ok
        assert "3 half(s)" in " ".join(finding.detail)

    def test_a_charms_cost_half_joins_one_effect_half_per_mode(self):
        """FR-029d: the widened link_id, one cost record and N mode halves."""
        records = _healthy() + [
            _activation("run.0-L1.920", link_id="charm.0"),
            _resolution("run.0-L1.921", link_id="charm.0", ability=(_mode(0),)),
            _resolution("run.0-L1.922", link_id="charm.0", ability=(_mode(2),)),
        ]
        finding = _named(validate_corpus(records), "link_id")
        assert finding.ok
        assert "3 half(s)" in " ".join(finding.detail)

    def test_mode_halves_of_two_different_charms_do_not_join(self):
        other = ProvenanceKey("cardsfolder/c/cryptic_command.txt", 0, "spell", 0, 1)
        records = _healthy() + [
            _activation(f"run.0-L1.{930 + i}", link_id=f"charm.{i}") for i in range(10)
        ] + [
            half
            for i in range(10)
            for half in (
                _resolution(f"run.0-L1.{950 + i}", link_id=f"charm.{i}", ability=(_mode(0),)),
                _resolution(f"run.0-L1.{970 + i}", link_id=f"charm.{i}", ability=(other,)),
            )
        ]
        assert not _named(validate_corpus(records), "link_id").ok

    def test_two_root_effect_halves_on_one_cost_do_not_join(self):
        records = _healthy() + [
            _activation(f"run.0-L1.{930 + i}", link_id=f"pair.{i}") for i in range(10)
        ] + [
            half
            for i in range(10)
            for half in (
                _resolution(f"run.0-L1.{950 + i}", link_id=f"pair.{i}"),
                _resolution(f"run.0-L1.{970 + i}", link_id=f"pair.{i}"),
            )
        ]
        assert not _named(validate_corpus(records), "link_id").ok

    def test_a_few_truncated_tails_stay_within_tolerance(self):
        """A killed worker loses its last block, so some halves have no partner."""
        paired = []
        for index in range(100):
            paired += [
                _activation(f"run.0-L1.{2 * index}", link_id=f"pair.{index}"),
                _resolution(f"run.0-L1.{2 * index + 1}", link_id=f"pair.{index}"),
            ]
        records = paired + [_activation("run.0-L1.999", link_id="tail")]
        finding = _named(validate_corpus(records), "link_id")
        assert finding.ok, finding.measured


class TestActingLines:
    def test_a_trigger_that_names_no_line_fails(self):
        """727,308 trigger records said only that some unnamed trigger saw an event."""
        records = _healthy() + [
            _trigger(f"run.0-L1.{index}", ability=None) for index in range(300, 310)
        ]
        finding = _named(validate_corpus(records), "trigger and rewrite")
        assert not finding.ok
        assert "carry the field" in finding.measured

    def test_a_rewrite_that_names_no_line_fails(self):
        records = _healthy() + [
            _rewrite(f"run.0-L1.{index}", ability=None) for index in range(300, 310)
        ]
        assert not _named(validate_corpus(records), "trigger and rewrite").ok

    def test_resolution_records_below_the_keyed_floor_fail(self):
        """41% of the first corpus's resolution records resolved to no key."""
        records = _healthy() + [
            _resolution(
                f"run.0-L1.{index}", ability=(), ability_unresolved="unindexable",
            )
            for index in range(400, 410)
        ]
        finding = _named(validate_corpus(records), "resolution records name")
        assert not finding.ok
        assert "unindexable" in " ".join(finding.detail)

    def test_a_reason_is_reported_under_its_own_kind_alone(self):
        records = _healthy() + [
            _resolution(
                f"run.0-L1.{index}", ability=(), ability_unresolved="unindexable",
            )
            for index in range(400, 410)
        ]
        findings = validate_corpus(records)
        assert "unindexable" in " ".join(_named(findings, "resolution records name").detail)
        assert "unindexable" not in " ".join(_named(findings, "trigger and rewrite").detail)

    def test_the_keyed_floor_is_configurable(self):
        records = _healthy() + [
            _resolution(
                f"run.0-L1.{index}", ability=(), ability_unresolved="engine_effect",
            )
            for index in range(400, 410)
        ]
        loose = Thresholds(min_keyed_rate=0.1)
        assert _named(validate_corpus(records, loose), "resolution records name").ok

    def test_a_kind_absent_from_the_window_is_not_a_failure(self):
        """A two-minute window need not contain a continuous record."""
        finding = _named(validate_corpus(_healthy()), "continuous records name")
        assert finding.ok
        assert "no records of this kind" in finding.measured

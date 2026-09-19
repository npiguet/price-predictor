"""Gate 1's stratum and its plumbing (T099).

Gate 1 blocks shipping, and until this existed it was reported only when its
baseline was *missing* — supplying the baseline produced no gate-1 line at all,
so the acceptance criterion for User Story 1 silently never ran.

The stratum is where the gate is won or lost. The identity baseline holds a free
vector per ability text, so on a text it saw in training it recalls the answer
without reading a word; comparing the two over such texts measures memorization
on both sides. Only texts absent from training separate them.
"""

from __future__ import annotations

import pytest

from effects.application.gate_one import (
    GateOneMetrics,
    acting_text,
    training_texts,
    unique_text_records,
)
from effects.domain.provenance import ProvenanceKey
from effects.domain.records import (
    CombatPayload,
    EffectRecord,
    Moment,
    RecordKind,
    ResolutionPayload,
)
from effects.domain.state_snapshot import GlobalState, StateSnapshot

_SNAPSHOT = StateSnapshot(
    global_=GlobalState(
        turn=1, phase="main1", active="P0", priority="P0", stack_size=0,
    ),
    players=(),
    entities=(),
)


class _Batcher:
    """Stands in for the real batcher: a key maps to a text by its index."""

    def __init__(self, texts: dict[int, str]) -> None:
        self._texts = texts

    def text_of(self, key):
        return self._texts.get(key.index_within_kind)


def _key(index: int) -> ProvenanceKey:
    return ProvenanceKey("cardsfolder/a/a.txt", 0, "spell", index)


def _record(
    index: int | None, *, kind: RecordKind = RecordKind.RESOLUTION,
    game: str = "g1",
) -> EffectRecord:
    payload = (
        ResolutionPayload() if kind is RecordKind.RESOLUTION
        else CombatPayload()
    )
    return EffectRecord(
        record_id=f"r.{index}", run_id="r", timestamp="t", game_id=game,
        kind=kind,
        moment=Moment.RESOLUTION if kind is RecordKind.RESOLUTION else None,
        actor_player="P0", state=_SNAPSHOT, payload=payload,
        # None, not (): the envelope requires the kinds without an acting line
        # to carry no ability field at all.
        ability=(_key(index),) if index is not None else None,
    )


@pytest.fixture
def batcher() -> _Batcher:
    return _Batcher({0: "deal 3 damage", 1: "draw a card", 2: "gain 2 life"})


class TestActingText:
    def test_the_first_key_that_resolves_wins(self, batcher):
        """A rendered line merged from several traits carries several keys, and
        they all name the same line."""
        record = _record(0)
        assert acting_text(record, batcher) == "deal 3 damage"

    def test_a_record_with_no_ability_has_no_text(self, batcher):
        assert acting_text(_record(None), batcher) is None

    def test_an_unresolvable_key_yields_no_text(self, batcher):
        assert acting_text(_record(99), batcher) is None


class TestTheStratum:
    def test_a_text_seen_in_training_is_excluded(self, batcher):
        """The baseline recalls it, so the comparison measures nothing."""
        training = [_record(0)]
        held_out = [_record(0), _record(1)]
        stratum = unique_text_records(
            held_out, batcher, seen=training_texts(training, batcher),
        )
        assert [acting_text(r, batcher) for r in stratum] == ["draw a card"]

    def test_a_text_absent_from_training_is_kept(self, batcher):
        stratum = unique_text_records(
            [_record(2)], batcher, seen=training_texts([_record(0)], batcher),
        )
        assert len(stratum) == 1

    def test_the_same_text_appearing_twice_is_still_kept(self, batcher):
        """The rule is "absent from training", not "seen once" — a text held out
        twice is held out twice, and both records score."""
        stratum = unique_text_records(
            [_record(1), _record(1)], batcher, seen=set(),
        )
        assert len(stratum) == 2

    def test_a_combat_record_is_not_in_the_stratum(self, batcher):
        """The envelope forbids it an acting line, so it could not be
        stratified by text even if the kind filter were removed."""
        combat = _record(None, kind=RecordKind.COMBAT)
        assert acting_text(combat, batcher) is None
        assert unique_text_records([combat], batcher, seen=set()) == []

    def test_a_record_whose_text_does_not_resolve_is_skipped(self, batcher):
        assert unique_text_records([_record(99)], batcher, seen=set()) == []

    def test_an_empty_training_set_keeps_every_resolution_record(self, batcher):
        stratum = unique_text_records(
            [_record(0), _record(1), _record(2)], batcher, seen=set(),
        )
        assert len(stratum) == 3


class TestTrainingTexts:
    def test_it_collects_every_acting_text(self, batcher):
        assert training_texts([_record(0), _record(1)], batcher) == {
            "deal 3 damage", "draw a card",
        }

    def test_records_without_an_acting_line_contribute_nothing(self, batcher):
        assert training_texts([_record(None)], batcher) == set()


class TestMetrics:
    def test_the_three_margins_are_one_object(self):
        """They must describe the same records, so they travel together."""
        metrics = GateOneMetrics(0.5, 0.4, 1.2)
        assert metrics.affected_gate_f1 == 0.5
        assert metrics.zone_outcome_accuracy == 0.4
        assert metrics.mean_poisson_deviance == 1.2

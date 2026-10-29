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
    training_texts,
    unique_text_records,
)
from effects.domain.provenance import ProvenanceKey, first_text
from effects.domain.records import (
    EffectRecord,
    RecordKind,
    acting_keys,
)
from effects.domain.state_snapshot import GlobalState, StateSnapshot
from tests.unit.effects.builders import make_record

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
    return make_record(
        kind, record_id=f"r.{index}", run_id="r", game_id=game,
        # None, not (): the envelope requires the kinds without an acting line
        # to carry no ability field at all.
        ability=(_key(index),) if index is not None else None,
    )


def _acting(records) -> set:
    return {acting_keys(record) for record in records}


def _text(record, batcher) -> str | None:
    return first_text(acting_keys(record), batcher.text_of)


@pytest.fixture
def batcher() -> _Batcher:
    return _Batcher({0: "deal 3 damage", 1: "draw a card", 2: "gain 2 life"})


class TestTheStratum:
    def test_a_text_seen_in_training_is_excluded(self, batcher):
        """The baseline recalls it, so the comparison measures nothing."""
        training = [_record(0)]
        held_out = [_record(0), _record(1)]
        stratum = unique_text_records(
            held_out, batcher.text_of, seen=training_texts(_acting(training), batcher.text_of),
        )
        assert [_text(r, batcher) for r in stratum] == ["draw a card"]

    def test_a_text_absent_from_training_is_kept(self, batcher):
        stratum = unique_text_records(
            [_record(2)], batcher.text_of,
            seen=training_texts(_acting([_record(0)]), batcher.text_of),
        )
        assert len(stratum) == 1

    def test_the_same_text_appearing_twice_is_still_kept(self, batcher):
        """The rule is "absent from training", not "seen once" — a text held out
        twice is held out twice, and both records score."""
        stratum = unique_text_records(
            [_record(1), _record(1)], batcher.text_of, seen=set(),
        )
        assert len(stratum) == 2

    def test_a_combat_record_is_not_in_the_stratum(self, batcher):
        """The envelope forbids it an acting line, so it could not be
        stratified by text even if the kind filter were removed."""
        combat = _record(None, kind=RecordKind.COMBAT)
        assert _text(combat, batcher) is None
        assert unique_text_records([combat], batcher.text_of, seen=set()) == []

    def test_a_record_whose_text_does_not_resolve_is_skipped(self, batcher):
        assert unique_text_records([_record(99)], batcher.text_of, seen=set()) == []

    def test_an_empty_training_set_keeps_every_resolution_record(self, batcher):
        stratum = unique_text_records(
            [_record(0), _record(1), _record(2)], batcher.text_of, seen=set(),
        )
        assert len(stratum) == 3


class TestTrainingTexts:
    def test_it_collects_every_acting_text(self, batcher):
        assert training_texts(_acting([_record(0), _record(1)]), batcher.text_of) == {
            "deal 3 damage", "draw a card",
        }

    def test_records_without_an_acting_line_contribute_nothing(self, batcher):
        assert training_texts(_acting([_record(None)]), batcher.text_of) == set()


class TestMarginAccumulator:
    """The three margins pool every chunk's entities, then reduce once."""

    def _chunk(self, affected_damage: float, observed_damage: float = 2.0):
        import torch

        from effects.domain.effect_head_layout import (
            FIELD_SLICES,
            GATE_INDEX,
            PER_ENTITY_WIDTH,
            ZONE_OUTCOMES,
        )
        from effects.domain.entity_field_targets import EntityFieldTargets

        gathered = torch.zeros(1, 2, PER_ENTITY_WIDTH)
        gathered[0, 0, GATE_INDEX] = 3.0   # predicted affected
        gathered[0, 1, GATE_INDEX] = -3.0  # predicted unaffected
        start, _ = FIELD_SLICES["zone_outcome"]
        gathered[0, 0, start + ZONE_OUTCOMES.index("died")] = 5.0
        damage, _ = FIELD_SLICES["damage_taken"]
        gathered[0, 0, damage] = torch.log(torch.tensor(affected_damage))
        targets = EntityFieldTargets.everywhere({
            "zone_outcome": torch.tensor(
                [[float(ZONE_OUTCOMES.index("died")), 0.0]],
            ),
            "damage_taken": torch.tensor([[observed_damage, 0.0]]),
        })
        return gathered, torch.tensor([[1.0, 0.0]]), targets, torch.ones(1, 2)

    def test_a_perfect_chunk_scores_perfectly(self):
        from effects.application.gate_one import MarginAccumulator
        from effects.domain.effect_head_layout import FIELDS_BY_NAME

        margins = MarginAccumulator(
            (FIELDS_BY_NAME["zone_outcome"], FIELDS_BY_NAME["damage_taken"]),
        )
        margins.add(*self._chunk(2.0))
        metrics = margins.metrics()
        assert metrics.affected_gate_f1 == pytest.approx(1.0)
        assert metrics.zone_outcome_accuracy == pytest.approx(1.0)
        assert metrics.mean_poisson_deviance == pytest.approx(0.0, abs=1e-6)

    def test_a_count_netted_negative_scores_as_none_as_it_trains(self):
        """A heal netted into ``damage_taken`` trains as zero damage, not as
        the heal's size."""
        from effects.application.gate_one import MarginAccumulator
        from effects.domain.effect_head_layout import FIELDS_BY_NAME

        margins = MarginAccumulator((FIELDS_BY_NAME["damage_taken"],))
        margins.add(*self._chunk(1e-6, observed_damage=-2.0))
        assert margins.metrics().mean_poisson_deviance == pytest.approx(0.0, abs=1e-4)

    def test_nothing_added_is_unmeasured_rather_than_perfect(self):
        import math

        from effects.application.gate_one import MarginAccumulator

        metrics = MarginAccumulator(()).metrics()
        assert math.isnan(metrics.affected_gate_f1)
        assert math.isnan(metrics.mean_poisson_deviance)

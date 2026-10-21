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
            held_out, batcher, seen=training_texts(_acting(training), batcher),
        )
        assert [acting_text(r, batcher) for r in stratum] == ["draw a card"]

    def test_a_text_absent_from_training_is_kept(self, batcher):
        stratum = unique_text_records(
            [_record(2)], batcher, seen=training_texts(_acting([_record(0)]), batcher),
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
        assert training_texts(_acting([_record(0), _record(1)]), batcher) == {
            "deal 3 damage", "draw a card",
        }

    def test_records_without_an_acting_line_contribute_nothing(self, batcher):
        assert training_texts(_acting([_record(None)]), batcher) == set()


class TestMetrics:
    def test_the_three_margins_are_one_object(self):
        """They must describe the same records, so they travel together."""
        metrics = GateOneMetrics(0.5, 0.4, 1.2)
        assert metrics.affected_gate_f1 == 0.5
        assert metrics.zone_outcome_accuracy == 0.4
        assert metrics.mean_poisson_deviance == 1.2


class TestPerRecordLosses:
    """``measure``'s per-record rows sum the trainer's own loss terms (T104)."""

    def test_a_record_s_total_is_the_per_entity_loss_of_that_record_alone(self):
        import torch

        from effects.application.breakdowns import TOTAL
        from effects.application.gate_one import _record_losses
        from effects.domain.effect_head_layout import (
            PER_ENTITY_FIELDS,
            PER_ENTITY_WIDTH,
            EntityFieldTargets,
            FieldType,
        )
        from effects.domain.effect_losses import per_entity_loss

        torch.manual_seed(0)
        fields = PER_ENTITY_FIELDS
        outputs = torch.randn(2, 3, PER_ENTITY_WIDTH)
        gate = torch.tensor([[1.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
        mask = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]])
        targets = EntityFieldTargets.everywhere({
            spec.name: torch.zeros(
                (2, 3, spec.arity) if spec.type is FieldType.MULTI_BINARY else (2, 3)
            )
            for spec in fields
        })
        for row in range(2):
            expected, _ = per_entity_loss(
                outputs[row : row + 1], gate[row : row + 1],
                targets.select(slice(row, row + 1)),
                mask[row : row + 1], fields=fields,
            )
            losses = _record_losses(row, outputs, gate, targets, mask, fields)
            assert losses[TOTAL] == pytest.approx(float(expected), rel=1e-5)

    def test_a_record_with_nothing_affected_supervises_only_the_gate(self):
        import torch

        from effects.application.breakdowns import TOTAL
        from effects.application.gate_one import _record_losses
        from effects.domain.effect_head_layout import (
            PER_ENTITY_FIELDS,
            PER_ENTITY_WIDTH,
            EntityFieldTargets,
        )

        outputs = torch.zeros(1, 2, PER_ENTITY_WIDTH)
        losses = _record_losses(
            0, outputs, torch.zeros(1, 2), EntityFieldTargets({}, {}), torch.ones(1, 2),
            PER_ENTITY_FIELDS,
        )
        assert set(losses) == {"gate", TOTAL}

    def test_every_term_equals_the_masked_sum_over_that_record(self):
        """Each reported term, not only the total, is the gate BCE over the real
        entities plus each field's loss over the target-affected ones — the
        same numbers a hand-masked sum gives, on targets that are not all zero."""
        import torch
        from torch.nn import functional

        from effects.application.breakdowns import TOTAL
        from effects.application.gate_one import _record_losses
        from effects.domain.effect_head_layout import (
            FIELD_SLICES,
            GATE_INDEX,
            PER_ENTITY_FIELDS,
            PER_ENTITY_WIDTH,
            EntityFieldTargets,
            FieldType,
        )
        from effects.domain.effect_losses import field_loss

        generator = torch.Generator().manual_seed(3)
        fields = PER_ENTITY_FIELDS
        outputs = torch.randn(3, 4, PER_ENTITY_WIDTH, generator=generator)
        gate = torch.tensor([
            [1.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
        ])
        mask = torch.tensor([
            [1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0],
        ])

        def target(spec):
            match spec.type:
                case FieldType.MULTI_BINARY:
                    return torch.randint(0, 2, (3, 4, spec.arity), generator=generator)
                case FieldType.CATEGORICAL:
                    return torch.randint(0, spec.arity, (3, 4), generator=generator)
                case FieldType.SIGNED_DELTA:
                    return torch.randint(-3, 4, (3, 4), generator=generator)
                case FieldType.COUNT:
                    return torch.randint(0, 5, (3, 4), generator=generator)
            return torch.randint(0, 2, (3, 4), generator=generator)

        values = {spec.name: target(spec).float() for spec in fields}
        targets = EntityFieldTargets.everywhere(values)
        for row in range(3):
            real = mask[row].bool()
            affected = real & gate[row].bool()
            expected = {"gate": float(functional.binary_cross_entropy_with_logits(
                outputs[row, :, GATE_INDEX][real], gate[row][real], reduction="sum",
            ))}
            if bool(affected.any()):
                for spec in fields:
                    start, end = FIELD_SLICES[spec.name]
                    expected[spec.name] = float(field_loss(
                        spec, outputs[row, :, start:end][affected],
                        values[spec.name][row][affected],
                    ))
            losses = _record_losses(row, outputs, gate, targets, mask, fields)
            assert set(losses) == {*expected, TOTAL}
            for name, value in expected.items():
                assert losses[name] == pytest.approx(value, rel=1e-6, abs=1e-6), name
            assert losses[TOTAL] == pytest.approx(sum(expected.values()), rel=1e-6)


class TestMarginAccumulator:
    """The three margins pool every chunk's entities, then reduce once."""

    def _chunk(self, affected_damage: float):
        import torch

        from effects.domain.effect_head_layout import (
            FIELD_SLICES,
            GATE_INDEX,
            PER_ENTITY_WIDTH,
            ZONE_OUTCOMES,
            EntityFieldTargets,
        )

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
            "damage_taken": torch.tensor([[2.0, 0.0]]),
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

    def test_nothing_added_is_unmeasured_rather_than_perfect(self):
        import math

        from effects.application.gate_one import MarginAccumulator

        metrics = MarginAccumulator(()).metrics()
        assert math.isnan(metrics.affected_gate_f1)
        assert math.isnan(metrics.mean_poisson_deviance)

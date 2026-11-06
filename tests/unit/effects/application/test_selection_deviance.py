"""The deviance sample's per-record constant-prediction losses (research E4)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import pytest
import torch

from effects.application.selection_deviance import TargetDevianceLosses
from effects.domain.effect_head_layout import FIELDS_BY_NAME, GATE_TERM
from effects.domain.entity_field_targets import EntityFieldTargets, EntityTargetBatch


@dataclass(frozen=True)
class _Record:
    record_id: str
    game_id: str


@dataclass(frozen=True)
class _Weights:
    elsewhere: float = 1.0
    gate_entities: frozenset | None = None

    def weight(self, output: str) -> float:
        return self.elsewhere


#: Per record: the gate at two entities, and ``tap_state`` at each.
_GATE = {"a": [1, 1], "b": [1, 0], "c": [0, 0], "d": [1, 1]}
_TAP = {"a": [1, 0], "b": [1, 0], "c": [0, 0], "d": [1, 1]}


def _prepare(records):
    gate = torch.tensor([_GATE[r.record_id] for r in records], dtype=torch.float32)
    tap = torch.tensor([_TAP[r.record_id] for r in records], dtype=torch.float32)
    ones = torch.ones_like(gate, dtype=torch.bool)
    return EntityTargetBatch(
        gate=gate,
        fields=EntityFieldTargets({"tap_state": tap}, {"tap_state": ones}),
        mask=torch.ones_like(gate), index=torch.zeros_like(gate, dtype=torch.long),
    ), {}


def _bce(p: float, y: float) -> float:
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))


def test_each_record_carries_the_constant_loss_of_its_scored_targets() -> None:
    records = [_Record(rid, f"g{rid}") for rid in "abcd"]
    weights = {"hard": _Weights(0.0), "soft": _Weights(0.5)}
    reader = TargetDevianceLosses(
        _prepare, {}, fields=(FIELDS_BY_NAME["tap_state"],), batch_size=3,
        weights_of=lambda record, mode: (_Weights() if record.record_id != "a"
                                         else weights[mode]))
    out = reader.losses(iter(records))
    # The gate trains on all eight entities, five of them affected.
    p = 5 / 8
    gate = [sum(_bce(p, y) for y in _GATE[r]) for r in "abcd"]
    assert out[GATE_TERM]["soft"] == pytest.approx([0.5 * gate[0], *gate[1:]], rel=1e-5)
    assert out[GATE_TERM]["hard"] == pytest.approx([0.0, *gate[1:]], rel=1e-5)
    # tap_state is scored only where the gate fired: a's two, b's first, d's two.
    q = 4 / 5
    tap = [_bce(q, 1) + _bce(q, 0), _bce(q, 1), 0.0, 2 * _bce(q, 1)]
    assert out["tap_state"]["soft"] == pytest.approx([0.5 * tap[0], *tap[1:]], rel=1e-5)


def test_a_field_whose_targets_never_vary_has_no_loss() -> None:
    records = [_Record("d", "g")]
    out = TargetDevianceLosses(_prepare, {}, fields=(FIELDS_BY_NAME["tap_state"],),
                               weights_of=lambda record, mode: _Weights()).losses(records)
    assert out["tap_state"]["hard"] == [0.0]


def test_the_hard_masks_gate_weighs_the_share_of_entities_its_rule_names() -> None:
    records = [_Record("a", "g"), _Record("b", "g")]
    out = TargetDevianceLosses(
        _prepare, {}, fields=(), modes=("hard",),
        weights_of=lambda record, mode: _Weights(0.0, frozenset({"E1"})),
    ).losses(records)
    p = 3 / 4
    whole = [_bce(p, 1) * 2, _bce(p, 1) + _bce(p, 0)]
    # One named entity of the record's two gate entities: half the record's loss.
    assert out[GATE_TERM]["hard"] == pytest.approx([w / 2 for w in whole], rel=1e-5)


# ── the record heads ────────────────────────────────────────────────────


@dataclass(frozen=True)
class _ByOutput:
    """A copy's weights, one per output name, one wherever unnamed."""

    weights: dict
    gate_entities: frozenset | None = None

    def weight(self, output: str) -> float:
        return self.weights.get(output, 1.0)


#: ``can_play`` per record (None: the record sets no verdict field), and
#: whether the record supervises the created-objects head with how many.
_CAN_PLAY = {"a": 1.0, "b": 0.0, "c": 1.0, "d": None}
_CREATED = {"a": 1, "b": 0, "c": 2, "d": None}


def _prepare_with_heads(records):
    from effects.domain.aux_heads import Head
    from effects.domain.effect_head_layout import (
        CREATED_OBJECTS_WIDTH,
        VERDICT_LAYOUT,
        CreatedSlot,
    )

    batch, _ = _prepare(records)
    verdict = torch.zeros(len(records), VERDICT_LAYOUT.target_width)
    flags = torch.zeros(len(records), len(VERDICT_LAYOUT.fields), dtype=torch.bool)
    created = torch.zeros(len(records), CREATED_OBJECTS_WIDTH)
    supervises = torch.zeros(len(records), dtype=torch.bool)
    for row, record in enumerate(records):
        value = _CAN_PLAY[record.record_id]
        if value is not None:
            verdict[row, VERDICT_LAYOUT.target_slices["can_play"][0]] = value
            flags[row, 0] = True
        made = _CREATED[record.record_id]
        if made is not None:
            supervises[row] = True
            created[row, CreatedSlot.PRESENT] = float(made > 0)
            created[row, CreatedSlot.COUNT] = float(made)
    return batch, {Head.VERDICT: (verdict, flags),
                   Head.CREATED_OBJECTS: (created, supervises, torch.ones_like(created))}


def test_verdict_fields_and_created_objects_terms_are_items_of_their_own() -> None:
    records = [_Record(rid, f"g{rid}") for rid in "abcd"]
    out = TargetDevianceLosses(_prepare_with_heads, {}, fields=(),
                               weights_of=lambda record, mode: _ByOutput({})).losses(records)
    assert "verdict.can_play" in out and "created_objects.count" in out
    assert "created_objects.present" in out and "gate" in out
    # A head no record supervises has no item, and a verdict field none sets neither.
    assert not any(name.startswith("global.") for name in out)
    assert "verdict.affordable" not in out
    # can_play: the constant is 2/3 over a, b and c; d sets no verdict field.
    p = 2 / 3
    assert out["verdict.can_play"]["hard"] == pytest.approx(
        [_bce(p, 1), _bce(p, 0), _bce(p, 1), 0.0], rel=1e-5)


def test_record_head_items_sum_to_the_constant_predictors_floor() -> None:
    from effects.domain.loss_floor import TargetSample, constant_predictor_losses
    from effects.domain.selection_set import deviance_items

    records = [_Record(rid, f"g{rid}") for rid in "abcd"]
    out = TargetDevianceLosses(_prepare_with_heads, {}, fields=(),
                               weights_of=lambda record, mode: _ByOutput({})).losses(records)
    sample = TargetSample()
    batch, heads = _prepare_with_heads(records)
    sample.add(batch, heads)
    floor = deviance_items(constant_predictor_losses(sample, fields=()))
    for item in ("verdict.can_play", "created_objects.present", "created_objects.count"):
        assert sum(out[item]["soft"]) / len(records) == pytest.approx(floor[item], rel=1e-4)


def test_a_record_heads_items_weigh_each_copy_as_the_mask_weighs_its_output() -> None:
    records = [_Record(rid, f"g{rid}") for rid in "abcd"]
    weights = {"hard": _ByOutput({"can_play": 0.0, "created_objects": 0.0}),
               "soft": _ByOutput({"can_play": 0.25, "created_objects": 0.5})}
    out = TargetDevianceLosses(
        _prepare_with_heads, {}, fields=(),
        weights_of=lambda record, mode: weights[mode] if record.record_id == "a" else _ByOutput({}),
    ).losses(records)
    whole = TargetDevianceLosses(_prepare_with_heads, {}, fields=(),
                                 weights_of=lambda record, mode: _ByOutput({})).losses(records)
    for item, share in (("verdict.can_play", 0.25), ("created_objects.count", 0.5)):
        assert out[item]["hard"] == pytest.approx([0.0, *whole[item]["hard"][1:]])
        assert out[item]["soft"] == pytest.approx(
            [share * whole[item]["soft"][0], *whole[item]["soft"][1:]])

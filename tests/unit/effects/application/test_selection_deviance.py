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
    )


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

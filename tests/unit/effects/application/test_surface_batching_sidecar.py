"""The batcher's sidecar wiring: dropped keys, baseline variants, memoised joins."""

from __future__ import annotations

import pytest
import torch

from effects.domain.effect_head_input import SlotKind
from tests.unit.effects.application.surface_batching_support import (
    _ACT,
    _ANTHEM,
    _LINES,
    _PHANTOM,
    _batcher,
    _record,
    _Sidecars,
)
from tests.unit.effects.builders import (
    make_batcher,
    slots_of_kind,
)


def test_a_dropped_key_gets_no_ability_token():
    rows = {"creatures you control get +1/+1": 0, "deals 3 damage to any target": 1}
    surface = _batcher().surface_builder(rows).build(_record())
    assert [slot.e for slot in slots_of_kind(surface, SlotKind.ABILITY)] == [0]


def _prepared(variant: str):
    """One record prepared under ``variant``'s masks, through the real pipeline."""
    from effects.domain.ability_tokenizer import AbilityTokenizer
    from effects.domain.baselines import baseline_masks
    from effects.domain.slot_widths import layout_widths

    record = _record()
    batcher = make_batcher(
        tokenizer=AbilityTokenizer({"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[MASK]": 3}),
        sidecars=_Sidecars(), masks=baseline_masks(variant), surface="script",
        e_dim=4, widths=layout_widths(), device=torch.device("cpu"),
    )
    prepared = batcher.prepare([record])
    return prepared.surfaces[0], prepared.inputs


def _columns(surface, kind):
    return [index for index, slot in enumerate(surface.slots) if slot.kind is kind]


def test_the_state_only_variant_keeps_the_zero_token_for_a_real_line():
    surface, inputs = _prepared("state-only")
    (ability,) = _columns(surface, SlotKind.ABILITY)
    assert inputs.e_rows[0, ability] == -1
    assert torch.all(inputs.e_vectors[0, ability] == 0.0)


def test_the_state_only_variant_zeroes_the_acting_lines_e():
    surface, inputs = _prepared("state-only")
    (act,) = _columns(surface, SlotKind.ACT)
    assert inputs.e_rows[0, act] == -1


def test_the_no_state_variant_keeps_the_acting_lines_e():
    """FR-094: every input zeroed except [ACT] and the record-kind flag."""
    surface, inputs = _prepared("no-state")
    (act,) = _columns(surface, SlotKind.ACT)
    (ability,) = _columns(surface, SlotKind.ABILITY)
    assert inputs.e_rows[0, act] >= 0
    assert inputs.e_rows[0, ability] == -1


def test_the_no_state_variant_keeps_only_the_record_kind_flags_of_global():
    from effects.domain.effect_head_input import KIND_FLAG_WIDTH

    full = _prepared("full")[1].slot_features[SlotKind.GLOBAL][0, 0]
    _surface, inputs = _prepared("no-state")
    blank = inputs.slot_features[SlotKind.GLOBAL][0, 0]
    assert torch.all(blank[:-KIND_FLAG_WIDTH] == 0.0)
    assert torch.equal(blank[-KIND_FLAG_WIDTH:], full[-KIND_FLAG_WIDTH:])
    assert full[:-KIND_FLAG_WIDTH].abs().sum() > 0
    assert torch.all(inputs.slot_features[SlotKind.CARD] == 0.0)


class _Counting(_Sidecars):
    """Records every join, which is what the memo exists to stop repeating."""

    def __init__(self) -> None:
        self.calls: list = []

    def line_for(self, key):
        self.calls.append(key)
        return _LINES.get(key)


class _Mismatching(_Sidecars):
    """The sidecar disagrees with the record: the fail-loudly case."""

    def line_for(self, key):
        raise KeyError(
            f"provenance key {key} appears in neither the lines nor the "
            "dropped_keys"
        )


def test_a_key_is_joined_once_however_often_the_batch_asks():
    # Three keys, asked for by `batch_texts` and then three more times per
    # record inside `SurfaceBuilder.build`, over two passes.
    sidecars = _Counting()
    batcher = _batcher(sidecars)
    records = [_record()]
    for _ in range(2):
        texts = batcher.batch_texts(records)
        rows = {text: row for row, text in enumerate(sorted(texts))}
        batcher.surface_builder(rows).build(records[0])

    assert sidecars.calls == [_ACT, _PHANTOM, _ANTHEM]


def test_a_sidecar_mismatch_is_never_memoised():
    # A mismatch is the one answer the memo must not keep: swallowing it once
    # would turn "the sidecar does not describe this card" into no text at all.
    batcher = _batcher(_Mismatching())
    for _ in range(3):
        with pytest.raises(KeyError, match="neither the lines"):
            batcher.text_of(_ACT)


def test_a_sidecar_mismatch_raises_out_of_the_batcher():
    batcher = _batcher(_Mismatching())
    with pytest.raises(KeyError, match="neither the lines"):
        batcher.batch_texts([_record()])

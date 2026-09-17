"""``SurfaceBatcher`` hands the builder the sidecar's answer to "is this a line".

The rule itself lives in the domain builder; this pins the wiring, because a
batcher that forgot to pass the predicate would silently keep every phantom
token and every number it reports would still move.
"""

from __future__ import annotations

import pytest
import torch

from effects.application.surface_batching import SurfaceBatcher
from effects.application.train_effect_model import VariantMasks
from effects.domain.effect_head_input import SlotKind
from effects.domain.provenance import ProvenanceKey, SidecarLine
from effects.domain.records import EffectRecord, Moment, RecordKind, ResolutionPayload
from effects.domain.state_snapshot import EntityState, GlobalState, StateSnapshot

_ACT = ProvenanceKey("cardsfolder/l/lightning_bolt.txt", 0, "spell", 0)
_ANTHEM = ProvenanceKey("cardsfolder/a/anthem.txt", 0, "static", 0)
_PHANTOM = ProvenanceKey("cardsfolder/a/anthem.txt", 0, "spell", 0)

_LINES = {
    _ACT: SidecarLine(line_index=0, line_kind="spell", provenance=(_ACT,),
                      script_text="deals 3 damage to any target"),
    _ANTHEM: SidecarLine(line_index=0, line_kind="static", provenance=(_ANTHEM,),
                         script_text="creatures you control get +1/+1"),
}


class _Sidecars:
    """The two answers a real cache gives: a line, or None for a dropped key."""

    def line_for(self, key):
        return _LINES.get(key)

    def prose_for(self, key):
        return None


def _record() -> EffectRecord:
    return EffectRecord(
        record_id="run.0.1", run_id="run", timestamp="t", game_id="run.0.1",
        kind=RecordKind.RESOLUTION, moment=Moment.RESOLUTION, actor_player="P0",
        ability=(_ACT,), payload=ResolutionPayload(),
        state=StateSnapshot(
            global_=GlobalState(turn=1, phase="main1", active="P0", priority="P0", stack_size=1),
            players=(),
            entities=(EntityState(id="E1", name="Anthem", zone="battlefield",
                                  controller="P0", printed=(_PHANTOM, _ANTHEM)),),
        ),
    )


def _batcher(**masks) -> SurfaceBatcher:
    return SurfaceBatcher(
        tokenizer=None, sidecars=_Sidecars(), masks=VariantMasks(**masks),
        surface="script", e_dim=4, widths={}, device=torch.device("cpu"),
    )


def test_a_dropped_key_gets_no_ability_token():
    rows = {"creatures you control get +1/+1": 0, "deals 3 damage to any target": 1}
    surface = _batcher().surface_for(_record(), rows)
    assert [slot.e for slot in surface.of_kind(SlotKind.ABILITY)] == [0]


def test_the_state_only_variant_keeps_the_zero_token_for_a_real_line():
    rows = {"creatures you control get +1/+1": 0, "deals 3 damage to any target": 1}
    surface = _batcher(zero_e=True).surface_for(_record(), rows)
    abilities = surface.of_kind(SlotKind.ABILITY)
    assert len(abilities) == 1
    assert abilities[0].e == (0.0,) * 4


def test_a_sidecar_mismatch_raises_out_of_the_batcher():
    class _Mismatching(_Sidecars):
        def line_for(self, key):
            raise KeyError(
                f"provenance key {key} appears in neither the lines nor the "
                "dropped_keys"
            )

    batcher = _batcher()
    batcher.sidecars = _Mismatching()
    with pytest.raises(KeyError, match="neither the lines"):
        batcher.batch_texts([_record()])

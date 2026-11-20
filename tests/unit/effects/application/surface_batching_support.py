"""Builders the surface-batching tests share: a sidecar double, one record, one batcher."""

from __future__ import annotations

import torch

from effects.application.surface_batching import SurfaceBatcher
from effects.domain.baselines import BaselineMasks
from effects.domain.provenance import ProvenanceKey, SidecarLine
from effects.domain.records import (
    EffectRecord,
)
from effects.domain.state_snapshot import (
    GlobalState,
)
from tests.unit.effects.builders import (
    lines_of,
    make_batcher,
    make_entity,
    make_record,
    make_snapshot,
)

_ACT = ProvenanceKey("cardsfolder/l/lightning_bolt.txt", 0, "spell", 0)
_ANTHEM = ProvenanceKey("cardsfolder/a/anthem.txt", 0, "static", 0)
_PHANTOM = ProvenanceKey("cardsfolder/a/anthem.txt", 0, "spell", 0)

_LINES = {
    _ACT: SidecarLine(line_index=0, line_kind="spell", provenance=(_ACT,),
                      script_text="deals 3 damage to any target", prose_offset=0),
    _ANTHEM: SidecarLine(line_index=0, line_kind="static", provenance=(_ANTHEM,),
                         script_text="creatures you control get +1/+1", prose_offset=0),
}


class _Sidecars:
    """The two answers a real cache gives: a line, or None for a dropped key."""

    def line_for(self, key):
        return _LINES.get(key)

    def prose_for(self, key):
        """A converted card's line has prose; here it spells the script, which
        the script surface reads."""
        line = _LINES.get(key)
        return line.script_text if line is not None else None

    def get(self, script_file):
        raise KeyError(f"{script_file}: no sidecar, so no charm options")


def _record() -> EffectRecord:
    return make_record(
        record_id="run.0.1", run_id="run", game_id="run.0.1", ability=(_ACT,),
        state=make_snapshot(
            global_=GlobalState(turn=1, phase="main1", active="P0", priority="P0", stack_size=1),
            entities=(make_entity("E1", "Anthem", lines=lines_of(printed=(_PHANTOM, _ANTHEM))),),
        ),
    )


def _batcher(sidecars=None, **masks) -> SurfaceBatcher:
    return make_batcher(
        tokenizer=None, sidecars=sidecars or _Sidecars(), masks=BaselineMasks(**masks),
        surface="script", e_dim=4, widths={}, device=torch.device("cpu"),
    )

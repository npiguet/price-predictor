"""What gate 2's tests share: the boards they score and the batcher that builds them.

One first-striking attacker and one blocker, whose lines a sidecar stand-in
answers for, built through a real batcher whose ``e`` comes from a free table.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch

from effects.application.surface_batching import IdentityTable, SurfaceBatcher
from effects.domain.baselines import BaselineMasks
from effects.domain.provenance import ProvenanceKey, SidecarLine
from effects.domain.records import CombatPayload, EffectRecord, RecordKind
from effects.domain.state_snapshot import (
    CardLine,
    CombatStatus,
    EntityState,
    PlayerState,
    Refs,
    StateSnapshot,
)
from tests.unit.effects.batch_builders import BatcherSettings, make_batcher
from tests.unit.effects.damage_step_boards import (
    CreatureTraits,
    creature,
    damage_step_global,
)

# ── a board: one first-striking attacker, one blocker ────────────────────

_FIRST_STRIKE = ProvenanceKey("cardsfolder/w/white_knight.txt", 0, "static", 0)
_PUMP = ProvenanceKey("cardsfolder/w/white_knight.txt", 0, "activated", 0)
_OTHER = ProvenanceKey("cardsfolder/g/grizzly_bears.txt", 0, "static", 0)
#: An Equipment's static granting first strike: on gen-3 the line it grants is
#: keyed by this static, and only the line's fragment says which keyword it is.
_EQUIPPED = ProvenanceKey("cardsfolder/b/bonesplitter.txt", 0, "static", 0)

_LINES = {
    _EQUIPPED: SidecarLine(
        line_index=0, line_kind="static", provenance=(_EQUIPPED,),
        script_text="Equipped creature has first strike.", prose_offset=0,
    ),
    _FIRST_STRIKE: SidecarLine(
        line_index=0, line_kind="static", provenance=(_FIRST_STRIKE,),
        script_text="First Strike", prose_offset=0,
    ),
    _PUMP: SidecarLine(
        line_index=1, line_kind="activated", provenance=(_PUMP,),
        script_text="W: this creature gets +1/+0", prose_offset=0,
    ),
    _OTHER: SidecarLine(
        line_index=0, line_kind="static", provenance=(_OTHER,),
        script_text="Trample", prose_offset=0,
    ),
}


class _Sidecars:
    def line_for(self, key):
        return _LINES.get(key)

    def line_or_none(self, key):
        return _LINES.get(key)

    def prose_for(self, key):
        """A converted card's line has prose; here it spells the script, which
        the script surface reads."""
        line = _LINES.get(key)
        return line.script_text if line is not None else None

    def get(self, script_file):
        raise KeyError(f"{script_file}: no sidecar, so no charm options")


def _active_statics(lines: tuple[CardLine, ...]) -> tuple[CardLine, ...]:
    """``lines`` as gen-3 writes them: a static line carries ``active``."""
    return tuple(replace(line, active=True) if line.kind == "static" else line
                 for line in lines)


def _creature(entity_id: str, power: int, toughness: int, **traits: Any) -> EntityState:
    return creature(entity_id, power, toughness, CreatureTraits(active_statics=True, **traits))


def _combat_record(
    *entities: EntityState, game: str = "g1", substep: str = "regular",
) -> EffectRecord:
    return EffectRecord(
        record_id=f"{game}.1", run_id="run", timestamp="t", game_id=game,
        kind=RecordKind.COMBAT, actor_player="P0",
        payload=CombatPayload(attackers=("A",)),
        state=StateSnapshot(
            global_=damage_step_global(substep),
            players=(
                PlayerState(id="P0", life=20, hand=3, library=30, graveyard=1,
                            max_land_plays=1),
                PlayerState(id="P1", life=17, hand=4, library=28, graveyard=2,
                            max_land_plays=1),
            ),
            entities=entities, refs=Refs(targets_on_stack=()),
        ),
    )


def _printed_board(
    game: str = "g1", substep: str = "first_strike",
) -> EffectRecord:
    """The attacker prints first strike; the blocker prints nothing relevant.

    The first-strike step's record by default, the one record the first-strike
    row reads: the attacker's 2 damage lands on the 2/2 blocker and kills it.
    """
    attacker = _creature(
        "A", 2, 2, printed=(_FIRST_STRIKE, _PUMP),
        combat=CombatStatus(attacking="P1", blocked_by=("B0",), became_blocked=True),
    )
    blocker = _creature(
        "B0", 2, 2, controller="P1", printed=(_OTHER,),
        combat=CombatStatus(blocking=("A",)),
    )
    return _combat_record(attacker, blocker, game=game, substep=substep)


def _batcher() -> SurfaceBatcher:
    """A real batcher whose ``e`` comes from a free table rather than a model.

    The identity baseline's own path, borrowed here because it needs no
    tokenizer and no encoder: what these tests are about is the surface, and an
    encoder would only be a second thing that could fail.
    """
    return make_batcher(BatcherSettings(
        tokenizer=None, sidecars=_Sidecars(),
        masks=BaselineMasks(identity_embedding=True),
        surface="script", e_dim=4, widths={}, device=torch.device("cpu"),
        identity_table=IdentityTable(
            [line.script_text for line in _LINES.values()], 4,
        ),
    ))


class _NoEncoder(torch.nn.Module):
    """The encoder the identity batcher never calls.

    It is still a module, because the forward puts both halves of the model
    into evaluation mode before it reads anything and back after.
    """

    def forward(self, **batch):
        raise AssertionError("the identity batcher must not encode")

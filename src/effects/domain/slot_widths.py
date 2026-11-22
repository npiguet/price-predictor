"""Each slot kind's feature width: a constant of the input layout.

The widths are what the slot builders of :mod:`effects.domain.effect_head_input`
emit, and no builder's width depends on the record it reads. They are computed
once per process, by running a blank record with one blank player and one
blank card through those builders, so they cannot drift from what a batch
holds; no record is ever read to find them (finding 19.1).
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import cache
from types import MappingProxyType

from effects.domain.effect_head_input import (
    ABILITY_FEATURE_WIDTH,
    SlotKind,
    act_features,
    card_features,
    global_features,
    player_features,
)
from effects.domain.records import EffectRecord, Moment, RecordKind, ResolutionPayload
from effects.domain.state_snapshot import EntityState, GlobalState, PlayerState, StateSnapshot


def _blank_record() -> EffectRecord:
    """A resolution with one blank player and one blank card and nothing else."""
    player = PlayerState(id="P1", life=0, hand=0, library=0, graveyard=0)
    card = EntityState(id="E1", name="", zone="battlefield", controller="P1")
    return EffectRecord(
        record_id="", run_id="", timestamp="", game_id="",
        kind=RecordKind.RESOLUTION, moment=Moment.RESOLUTION, actor_player="P1",
        state=StateSnapshot(
            global_=GlobalState(turn=0, phase="", active="P1", priority=None, stack_size=0),
            players=(player,), entities=(card,),
        ),
        payload=ResolutionPayload(),
    )


@cache
def _widths() -> Mapping[SlotKind, int]:
    record = _blank_record()
    return MappingProxyType({
        SlotKind.GLOBAL: len(global_features(record)),
        SlotKind.ACT: len(act_features(record)),
        SlotKind.PLAYER: len(player_features(record.state.players[0], record)),
        SlotKind.CARD: len(card_features(record.state.entities[0], record)),
        SlotKind.ABILITY: ABILITY_FEATURE_WIDTH,
    })


def layout_widths() -> dict[SlotKind, int]:
    """Each slot kind's feature width, the trunk's input sizes; a fresh dict
    on every call, computed once per process."""
    return dict(_widths())

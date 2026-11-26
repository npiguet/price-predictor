"""Builders the ``test_effect_targets*`` modules share: a record around a
payload, a resolution of events, a record on a board of entities and two
players, and the head layout's lookups by name."""

from __future__ import annotations

from dataclasses import dataclass

from effects.domain.effect_head_layout import ZONE_OUTCOMES
from effects.domain.records import (
    EffectRecord,
    RecordKind,
    ResolutionPayload,
)
from effects.domain.state_snapshot import EntityState, Refs
from tests.unit.effects.builders import make_record


def _record(payload, kind=RecordKind.RESOLUTION, **overrides) -> EffectRecord:
    defaults: dict = {"record_id": "r.0.1", "run_id": "r", "game_id": "r.0.1"}
    defaults.update(overrides)
    return make_record(kind, payload=payload, **defaults)


def _resolution(*events) -> EffectRecord:
    return _record(ResolutionPayload(events=tuple(events)))


def _players(*ids, library=30):
    from effects.domain.state_snapshot import PlayerState

    return tuple(
        PlayerState(id=pid, life=20, hand=5, library=library, graveyard=0) for pid in ids
    )


@dataclass(frozen=True, slots=True)
class Board:
    """What a record's snapshot holds beside its two players, P0 and P1: the
    entities, and the refs of the acting ability."""

    entities: tuple[EntityState, ...] = ()
    refs: Refs = Refs()

    @classmethod
    def of(cls, *entities: EntityState, refs: Refs = Refs()) -> Board:
        return cls(entities=tuple(entities), refs=refs)


def _on_board(payload, board: Board, *, kind=RecordKind.RESOLUTION, **overrides):
    from tests.unit.effects.builders import make_snapshot

    state = make_snapshot(
        players=_players("P0", "P1"), entities=board.entities, refs=board.refs,
    )
    return _record(payload, kind=kind, state=state, **overrides)


def _entity(entity_id, **overrides):
    from tests.unit.effects.builders import make_entity

    return make_entity(entity_id, entity_id, **overrides)


def _zone(name):
    return ZONE_OUTCOMES.index(name)


def _spec(name):
    from effects.domain.effect_head_layout import FIELDS_BY_NAME

    return FIELDS_BY_NAME[name]

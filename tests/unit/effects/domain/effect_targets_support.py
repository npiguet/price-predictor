"""Builders the ``test_effect_targets*`` modules share: a record around a
payload, a resolution of events, a record on a board of entities and two
players, and the head layout's lookups by name."""

from __future__ import annotations

from effects.domain.effect_head_layout import ZONE_OUTCOMES
from effects.domain.records import (
    EffectRecord,
    RecordKind,
    ResolutionPayload,
)
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


def _on_board(payload, *entities, kind=RecordKind.RESOLUTION, refs=None, **overrides):
    from effects.domain.state_snapshot import Refs
    from tests.unit.effects.builders import make_snapshot

    state = make_snapshot(
        players=_players("P0", "P1"), entities=tuple(entities), refs=refs or Refs(),
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

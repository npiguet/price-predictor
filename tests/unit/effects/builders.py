"""Shared builders for the effect-record unit tests.

Each test module used to hand-roll its own ``_record`` / ``_snapshot`` /
``_entity``. These three build the plain case; a test that needs something
specific passes it as an override, or keeps a small local wrapper that does.
"""

from __future__ import annotations

from effects.domain.event_schema import Event, EventType
from effects.domain.records import (
    ActivationPayload,
    CombatPayload,
    ContinuousPayload,
    Costs,
    EffectRecord,
    Moment,
    PlayabilityAttackersPayload,
    PlayabilityBlockersPayload,
    PlayabilityDecisionPayload,
    PlayabilitySubkind,
    RecordKind,
    ResolutionOutcome,
    ResolutionPayload,
    RewritePayload,
    TriggerPayload,
)
from effects.domain.state_snapshot import EntityState, GlobalState, StateSnapshot

_EVENT = Event(type=EventType.LIFE_CHANGE, params={"delta": 1})


def make_snapshot(**overrides) -> StateSnapshot:
    """An empty turn-1 board; ``overrides`` replace any ``StateSnapshot`` field."""
    defaults: dict = {
        "global_": GlobalState(
            turn=1, phase="main1", active="P0", priority="P0", stack_size=0,
        ),
        "players": (),
        "entities": (),
    }
    defaults.update(overrides)
    return StateSnapshot(**defaults)


def make_entity(entity_id: str, name: str, **overrides) -> EntityState:
    """A battlefield permanent controlled by ``P0``."""
    defaults: dict = {
        "id": entity_id, "name": name, "zone": "battlefield", "controller": "P0",
    }
    defaults.update(overrides)
    return EntityState(**defaults)


def _default_payload(kind: RecordKind, moment, subkind):
    if kind is RecordKind.RESOLUTION:
        if moment is Moment.ACTIVATION:
            return ActivationPayload(
                costs=Costs(), outcome=ResolutionOutcome.RESOLVED,
            )
        return ResolutionPayload()
    if kind is RecordKind.PLAYABILITY:
        return {
            PlayabilitySubkind.DECISION: PlayabilityDecisionPayload,
            PlayabilitySubkind.ATTACKERS: PlayabilityAttackersPayload,
            PlayabilitySubkind.BLOCKERS: lambda: PlayabilityBlockersPayload(
                anchor_attacker="E1",
            ),
        }[subkind]()
    return {
        RecordKind.COMBAT: CombatPayload,
        RecordKind.CONTINUOUS: ContinuousPayload,
        RecordKind.TRIGGER: lambda: TriggerPayload(event=_EVENT, fired=True),
        RecordKind.REWRITE: lambda: RewritePayload(
            incoming=_EVENT, outgoing=_EVENT,
        ),
    }[kind]()


def make_record(kind: RecordKind = RecordKind.RESOLUTION, **overrides) -> EffectRecord:
    """A valid record of ``kind`` with every envelope field defaulted.

    ``moment`` (resolution) and ``subkind`` (playability) select the default
    payload; any field, ``payload`` included, can be overridden.
    """
    game_id = overrides.get("game_id", "g1")
    defaults: dict = {
        "record_id": f"{game_id}.1", "run_id": "run", "timestamp": "t",
        "game_id": game_id, "kind": kind, "actor_player": "P0",
        "state": make_snapshot(),
    }
    moment = overrides.pop("moment", None)
    subkind = overrides.pop("subkind", None)
    if kind is RecordKind.RESOLUTION:
        moment = moment or Moment.RESOLUTION
    elif kind is RecordKind.PLAYABILITY:
        subkind = subkind or PlayabilitySubkind.DECISION
    # A discriminator on the wrong kind is forwarded as given, so a test can
    # still build the record the envelope check is meant to refuse.
    if moment is not None:
        defaults["moment"] = moment
    if subkind is not None:
        defaults["subkind"] = subkind
    if "payload" not in overrides:
        defaults["payload"] = _default_payload(kind, moment, subkind)
    defaults.update(overrides)
    return EffectRecord(**defaults)


def slots_of_kind(surface, kind) -> tuple:
    """A token surface's slots of one ``SlotKind``, in surface order."""
    return tuple(slot for slot in surface.slots if slot.kind is kind)

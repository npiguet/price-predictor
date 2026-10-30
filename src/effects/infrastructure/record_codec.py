"""The JSON codec for one effect record: its wire form and back.

Every conversion is written out rather than reflected from the dataclass, so
a field renamed in Python never silently renames itself in a corpus that
cannot be rebuilt. Reading and writing whole shards is ``shard_io``'s.

Each block of the record is one ``_x_to_json`` / ``_x_from_json`` pair here,
except the two values whose wire form more than one file shares: an
:class:`Event` (records, and the validator's event digests) and a
:class:`ProvenanceKey` (records, sidecars, checkpoints' probe sets) serialise
themselves beside their fields, in the domain, so every format writes them
alike.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import orjson

from effects.domain.event_schema import Event
from effects.domain.keyword_names import normalize_keyword
from effects.domain.provenance import ProvenanceKey
from effects.domain.records import (
    PAYLOAD_TYPES,
    ActivationPayload,
    Candidate,
    CollectionMode,
    CombatPayload,
    ContinuousPayload,
    Contribution,
    Costs,
    EffectRecord,
    ForbiddenEntity,
    Moment,
    PlayabilityAttackersPayload,
    PlayabilityBlockersPayload,
    PlayabilityDecisionPayload,
    PlayabilitySubkind,
    RecordKind,
    ResolutionOutcome,
    ResolutionPayload,
    RewritePayload,
    RewriteResult,
    TriggerPayload,
)
from effects.domain.state_snapshot import (
    CombatStatus,
    EntityState,
    GlobalState,
    GrantedTemporary,
    InclusionTier,
    PlayerState,
    PowerToughness,
    Refs,
    StackExtras,
    StateSnapshot,
    normalize_counter_type,
)

#: Envelope keys this code models. Anything else on a line is a field a newer
#: writer added and is kept in ``EffectRecord.extra_fields``.
_KNOWN_ENVELOPE_KEYS = frozenset({
    "record_id", "run_id", "timestamp", "game_id", "kind", "moment", "subkind",
    "link_id", "mirror_of", "variant_of", "mode", "interventional", "fork",
    "synthetic", "random_seat", "what_if", "actor_player", "ability",
    "ability_unresolved", "state", "payload",
})


# ── provenance keys ─────────────────────────────────────────────────────


def _keys_to_json(keys: tuple[ProvenanceKey, ...]) -> list[dict]:
    return [key.as_dict() for key in keys]


#: Every key parsed so far, by its raw JSON fields: a shard repeats the same
#: few hundred keys thousands of times. Bounded by the card pool's traits.
_INTERNED_KEYS: dict[tuple, ProvenanceKey] = {}


def _key_from_json(item: dict) -> ProvenanceKey:
    raw = (
        item.get("script_file"), item.get("face"), item.get("trait_kind"),
        item.get("index_within_kind"), item.get("option"),
    )
    key = _INTERNED_KEYS.get(raw)
    if key is None:
        key = _INTERNED_KEYS[raw] = ProvenanceKey.from_dict(item)
    return key


def _keys_from_json(data: Any) -> tuple[ProvenanceKey, ...]:
    return tuple(_key_from_json(item) for item in data or ())


# ── events ──────────────────────────────────────────────────────────────


def _events_to_json(events: tuple[Event, ...]) -> list[dict]:
    return [event.as_dict() for event in events]


def _events_from_json(data: Any) -> tuple[Event, ...]:
    return tuple(_event_from_json(item) for item in data or ())


def _event_from_json(item: Any) -> Event:
    """One event, with its counter type in this package's spelling.

    Normalized on read, like keywords, because collected shards carry Forge's
    printed spelling and cannot be rewritten.
    """
    event = Event.from_dict(item)
    counter = event.params.get("counter_type")
    if isinstance(counter, str):
        return replace(
            event,
            params={**event.params, "counter_type": normalize_counter_type(counter)},
        )
    return event


# ── snapshot ────────────────────────────────────────────────────────────


def _optional[T, R](convert: Callable[[T], R], value: T | None) -> R | None:
    """``convert(value)``, or None for an absent block: null, or empty on read."""
    return convert(value) if value else None


def _pt_to_json(pt: PowerToughness) -> dict:
    return {
        "base": list(pt.base),
        "boosts": list(pt.boosts),
        "counters": list(pt.counters),
    }


def _pt_from_json(data: dict) -> PowerToughness:
    return PowerToughness(
        base=tuple(data["base"]),
        boosts=tuple(data.get("boosts", (0, 0))),
        counters=tuple(data.get("counters", (0, 0))),
    )


def _combat_status_to_json(combat: CombatStatus) -> dict:
    return {
        "attacking": combat.attacking,
        "blocking": list(combat.blocking),
        "blocked_by": list(combat.blocked_by),
        "became_blocked": combat.became_blocked,
    }


def _combat_status_from_json(data: dict) -> CombatStatus:
    return CombatStatus(
        attacking=data.get("attacking"),
        blocking=tuple(data.get("blocking", ())),
        blocked_by=tuple(data.get("blocked_by", ())),
        became_blocked=data.get("became_blocked", False),
    )


def _granted_temporary_to_json(granted: GrantedTemporary) -> dict:
    return {
        "keywords": list(granted.keywords),
        "abilities": _keys_to_json(granted.abilities),
    }


def _granted_temporary_from_json(data: dict) -> GrantedTemporary:
    return GrantedTemporary(
        keywords=tuple(normalize_keyword(k) for k in data.get("keywords", ())),
        abilities=_keys_from_json(data.get("abilities")),
    )


def _stack_extras_to_json(extras: StackExtras) -> dict:
    return {
        "targets": list(extras.targets),
        "per_target_amounts": dict(extras.per_target_amounts),
        "up_to_counts": dict(extras.up_to_counts),
    }


def _stack_extras_from_json(data: dict) -> StackExtras:
    return StackExtras(
        targets=tuple(data.get("targets", ())),
        per_target_amounts=dict(data.get("per_target_amounts", {})),
        up_to_counts=dict(data.get("up_to_counts", {})),
    )


def _entity_to_json(entity: EntityState) -> dict:
    return {
        "id": entity.id, "name": entity.name, "zone": entity.zone,
        "controller": entity.controller, "face": entity.face,
        "copy_source": entity.copy_source,
        "token_script_id": entity.token_script_id,
        "types": list(entity.types), "subtypes": list(entity.subtypes),
        "supertypes": list(entity.supertypes), "colors": list(entity.colors),
        "mana_value": entity.mana_value,
        "pt": _optional(_pt_to_json, entity.pt),
        "tapped": entity.tapped, "sick": entity.sick, "damage": entity.damage,
        "counters": dict(entity.counters),
        "combat": _optional(_combat_status_to_json, entity.combat),
        "attached_to": entity.attached_to,
        "face_down": entity.face_down,
        "granted_attached": _keys_to_json(entity.granted_attached),
        "granted_temporary": _granted_temporary_to_json(entity.granted_temporary),
        "printed": _keys_to_json(entity.printed),
        "stack_extras": _optional(_stack_extras_to_json, entity.stack_extras),
    }


def _entity_from_json(data: dict) -> EntityState:
    return EntityState(
        id=data["id"], name=data["name"], zone=data["zone"],
        controller=data["controller"], face=data.get("face", 0),
        copy_source=data.get("copy_source"),
        token_script_id=data.get("token_script_id"),
        types=tuple(data.get("types", ())),
        subtypes=tuple(data.get("subtypes", ())),
        supertypes=tuple(data.get("supertypes", ())),
        colors=tuple(data.get("colors", ())),
        mana_value=data.get("mana_value", 0),
        pt=_optional(_pt_from_json, data.get("pt")),
        tapped=data.get("tapped", False), sick=data.get("sick", False),
        damage=data.get("damage", 0), counters=dict(data.get("counters", {})),
        combat=_optional(_combat_status_from_json, data.get("combat")),
        attached_to=data.get("attached_to"),
        face_down=data.get("face_down", False),
        granted_attached=_keys_from_json(data.get("granted_attached")),
        granted_temporary=_granted_temporary_from_json(
            data.get("granted_temporary") or {}
        ),
        printed=_keys_from_json(data.get("printed")),
        stack_extras=_optional(_stack_extras_from_json, data.get("stack_extras")),
    )


def _global_to_json(global_: GlobalState) -> dict:
    return {
        "turn": global_.turn,
        "phase": global_.phase,
        "active": global_.active,
        "priority": global_.priority,
        "stack_size": global_.stack_size,
        "combat_substep": global_.combat_substep,
        "emblems": _keys_to_json(global_.emblems),
    }


def _global_from_json(data: dict) -> GlobalState:
    return GlobalState(
        turn=data["turn"], phase=data["phase"], active=data["active"],
        priority=data.get("priority"), stack_size=data.get("stack_size", 0),
        combat_substep=data.get("combat_substep"),
        emblems=_keys_from_json(data.get("emblems")),
    )


def _player_to_json(player: PlayerState) -> dict:
    return {
        "id": player.id, "life": player.life, "hand": player.hand,
        "library": player.library, "graveyard": player.graveyard,
        "poison": player.poison, "energy": player.energy,
        "this_turn": dict(player.this_turn),
        "floating_mana": dict(player.floating_mana),
        "untapped_production": dict(player.untapped_production),
    }


def _player_from_json(data: dict) -> PlayerState:
    return PlayerState(
        id=data["id"], life=data["life"], hand=data["hand"],
        library=data["library"], graveyard=data["graveyard"],
        poison=data.get("poison", 0), energy=data.get("energy", 0),
        this_turn=dict(data.get("this_turn", {})),
        floating_mana=dict(data.get("floating_mana", {})),
        untapped_production=dict(data.get("untapped_production", {})),
    )


def _refs_to_json(refs: Refs) -> dict:
    return {
        "targets": list(refs.targets),
        "source": refs.source,
        "modes": list(refs.modes),
        "x": refs.x,
        "choices": dict(refs.choices),
    }


def _refs_from_json(data: dict) -> Refs:
    return Refs(
        targets=tuple(data.get("targets", ())),
        source=data.get("source"),
        modes=tuple(data.get("modes", ())),
        x=data.get("x"),
        choices=dict(data.get("choices", {})),
    )


def _snapshot_to_json(state: StateSnapshot) -> dict:
    return {
        "global": _global_to_json(state.global_),
        "players": [_player_to_json(player) for player in state.players],
        "entities": [_entity_to_json(e) for e in state.entities],
        "refs": _refs_to_json(state.refs),
        "pending_event": (
            state.pending_event.as_dict() if state.pending_event else None
        ),
        "tiers": sorted(int(t) for t in state.tiers),
    }


def _snapshot_from_json(data: dict) -> StateSnapshot:
    pending = data.get("pending_event")
    return StateSnapshot(
        global_=_global_from_json(data["global"]),
        players=tuple(_player_from_json(p) for p in data.get("players", ())),
        entities=tuple(_entity_from_json(e) for e in data.get("entities", ())),
        refs=_refs_from_json(data.get("refs") or {}),
        pending_event=_event_from_json(pending) if pending else None,
        tiers=frozenset(InclusionTier(t) for t in data.get("tiers", ())),
    )


# ── payloads ────────────────────────────────────────────────────────────


def _costs_to_json(costs: Costs) -> dict:
    return {
        "mana_by_color": dict(costs.mana_by_color),
        "tapped": list(costs.tapped),
        "life": costs.life,
        "sacrificed": list(costs.sacrificed),
        "discarded": list(costs.discarded),
        "exiled": list(costs.exiled),
    }


def _costs_from_json(data: dict) -> Costs:
    return Costs(
        mana_by_color=dict(data.get("mana_by_color", {})),
        tapped=tuple(data.get("tapped", ())),
        life=data.get("life", 0),
        sacrificed=tuple(data.get("sacrificed", ())),
        discarded=tuple(data.get("discarded", ())),
        exiled=tuple(data.get("exiled", ())),
    )


def _activation_to_json(payload: ActivationPayload) -> dict:
    return {"costs": _costs_to_json(payload.costs), "outcome": payload.outcome.value}


def _activation_from_json(data: dict) -> ActivationPayload:
    return ActivationPayload(
        costs=_costs_from_json(data.get("costs") or {}),
        outcome=ResolutionOutcome(data["outcome"]),
    )


def _resolution_to_json(payload: ResolutionPayload) -> dict:
    return {"events": _events_to_json(payload.events)}


def _resolution_from_json(data: dict) -> ResolutionPayload:
    return ResolutionPayload(events=_events_from_json(data.get("events")))


def _rewrite_to_json(payload: RewritePayload) -> dict:
    return {
        "incoming": payload.incoming.as_dict(),
        "outgoing": payload.outgoing.as_dict() if payload.outgoing else None,
        "result": payload.result.value if payload.result else None,
        "replaced_by": _keys_to_json(payload.replaced_by),
    }


def _rewrite_from_json(data: dict) -> RewritePayload:
    outgoing = data.get("outgoing")
    result = data.get("result")
    return RewritePayload(
        incoming=_event_from_json(data["incoming"]),
        outgoing=_event_from_json(outgoing) if outgoing else None,
        result=RewriteResult(result) if result else None,
        replaced_by=_keys_from_json(data.get("replaced_by")),
    )


def _contribution_to_json(contribution: Contribution) -> dict:
    return {
        "entity": contribution.entity, "pt_boost": list(contribution.pt_boost),
        "keywords": list(contribution.keywords), "types": list(contribution.types),
        "colors": list(contribution.colors), "name": contribution.name,
    }


def _contribution_from_json(data: dict) -> Contribution:
    return Contribution(
        entity=data["entity"],
        pt_boost=tuple(data.get("pt_boost", (0, 0))),
        keywords=tuple(normalize_keyword(k) for k in data.get("keywords", ())),
        types=tuple(data.get("types", ())),
        colors=tuple(data.get("colors", ())),
        name=data.get("name"),
    )


def _continuous_to_json(payload: ContinuousPayload) -> dict:
    return {
        "contributions": [_contribution_to_json(c) for c in payload.contributions],
        "board_hash": payload.board_hash,
    }


def _continuous_from_json(data: dict) -> ContinuousPayload:
    return ContinuousPayload(
        contributions=tuple(
            _contribution_from_json(c) for c in data.get("contributions", ())
        ),
        board_hash=data.get("board_hash", ""),
    )


def _combat_to_json(payload: CombatPayload) -> dict:
    return {
        "attackers": list(payload.attackers),
        "blocks": {k: list(v) for k, v in payload.blocks.items()},
        "assignment_choices": {
            k: dict(v) for k, v in payload.assignment_choices.items()
        },
        "events": _events_to_json(payload.events),
        "probed_keyword": payload.probed_keyword,
        "probed_entity": payload.probed_entity,
    }


def _combat_from_json(data: dict) -> CombatPayload:
    return CombatPayload(
        attackers=tuple(data.get("attackers", ())),
        blocks={k: tuple(v) for k, v in (data.get("blocks") or {}).items()},
        assignment_choices={
            k: dict(v)
            for k, v in (data.get("assignment_choices") or {}).items()
        },
        events=_events_from_json(data.get("events")),
        probed_keyword=data.get("probed_keyword"),
        probed_entity=data.get("probed_entity"),
    )


def _trigger_to_json(payload: TriggerPayload) -> dict:
    return {"event": payload.event.as_dict(), "fired": payload.fired}


def _trigger_from_json(data: dict) -> TriggerPayload:
    return TriggerPayload(
        event=_event_from_json(data["event"]), fired=bool(data["fired"]),
    )


def _candidate_to_json(candidate: Candidate) -> dict:
    return {
        "ability": _keys_to_json(candidate.ability),
        "verdict": {
            "can_play": candidate.can_play,
            "affordable": candidate.affordable,
            "has_legal_target": candidate.has_legal_target,
        },
        "legal_targets": list(candidate.legal_targets),
        "cost_after_adjustment": dict(candidate.cost_after_adjustment),
        "responsible_static": _keys_to_json(candidate.responsible_static),
    }


def _candidate_from_json(data: dict) -> Candidate:
    verdict = data["verdict"]
    return Candidate(
        ability=_keys_from_json(data.get("ability")),
        can_play=verdict["can_play"],
        affordable=verdict["affordable"],
        has_legal_target=verdict["has_legal_target"],
        legal_targets=tuple(data.get("legal_targets", ())),
        cost_after_adjustment=dict(data.get("cost_after_adjustment", {})),
        responsible_static=_keys_from_json(data.get("responsible_static")),
    )


def _decision_to_json(payload: PlayabilityDecisionPayload) -> dict:
    return {"candidates": [_candidate_to_json(c) for c in payload.candidates]}


def _decision_from_json(data: dict) -> PlayabilityDecisionPayload:
    return PlayabilityDecisionPayload(
        candidates=tuple(_candidate_from_json(c) for c in data.get("candidates", ())),
    )


def _forbidden_to_json(forbidden: tuple[ForbiddenEntity, ...]) -> list[dict]:
    return [
        {
            "entity": f.entity,
            "responsible_static": _keys_to_json(f.responsible_static),
        }
        for f in forbidden
    ]


def _forbidden_from_json(data: Any) -> tuple[ForbiddenEntity, ...]:
    return tuple(
        ForbiddenEntity(
            entity=f["entity"],
            responsible_static=_keys_from_json(f.get("responsible_static")),
        )
        for f in data or ()
    )


def _attackers_to_json(payload: PlayabilityAttackersPayload) -> dict:
    return {
        "legal_attackers": list(payload.legal_attackers),
        "forbidden": _forbidden_to_json(payload.forbidden),
    }


def _attackers_from_json(data: dict) -> PlayabilityAttackersPayload:
    return PlayabilityAttackersPayload(
        legal_attackers=tuple(data.get("legal_attackers", ())),
        forbidden=_forbidden_from_json(data.get("forbidden")),
    )


def _blockers_to_json(payload: PlayabilityBlockersPayload) -> dict:
    return {
        "anchor_attacker": payload.anchor_attacker,
        "legal_blockers": list(payload.legal_blockers),
        "forbidden": _forbidden_to_json(payload.forbidden),
        "min_blockers": payload.min_blockers,
    }


def _blockers_from_json(data: dict) -> PlayabilityBlockersPayload:
    return PlayabilityBlockersPayload(
        anchor_attacker=data["anchor_attacker"],
        legal_blockers=tuple(data.get("legal_blockers", ())),
        forbidden=_forbidden_from_json(data.get("forbidden")),
        min_blockers=data.get("min_blockers", 0),
    )


@dataclass(frozen=True, slots=True)
class _PayloadCodec:
    encode: Callable[[Any], dict]
    decode: Callable[[dict], Any]


#: Every payload type's wire form: the values of ``PAYLOAD_TYPES``, each once.
_PAYLOAD_CODECS: dict[type, _PayloadCodec] = {
    ActivationPayload: _PayloadCodec(_activation_to_json, _activation_from_json),
    ResolutionPayload: _PayloadCodec(_resolution_to_json, _resolution_from_json),
    RewritePayload: _PayloadCodec(_rewrite_to_json, _rewrite_from_json),
    ContinuousPayload: _PayloadCodec(_continuous_to_json, _continuous_from_json),
    CombatPayload: _PayloadCodec(_combat_to_json, _combat_from_json),
    TriggerPayload: _PayloadCodec(_trigger_to_json, _trigger_from_json),
    PlayabilityDecisionPayload: _PayloadCodec(_decision_to_json, _decision_from_json),
    PlayabilityAttackersPayload: _PayloadCodec(_attackers_to_json, _attackers_from_json),
    PlayabilityBlockersPayload: _PayloadCodec(_blockers_to_json, _blockers_from_json),
}


def _payload_to_json(payload) -> dict:
    codec = _PAYLOAD_CODECS.get(type(payload))
    if codec is None:
        raise TypeError(f"no wire form for payload type {type(payload).__name__}")
    return codec.encode(payload)


def _payload_from_json(
    data: dict, kind: RecordKind, discriminator: Moment | PlayabilitySubkind | None,
):
    payload_type = PAYLOAD_TYPES.get((kind, discriminator))
    if payload_type is None:
        # A ValueError, as the record's own discriminator check raises.
        raise ValueError(
            f"no payload is defined for kind={kind.value} with "
            f"moment/subkind={discriminator.value if discriminator else None}"
        )
    return _PAYLOAD_CODECS[payload_type].decode(data)


# ── records ─────────────────────────────────────────────────────────────


def record_to_dict(record: EffectRecord) -> dict:
    """One record's JSON-serializable dict form, envelope keys first."""
    out: dict = {
        "record_id": record.record_id,
        "run_id": record.run_id,
        "timestamp": record.timestamp,
        "game_id": record.game_id,
        "kind": record.kind.value,
        "moment": record.moment.value if record.moment else None,
        "subkind": record.subkind.value if record.subkind else None,
        "link_id": record.link_id,
        "mirror_of": record.mirror_of,
        "variant_of": record.variant_of,
        "mode": record.mode.value,
        "interventional": record.interventional,
        "fork": record.fork,
        "synthetic": record.synthetic,
    }
    # Written only where the source carried it: its presence makes a shard
    # gen-2 (``shard_generation``), so a rewritten gen-1 record stays gen-1.
    if record.random_seat is not None:
        out["random_seat"] = record.random_seat
    if record.what_if is not None:
        out["what_if"] = record.what_if
    out |= {
        "actor_player": record.actor_player,
        "ability": (
            _keys_to_json(record.ability) if record.ability is not None else None
        ),
        "ability_unresolved": record.ability_unresolved,
        "state": _snapshot_to_json(record.state),
        "payload": _payload_to_json(record.payload),
    }
    for key, value in record.extra_fields.items():
        out.setdefault(key, value)
    return out


def record_from_dict(data: dict) -> EffectRecord:
    """Parse one shard line into an :class:`EffectRecord`."""
    kind = RecordKind(data["kind"])
    moment = Moment(data["moment"]) if data.get("moment") else None
    subkind = PlayabilitySubkind(data["subkind"]) if data.get("subkind") else None
    ability = data.get("ability")
    return EffectRecord(
        record_id=data["record_id"],
        run_id=data["run_id"],
        timestamp=data["timestamp"],
        game_id=data["game_id"],
        kind=kind,
        actor_player=data["actor_player"],
        state=_snapshot_from_json(data["state"]),
        payload=_payload_from_json(data["payload"], kind, moment or subkind),
        mode=CollectionMode(data.get("mode", CollectionMode.DEGRADED.value)),
        moment=moment,
        subkind=subkind,
        link_id=data.get("link_id"),
        mirror_of=data.get("mirror_of"),
        variant_of=data.get("variant_of"),
        interventional=data.get("interventional", False),
        fork=data.get("fork", False),
        synthetic=data.get("synthetic", False),
        random_seat=data.get("random_seat"),
        what_if=data.get("what_if"),
        ability=_keys_from_json(ability) if ability is not None else None,
        ability_unresolved=data.get("ability_unresolved"),
        extra_fields={
            k: v for k, v in data.items() if k not in _KNOWN_ENVELOPE_KEYS
        },
    )


def format_record_line(record: EffectRecord) -> str:
    """Render one shard line: compact JSON in field order, no trailing newline.

    ``orjson``, as the reader is; non-ASCII text is written as UTF-8 rather
    than ``\\u`` escapes, which every reader decodes alike. A non-string key
    is written as its string, as ``json`` would.
    """
    return orjson.dumps(record_to_dict(record), option=orjson.OPT_NON_STR_KEYS).decode()


def parse_record_line(text: str) -> dict:
    """One shard line as the dict ``record_from_dict`` reads.

    ``orjson`` for speed; it gives the same dicts as ``json``, and its decode
    error subclasses ``json.JSONDecodeError``.
    """
    return orjson.loads(text)

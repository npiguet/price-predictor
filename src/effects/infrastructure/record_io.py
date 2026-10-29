"""JSONL shard IO for the effect-record corpus.

The Java collector writes raw shards; Python reads them and writes the
curated shards ``build-corpus`` produces. Every conversion is written out
rather than reflected from the dataclass, so a field renamed in Python never
silently renames itself in a corpus that cannot be rebuilt.
"""

from __future__ import annotations

import gzip
import json
import logging
import math
import random
import re
import shutil
import zlib
from collections import defaultdict
from collections.abc import Callable, Collection, Iterable, Iterator, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import IO, Any, NamedTuple

import orjson

from effects.domain.corpus_curation import record_hash
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
from price_predictor.infrastructure.append_only import (
    count_complete_lines,
    iter_complete_lines,
)

logger = logging.getLogger(__name__)

_JSON_SEPARATORS = (",", ":")  # compact, newline-free

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


def _entity_to_json(entity: EntityState) -> dict:
    return {
        "id": entity.id, "name": entity.name, "zone": entity.zone,
        "controller": entity.controller, "face": entity.face,
        "copy_source": entity.copy_source,
        "token_script_id": entity.token_script_id,
        "types": list(entity.types), "subtypes": list(entity.subtypes),
        "supertypes": list(entity.supertypes), "colors": list(entity.colors),
        "mana_value": entity.mana_value,
        "pt": (
            {
                "base": list(entity.pt.base),
                "boosts": list(entity.pt.boosts),
                "counters": list(entity.pt.counters),
            }
            if entity.pt
            else None
        ),
        "tapped": entity.tapped, "sick": entity.sick, "damage": entity.damage,
        "counters": dict(entity.counters),
        "combat": (
            {
                "attacking": entity.combat.attacking,
                "blocking": list(entity.combat.blocking),
                "blocked_by": list(entity.combat.blocked_by),
                "became_blocked": entity.combat.became_blocked,
            }
            if entity.combat
            else None
        ),
        "attached_to": entity.attached_to,
        "face_down": entity.face_down,
        "granted_attached": _keys_to_json(entity.granted_attached),
        "granted_temporary": {
            "keywords": list(entity.granted_temporary.keywords),
            "abilities": _keys_to_json(entity.granted_temporary.abilities),
        },
        "printed": _keys_to_json(entity.printed),
        "stack_extras": (
            {
                "targets": list(entity.stack_extras.targets),
                "per_target_amounts": dict(entity.stack_extras.per_target_amounts),
                "up_to_counts": dict(entity.stack_extras.up_to_counts),
            }
            if entity.stack_extras
            else None
        ),
    }


def _entity_from_json(data: dict) -> EntityState:
    pt = data.get("pt")
    combat = data.get("combat")
    extras = data.get("stack_extras")
    granted = data.get("granted_temporary") or {}
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
        pt=(
            PowerToughness(
                base=tuple(pt["base"]),
                boosts=tuple(pt.get("boosts", (0, 0))),
                counters=tuple(pt.get("counters", (0, 0))),
            )
            if pt
            else None
        ),
        tapped=data.get("tapped", False), sick=data.get("sick", False),
        damage=data.get("damage", 0), counters=dict(data.get("counters", {})),
        combat=(
            CombatStatus(
                attacking=combat.get("attacking"),
                blocking=tuple(combat.get("blocking", ())),
                blocked_by=tuple(combat.get("blocked_by", ())),
                became_blocked=combat.get("became_blocked", False),
            )
            if combat
            else None
        ),
        attached_to=data.get("attached_to"),
        face_down=data.get("face_down", False),
        granted_attached=_keys_from_json(data.get("granted_attached")),
        granted_temporary=GrantedTemporary(
            keywords=tuple(
                normalize_keyword(k) for k in granted.get("keywords", ())
            ),
            abilities=_keys_from_json(granted.get("abilities")),
        ),
        printed=_keys_from_json(data.get("printed")),
        stack_extras=(
            StackExtras(
                targets=tuple(extras.get("targets", ())),
                per_target_amounts=dict(extras.get("per_target_amounts", {})),
                up_to_counts=dict(extras.get("up_to_counts", {})),
            )
            if extras
            else None
        ),
    )


def _snapshot_to_json(state: StateSnapshot) -> dict:
    return {
        "global": {
            "turn": state.global_.turn,
            "phase": state.global_.phase,
            "active": state.global_.active,
            "priority": state.global_.priority,
            "stack_size": state.global_.stack_size,
            "combat_substep": state.global_.combat_substep,
            "emblems": _keys_to_json(state.global_.emblems),
        },
        "players": [
            {
                "id": p.id, "life": p.life, "hand": p.hand,
                "library": p.library, "graveyard": p.graveyard,
                "poison": p.poison, "energy": p.energy,
                "this_turn": dict(p.this_turn),
                "floating_mana": dict(p.floating_mana),
                "untapped_production": dict(p.untapped_production),
            }
            for p in state.players
        ],
        "entities": [_entity_to_json(e) for e in state.entities],
        "refs": {
            "targets": list(state.refs.targets),
            "source": state.refs.source,
            "modes": list(state.refs.modes),
            "x": state.refs.x,
            "choices": dict(state.refs.choices),
        },
        "pending_event": (
            state.pending_event.as_dict() if state.pending_event else None
        ),
        "tiers": sorted(int(t) for t in state.tiers),
    }


def _snapshot_from_json(data: dict) -> StateSnapshot:
    g = data["global"]
    refs = data.get("refs") or {}
    pending = data.get("pending_event")
    return StateSnapshot(
        global_=GlobalState(
            turn=g["turn"], phase=g["phase"], active=g["active"],
            priority=g.get("priority"), stack_size=g.get("stack_size", 0),
            combat_substep=g.get("combat_substep"),
            emblems=_keys_from_json(g.get("emblems")),
        ),
        players=tuple(
            PlayerState(
                id=p["id"], life=p["life"], hand=p["hand"],
                library=p["library"], graveyard=p["graveyard"],
                poison=p.get("poison", 0), energy=p.get("energy", 0),
                this_turn=dict(p.get("this_turn", {})),
                floating_mana=dict(p.get("floating_mana", {})),
                untapped_production=dict(p.get("untapped_production", {})),
            )
            for p in data.get("players", ())
        ),
        entities=tuple(_entity_from_json(e) for e in data.get("entities", ())),
        refs=Refs(
            targets=tuple(refs.get("targets", ())),
            source=refs.get("source"),
            modes=tuple(refs.get("modes", ())),
            x=refs.get("x"),
            choices=dict(refs.get("choices", {})),
        ),
        pending_event=_event_from_json(pending) if pending else None,
        tiers=frozenset(InclusionTier(t) for t in data.get("tiers", ())),
    )


# ── payloads ────────────────────────────────────────────────────────────


def _activation_to_json(payload: ActivationPayload) -> dict:
    return {
        "costs": {
            "mana_by_color": dict(payload.costs.mana_by_color),
            "tapped": list(payload.costs.tapped),
            "life": payload.costs.life,
            "sacrificed": list(payload.costs.sacrificed),
            "discarded": list(payload.costs.discarded),
            "exiled": list(payload.costs.exiled),
        },
        "outcome": payload.outcome.value,
    }


def _activation_from_json(data: dict) -> ActivationPayload:
    costs = data.get("costs") or {}
    return ActivationPayload(
        costs=Costs(
            mana_by_color=dict(costs.get("mana_by_color", {})),
            tapped=tuple(costs.get("tapped", ())),
            life=costs.get("life", 0),
            sacrificed=tuple(costs.get("sacrificed", ())),
            discarded=tuple(costs.get("discarded", ())),
            exiled=tuple(costs.get("exiled", ())),
        ),
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


def _continuous_to_json(payload: ContinuousPayload) -> dict:
    return {
        "contributions": [
            {
                "entity": c.entity, "pt_boost": list(c.pt_boost),
                "keywords": list(c.keywords), "types": list(c.types),
                "colors": list(c.colors), "name": c.name,
            }
            for c in payload.contributions
        ],
        "board_hash": payload.board_hash,
    }


def _continuous_from_json(data: dict) -> ContinuousPayload:
    return ContinuousPayload(
        contributions=tuple(
            Contribution(
                entity=c["entity"],
                pt_boost=tuple(c.get("pt_boost", (0, 0))),
                keywords=tuple(
                    normalize_keyword(k) for k in c.get("keywords", ())
                ),
                types=tuple(c.get("types", ())),
                colors=tuple(c.get("colors", ())),
                name=c.get("name"),
            )
            for c in data.get("contributions", ())
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


def _decision_to_json(payload: PlayabilityDecisionPayload) -> dict:
    return {
        "candidates": [
            {
                "ability": _keys_to_json(c.ability),
                "verdict": {
                    "can_play": c.can_play,
                    "affordable": c.affordable,
                    "has_legal_target": c.has_legal_target,
                },
                "legal_targets": list(c.legal_targets),
                "cost_after_adjustment": dict(c.cost_after_adjustment),
                "responsible_static": _keys_to_json(c.responsible_static),
            }
            for c in payload.candidates
        ]
    }


def _decision_from_json(data: dict) -> PlayabilityDecisionPayload:
    return PlayabilityDecisionPayload(
        candidates=tuple(
            Candidate(
                ability=_keys_from_json(c.get("ability")),
                can_play=c["verdict"]["can_play"],
                affordable=c["verdict"]["affordable"],
                has_legal_target=c["verdict"]["has_legal_target"],
                legal_targets=tuple(c.get("legal_targets", ())),
                cost_after_adjustment=dict(c.get("cost_after_adjustment", {})),
                responsible_static=_keys_from_json(c.get("responsible_static")),
            )
            for c in data.get("candidates", ())
        )
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
    """Render one shard line (compact JSON, no trailing newline)."""
    return json.dumps(record_to_dict(record), separators=_JSON_SEPARATORS)


def _write_line(handle: IO[str], record: EffectRecord) -> None:
    handle.write(format_record_line(record))
    handle.write("\n")


def write_shard(path: Path, records: Iterable[EffectRecord]) -> int:
    """Write a whole shard at once, gzipped; return the record count.

    One gzip member: per-block members guard a worker killed mid-run, and a
    curated shard is written by a process that finishes or leaves nothing.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for record in records:
            _write_line(handle, record)
            written += 1
    return written


#: How many group files the first deal holds open: a large stratum has more
#: shards than one process may hold files open.
_GROUPS = 256


def repack_shards(
    parts: Sequence[Path], out_dir: Path, *, shard_records: int,
    seed: int = 0, records: int | None = None, prefix: str = "shard",
) -> list[Path]:
    """Deal ``parts``' games into shards of about ``shard_records`` records each.

    Whole games are dealt by a seeded hash of ``game_id``, so a game never
    spans two shards and each shard mixes games from across the stratum
    rather than one run's consecutive lifetimes (the trainer shuffles only
    inside the shard it holds). Each shard is then shuffled under ``seed``.

    Streamed through at most :data:`_GROUPS` compressed group files beside
    ``out_dir``, since a stratum does not fit in memory. ``records`` is the
    parts' record count when the caller knows it. ``shard_records <= 0``
    writes one shard per part, unshuffled. Parts are not deleted.
    """
    parts = [Path(p) for p in parts]
    if not parts:
        return []
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if shard_records <= 0:
        return _copy_parts_unshuffled(parts, out_dir, prefix)

    if records is None:
        records = sum(1 for part in parts for _ in iter_shard_lines(part))
    count = max(1, math.ceil(records / shard_records))
    scratch = out_dir.with_name(f".{out_dir.name}.buckets")
    shutil.rmtree(scratch, ignore_errors=True)
    scratch.mkdir(parents=True)
    try:
        filled = _deal_into_buckets(parts, scratch, seed=seed, count=count)
        # Named before writing: groups write shards out of index order.
        names = {
            index: out_dir / f"{prefix}-{number:05d}.jsonl.gz"
            for number, index in enumerate(sorted(filled), start=1)
        }
        _write_shuffled_buckets(scratch, names, seed=seed, count=count)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return list(names.values())


def _copy_parts_unshuffled(parts: list[Path], out_dir: Path, prefix: str) -> list[Path]:
    written = []
    for number, part in enumerate(parts, start=1):
        path = out_dir / f"{prefix}-{number:05d}.jsonl.gz"
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            for line in iter_shard_lines(part):
                handle.write(line)
        written.append(path)
    return written


def _bucket_of(line: str, *, seed: int, count: int) -> int:
    """The index of the output shard a line's game is dealt to."""
    return record_hash(_game_of_line(line), seed=seed) % count


def _group_path(scratch: Path, group: int) -> Path:
    return scratch / f"{group:05d}.gz"


def _deal_into_buckets(
    parts: list[Path], scratch: Path, *, seed: int, count: int,
) -> set[int]:
    """Deal every line into the group of its shard's index; return the indices.

    A group keeps the parts' order, so each shuffle sees a fixed input order.
    """
    groups = min(count, _GROUPS)
    filled: set[int] = set()
    handles = [
        gzip.open(_group_path(scratch, group), "wt", encoding="utf-8", compresslevel=1)
        for group in range(groups)
    ]
    try:
        for part in parts:
            for line in iter_shard_lines(part):
                if not line.strip():
                    continue
                index = _bucket_of(line, seed=seed, count=count)
                filled.add(index)
                handles[index % groups].write(line)
    finally:
        for handle in handles:
            handle.close()
    return filled


def _write_shuffled_buckets(
    scratch: Path, names: dict[int, Path], *, seed: int, count: int,
) -> None:
    """Split each group into its shards, shuffle each under ``seed``, write it.

    A group is held whole, which is about ``count / _GROUPS`` shards' lines.
    """
    for group in range(min(count, _GROUPS)):
        buckets: dict[int, list[str]] = defaultdict(list)
        with gzip.open(_group_path(scratch, group), "rt", encoding="utf-8") as handle:
            for line in handle:
                buckets[_bucket_of(line, seed=seed, count=count)].append(line)
        for index, lines in buckets.items():
            random.Random(f"{seed}:{index}").shuffle(lines)
            with gzip.open(names[index], "wt", encoding="utf-8") as handle:
                handle.writelines(lines)


def _game_of_line(line: str) -> str:
    """A shard line's ``game_id``, read off the envelope head without a parse."""
    head = peek_envelope(line)
    if head is not None:
        return head.game_id
    return parse_record_line(line.strip())["game_id"]


def parse_record_line(text: str) -> dict:
    """One shard line as the dict ``record_from_dict`` reads.

    ``orjson`` for speed; it gives the same dicts as ``json``, and its decode
    error subclasses ``json.JSONDecodeError``.
    """
    return orjson.loads(text)


#: Both shard spellings: plain `.jsonl` predates compression.
SHARD_GLOBS: tuple[str, ...] = ("*.jsonl.gz", "*.jsonl")


def iter_shard_lines(path: Path) -> Iterator[str]:
    """Every complete line of a shard, compressed or not.

    A truncated final gzip member, or a trailing partial line, ends the
    stream.
    """
    path = Path(path)
    if path.suffix != ".gz":
        yield from iter_complete_lines(path)
        return

    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                # A line without its newline can only be a member cut short.
                if line.endswith("\n"):
                    yield line
    except (EOFError, gzip.BadGzipFile, zlib.error):
        return


def iter_record_dicts(path: Path) -> Iterator[dict]:
    """Every complete record of one shard, as the dict its line parses to.

    For a caller that needs the raw JSON, such as a generation check. Blank
    lines are skipped.
    """
    for line in iter_shard_lines(Path(path)):
        stripped = line.strip()
        if stripped:
            yield parse_record_line(stripped)


def read_window(
    directory: Path, limit: int = 0,
    observe: Callable[[str, dict], None] | None = None,
) -> Iterator[EffectRecord]:
    """The first ``limit`` records under ``directory``; 0 reads all of them.

    ``observe``, when given, sees each record's raw dict, with its shard's
    path relative to ``directory``, before it is parsed: a parsed record
    cannot say whether a field was on the wire.
    """
    root = Path(directory)
    seen = 0
    for shard in iter_shards(root):
        name = shard.relative_to(root).as_posix()
        for data in iter_record_dicts(shard):
            if observe is not None:
                observe(name, data)
            yield record_from_dict(data)
            seen += 1
            if limit and seen >= limit:
                return


def read_shard(path: Path) -> Iterator[EffectRecord]:
    """Yield every complete record in one shard; skip a trailing partial line."""
    for data in iter_record_dicts(path):
        yield record_from_dict(data)


def iter_shards(directory: Path) -> list[Path]:
    """Every shard at or under ``directory``, in path order, either spelling.

    Recursive, so collection runs kept in subdirectories read as one corpus.
    Path order rather than name order, so every reader walks a corpus in the
    same order on every run.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return []
    found: list[Path] = []
    for pattern in SHARD_GLOBS:
        found.extend(
            path for path in directory.rglob(pattern)
            # "*.jsonl" also matches "x.jsonl.gz" on some platforms; the
            # compressed pattern already claimed those.
            if path.suffix != ".gz" or pattern.endswith(".gz")
        )
    return sorted(set(found))


def read_records(directory: Path) -> Iterator[EffectRecord]:
    """Stream every record under ``directory``, shard by shard in path order.

    A missing directory yields nothing: an empty corpus, not an error.
    """
    for shard in iter_shards(directory):
        yield from read_shard(shard)


#: The envelope's head as both writers lay it out: the Java
#: ``EffectRecord.toJson`` and :func:`record_to_dict` emit ``record_id``,
#: ``run_id``, ``timestamp``, ``game_id`` and ``kind`` first, in that order.
_ENVELOPE_HEAD = re.compile(
    r'"record_id":"([^"]*)",.*?"game_id":"([^"]*)","kind":"([a-z_]+)"'
)

#: How far into a line the head is looked for. The five leading fields are a
#: few hundred characters; the state snapshot that follows is the bulk.
_ENVELOPE_HEAD_SPAN = 512


class EnvelopeHead(NamedTuple):
    """A shard line's leading envelope fields, read without parsing it."""

    record_id: str
    game_id: str
    kind: str


def peek_envelope(line: str) -> EnvelopeHead | None:
    """The head of one shard line, or None where it has another shape, which
    the caller then parses: a reordered envelope costs speed, never records."""
    head = _ENVELOPE_HEAD.search(line, 0, _ENVELOPE_HEAD_SPAN)
    return None if head is None else EnvelopeHead(*head.groups())


def read_records_matching(
    directory: Path,
    *,
    game_ids: Collection[str] | None = None,
    kinds: Collection[str] | None = None,
) -> Iterator[EffectRecord]:
    """The records under ``directory`` of the named games and kinds.

    The same records a filter over :func:`read_records` keeps, without
    parsing the rest; a line :func:`peek_envelope` cannot read is parsed and
    filtered the slow way.
    """
    shards = iter_shards(directory)
    for done, shard in enumerate(shards):
        if done and done % 250 == 0:
            logger.info("scanned %d of %d shards under %s", done, len(shards), directory)
        for line in iter_shard_lines(shard):
            head = peek_envelope(line)
            if head is not None and not _wanted(head.game_id, head.kind, game_ids, kinds):
                continue
            stripped = line.strip()
            if not stripped:
                continue
            record = record_from_dict(parse_record_line(stripped))
            if head is None and not _wanted(record.game_id, record.kind.value, game_ids, kinds):
                continue
            yield record


def _wanted(
    game_id: str, kind: str, game_ids: Collection[str] | None, kinds: Collection[str] | None,
) -> bool:
    """Whether a record of ``game_id`` and ``kind`` passes both filters."""
    return (game_ids is None or game_id in game_ids) and (kinds is None or kind in kinds)


def count_records(directory: Path, *, ceiling: int = 0) -> int:
    """Complete records under ``directory``, for progress and resume.

    Counts complete lines without parsing them. A non-zero ``ceiling`` stops
    the count there and returns it, since decompressing a full corpus takes
    minutes.
    """
    total = 0
    for shard in iter_shards(directory):
        total += count_shard_records(shard)
        if ceiling and total >= ceiling:
            return ceiling
    return total


def count_shard_records(shard: Path) -> int:
    """Complete records in one shard, counted as :func:`count_records` does."""
    if shard.suffix == ".gz":
        return sum(1 for line in iter_shard_lines(shard) if line.strip())
    return count_complete_lines(shard)


# ── shard generation (FR-033) ───────────────────────────────────────────

#: The two record generations a shard can hold. A gen-2 record always carries
#: ``random_seat``; a gen-1 one never does, and on gen-1 ``playability``
#: records ``actor_player`` keeps feature 023's meaning rather than naming the
#: deciding player (FR-030a).
GEN_1, GEN_2 = "gen-1", "gen-2"


class MixedGenerationsError(ValueError):
    """A records set holds both gen-1 and gen-2 shards."""


def shard_generation(path: Path) -> str | None:
    """``gen-1`` or ``gen-2`` by the shard's first complete record; None if empty.

    One record speaks for its shard: ``validate-corpus`` fails a shard whose
    records disagree.
    """
    for data in iter_record_dicts(path):
        return GEN_2 if "random_seat" in data else GEN_1
    return None


def refuse_mixed_generations(paths: Iterable[Path]) -> str | None:
    """The one generation ``paths`` hold, or raise when they hold both.

    Every reader of a records set calls this first (FR-033): the redefined
    ``actor_player`` has no version field of its own. None for a set of empty
    shards.
    """
    seen: dict[str, Path] = {}
    for path in paths:
        generation = shard_generation(path)
        if generation is not None:
            seen.setdefault(generation, Path(path))
    if len(seen) > 1:
        raise MixedGenerationsError(
            "this records set mixes gen-1 shards (no random_seat field) with "
            f"gen-2 shards, e.g. {seen[GEN_1]} and {seen[GEN_2]}. The two read "
            "actor_player on playability records differently (FR-030a); keep "
            "them in separate directories and read one at a time."
        )
    return next(iter(seen), None)

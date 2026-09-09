"""JSONL shard IO for the effect-record corpus.

One ``{run_id}.{worker}-{lifetime}.jsonl`` shard per worker JVM, all read as a
directory. One shard per worker rather than one shared file because records are
far larger than a match-outcome row, so cross-process appends would interleave
mid-record; and line-oriented rather than columnar because the writers are Forge
JVMs that are expected to crash mid-write, which is exactly what a line format
survives.

A shard per JVM *lifetime*, not per worker slot: the supervisor recycles workers
on a timer, and the ids inside a shard are counters that restart with the JVM.

Serialization lives here; the in-memory shapes are the pure dataclasses in
``effects/domain/``. Every conversion is written out rather than reflected from
the dataclass, because the wire format is a frozen contract and a field renamed
in Python must not silently rename itself in a corpus that cannot be rebuilt.

**Widening a field's value set is not redefining the field**, and most of what a
collector fix changes is the former. ``outcome`` gained no new member when the
writer stopped hardcoding ``resolved``: the enum always held all five and the
corpus only ever showed one. ``zone_change`` gained no new param when
``from_zone`` started being written; the key was already in ``EVENT_PARAMS`` and
always empty. ``attributed_to`` gained two sentinel *values* (``root``,
``unresolved``) and stayed a nullable string, so a reader that predates them
parses them as the strings they are. Compatibility rule 1 forbids only the other
direction — nothing here may change what an existing field *means*, because the
corpus is append-only and a reinterpreted field silently reinterprets hours of
records that cannot be recollected.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import IO, Any

from effects.domain.event_schema import Event
from effects.domain.provenance import ProvenanceKey
from effects.domain.records import (
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
    normalize_keyword,
)
from price_predictor.infrastructure.append_only import (
    count_complete_lines,
    iter_complete_lines,
)

_JSON_SEPARATORS = (",", ":")  # compact, newline-free

#: Envelope keys this code models. Anything else on a line is a field a newer
#: writer added and is kept in ``EffectRecord.extra_fields``.
_KNOWN_ENVELOPE_KEYS = frozenset({
    "record_id", "run_id", "timestamp", "game_id", "kind", "moment", "subkind",
    "link_id", "mirror_of", "variant_of", "mode", "interventional", "fork",
    "synthetic", "actor_player", "ability", "ability_unresolved", "state",
    "payload",
})


# ── provenance keys ─────────────────────────────────────────────────────


def _keys_to_json(keys: tuple[ProvenanceKey, ...]) -> list[dict]:
    return [key.as_dict() for key in keys]


def _keys_from_json(data: Any) -> tuple[ProvenanceKey, ...]:
    return tuple(ProvenanceKey.from_dict(item) for item in data or ())


# ── events ──────────────────────────────────────────────────────────────


def _events_to_json(events: tuple[Event, ...]) -> list[dict]:
    return [event.as_dict() for event in events]


def _events_from_json(data: Any) -> tuple[Event, ...]:
    return tuple(_event_from_json(item) for item in data or ())


def _event_from_json(item: Any) -> Event:
    """One event, with its counter type in this package's spelling.

    Normalized on read for the same reason keywords are: Forge writes what a
    card prints and the head reads a canonical name, and a corpus already
    collected cannot be rewritten. Doing it here rather than at the head's
    lookup means every consumer gets the same answer.
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
        # Explicit, because "absent tier means uncollected" is unanswerable
        # without it: an empty entity list and an uncollected tier look alike.
        "tiers": sorted(int(t) for t in state.tiers),
    }


def _entity_to_json(entity: EntityState) -> dict:
    out: dict = {
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
    return out


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
            # Normalized on read rather than on write: the corpus is
            # append-only, so shards collected before this was noticed carry
            # Forge's spelling and have to keep resolving.
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
        pending_event=Event.from_dict(pending) if pending else None,
        tiers=frozenset(InclusionTier(t) for t in data.get("tiers", ())),
    )


# ── payloads ────────────────────────────────────────────────────────────


def _payload_to_json(payload) -> dict:
    match payload:
        case ActivationPayload():
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
        case ResolutionPayload():
            return {"events": _events_to_json(payload.events)}
        case RewritePayload():
            return {
                "incoming": payload.incoming.as_dict(),
                "outgoing": payload.outgoing.as_dict(),
            }
        case ContinuousPayload():
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
        case CombatPayload():
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
        case TriggerPayload():
            return {"event": payload.event.as_dict(), "fired": payload.fired}
        case PlayabilityDecisionPayload():
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
        case PlayabilityAttackersPayload():
            return {
                "legal_attackers": list(payload.legal_attackers),
                "forbidden": _forbidden_to_json(payload.forbidden),
            }
        case PlayabilityBlockersPayload():
            return {
                "anchor_attacker": payload.anchor_attacker,
                "legal_blockers": list(payload.legal_blockers),
                "forbidden": _forbidden_to_json(payload.forbidden),
                "min_blockers": payload.min_blockers,
            }
    raise TypeError(f"no wire form for payload type {type(payload).__name__}")


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


def _payload_from_json(
    data: dict, kind: RecordKind, discriminator: Moment | PlayabilitySubkind | None,
):
    if kind is RecordKind.RESOLUTION:
        if discriminator is Moment.ACTIVATION:
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
        return ResolutionPayload(events=_events_from_json(data.get("events")))
    if kind is RecordKind.REWRITE:
        return RewritePayload(
            incoming=Event.from_dict(data["incoming"]),
            outgoing=Event.from_dict(data["outgoing"]),
        )
    if kind is RecordKind.CONTINUOUS:
        return ContinuousPayload(
            contributions=tuple(
                Contribution(
                    entity=c["entity"],
                    pt_boost=tuple(c.get("pt_boost", (0, 0))),
                    # Same spelling rule as the overlay channel: Forge writes
                    # "first strike", everything downstream reads first_strike.
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
    if kind is RecordKind.COMBAT:
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
    if kind is RecordKind.TRIGGER:
        return TriggerPayload(
            event=Event.from_dict(data["event"]), fired=bool(data["fired"]),
        )
    if discriminator is PlayabilitySubkind.DECISION:
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
    if discriminator is PlayabilitySubkind.ATTACKERS:
        return PlayabilityAttackersPayload(
            legal_attackers=tuple(data.get("legal_attackers", ())),
            forbidden=_forbidden_from_json(data.get("forbidden")),
        )
    return PlayabilityBlockersPayload(
        anchor_attacker=data["anchor_attacker"],
        legal_blockers=tuple(data.get("legal_blockers", ())),
        forbidden=_forbidden_from_json(data.get("forbidden")),
        min_blockers=data.get("min_blockers", 0),
    )


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
        "actor_player": record.actor_player,
        "ability": (
            _keys_to_json(record.ability) if record.ability is not None else None
        ),
        "ability_unresolved": record.ability_unresolved,
        "state": _snapshot_to_json(record.state),
        "payload": _payload_to_json(record.payload),
    }
    # Fields a newer writer added, carried through untouched.
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
        ability=_keys_from_json(ability) if ability is not None else None,
        ability_unresolved=data.get("ability_unresolved"),
        extra_fields={
            k: v for k, v in data.items() if k not in _KNOWN_ENVELOPE_KEYS
        },
    )


def format_record_line(record: EffectRecord) -> str:
    """Render one shard line (compact JSON, no trailing newline)."""
    return json.dumps(record_to_dict(record), separators=_JSON_SEPARATORS)


def append_record(out: IO[str], record: EffectRecord) -> None:
    """Append one record to an open text handle and flush.

    The caller owns the handle (opened in append mode) so a long collection run
    keeps one open per worker across many games.
    """
    out.write(format_record_line(record))
    out.write("\n")
    out.flush()


def shard_path(directory: Path, run_id: str, worker: int, lifetime: str) -> Path:
    """``{run_id}.{worker}-{lifetime}.jsonl`` under ``directory``.

    ``lifetime`` is required rather than defaulted: a caller that does not say
    which JVM lifetime it is writing would reopen a previous one's shard and
    reissue ids it already used, which is the defect the segment exists to
    prevent.
    """
    return Path(directory) / f"{run_id}.{worker}-{lifetime}.jsonl"


class ShardWriter:
    """Append-only writer for one worker's shard.

    Used as a context manager so the handle closes on a clean shutdown; a crash
    leaves at most a trailing partial line, which every reader here skips.
    """

    def __init__(
        self, directory: Path, run_id: str, worker: int, lifetime: str,
    ) -> None:
        self.path = shard_path(directory, run_id, worker, lifetime)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle: IO[str] | None = None

    def __enter__(self) -> ShardWriter:
        self._handle = self.path.open("a", encoding="utf-8")
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def write(self, record: EffectRecord) -> None:
        if self._handle is None:
            self._handle = self.path.open("a", encoding="utf-8")
        append_record(self._handle, record)

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


#: Both shard spellings. `.jsonl.gz` is what the writer produces; plain
#: `.jsonl` is read so a corpus collected before compression keeps working,
#: because the corpus is append-only and cannot be regenerated.
SHARD_GLOBS: tuple[str, ...] = ("*.jsonl.gz", "*.jsonl")


def iter_shard_lines(path: Path) -> Iterator[str]:
    """Every complete line of a shard, compressed or not.

    A compressed shard is a concatenation of gzip members, one per block of
    records. Every member stands alone, so a worker killed mid-write truncates
    the last one and leaves the rest readable — and a truncated member raises
    on read, which is where the stream stops. That is the same rule a plain
    shard's trailing partial line follows, applied a block at a time.
    """
    path = Path(path)
    if path.suffix != ".gz":
        yield from iter_complete_lines(path)
        return

    import gzip
    import zlib

    pending = ""
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if line.endswith("\n"):
                    if pending:
                        line = pending + line
                        pending = ""
                    yield line
                else:
                    # No newline: either the file ends here or the member was
                    # cut short. Held back rather than yielded, because half a
                    # record parses as nothing.
                    pending = line
    except (EOFError, gzip.BadGzipFile, zlib.error):
        # The final member was truncated by a killed worker. Everything before
        # it has already been yielded.
        return


def read_shard(path: Path) -> Iterator[EffectRecord]:
    """Yield every complete record in one shard; skip a trailing partial line."""
    for line in iter_shard_lines(Path(path)):
        stripped = line.strip()
        if not stripped:
            continue
        yield record_from_dict(json.loads(stripped))


def iter_shards(directory: Path) -> list[Path]:
    """Every shard under ``directory``, in name order, either spelling."""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    found: list[Path] = []
    for pattern in SHARD_GLOBS:
        found.extend(
            path for path in directory.glob(pattern)
            # "*.jsonl" also matches "x.jsonl.gz" on some platforms; the
            # compressed pattern already claimed those.
            if path.suffix != ".gz" or pattern.endswith(".gz")
        )
    return sorted(set(found))


def read_records(directory: Path) -> Iterator[EffectRecord]:
    """Stream every record under ``directory``, shard by shard in name order.

    A missing directory yields nothing rather than raising: a collection run
    that has not started yet is an empty corpus, not an error.
    """
    for shard in iter_shards(directory):
        yield from read_shard(shard)


def count_records(directory: Path) -> int:
    """Complete records under ``directory``, for progress and resume.

    Counts complete lines rather than parsed records: the count is a number,
    and building a record dataclass per line to arrive at it would parse the
    whole corpus's JSON — the variant collector asks for this before every run
    just to size its budget.

    A compressed shard still has to be decompressed to be counted, so the saving
    there is only the JSON parse.
    """
    total = 0
    for shard in iter_shards(directory):
        if shard.suffix == ".gz":
            total += sum(1 for line in iter_shard_lines(shard) if line.strip())
        else:
            total += count_complete_lines(shard)
    return total

"""The JSON codec for one effect record: its wire form and back.

Every conversion is written out rather than reflected from the dataclass, so
a field renamed in Python never silently renames itself in a corpus that
cannot be rebuilt. Reading and writing whole shards is ``shard_io``'s.

This module holds the envelope and the record as a whole. Each block of the
record is one ``_x_to_json`` / ``_x_from_json`` pair in a module beside it:
``record_codec_values`` (provenance keys, events, counter maps),
``record_codec_entity``, ``record_codec_snapshot``, ``record_codec_payloads``
(each kind's payload and the payload table) and ``record_codec_playability``.
The two values whose wire form more than one file shares, an :class:`Event`
(records, and the validator's event digests) and a :class:`ProvenanceKey`
(records, sidecars, the knowledge probes' frozen sets), serialise themselves
beside their fields, in the domain, so every format writes them alike.
"""

from __future__ import annotations

import orjson

from effects.domain.fork_operation import ForkOperation
from effects.domain.gen3_members import Site, require_always
from effects.domain.records import (
    SUBKINDS_OF,
    CollectionMode,
    Curation,
    DirectedTag,
    EffectRecord,
    ForkKind,
    Moment,
    PlayabilitySubkind,
    RecordKind,
)
from effects.domain.shard_generation import require_gen3

# The blocks' converters live beside this module; the names other modules and
# tests read through this one stay importable from it.
from effects.infrastructure.record_codec_entity import _entity_from_json as _entity_from_json
from effects.infrastructure.record_codec_payloads import _PAYLOAD_CODECS as _PAYLOAD_CODECS
from effects.infrastructure.record_codec_payloads import _is_observed_combat, _payload_to_json
from effects.infrastructure.record_codec_payloads import _payload_from_json as _payload_from_json
from effects.infrastructure.record_codec_snapshot import _player_from_json as _player_from_json
from effects.infrastructure.record_codec_snapshot import _snapshot_from_json, _snapshot_to_json
from effects.infrastructure.record_codec_values import _counters_from_json as _counters_from_json
from effects.infrastructure.record_codec_values import _keys_from_json as _keys_from_json
from effects.infrastructure.record_codec_values import _keys_to_json

#: Envelope keys this code models. Anything else on a line is a field a newer
#: writer added and is kept in ``EffectRecord.extra_fields``.
_KNOWN_ENVELOPE_KEYS = frozenset({
    "record_id", "generation", "run_id", "timestamp", "game_id", "kind", "moment", "subkind",
    "link_id", "mirror_of", "variant_of", "mode", "interventional", "fork", "fork_kind",
    "synthetic", "random_seat", "what_if", "aimed_at", "directed", "copy_dropped", "curation",
    "actor_player",
    "ability",
    "ability_unresolved", "state", "payload",
})


# ── records ─────────────────────────────────────────────────────────────


def record_to_dict(record: EffectRecord) -> dict:
    """One record's JSON-serializable dict form, envelope keys first."""
    out: dict = {
        "record_id": record.record_id,
        "generation": record.generation,
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
    }
    # A fork's record alone, beside the flag it says more about.
    if record.fork_kind is not None:
        out["fork_kind"] = record.fork_kind.value
    out |= {"synthetic": record.synthetic, "random_seat": record.random_seat}
    # On a blockers record alone, the one kind that carries it.
    if record.what_if is not None:
        out["what_if"] = record.what_if
    if record.aimed_at is not None:
        out["aimed_at"] = list(record.aimed_at)
    if record.directed is not None:
        out["directed"] = {"variant": record.directed.variant,
                           "operation": record.directed.operation.value}
    if record.copy_dropped is not None:
        out["copy_dropped"] = dict(record.copy_dropped)
    if record.curation is not None:
        out["curation"] = _curation_to_json(record.curation)
    out |= {
        "actor_player": record.actor_player,
        "ability": (
            _keys_to_json(record.ability) if record.ability is not None else None
        ),
        "ability_unresolved": record.ability_unresolved,
        "state": _snapshot_to_json(record.state),
        "payload": _payload_to_json(record.payload),
    }
    if _is_observed_combat(record.payload):
        # The collector writes no probe member on an observed combat.
        del out["payload"]["probed_keyword"], out["payload"]["probed_entity"]
    for key, value in record.extra_fields.items():
        out.setdefault(key, value)
    return out


def _curation_to_json(curation: Curation) -> dict:
    return {
        "natural_share": curation.natural_share,
        "copies": curation.copies,
        "drawn_rule": curation.drawn_rule,
        "drawn_rules": list(curation.drawn_rules),
        "condition_entities": list(curation.condition_entities),
        "outcome_entities": list(curation.outcome_entities),
        "class_weight": curation.class_weight,
    }


def _curation_from_json(data: dict) -> Curation:
    return Curation(
        natural_share=int(data["natural_share"]),
        copies=int(data["copies"]),
        drawn_rule=data.get("drawn_rule"),
        drawn_rules=tuple(data.get("drawn_rules") or ()),
        condition_entities=tuple(data.get("condition_entities") or ()),
        outcome_entities=tuple(data.get("outcome_entities") or ()),
        class_weight=float(data["class_weight"]),
    )


class SupersededPayloadError(ValueError):
    """A gen-3 resolution record in the payload shape gen-3 replaced."""


def check_gen3_payload_shape(data: dict) -> None:
    """Refuse a resolution half in the superseded shape: a cost half holds
    ``events`` and no ``outcome`` (``{events, costs}``), an effect half
    carries ``outcome`` (null on a mana record and the rescue record)."""
    if data.get("kind") != RecordKind.RESOLUTION:
        return
    payload = data.get("payload") or {}
    moment = data.get("moment")
    if moment == Moment.ACTIVATION and ("events" not in payload or "outcome" in payload):
        raise SupersededPayloadError(
            "a gen-3 cost half is {events, costs}; this one is the superseded "
            f"{{{', '.join(payload)}}}")
    if moment == Moment.RESOLUTION and "outcome" not in payload:
        raise SupersededPayloadError(
            "a gen-3 effect half carries outcome; this one, of the superseded "
            "shape, carries none")


def record_from_dict(data: dict) -> EffectRecord:
    """Parse one shard line into an :class:`EffectRecord`.

    A record of an older generation, with no ``generation`` member, is refused
    naming it (:class:`~effects.domain.shard_generation.OlderGenerationError`),
    and so is a ``generation`` other than the integer 3
    (:class:`~effects.domain.shard_generation.UnknownGenerationError`) and a
    resolution half in the payload shape gen 3 replaced
    (:class:`SupersededPayloadError`). A block without a member
    :data:`~effects.domain.gen3_members.ALWAYS` owes its site is refused
    naming both (:class:`~effects.domain.gen3_members.MissingMemberError`);
    those members are read by index, never defaulted.
    """
    require_gen3(data)
    require_always(Site.RECORD, data)
    check_gen3_payload_shape(data)
    kind = RecordKind(data["kind"])
    moment = Moment(data["moment"]) if data["moment"] else None
    subkinds = SUBKINDS_OF.get(kind, PlayabilitySubkind)
    subkind = subkinds(data["subkind"]) if data["subkind"] else None
    ability = data["ability"]
    aimed_at = data.get("aimed_at")
    directed = data.get("directed")
    copy_dropped = data.get("copy_dropped")
    fork_kind = data.get("fork_kind")
    curation = data.get("curation")
    return EffectRecord(
        record_id=data["record_id"],
        run_id=data["run_id"],
        timestamp=data["timestamp"],
        game_id=data["game_id"],
        kind=kind,
        actor_player=data["actor_player"],
        state=_snapshot_from_json(data["state"]),
        payload=_payload_from_json(data, kind, moment or subkind),
        mode=CollectionMode(data["mode"]),
        moment=moment,
        subkind=subkind,
        link_id=data["link_id"],
        mirror_of=data["mirror_of"],
        variant_of=data["variant_of"],
        interventional=data["interventional"],
        fork=data["fork"],
        fork_kind=ForkKind(fork_kind) if fork_kind is not None else None,
        synthetic=data["synthetic"],
        random_seat=data["random_seat"],
        what_if=data.get("what_if"),
        generation=data["generation"],
        aimed_at=tuple(aimed_at) if aimed_at is not None else None,
        directed=(DirectedTag(directed["variant"], ForkOperation(directed["operation"]))
                  if directed is not None else None),
        copy_dropped=(tuple(sorted((str(kind), int(count)) for kind, count in copy_dropped.items()))
                      if copy_dropped is not None else None),
        curation=_curation_from_json(curation) if curation is not None else None,
        ability=_keys_from_json(ability) if ability is not None else None,
        ability_unresolved=data["ability_unresolved"],
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

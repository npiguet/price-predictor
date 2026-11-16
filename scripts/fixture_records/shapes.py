"""Reading a raw record dict (as the collector wrote it) the way every
category reads it: its events, its acting key, its decision candidate, the
lines and keywords of its entities."""

from __future__ import annotations

from collections.abc import Iterable, Iterator


def events(record: dict) -> list[dict]:
    return record["payload"].get("events") or []


def subjects(event: dict) -> list[str]:
    return event.get("subjects") or []


def params(event: dict) -> dict:
    return event.get("params") or {}


def entities(record: dict) -> list[dict]:
    return record["state"]["entities"]


def entities_by_id(record: dict) -> dict[str, dict]:
    return {entity["id"]: entity for entity in entities(record)}


def players(record: dict) -> list[dict]:
    return record["state"]["players"]


def global_state(record: dict) -> dict:
    return record["state"]["global"]


def first_key(keys: list[dict] | None) -> dict | None:
    return keys[0] if keys else None


def acting_key(record: dict) -> dict | None:
    """The first provenance key of the ability the record is about."""
    return first_key(record.get("ability"))


def acting_kind(record: dict) -> str | None:
    key = acting_key(record)
    return key["trait_kind"] if key is not None else None


def is_effect_resolution(record: dict) -> bool:
    """An observed effect half: a resolution record at its resolution moment."""
    return (record["kind"] == "resolution" and record.get("moment") == "resolution"
            and not record["fork"])


def is_game_end(record: dict) -> bool:
    return record["kind"] == "turn_action" and record.get("subkind") == "game_end"


def candidate(record: dict) -> dict | None:
    """A decision record's one candidate."""
    if record["kind"] != "playability" or record.get("subkind") != "decision":
        return None
    candidates = record["payload"].get("candidates") or ()
    return candidates[0] if candidates else None


def candidate_key(record: dict) -> dict | None:
    """A decision record's one candidate's first key."""
    found = candidate(record)
    return first_key(found.get("ability")) if found is not None else None


def failed_checks(found: dict) -> list[str]:
    return [entry["check"] for entry in found.get("failed_checks") or ()]


def lines(entity: dict) -> list[dict]:
    return entity.get("lines") or []


def fragments(entity: dict) -> list[str]:
    """The keyword lines' fragments, lower-cased (``ward:2``, ``flying``)."""
    return [str(line.get("fragment") or "").lower() for line in lines(entity)
            if line["kind"] == "keyword"]


def has_keyword(entity: dict, *names: str) -> bool:
    return any(fragment.split(":", 1)[0] in names for fragment in fragments(entity))


def on_battlefield(entity: dict) -> bool:
    return entity.get("zone") == "battlefield" and entity.get("role") is None


def battlefield(record: dict) -> Iterator[dict]:
    return (entity for entity in entities(record) if on_battlefield(entity))


def is_creature(entity: dict) -> bool:
    return "creature" in (entity.get("types") or ())


def is_death(event: dict, *, unresolved: bool = True) -> bool:
    """A battlefield-to-graveyard move (no clause's doing, by default)."""
    if event["type"] != "zone_change":
        return False
    moved = params(event)
    if moved.get("from_zone") != "battlefield" or moved.get("to_zone") != "graveyard":
        return False
    return not unresolved or event.get("attributed_to") == "unresolved"


def deaths(found: Iterable[dict] | None, *, unresolved: bool = True) -> set[str]:
    """Subjects of battlefield-to-graveyard moves (no clause's doing, by default)."""
    return {subject for event in found or () if is_death(event, unresolved=unresolved)
            for subject in subjects(event)}

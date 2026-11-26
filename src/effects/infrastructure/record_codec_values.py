"""The wire forms every block of a record shares: provenance keys, events,
counter maps, and the small converters for optional members.

Part of :mod:`effects.infrastructure.record_codec`, which reads and writes
a whole record through these.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

from effects.domain.event_schema import Event
from effects.domain.gen3_members import Site, owes
from effects.domain.provenance import ProvenanceKey
from effects.domain.state_snapshot import normalize_counter_type

# ── provenance keys ─────────────────────────────────────────────────────


def _keys_to_json(keys: tuple[ProvenanceKey, ...]) -> list[dict]:
    return [key.as_dict() for key in keys]


#: Every key parsed so far, by its raw JSON fields: a shard repeats the same
#: few hundred keys thousands of times. Bounded by the card pool's traits.
_INTERNED_KEYS: dict[tuple, ProvenanceKey] = {}


def _key_from_json(item: dict) -> ProvenanceKey:
    raw = (
        item.get("script_file"), item.get("face"), item.get("trait_kind"),
        item.get("index_within_kind"), item.get("option"), _built_identity(item.get("built")),
    )
    key = _INTERNED_KEYS.get(raw)
    if key is None:
        key = _INTERNED_KEYS[raw] = ProvenanceKey.from_dict(item)
    return key


def _built_identity(built: dict | None) -> tuple | None:
    """A key's ``built`` member as part of its interning identity; a dict is unhashable."""
    return None if built is None else (built.get("kind"), built.get("index"))


def _keys_from_json(data: Any) -> tuple[ProvenanceKey, ...]:
    return tuple(_key_from_json(item) for item in data or ())


# ── events ──────────────────────────────────────────────────────────────


def _events_to_json(events: tuple[Event, ...]) -> list[dict]:
    return [event.as_dict() for event in events]


def _events_from_json(data: Any) -> tuple[Event, ...]:
    return tuple(_event_from_json(item) for item in data or ())


@owes(Site.EVENT)
def _event_from_json(item: Any) -> Event:
    """One event, with every counter type it names in this package's spelling:
    its ``counter_type`` and a created token's ``characteristics.counters``.

    Normalized on read, like keywords, because collected shards carry Forge's
    printed spelling and cannot be rewritten.
    """
    event = Event.from_dict(item)
    params = event.params
    counter = params.get("counter_type")
    if isinstance(counter, str):
        params = {**params, "counter_type": normalize_counter_type(counter)}
    traits = params.get("characteristics")
    if isinstance(traits, dict) and isinstance(traits.get("counters"), dict):
        traits = {**traits, "counters": _counters_from_json(traits["counters"])}
        params = {**params, "characteristics": traits}
    return event if params is event.params else replace(event, params=params)


def _counters_from_json(counters: dict) -> dict[str, int]:
    """A record's ``{counter type: count}`` map with each type in this
    package's spelling, in the record's order; two spellings of one type add up."""
    named: dict[str, int] = {}
    for raw, count in counters.items():
        name = normalize_counter_type(raw)
        named[name] = named[name] + count if name in named else count
    return named


def _optional[T, R](convert: Callable[[T], R], value: T | None) -> R | None:
    """``convert(value)``, or None for an absent block: null, or empty on read."""
    return convert(value) if value else None


def _tuple_or_none(data: Any) -> tuple | None:
    """A list member as a tuple, keeping an absent or null one None."""
    return tuple(data) if data is not None else None


def _list_or_none(values: tuple | None) -> list | None:
    return list(values) if values is not None else None


def _absent_or[T, R](convert: Callable[[T], R], value: T | None) -> R | None:
    """``convert(value)``, or None for null; unlike :func:`_optional`, an
    empty block (gen-3's ``chosen: {}``) is a value."""
    return convert(value) if value is not None else None

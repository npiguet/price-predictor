"""Builders the ``test_effect_head_input_*`` modules share: a record on a
two-player board with two cards, the surface built from it over a line
lookup of plain callables, a card's lines and parts, and the lookups of
a surface's slots and relations by what they stand for."""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, replace

from effects.domain.card_features import card_features, card_segments
from effects.domain.input_layout import SlotKind
from effects.domain.provenance import ProvenanceKey
from effects.domain.records import (
    EffectRecord,
    RecordKind,
)
from effects.domain.state_snapshot import (
    EntityState,
    GlobalState,
    PlayerState,
    PowerToughness,
    StateSnapshot,
)
from effects.domain.surface_builder import NO_STRIP, SurfaceBuilder
from tests.unit.effects.builders import (
    lines_of,
    make_entity,
    make_record,
    make_snapshot,
)

E_DIM = 4
_ZERO = (0.0,) * E_DIM

_BOLT = ProvenanceKey("cardsfolder/l/lightning_bolt.txt", 0, "spell", 0)
_ANTHEM = ProvenanceKey("cardsfolder/a/anthem.txt", 0, "static", 0)
_WARD = ProvenanceKey("cardsfolder/w/ward.txt", 0, "static", 1)

_VECTORS = {
    _BOLT: (1.0, 0.0, 0.0, 0.0),
    _ANTHEM: (0.0, 1.0, 0.0, 0.0),
    _WARD: (0.0, 0.0, 1.0, 0.0),
}


def _e_for(key: ProvenanceKey):
    return _VECTORS.get(key)


@dataclass(frozen=True)
class _Lines:
    """A :class:`LineLookup` from plain callables: every key has a line, is
    no keyword, has no modes, is one slot and no dungeon room unless a test
    says otherwise."""

    e_for: Callable = _e_for
    has_line: Callable = lambda key: True
    keyword_of: Callable = lambda key: None
    options_for: Callable = lambda key: ()
    board_keys: Callable = lambda key: (key,)
    room_of: Callable = lambda key: None


def _flags(slot) -> dict[str, float]:
    """An ``[ABILITY]`` slot's line flags by name."""
    active, granted, by_effect, cast, count, _, chosen = slot.features
    return {"active": active, "granted": granted, "by_effect": by_effect,
            "cast_with": cast, "count": count, "chosen": chosen}


def _entity(entity_id: str, **overrides) -> EntityState:
    defaults = {
        "name": "Serra Angel", "types": ("Creature",), "colors": ("W",),
        "mana_value": 5, "pt": PowerToughness(base=(4, 4)),
    }
    defaults.update(overrides)
    return make_entity(entity_id, defaults.pop("name"), **defaults)


def _snapshot(**overrides) -> StateSnapshot:
    defaults = {
        "global_": GlobalState(
            turn=7, phase="main1", active="P0", priority="P0", stack_size=1,
        ),
        "players": (
            PlayerState(id="P0", life=14, hand=3, library=21, graveyard=9),
            PlayerState(id="P1", life=8, hand=1, library=30, graveyard=4),
        ),
        "entities": (
            _entity("E1", lines=lines_of(printed=(_ANTHEM,))),
            _entity("E2", controller="P1", lines=lines_of(printed=(_WARD,))),
        ),
    }
    defaults.update(overrides)
    return make_snapshot(**defaults)


def _record(**overrides) -> EffectRecord:
    kind = overrides.pop("kind", RecordKind.RESOLUTION)
    defaults: dict = {
        "record_id": "run.0.1", "game_id": "run.0.1", "state": _snapshot(),
    }
    if kind is RecordKind.RESOLUTION:
        defaults["ability"] = (_BOLT,)
    defaults.update(overrides)
    return make_record(kind, **defaults)


def _build(
    record, *, rng=None, e_for=_e_for, has_line=None, options_for=None,
    context_dropout=0.0, stripped=NO_STRIP,
):
    lines = _Lines(e_for=e_for)
    if has_line is not None:
        lines = replace(lines, has_line=has_line)
    if options_for is not None:
        lines = replace(lines, options_for=options_for)
    return SurfaceBuilder(
        lines, e_dim=E_DIM, rng=rng or random.Random(0), context_dropout=context_dropout,
    ).build(record, stripped)


_CHOICE = ProvenanceKey("cardsfolder/c/charm.txt", 0, "spell", 0)


def _line(key, kind="static", granted=False, **fields):
    from effects.domain.state_snapshot import CardLine

    return CardLine(keys=(key,) if key else (), kind=kind, granted=granted, **fields)


def _card_part(entity, record, name):
    return card_features(entity, record)[card_segments(entity, record)[name]]


_RUN = "3550376f-3c30-4231-90cd-cc0f1b308e87"


def _relation(family, direction="forward", index=None):
    from effects.domain.relation_types import relation_type

    return relation_type(family, direction, index)


def _subject_slot(surface, subject):
    """The slot of the card or player ``subject`` names."""
    return next(index for index, slot in enumerate(surface.slots)
                if slot.kind in (SlotKind.CARD, SlotKind.PLAYER) and slot.subject_id == subject)


def _linked(surface, a, b, family, index=None) -> bool:
    """Whether the surface relates slots ``a`` and ``b`` by ``family``, both ways."""
    relations = set(surface.relations)
    return ((a, b, _relation(family, "forward", index)) in relations
            and (b, a, _relation(family, "reverse", index)) in relations)


def _gen3_entity(entity_id, name="Serra Angel", lines=(), **fields):
    return _entity(entity_id, name=name, lines=lines, **fields)


def _gen3_record(entities, links=(), players=None, **snapshot):
    """A gen-3 record: a snapshot with ``links`` is one."""
    defaults = {"entities": tuple(entities), "links": tuple(links)}
    if players is not None:
        defaults["players"] = players
    defaults.update(snapshot)
    return _record(state=_snapshot(**defaults))

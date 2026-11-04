"""Snapshot to the effect head's one token surface.

    [GLOBAL] [ACT] [PLAYER] [PLAYER] [CARD] e e … [CARD] e …

Position ids reset at each ``[CARD]``, so an entity and its abilities read the
same wherever the entity sits on the board. Every record kind is a variant of
this one surface. Controller tags are derived here, relative to the record's
``actor_player``, so one snapshot serves records with different actors.

Each ``[ABILITY]`` slot is one line of its card with the line's flags (root
spec § 9.2): every line the card carries, in force or not, temporary grants
included. A keyword line the record names no script for, such as a keyword a
resolved effect granted or a keyword counter, reads its keyword's reminder
text through a :func:`keyword_line_key`. Three card-slot inputs are lists of
ids rather than features, each id a learned row in ``effect_model.py``:
subtypes, chosen creature types (the subtypes' rows) and named counters (a
row per counter type, scaled by the count).

The board's cards include the gen-3 snapshot's extras (library cards, cards
exiled with a permanent, a mutated permanent's components, the pending
event's last known subject) and, after them, a command-zone card per floating
effect or delayed trigger the snapshot lists no entity for, whose ability
slots are the source lines that made it (root spec § 9.3). Every link between
two slots is a relation, a ``(from slot, to slot, type)`` triple acting only
inside the trunk's attention (root spec § 10).
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import IntEnum
from functools import lru_cache
from types import MappingProxyType
from typing import Protocol

from effects.domain.event_schema import EventType
from effects.domain.forge_vocabularies import KEYWORD_COUNTERS, NAMED_COUNTERS, SUBTYPES
from effects.domain.kept_keyword_flags import KEPT_KEYWORD_FLAGS
from effects.domain.keyword_names import normalize_keyword, overlay_keyword
from effects.domain.provenance import ProvenanceKey, TraitKind, first_text
from effects.domain.records import (
    CombatPayload,
    ContinuousPayload,
    EffectRecord,
    Moment,
    PlayabilitySubkind,
    RecordKind,
    acting_keys,
)
from effects.domain.relation_types import (
    RELATION_TYPES,
    TARGET_INDICES,
    Direction,
    relation_type,
)
from effects.domain.state_snapshot import (
    COLORS,
    CardLine,
    CombatSubstep,
    CommandZoneObject,
    DayNight,
    EffectDuration,
    EntityState,
    FaceState,
    LineKind,
    LineRef,
    PlayerState,
    TurnCounts,
    Zone,
    normalize_counter_type,
)


class SlotKind(IntEnum):
    GLOBAL = 0
    ACT = 1
    PLAYER = 2
    CARD = 3
    ABILITY = 4


# Each vocabulary's one-hot ends in an "other" slot. Never edit these tuples,
# nor the enums two of them are read off, without a new INPUT_LAYOUT: a
# checkpoint trained on them reads them bit for bit.

PHASES: tuple[str, ...] = (
    "untap", "upkeep", "draw", "main1", "begin_combat", "declare_attackers",
    "declare_blockers", "first_strike_damage", "combat_damage", "end_combat",
    "main2", "end_of_turn", "cleanup",
)
COMBAT_SUBSTEPS: tuple[str, ...] = tuple(substep.value for substep in CombatSubstep)
ZONES: tuple[str, ...] = tuple(zone.value for zone in Zone)
CARD_TYPES: tuple[str, ...] = (
    "artifact", "battle", "creature", "enchantment", "instant", "land",
    "planeswalker", "sorcery", "kindred", "emblem",
)
SUPERTYPES: tuple[str, ...] = ("basic", "legendary", "snow", "world")
#: The counter types the head's ``counters_delta`` outputs name; the input
#: reads every named counter (:data:`NAMED_COUNTERS`) instead.
COUNTER_TYPES: tuple[str, ...] = (
    "P1P1", "M1M1", "LOYALTY", "CHARGE", "TIME", "STUN", "SHIELD", "OIL",
    "POISON", "EXPERIENCE", "KI", "LEVEL", "LORE", "DEFENSE",
)
#: Keywords the head's ``keywords_gained``/``keywords_lost`` outputs track; the
#: eight damage-step keywords must be among them. The input's keyword flags
#: are :data:`KEPT_KEYWORD_FLAGS` instead (root spec § 9.6).
OVERLAY_KEYWORDS: tuple[str, ...] = (
    "first_strike", "double_strike", "deathtouch", "lifelink", "trample",
    "indestructible", "wither", "infect",
    "flying", "reach", "vigilance", "haste", "menace", "hexproof", "shroud",
    "defender", "protection", "ward", "flash", "banding", "fear", "intimidate",
    "horsemanship", "skulk", "shadow", "unblockable", "annihilator",
)
THIS_TURN_COUNTERS: tuple[str, ...] = (
    "creatures_died", "spells_cast", "lands_played",
)
#: A gen-3 player's turn and game counts (``in.history``), in record order.
TURN_COUNTS: tuple[str, ...] = tuple(TurnCounts.__dataclass_fields__)
FACE_STATES: tuple[str, ...] = tuple(face.value for face in FaceState)
#: Day, night, or neither (the snapshot's null).
DAY_NIGHT: tuple[str, ...] = (DayNight.DAY.value, DayNight.NIGHT.value, "neither")
#: A command-zone object's duration, over the three values the collector buckets.
EFFECT_DURATIONS: tuple[str, ...] = tuple(duration.value for duration in EffectDuration)
#: Forge's closed designation list, as ``CardFacts.designationTable`` writes it.
DESIGNATIONS: tuple[str, ...] = (
    "bestowed", "cloaked", "commander", "foretold", "goaded", "harnessed",
    "manifested", "monstrous", "phased_out", "plotted", "renowned", "ring_bearer",
    "saddled", "solved", "specialized", "suspected", "unearthed", "warped",
)
#: A mana cost's symbol kinds (Forge's ``ManaCostShard`` short strings, plus
#: ``generic``): colours, colourless, the hybrid pairs, the two-generic and
#: colourless hybrids, snow, X and Phyrexian per colour. A hybrid Phyrexian
#: symbol lands in the trailing "other".
MANA_SYMBOLS: tuple[str, ...] = (
    "generic", "W", "U", "B", "R", "G", "C",
    "W/U", "W/B", "U/B", "U/R", "B/R", "B/G", "R/W", "R/G", "G/W", "G/U",
    "2/W", "2/U", "2/B", "2/R", "2/G",
    "C/W", "C/U", "C/B", "C/R", "C/G",
    "S", "X",
    "W/P", "U/P", "B/P", "R/P", "G/P",
)
#: A chosen colour's name, as the snapshot spells it, to its :data:`COLORS` symbol.
COLOR_NAMES: Mapping[str, str] = MappingProxyType({
    "white": "W", "blue": "U", "black": "B", "red": "R", "green": "G", "colorless": "C",
})
CHOSEN_DIRECTIONS: tuple[str, ...] = ("left", "right")
CHOSEN_PARITIES: tuple[str, ...] = ("even", "odd")
EVENT_TYPES: tuple[str, ...] = tuple(event.value for event in EventType)

# Forge's own spellings of phases, counters and protection miss the
# vocabularies above and would land in "other"; the layout aliases them.

#: Where a checkpoint's ``training_settings`` record the layout it trained on.
INPUT_LAYOUT_KEY = "input_layout"

PHASE_ALIASES: dict[str, str] = {
    "combat_begin": "begin_combat",
    "combat_declare_attackers": "declare_attackers",
    "combat_declare_blockers": "declare_blockers",
    "combat_first_strike_damage": "first_strike_damage",
    "combat_end": "end_combat",
}


#: The input layout every training run builds and records, and the only one a
#: loader accepts (FR-045): a checkpoint recording another layout, or none, was
#: trained on inputs this code no longer builds. Gen-3 is the card's lines as
#: ability slots, its gen-3 card, player, global and act inputs, command-zone
#: slots and the relations (root spec § 9 – § 10); a gen-2 checkpoint is refused.
INPUT_LAYOUT = "gen-3"


class CheckpointLayoutError(ValueError):
    """A checkpoint trained on an input layout this code does not build."""


#: Where a checkpoint records the trunk's relation types, by name in table-row
#: order, and the ``targets`` index cap (research M3, M4).
RELATION_TYPES_KEY = "relation_types"
RELATION_TARGET_INDICES_KEY = "relation_target_indices"
#: Where a checkpoint records the subtype and named-counter lists whose order
#: fixes its embedding rows.
SUBTYPES_KEY = "subtypes"
NAMED_COUNTERS_KEY = "named_counters"


def layout_settings() -> dict:
    """What a run records about the inputs it was built on: the input layout,
    the relation types whose order fixes the trunk's table rows, and the
    subtypes and named counters whose order fixes the card slot's rows."""
    return {
        INPUT_LAYOUT_KEY: INPUT_LAYOUT,
        RELATION_TYPES_KEY: list(RELATION_TYPES),
        RELATION_TARGET_INDICES_KEY: TARGET_INDICES,
        SUBTYPES_KEY: list(SUBTYPES),
        NAMED_COUNTERS_KEY: list(NAMED_COUNTERS),
    }


def require_input_layout(training_settings) -> None:
    """Refuse a checkpoint trained on another input layout, relation-type table
    or card-slot vocabulary."""
    settings = training_settings or {}
    recorded = settings.get(INPUT_LAYOUT_KEY)
    if recorded != INPUT_LAYOUT:
        raise CheckpointLayoutError(
            f"the checkpoint was trained on input layout {recorded!r}; this code "
            f"builds {INPUT_LAYOUT!r} only. Retrain it on the current code."
        )
    for key, value in layout_settings().items():
        if settings.get(key) != value:
            raise CheckpointLayoutError(
                f"the checkpoint records another {key} than this code's ({INPUT_LAYOUT!r} "
                "layout); its embedding tables would be read in the wrong rows. Retrain it."
            )


#: The largest magnitude a count or a signed delta is scored at, by the field
#: losses, the floor and the evaluation's deviance alike. Above it an amount is
#: the game state compounding, and one such target would dominate the Poisson
#: loss.
MAGNITUDE_CAP = 40.0

THERMOMETER_STEPS = int(MAGNITUDE_CAP)


def thermometer(value: float) -> list[float]:
    """``value`` as :data:`THERMOMETER_STEPS` positions, position k set when
    ``value >= k``, so "amount >= t" is one dot product with a one-hot of t."""
    return [1.0 if value >= k else 0.0 for k in range(1, THERMOMETER_STEPS + 1)]


_NO_THERMOMETER: tuple[float, ...] = (0.0,) * THERMOMETER_STEPS


def _phase_in(phase: str | None) -> str | None:
    if phase is None:
        return phase
    return PHASE_ALIASES.get(phase.strip().lower(), phase)


@lru_cache(maxsize=None)
def _positions(vocabulary: tuple[str, ...]) -> dict[str, int]:
    """``normalized member -> its slot``, built once per vocabulary."""
    positions: dict[str, int] = {}
    for index, member in enumerate(vocabulary):
        positions.setdefault(_spelling(member), index)
    return positions


def _spelling(value: str) -> str:
    """A value in its vocabulary's spelling: lower case, underscores."""
    return value.strip().lower().replace(" ", "_")


def vocabulary_slot(value: str, vocabulary: tuple[str, ...]) -> int | None:
    """``value``'s slot in ``vocabulary``, matched in the vocabulary's
    spelling; None for a value it does not list. The input surface and the
    head's targets both look a member up here, so they cannot normalize apart."""
    return _positions(vocabulary).get(_spelling(value))


def _one_hot(value: str | None, vocabulary: tuple[str, ...]) -> list[float]:
    """One-hot over ``vocabulary`` plus a trailing "other/absent" slot."""
    out = [0.0] * (len(vocabulary) + 1)
    if value is None:
        return out
    index = vocabulary_slot(value, vocabulary)
    out[-1 if index is None else index] = 1.0
    return out


#: What :func:`_push_scalar` appends per scalar.
_SCALAR_WIDTH = 2


def _one_hot_width(vocabulary: tuple[str, ...]) -> int:
    """What :func:`_one_hot` returns over ``vocabulary``."""
    return len(vocabulary) + 1


def _push_scalar(out: list[float], value: float) -> None:
    """A scalar as raw plus signed ``log1p``, onto the caller's list; nothing
    is binned (FR-074). Appended rather than returned: a list per scalar made
    the allocation the cost."""
    raw = float(value)
    out.append(raw)
    out.append(math.copysign(math.log1p(abs(raw)), raw))


def _push_optional(out: list[float], value: float | None) -> None:
    """A present flag, then the scalar (0 when absent)."""
    out.append(0.0 if value is None else 1.0)
    _push_scalar(out, 0 if value is None else value)


#: Power and toughness of the base, boosts and counters, then the present flag.
_NO_PT: tuple[float, ...] = (0.0,) * (3 * 2 * _SCALAR_WIDTH + 1)
#: Targets, per-target amounts and up-to counts.
_NO_STACK_EXTRAS: tuple[float, ...] = (0.0,) * (3 * _SCALAR_WIDTH)


def _multi_hot(present, vocabulary: tuple[str, ...]) -> list[float]:
    """Multi-hot over ``vocabulary`` plus a count of everything else."""
    positions = _positions(vocabulary)
    out = [0.0] * (len(vocabulary) + 1)
    seen: set[str] = set()
    other = 0
    for value in present:
        normalized = _spelling(value)
        if normalized in seen:
            continue
        seen.add(normalized)
        index = positions.get(normalized)
        if index is None:
            other += 1
        else:
            out[index] = 1.0
    out[-1] = float(other)
    return out


# ── the id lists: subtypes, chosen types, named counters ─────────────────

#: Row 0 of each id table is padding; a value its list does not name takes the
#: last row.
ID_PADDING = 0
#: Rows of the subtype table: padding, one per :data:`SUBTYPES`, other.
SUBTYPE_ROWS = len(SUBTYPES) + 2
#: Rows of the named-counter table: padding, one per :data:`NAMED_COUNTERS`, other.
COUNTER_ROWS = len(NAMED_COUNTERS) + 2


def subtype_id(subtype: str) -> int:
    """A subtype's row of the subtype table (the chosen types' too)."""
    index = vocabulary_slot(subtype, SUBTYPES)
    return SUBTYPE_ROWS - 1 if index is None else index + 1


@lru_cache(maxsize=None)
def _counter_positions() -> dict[str, int]:
    return {name: index for index, name in enumerate(NAMED_COUNTERS)}


def counter_id(counter_type: str) -> int:
    """A named counter's row of the counter table."""
    index = _counter_positions().get(normalize_counter_type(counter_type))
    return COUNTER_ROWS - 1 if index is None else index + 1


_KEYWORD_COUNTERS = frozenset(KEYWORD_COUNTERS)


def is_keyword_counter(counter_type: str) -> bool:
    """Whether a counter grants a keyword (rule 122.1b), which reads as a
    granted keyword line rather than as a counter (root spec § 9.6)."""
    return normalize_keyword(counter_type) in _KEYWORD_COUNTERS


@dataclass(frozen=True, slots=True)
class Slot:
    """One token of the input surface."""

    kind: SlotKind
    position: int
    features: tuple[float, ...] = ()
    #: An ``ACT`` or ``ABILITY`` slot's ability vector: the values (from the
    #: cache) or, in training, an ``int`` row of the batch's live encoder
    #: output. Told apart by type, never by length.
    e: tuple[float, ...] | int | None = None
    entity_id: str | None = None
    player_id: str | None = None
    #: A charm mode's ``option`` line (FR-063a); the flag plus following its
    #: root's slot is the whole link between a mode and its charm.
    option: bool = False
    #: A ``CARD`` slot's subtypes, as rows of the subtype table.
    subtypes: tuple[int, ...] = ()
    #: A ``CARD`` slot's chosen creature types, as rows of the same table.
    chosen_types: tuple[int, ...] = ()
    #: A ``CARD`` slot's named counters other than keyword counters, as
    #: ``(row of the counter table, count)``.
    counters: tuple[tuple[int, float], ...] = ()

    @property
    def subject_id(self) -> str | None:
        """The entity or player a ``CARD`` or ``PLAYER`` slot stands for: the
        id its per-entity targets and outputs are keyed by."""
        return self.entity_id or self.player_id


@dataclass(frozen=True, slots=True)
class EffectHeadInput:
    """The full token surface for one record."""

    slots: tuple[Slot, ...]
    #: Slot indices of the per-entity head's targets: every CARD and PLAYER.
    entity_slots: tuple[int, ...] = field(default_factory=tuple)
    #: ``(from slot, to slot, relation type)`` triples over ``slots`` (root spec § 10.1).
    relations: tuple[tuple[int, int, int], ...] = field(default_factory=tuple)


#: The kinds a model input can be: every kind but ``turn_action``, which no
#: curated stratum holds and so never reaches the model.
_RECORD_KINDS: tuple[str, ...] = tuple(
    k.value for k in RecordKind if k is not RecordKind.TURN_ACTION
)
_MOMENTS: tuple[str, ...] = tuple(m.value for m in Moment)
_PLAYABILITY_SUBKINDS: tuple[str, ...] = tuple(
    s.value for s in PlayabilitySubkind
)


def _event_amount(params: Mapping) -> float:
    """The amount a pending event carries: its ``amount``, else its ``delta``."""
    for name in ("amount", "delta"):
        value = params.get(name)
        if isinstance(value, int | float) and not isinstance(value, bool):
            return float(value)
    return 0.0


def global_features(record: EffectRecord) -> tuple[float, ...]:
    """Turn structure, day or night, the pending event, plus the record's own
    kind flags, which stay last (:data:`KIND_FLAG_WIDTH`)."""
    state = record.state
    g = state.global_
    out: list[float] = []
    _push_scalar(out, g.turn)
    _push_scalar(out, g.stack_size)
    out += _one_hot(_phase_in(g.phase), PHASES)
    out += _one_hot(g.combat_substep, COMBAT_SUBSTEPS)
    out.append(1.0 if g.active == record.actor_player else 0.0)
    out.append(1.0 if g.priority == record.actor_player else 0.0)
    _push_scalar(out, len(g.emblems))
    out += _one_hot(g.day_night or "neither", DAY_NIGHT)
    pending = state.pending_event
    out += _one_hot(pending.type.value if pending else None, EVENT_TYPES)
    _push_scalar(out, _event_amount(pending.params) if pending else 0)
    out += _one_hot(record.kind.value, _RECORD_KINDS)
    out += _one_hot(
        record.moment.value if record.moment else None, _MOMENTS,
    )
    out += _one_hot(
        record.subkind.value if record.subkind else None,
        _PLAYABILITY_SUBKINDS,
    )
    return tuple(out)


#: Width of the kind, moment and subkind one-hots that end every ``[GLOBAL]``
#: slot; the ``no-state`` baseline keeps exactly these (FR-094).
KIND_FLAG_WIDTH = sum(map(_one_hot_width, (_RECORD_KINDS, _MOMENTS, _PLAYABILITY_SUBKINDS)))

ACT_SLOT = 1


def act_features(record: EffectRecord) -> tuple[float, ...]:
    """Announced values and the dice and coins the resolution rolled and
    flipped (``in.random_results``); zeros for ``combat`` and the
    ``attackers``/``blockers`` subkinds, where no single line acts. Whether a
    cost half's spell resolved is an output, not an input (root spec § 11.5,
    FR-052)."""
    if not record.has_act_line:
        return tuple([0.0] * _ACT_FEATURE_WIDTH)
    refs = record.state.refs
    out: list[float] = [1.0]  # slot is populated
    _push_scalar(out, refs.x if refs.x is not None else 0)
    out.append(1.0 if refs.x is not None else 0.0)
    _push_scalar(out, len(refs.modes))
    _push_scalar(out, len(refs.targets))
    _push_scalar(out, len(refs.choices))
    _push_random_results(out, refs.random)
    return tuple(out)


def _push_random_results(out: list[float], random_results) -> None:
    """The dice rolled (how many, their sum, the highest) and the coin flips
    won and lost; zeros where the record carries none."""
    dice = random_results.dice if random_results is not None else ()
    _push_scalar(out, len(dice))
    _push_scalar(out, sum(dice))
    _push_scalar(out, max(dice, default=0))
    _push_scalar(out, random_results.coin_flips_won if random_results is not None else 0)
    _push_scalar(out, random_results.coin_flips_lost if random_results is not None else 0)


#: The populated and X-present flags, the X, mode, target and choice scalars,
#: and the five random-result scalars.
_ACT_FEATURE_WIDTH = 2 + 9 * _SCALAR_WIDTH


def player_features(player: PlayerState, record: EffectRecord) -> tuple[float, ...]:
    """One player's resources and designations, plus their relation to the
    acting player and to the turn (root spec § 9.4, § 9.5)."""
    out: list[float] = []
    for value in (player.life, player.hand, player.library, player.graveyard,
                  player.poison, player.energy):
        _push_scalar(out, value)
    for key in THIS_TURN_COUNTERS:
        _push_scalar(out, player.this_turn.get(key, 0))
    # The land plays the turn allows, beside the lands played (FR-044); 0 and
    # the flag where a static lifts the limit.
    _push_scalar(out, player.max_land_plays or 0)
    out.append(1.0 if player.land_plays_unlimited else 0.0)
    for color in COLORS:
        _push_scalar(out, player.floating_mana.get(color, 0))
    for color in COLORS:
        _push_scalar(out, player.untapped_production.get(color, 0))
    _push_scalar(out, player.speed)
    _push_scalar(out, player.rad)
    out.append(1.0 if player.city_blessing else 0.0)
    out.append(1.0 if player.monarch else 0.0)
    out.append(1.0 if player.initiative else 0.0)
    counts = player.turn_counts
    for name in TURN_COUNTS:
        _push_scalar(out, getattr(counts, name) if counts is not None else 0)
    g = record.state.global_
    out.append(1.0 if player.id == record.actor_player else 0.0)
    out.append(0.0 if player.id == record.actor_player else 1.0)
    out.append(1.0 if player.id == g.active else 0.0)
    out.append(1.0 if player.id == g.priority else 0.0)
    out.append(1.0 if player.id in record.state.refs.targets else 0.0)
    out += thermometer(player.life)
    return tuple(out)


# ── the card slot ─────────────────────────────────────────────────────────


def card_features(
    entity: EntityState,
    record: EffectRecord,
    *,
    masked_keywords: frozenset[str] = frozenset(),
) -> tuple[float, ...]:
    """One entity's structured characteristics and state (root spec § 9.4);
    ``masked_keywords`` removes keywords from its keyword flags."""
    out: list[float] = []
    for _, push in _CARD_PARTS:
        push(out, entity, record, masked_keywords)
    return tuple(out)


def card_segments(
    entity: EntityState, record: EffectRecord,
) -> dict[str, slice]:
    """Where each named part of :func:`card_features` sits; every part has a
    fixed width, so any entity gives the same slices."""
    out: list[float] = []
    segments: dict[str, slice] = {}
    for name, push in _CARD_PARTS:
        start = len(out)
        push(out, entity, record, frozenset())
        segments[name] = slice(start, len(out))
    return segments


def _push_characteristics(out, entity: EntityState, record, masked) -> None:
    """Types, supertypes, colours, mana value and the P/T breakdown; whether
    it is a copy and whether a token (subtypes are the slot's ids)."""
    out += _multi_hot(entity.types, CARD_TYPES)
    out += _multi_hot(entity.supertypes, SUPERTYPES)
    out += _multi_hot(entity.colors, COLORS)
    _push_scalar(out, entity.mana_value)
    pt = entity.pt
    if pt is None:
        out += _NO_PT
    else:
        for pair in (pt.base, pt.boosts, pt.counters):
            _push_scalar(out, pair[0])
            _push_scalar(out, pair[1])
        out.append(1.0)
    out.append(1.0 if entity.copy_source is not None else 0.0)
    out.append(1.0 if entity.token_script_id is not None else 0.0)


def face_state_of(entity: EntityState) -> str:
    """Which face is up: the record's ``face_state``, or on a gen-1 or gen-2
    entity, which writes none, its face-down flag and face index."""
    if entity.face_state is not None:
        return entity.face_state
    if entity.face_down:
        return FaceState.FACE_DOWN.value
    return FaceState.FRONT.value if entity.face == 0 else FaceState.BACK.value


def _push_face_state(out, entity: EntityState, record, masked) -> None:
    """Face state one-hot (``in.face_state``). A face-down card's real face
    reaches no card input: its lines enter only as inactive ability slots."""
    out += _one_hot(face_state_of(entity), FACE_STATES)


def _push_mana_cost(out, entity: EntityState, record, masked) -> None:
    """A scalar per mana symbol kind and one for every other symbol, after a
    flag for having a mana cost at all (``in.mana_cost``)."""
    cost = entity.mana_cost
    out.append(0.0 if cost is None else 1.0)
    cost = cost or {}
    for symbol in MANA_SYMBOLS:
        _push_scalar(out, cost.get(symbol, 0))
    _push_scalar(out, sum(n for symbol, n in cost.items() if symbol not in _MANA_SYMBOL_SET))


_MANA_SYMBOL_SET = frozenset(MANA_SYMBOLS)


def _push_printed_loyalty_defense(out, entity: EntityState, record, masked) -> None:
    """Printed loyalty and defense, each a scalar with a thermometer, as
    toughness has (``in.printed_loyalty_defense``)."""
    for value in (entity.printed_loyalty, entity.printed_defense):
        _push_optional(out, value)
        out += _NO_THERMOMETER if value is None else thermometer(value)


def _push_status(out, entity: EntityState, record, masked) -> None:
    """Zone, tapped, summoning sick, marked damage, combat."""
    out += _one_hot(entity.zone, ZONES)
    out.append(1.0 if entity.tapped else 0.0)
    out.append(1.0 if entity.sick else 0.0)
    _push_scalar(out, entity.damage)
    combat = entity.combat
    out.append(1.0 if combat and combat.attacking else 0.0)
    out.append(1.0 if combat and combat.blocking else 0.0)
    out.append(1.0 if combat and combat.became_blocked else 0.0)
    _push_scalar(out, len(combat.blocked_by) if combat else 0)


def keywords_of(entity: EntityState) -> set[str]:
    """Every keyword the card has now, in :func:`overlay_keyword`'s spelling:
    its keyword list, its keyword lines in force and its keyword counters."""
    found = {overlay_keyword(keyword) for keyword in entity.granted_temporary.keywords}
    for line in entity.keyword_lines:
        if line.in_force and line.keyword:
            found.add(overlay_keyword(line.keyword))
    for name, count in entity.counters.items():
        if count > 0 and is_keyword_counter(name):
            found.add(overlay_keyword(name))
    return found


def _push_keyword_flags(out, entity: EntityState, record, masked) -> None:
    """One flag per kept keyword (root spec § 9.6), and nothing for any other:
    a keyword without a flag is read through its line."""
    flags = [0.0] * len(KEPT_KEYWORD_FLAGS)
    positions = _positions(KEPT_KEYWORD_FLAGS)
    hidden = {overlay_keyword(keyword) for keyword in masked}
    for keyword in keywords_of(entity) - hidden:
        index = positions.get(_spelling(keyword))
        if index is not None:
            flags[index] = 1.0
    out += flags


def _push_relation_to_act(out, entity: EntityState, record: EffectRecord, masked) -> None:
    """Its relation to the acting player, the targets and the source, then the
    turn, priority and control inputs of root spec § 9.5."""
    state = record.state
    g = state.global_
    out.append(1.0 if entity.controller == record.actor_player else 0.0)
    out.append(1.0 if entity.id in state.refs.targets else 0.0)
    out.append(1.0 if entity.id == state.refs.source else 0.0)
    out.append(1.0 if entity.controller == g.active else 0.0)
    out.append(1.0 if entity.controller == g.priority else 0.0)
    out.append(1.0 if entity.owner is not None and entity.owner == record.actor_player else 0.0)
    out.append(1.0 if entity.owner is not None and entity.owner != entity.controller else 0.0)


def _push_designations(out, entity: EntityState, record, masked) -> None:
    out += _multi_hot(entity.designations, DESIGNATIONS)


def _push_x_paid(out, entity: EntityState, record, masked) -> None:
    _push_optional(out, entity.x_paid)


def _push_chosen(out, entity: EntityState, record, masked) -> None:
    """What the card remembers choosing (``in.chosen_values``): colours, card
    types, a number, whether a player, cards or names, a direction, even or
    odd, a sector or a mode, and stored die results. A creature type is a row
    of the subtype table, and a player, card or name a relation."""
    chosen = entity.chosen
    if chosen is None:
        out += _NO_CHOSEN
        return
    colors = [COLOR_NAMES.get(_spelling(color), color) for color in chosen.colors or ()]
    out += _multi_hot(colors, COLORS)
    card_types = [t for t in (chosen.type, chosen.type2)
                  if t is not None and vocabulary_slot(t, CARD_TYPES) is not None]
    out += _multi_hot(card_types, CARD_TYPES)
    _push_optional(out, chosen.number)
    out.append(1.0 if chosen.player is not None else 0.0)
    _push_scalar(out, len(chosen.cards or ()))
    _push_scalar(out, len(chosen.names or ()))
    out += _one_hot(chosen.direction, CHOSEN_DIRECTIONS)
    out += _one_hot(chosen.even_odd, CHOSEN_PARITIES)
    out.append(1.0 if chosen.sector else 0.0)
    out.append(1.0 if chosen.mode else 0.0)
    rolls = chosen.stored_rolls or ()
    _push_scalar(out, len(rolls))
    _push_scalar(out, sum(rolls))


#: The chosen block of a card that chose nothing.
_NO_CHOSEN: tuple[float, ...] = (0.0,) * (
    _one_hot_width(COLORS) + _one_hot_width(CARD_TYPES) + 1 + _SCALAR_WIDTH + 1
    + 2 * _SCALAR_WIDTH + _one_hot_width(CHOSEN_DIRECTIONS)
    + _one_hot_width(CHOSEN_PARITIES) + 2 + 2 * _SCALAR_WIDTH
)


def chosen_subtypes(entity: EntityState) -> tuple[int, ...]:
    """The subtype rows of a card's chosen types that are no card type."""
    chosen = entity.chosen
    if chosen is None:
        return ()
    return tuple(
        subtype_id(t) for t in (chosen.type, chosen.type2)
        if t is not None and vocabulary_slot(t, CARD_TYPES) is None
    )


def _push_costs_paid(out, entity: EntityState, record, masked) -> None:
    """The mana spent per colour, the zone cast from and the mana rider, after
    a flag for having been cast (``in.costs_paid``)."""
    paid = entity.costs_paid
    out.append(0.0 if paid is None else 1.0)
    mana = paid.mana if paid is not None else {}
    for color in COLORS:
        _push_scalar(out, mana.get(color, 0))
    out += _one_hot(paid.cast_from if paid is not None else None, ZONES)
    out.append(1.0 if paid is not None and paid.mana_rider else 0.0)


def _push_history(out, entity: EntityState, record, masked) -> None:
    """What the card did this turn (``in.history``)."""
    history = entity.history
    for done in (
        history.entered_this_turn if history else False,
        history.attacked_this_turn if history else False,
        history.blocked_this_turn if history else False,
        history.loyalty_activated if history else False,
    ):
        out.append(1.0 if done else 0.0)


def _push_order_rank(out, entity: EntityState, record, masked) -> None:
    _push_optional(out, entity.order_rank)


def _push_stack_extras(out, entity: EntityState, record, masked) -> None:
    extras = entity.stack_extras
    if extras is None:
        out += _NO_STACK_EXTRAS
        return
    _push_scalar(out, len(extras.targets))
    _push_scalar(out, sum(extras.per_target_amounts.values()))
    _push_scalar(out, sum(extras.up_to_counts.values()))


def _push_damage_assignment(out, entity: EntityState, record: EffectRecord, masked) -> None:
    """A combat record's declared combat and damage-assignment choices."""
    payload = record.payload
    if not isinstance(payload, CombatPayload):
        out += (0.0, 0.0, 0.0)
        return
    assigned = payload.assignment_choices.get(entity.id, {})
    out.append(1.0 if entity.id in payload.attackers else 0.0)
    out.append(float(len(assigned)))
    out.append(float(sum(assigned.values())))


def _push_pending_event(out, entity: EntityState, record: EffectRecord, masked) -> None:
    """Whether a ``rewrite`` or ``trigger`` record has a pending event, and
    whether this card is its subject; the event's type and amount are the
    ``[GLOBAL]`` slot's."""
    pending = record.state.pending_event
    if pending is None:
        out += (0.0, 0.0)
        return
    out.append(1.0)
    out.append(1.0 if entity.id in pending.subjects else 0.0)


def command_zone_object(record: EffectRecord, object_id: str) -> CommandZoneObject | None:
    """The floating effect or pending delayed trigger named ``object_id``;
    None for any other id, and on a gen-1 or gen-2 record, which lists none."""
    g = record.state.global_
    for found in (*g.floating_effects, *g.delayed_triggers):
        if found.id == object_id:
            return found
    return None


def _push_command_zone(out, entity: EntityState, record: EffectRecord, masked) -> None:
    """What a command-zone object's text cannot say (root spec § 9.3): its
    duration over the three values, and the amount left on a counting-down
    shield after a present flag; zeros on any other card. Its links are
    relations, and what it does is its source lines' ``e``."""
    found = command_zone_object(record, entity.id)
    out += _one_hot(found.duration if found is not None else None, EFFECT_DURATIONS)
    _push_optional(out, found.shield_left if found is not None else None)


def _push_comparable_amounts(out, entity: EntityState, record, masked) -> None:
    """Thermometers of total power, total toughness, and the toughness left
    after marked damage, subtracted here so the trunk need not learn to."""
    if entity.pt is None:
        out += _NO_THERMOMETER * 3
        return
    power, toughness = entity.pt.total
    out += thermometer(power)
    out += thermometer(toughness)
    out += thermometer(toughness - entity.damage)


#: :func:`card_features`, part by part, in order.
_CARD_PARTS = (
    ("characteristics", _push_characteristics),
    ("face_state", _push_face_state),
    ("mana_cost", _push_mana_cost),
    ("printed_loyalty_defense", _push_printed_loyalty_defense),
    ("status", _push_status),
    ("keyword_flags", _push_keyword_flags),
    ("relation_to_act", _push_relation_to_act),
    ("designations", _push_designations),
    ("x_paid", _push_x_paid),
    ("chosen", _push_chosen),
    ("costs_paid", _push_costs_paid),
    ("history", _push_history),
    ("order_rank", _push_order_rank),
    ("stack_extras", _push_stack_extras),
    ("damage_assignment", _push_damage_assignment),
    ("pending_event", _push_pending_event),
    ("command_zone", _push_command_zone),
    ("comparable_amounts", _push_comparable_amounts),
)


def counter_rows(entity: EntityState) -> tuple[tuple[int, float], ...]:
    """A card's named counters but its keyword counters, as rows of the
    counter table with their counts; two spellings of one type add up."""
    rows: dict[int, float] = {}
    for name, count in entity.counters.items():
        if count and not is_keyword_counter(name):
            row = counter_id(name)
            rows[row] = rows.get(row, 0.0) + float(count)
    return tuple(rows.items())


# ── the command zone ──────────────────────────────────────────────────────


def command_zone_entities(record: EffectRecord) -> tuple[EntityState, ...]:
    """A card for each floating effect and pending delayed trigger that is no
    card of the snapshot (a timestamped ``F`` effect, a ``D`` trigger), read
    like a permanent in the command zone (root spec § 9.3): its lines are the
    source lines that made it, granted, and its controller is that of the card
    its ``created_by`` link ends at. An effect card (``E``) is a snapshot
    entity already, its lines its source lines. None on gen-1 and gen-2."""
    state = record.state
    g = state.global_
    if not (g.floating_effects or g.delayed_triggers):
        return ()
    known = {entity.id for entity in state.entities}
    controllers = {entity.id: entity.controller for entity in state.entities}
    makers: dict[str, str] = {}
    for link in state.links or ():
        if link.type == "created_by" and isinstance(link.from_, str):
            end = link.to.entity if isinstance(link.to, LineRef) else link.to
            makers.setdefault(link.from_, end)
    return tuple(
        EntityState(
            id=found.id, name="", zone=Zone.COMMAND.value,
            controller=controllers.get(makers.get(found.id, ""), ""),
            lines=tuple(
                CardLine(keys=(key,), kind=key.trait_kind, granted=True)
                for key in found.source_lines
            ),
        )
        for found in (*g.floating_effects, *g.delayed_triggers)
        if found.id not in known
    )


def board_entities(record: EffectRecord) -> tuple[EntityState, ...]:
    """Every card the surface gives a ``[CARD]`` slot: the snapshot's
    entities, then :func:`command_zone_entities`."""
    return (*record.state.entities, *command_zone_entities(record))


# ── the ability slots ─────────────────────────────────────────────────────

#: The tree of a :func:`keyword_line_key`: no folder, as ``counterfactual``
#: is none; the batcher reads the key's text as its keyword's reminder text.
KEYWORD_LINE_TREE = "keyword-line"


def keyword_line_key(fragment: str) -> ProvenanceKey:
    """The key a keyword line with no script of its own is read through: a
    keyword a resolved effect granted, a keyword counter's keyword. Its text
    is the keyword's reminder text (root spec § 9.2)."""
    return ProvenanceKey(
        f"{KEYWORD_LINE_TREE}/{fragment}", 0, TraitKind.KEYWORD.value, 0,
    )


def keyword_line_fragment(key: ProvenanceKey) -> str | None:
    """The keyword fragment a :func:`keyword_line_key` reads; None for any
    other key."""
    tree, _, fragment = key.script_file.partition("/")
    return fragment if tree == KEYWORD_LINE_TREE else None


@dataclass(frozen=True, slots=True)
class SlotLine:
    """One line of a card as an ``[ABILITY]`` slot reads it: its key and its
    line flags (root spec § 9.2)."""

    key: ProvenanceKey
    #: The keyword the line is, where the record says (a gen-3 keyword line).
    keyword: str | None = None
    #: In force: its face is up, its static's condition holds.
    active: bool = True
    #: Not printed on the card.
    granted: bool = False
    #: Granted by an effect rather than by a static: until end of turn, say.
    temporary: bool = False
    #: How many times the spell paid this line's cost; None where it did not.
    cast_with: int | None = None
    #: The ``Choices$`` positions a modal line chose.
    chosen: tuple[int, ...] = ()

    def merged(self, other: SlotLine) -> SlotLine:
        """Two lines of one key as one slot: in force where either is."""
        cast_with = max((n for n in (self.cast_with, other.cast_with) if n is not None),
                        default=None)
        return SlotLine(
            key=self.key, keyword=self.keyword or other.keyword,
            active=self.active or other.active, granted=self.granted and other.granted,
            temporary=self.temporary and other.temporary, cast_with=cast_with,
            chosen=tuple(dict.fromkeys((*self.chosen, *other.chosen))),
        )


def slot_lines(entity: EntityState) -> tuple[SlotLine, ...]:
    """A card's lines as ``[ABILITY]`` slots, each key once, in line order.

    Gen-3: every line, in force or not (a face not up, an unmet static, the
    real face of a face-down card are slots flagged inactive). A line with no
    key is read through :func:`keyword_line_key` where it is a keyword line,
    and has no slot otherwise. A granted line is temporary unless a static
    grants it. Gen-1 and gen-2: the printed and attached lines, then the
    abilities a layer-6 change granted, flagged temporary. Either way a
    keyword counter whose keyword no line carries adds a granted keyword line.
    """
    merged: dict[ProvenanceKey, SlotLine] = {}

    def add(line: SlotLine) -> None:
        earlier = merged.get(line.key)
        merged[line.key] = line if earlier is None else earlier.merged(line)

    if entity.lines is None:
        attached = set(entity.granted_attached)
        for key in (*entity.printed, *entity.granted_attached):
            add(SlotLine(key, granted=key in attached))
        for key in entity.granted_temporary.abilities:
            add(SlotLine(key, granted=True, temporary=True))
    else:
        for line in entity.lines:
            if line.keys:
                key = line.keys[0]
            elif line.kind == LineKind.KEYWORD and line.fragment:
                key = keyword_line_key(line.fragment)
            else:
                continue
            add(SlotLine(
                key, keyword=line.keyword, active=line.in_force, granted=line.granted,
                temporary=line.granted and (
                    not line.keys or key.trait_kind != TraitKind.STATIC.value),
                cast_with=line.cast_with, chosen=line.chosen or (),
            ))
    carried = {overlay_keyword(line.keyword) for line in merged.values() if line.keyword}
    for name, count in entity.counters.items():
        keyword = normalize_keyword(name)
        if count > 0 and keyword in _KEYWORD_COUNTERS and keyword not in carried:
            carried.add(keyword)
            add(SlotLine(keyword_line_key(name), keyword=keyword, granted=True))
    return tuple(merged.values())


def ability_features(line: SlotLine, *, chosen: bool = False) -> tuple[float, ...]:
    """An ``[ABILITY]`` slot's line flags: active, granted, granted
    temporarily, cast with (a flag and the count), and chosen, which a charm's
    mode slot sets where the spell chose that mode."""
    out = [
        1.0 if line.active else 0.0,
        1.0 if line.granted else 0.0,
        1.0 if line.temporary else 0.0,
        0.0 if line.cast_with is None else 1.0,
    ]
    _push_scalar(out, line.cast_with or 0)
    out.append(1.0 if chosen else 0.0)
    return tuple(out)


ABILITY_FEATURE_WIDTH = 4 + _SCALAR_WIDTH + 1


class LineLookup(Protocol):
    """What the surface asks of each ability key's rendered line."""

    def e_for(self, key: ProvenanceKey) -> tuple[float, ...] | int | None:
        """The line's ``e`` (see :attr:`Slot.e`). None gives a zero vector
        rather than dropping the token, so a variant that masks ``e`` keeps
        the geometry."""

    def has_line(self, key: ProvenanceKey) -> bool:
        """False for a key with no rendered line (the implicit permanent
        spell), which gets no token."""

    def keyword_of(self, key: ProvenanceKey) -> str | None:
        """Which keyword the line *is*."""

    def options_for(self, key: ProvenanceKey) -> tuple[ProvenanceKey, ...]:
        """The charm-mode keys whose slots follow the key's (FR-063a)."""


#: The ``stripped`` map of a record built as collected: no entity loses a keyword.
NO_STRIP: Mapping[str, frozenset[str]] = MappingProxyType({})


# ── relations ─────────────────────────────────────────────────────────────


@lru_cache(maxsize=None)
def _relation_pair(family: str, target_index: int | None) -> tuple[int, int]:
    """A family's forward and reverse type rows (one target index's for ``targets``)."""
    return (relation_type(family, Direction.FORWARD, target_index),
            relation_type(family, Direction.REVERSE, target_index))


def _name_key(name: str) -> str:
    return name.strip().lower()


class _SlotRelations:
    """One surface's relations (root spec § 10.3), gathered while its slots
    are laid out and built once they are fixed.

    Every link is two triples, the family's forward type from the end the
    table names and its reverse type from the other, each triple once. The
    reader builds four families itself: ``line_of`` (each ability slot to its
    card), ``line_of_the_same_card`` (each pair of one card's ability slots),
    ``mode_of`` (a charm's mode slots to their root) and ``has_the_same_name``
    (each pair of cards whose names match, and a card that chose a name to
    each card of that name). The rest are the record's ``links``. ``[ACT]``
    reads ``targets`` to the acting ability's targets and ``source_of`` from
    its source, as gen-2's target and source flags do (root spec § 10.4).

    A link end that is a line reads that line's slot; where the line has no
    slot (context dropout, gate 2's strip, a key with no rendered line) it
    reads the line's card, as the collector writes a link whose line Forge
    does not record. A link to an object the surface has no slot for is
    dropped, and so is a link both of whose ends read one slot.
    """

    def __init__(self) -> None:
        #: A player's or card's id to its slot.
        self.subjects: dict[str, int] = {}
        self._lines: dict[tuple[str, ProvenanceKey], int] = {}
        self._named: dict[str, list[int]] = {}
        self._chosen_names: list[tuple[int, tuple[str, ...]]] = []
        self._triples: dict[tuple[int, int, int], None] = {}

    def link(self, a: int | None, b: int | None, family: str,
             target_index: int | None = None) -> None:
        if a is None or b is None or a == b:
            return
        forward, reverse = _relation_pair(family, target_index)
        self._triples[(a, b, forward)] = None
        self._triples[(b, a, reverse)] = None

    def _pairs(self, rows: list[int], family: str) -> None:
        for i, a in enumerate(rows):
            for b in rows[i + 1:]:
                self.link(a, b, family)

    def add_card(self, entity: EntityState, start: int, block: list[Slot],
                 keys: list[ProvenanceKey | None]) -> None:
        """One card's block, laid out from slot ``start``: its own lines'
        relations now, the rest noted for :meth:`build`."""
        self.subjects.setdefault(entity.id, start)
        lines: list[int] = []
        root = start
        for offset in range(1, len(block)):
            at = start + offset
            self._lines.setdefault((entity.id, keys[offset]), at)
            self.link(at, start, "line_of")
            if block[offset].option:
                self.link(at, root, "mode_of")
            else:
                root = at
            lines.append(at)
        self._pairs(lines, "line_of_the_same_card")
        # A face-down card's name would be its real face's: it has none here.
        if entity.name and face_state_of(entity) != FaceState.FACE_DOWN:
            self._named.setdefault(_name_key(entity.name), []).append(start)
        if entity.chosen is not None and entity.chosen.names:
            self._chosen_names.append((start, entity.chosen.names))

    def _end(self, end) -> int | None:
        if isinstance(end, LineRef):
            at = self._lines.get((end.entity, end.key))
            return at if at is not None else self.subjects.get(end.entity)
        return self.subjects.get(end)

    def build(self, record: EffectRecord) -> tuple[tuple[int, int, int], ...]:
        for rows in self._named.values():
            self._pairs(rows, "has_the_same_name")
        for chooser, names in self._chosen_names:
            for name in names:
                for at in self._named.get(_name_key(name), ()):
                    self.link(chooser, at, "has_the_same_name")
        state = record.state
        for found in state.links or ():
            index = (found.index or 0) if found.type == "targets" else None
            self.link(self._end(found.from_), self._end(found.to), found.type, index)
        if record.has_act_line:
            refs = state.refs
            for index, target in enumerate(refs.targets):
                self.link(ACT_SLOT, self.subjects.get(target), "targets", index)
            if refs.source is not None:
                self.link(self.subjects.get(refs.source), ACT_SLOT, "source_of")
        return tuple(self._triples)


class SurfaceBuilder:
    """Lays records out as the token surface, under one batch's constants.

    ``rng`` draws the context dropout: each entity's abilities but the acting
    one's lose their token with probability ``context_dropout``, drawn in
    board order.
    """

    def __init__(
        self, lines: LineLookup, *, e_dim: int, rng: random.Random,
        context_dropout: float = 0.0,
    ) -> None:
        self.lines = lines
        self.rng = rng
        self.context_dropout = context_dropout
        self._zero = tuple([0.0] * e_dim)

    def build(
        self, record: EffectRecord, stripped: Mapping[str, frozenset[str]] = NO_STRIP,
    ) -> EffectHeadInput:
        """One record's surface.

        A ``decision`` record's ``[ACT]`` reads its candidate's ``e``
        (FR-060a), and a ``continuous`` record's own keywords are hidden from
        the keyword flags (:func:`continuous_masked_keywords`). ``stripped`` is
        gate 2's perturbation: per-entity keywords removed from the flags *and*
        as ability slots, the keyword's lines whether printed or granted; the
        continuous mask must not silence a printed line of the same keyword,
        so the two stay apart.

        A floating effect or delayed trigger that is no snapshot entity gets a
        ``[CARD]`` slot after the board's (:func:`command_zone_entities`), but
        no per-entity target: nothing the head predicts happens to it.
        Relations are built last, over the slots as they stand after context
        dropout and the strip (:class:`_SlotRelations`).
        """
        slots = [
            Slot(
                kind=SlotKind.GLOBAL, position=0,
                features=global_features(record),
            ),
            Slot(
                kind=SlotKind.ACT, position=ACT_SLOT,
                features=act_features(record), e=self._act_vector(record),
            ),
        ]
        relations = _SlotRelations()
        entity_slots: list[int] = []
        for player in record.state.players:
            entity_slots.append(len(slots))
            relations.subjects[player.id] = len(slots)
            slots.append(Slot(
                kind=SlotKind.PLAYER, position=len(slots),
                features=player_features(player, record),
                player_id=player.id,
            ))

        masked = continuous_masked_keywords(record)
        source_id = record.state.refs.source
        for entity in record.state.entities:
            gone = stripped.get(entity.id, frozenset())
            entity_slots.append(len(slots))
            block, keys = self._entity_block(
                entity, record, masked=masked.get(entity.id, frozenset()) | gone,
                gone=frozenset(overlay_keyword(keyword) for keyword in gone),
                dropout=self.context_dropout if entity.id != source_id else 0.0,
            )
            relations.add_card(entity, len(slots), block, keys)
            slots += block
        for entity in command_zone_entities(record):
            block, keys = self._entity_block(
                entity, record, masked=frozenset(), gone=frozenset(),
                dropout=self.context_dropout,
            )
            relations.add_card(entity, len(slots), block, keys)
            slots += block
        return EffectHeadInput(
            slots=tuple(slots), entity_slots=tuple(entity_slots),
            relations=relations.build(record),
        )

    def _entity_block(
        self, entity: EntityState, record: EffectRecord, *,
        masked: frozenset[str], gone: frozenset[str], dropout: float,
    ) -> tuple[list[Slot], list[ProvenanceKey | None]]:
        """One entity's ``CARD`` slot, then an ``ABILITY`` slot per line it
        keeps, each charm root followed by its mode lines; positions restart.
        Beside the block, the key each slot reads (None for the card)."""
        keys: list[ProvenanceKey | None] = [None]
        block = [Slot(
            kind=SlotKind.CARD, position=0,
            features=card_features(entity, record, masked_keywords=masked),
            entity_id=entity.id,
            subtypes=tuple(subtype_id(subtype) for subtype in entity.subtypes),
            chosen_types=chosen_subtypes(entity),
            counters=counter_rows(entity),
        )]
        for line in slot_lines(entity):
            if not self._keeps(line, gone=gone, dropout=dropout):
                continue
            modes = self.lines.options_for(line.key)
            for mode_key, option in ((line.key, False), *((mode, True) for mode in modes)):
                resolved = self.lines.e_for(mode_key)
                block.append(Slot(
                    kind=SlotKind.ABILITY, position=len(block),
                    features=ability_features(
                        line, chosen=option and mode_key.option in line.chosen),
                    # Explicitly against None: row 0 is a valid matrix row and
                    # ``or`` would read it as a miss.
                    e=self._zero if resolved is None else resolved,
                    entity_id=entity.id, option=option,
                ))
                keys.append(mode_key)
        return block, keys

    def _keeps(self, line: SlotLine, *, gone: frozenset[str], dropout: float) -> bool:
        """Whether one of an entity's lines gets a slot."""
        if not self.lines.has_line(line.key):
            return False
        if gone:
            keyword = line.keyword or self.lines.keyword_of(line.key)
            if keyword is not None and overlay_keyword(keyword) in gone:
                # Dropped rather than zeroed: a zero token still says an
                # ability is here.
                return False
        return not (dropout > 0.0 and self.rng.random() < dropout)

    def _act_vector(self, record: EffectRecord):
        """The acting line's ``e``: the first of :func:`acting_keys` that
        resolves, so ``[ACT]`` reads exactly the line rarity and gate 1 key on.
        A merged line's keys all name the same row."""
        if not record.has_act_line:
            return self._zero
        vector = first_text(acting_keys(record), self.lines.e_for)
        return self._zero if vector is None else vector


def continuous_masked_keywords(record: EffectRecord) -> dict[str, frozenset[str]]:
    """Keywords a ``continuous`` record's own static contributed (FR-020),
    hidden so the input does not already contain the answer."""
    match record.payload:
        case ContinuousPayload(contributions=contributions):
            return {
                contribution.entity: frozenset(contribution.keywords)
                for contribution in contributions
                if contribution.keywords
            }
    return {}


def feature_widths(records) -> dict[SlotKind, int]:
    """Each slot kind's width, measured from real records; 0 for a kind no
    record has."""
    probe = records[0]
    widths = {
        SlotKind.GLOBAL: len(global_features(probe)),
        SlotKind.ACT: len(act_features(probe)),
        SlotKind.PLAYER: 0,
        SlotKind.CARD: 0,
        SlotKind.ABILITY: ABILITY_FEATURE_WIDTH,
    }
    for record in records:
        if record.state.players:
            widths[SlotKind.PLAYER] = len(
                player_features(record.state.players[0], record)
            )
        if record.state.entities:
            widths[SlotKind.CARD] = len(
                card_features(record.state.entities[0], record)
            )
        if widths[SlotKind.PLAYER] and widths[SlotKind.CARD]:
            break
    return widths

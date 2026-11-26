"""The closed name lists the input surface one-hots over, and the id tables.

Each list's one-hot ends in an "other" slot; a value is looked up in its
list's spelling (:func:`vocabulary_slot`), which the input surface and the
head's targets share. Three card-slot inputs are lists of ids rather than
features, each id a learned row in ``effect_model.py``: subtypes, chosen
creature types (the subtypes' rows) and named counters (a row per counter
type, scaled by the count).
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from types import MappingProxyType

from effects.domain.event_schema import EventType
from effects.domain.forge_vocabularies import KEYWORD_COUNTERS, NAMED_COUNTERS, SUBTYPES
from effects.domain.keyword_names import normalize_keyword
from effects.domain.records import (
    Moment,
    PlayabilitySubkind,
    RecordKind,
)
from effects.domain.state_snapshot import (
    CombatSubstep,
    DayNight,
    EffectDuration,
    EntityRole,
    FaceState,
    TurnCounts,
    Zone,
    normalize_counter_type,
)

# Each vocabulary's one-hot ends in an "other" slot. A checkpoint records the
# order of each of these tuples, and of the enums they are read off
# (``checkpoint_layout.LAYOUT_VOCABULARIES``), and a loader refuses another.

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


THIS_TURN_COUNTERS: tuple[str, ...] = (
    "creatures_died", "spells_cast", "lands_played",
)
#: A gen-3 player's turn and game counts (``in.history``), in record order.
TURN_COUNTS: tuple[str, ...] = tuple(TurnCounts.__dataclass_fields__)
FACE_STATES: tuple[str, ...] = tuple(face.value for face in FaceState)
#: A card slot's role (``in.role``): an entity of the four inclusion tiers, a
#: command-zone card among them, is ``tier``; a gen-3 extra is its
#: :class:`EntityRole`, in the enum's order.
TIER_ROLE = "tier"
ENTITY_ROLES: tuple[str, ...] = (TIER_ROLE, *(role.value for role in EntityRole))
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
#: The pending event's ``mode`` (the engine's name for the hook that wrote
#: it: a ``state_flag_change`` is a phase, a room entered or an attacker
#: unblocked only by it): every mode the gen-3 fixture's and the five
#: 2026-11-21/22 review pilots' pending events carry, in case-blind
#: alphabetical order. The corpus manifest lists no event modes.
PENDING_EVENT_MODES: tuple[str, ...] = (
    "AbilityCast", "AddCounter", "Always", "Attached", "AttackerBlocked",
    "AttackerBlockedByCreature", "AttackersDeclared", "AttackersDeclaredOneTarget",
    "AttackerUnblocked", "Attacks", "BecomeMonstrous", "BecomesSaddled",
    "BecomesTarget", "BeginPhase", "Blocks", "ChangesController", "ChangesZone",
    "ChangesZoneAll", "ClassLevelGained", "CollectEvidence", "CommitCrime",
    "CounterAdded", "CounterAddedOnce", "CounterRemoved", "CounterRemovedOnce",
    "CreateToken", "Crewed", "Cycled", "DamageDealtOnce", "DamageDone",
    "DamageDoneOnce", "DamageDoneOnceByController", "DayTimeChanges", "Destroy",
    "Discarded", "DiscardedAll", "Draw", "Drawn", "Enlisted", "Evolved",
    "ExcessDamageAll", "Exerted", "Exiled", "Exploited", "GainLife", "GameLoss",
    "Immediate", "LandPlayed", "LifeGained", "LifeLost", "LifeLostAll", "LifeReduced",
    "ManaExpend", "ManifestDread", "Moved", "Mutates", "PayCumulativeUpkeep", "PayEcho",
    "Phase", "Proliferate", "RingTemptsYou", "RollDice", "RolledDieOnce", "RoomEntered",
    "Sacrificed", "SacrificedOnce", "SpellCast", "SpellCastOrCopy", "Surveil",
    "TakesInitiative", "Taps", "TapsForMana", "TokenCreated", "Transformed",
    "TurnBegin", "TurnFaceUp", "Unattached", "UnlockDoor", "Untap", "Untaps", "Vote",
)

# Forge's own spellings of phases, counters and protection miss the
# vocabularies above and would land in "other"; the layout aliases them.

PHASE_ALIASES: dict[str, str] = {
    "combat_begin": "begin_combat",
    "combat_declare_attackers": "declare_attackers",
    "combat_declare_blockers": "declare_blockers",
    "combat_first_strike_damage": "first_strike_damage",
    "combat_end": "end_combat",
}


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


def _one_hot_width(vocabulary: tuple[str, ...]) -> int:
    """What :func:`_one_hot` returns over ``vocabulary``."""
    return len(vocabulary) + 1


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


#: The kinds a model input can be: every kind but ``turn_action``, which no
#: curated stratum holds and so never reaches the model.
RECORD_KINDS: tuple[str, ...] = tuple(
    k.value for k in RecordKind if k is not RecordKind.TURN_ACTION
)
MOMENTS: tuple[str, ...] = tuple(m.value for m in Moment)
PLAYABILITY_SUBKINDS: tuple[str, ...] = tuple(
    s.value for s in PlayabilitySubkind
)

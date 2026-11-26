"""What the rules of the game decide, as pure functions of a board.

Each function here states one rule's outcome from the facts it depends on, so
that the code labelling counterfactual copies, the probe grids that edit a
board and score the model against the rule, and the rule-test detectors all
read one statement of the rule rather than three. Rules covered:

- 704.5f-g, a creature with toughness 0 or less, or with lethal damage marked,
  is put into its owner's graveyard (:func:`lethal`);
- 510, 702.2, 702.4, 702.7 and 702.12, which creatures die in one damage step of
  a one-on-one block (:func:`step_deaths`);
- 702.9, 702.13, 702.16, 702.28, 702.31 and 702.36, whether one creature may
  block another as far as evasion and protection decide (:func:`may_block`);
- 702.16b, c, e and j, what protection from a quality stops: being targeted
  (:func:`may_target`), being enchanted (:func:`may_enchant`) and being dealt
  damage (:func:`damage_prevented`), over the qualities
  :class:`ProtectionQuality` reads: a colour, a card type, a creature type, a
  condition, everything;
- 500.1, which phase and step a turn is in, as Forge's scripts name a set of
  them (:func:`forge_phases`), and 307.1 and 117.1a, when a sorcery-speed
  play's timing allows it (:func:`sorcery_timing`);
- 702.61a, split second: while a spell with it is on the stack, no other spell
  may be cast and no ability but a mana ability activated
  (:func:`split_second_forbids`).

Torch-free.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import NamedTuple

from effects.domain.forge_vocabularies import SUBTYPES
from effects.domain.keyword_names import HEXPROOF_FROM, PROTECTION_FROM, normalize_keyword
from effects.domain.state_snapshot import EntityState

# ── 704.5f-g ─────────────────────────────────────────────────────────────


class EffectKind(StrEnum):
    """What one amount does to a creature it hits."""

    DAMAGE = "damage"  # marks damage on it
    SHRINK = "shrink"  # lowers its toughness, -X/-X style


def damage_lethal(toughness: int, damage: int, amount: int) -> bool:
    """Whether ``amount`` more damage kills a creature (704.5f-g)."""
    return toughness <= 0 or damage + amount >= toughness


def shrink_lethal(toughness: int, damage: int, amount: int) -> bool:
    """Whether losing ``amount`` toughness kills a creature (704.5f-g)."""
    left = toughness - amount
    return left <= 0 or (damage > 0 and damage >= left)


_LETHAL = {EffectKind.DAMAGE: damage_lethal, EffectKind.SHRINK: shrink_lethal}


def lethal(kind: EffectKind, amount: int, entity: EntityState) -> bool:
    """Whether ``amount`` kills ``entity`` under rule 704.5f-g."""
    return _LETHAL[kind](entity.toughness, entity.damage, amount)


# ── combat damage ────────────────────────────────────────────────────────


def strikes_in_step(keywords: Collection[str], step: str) -> bool:
    """Whether a creature with ``keywords`` deals its combat damage in ``step``.

    Rules 510, 702.7 and 702.4: a first or double striker in the first-strike step,
    a creature without first strike or with double strike in the regular step.
    """
    keywords = set(keywords)
    if step == "first_strike":
        return bool(keywords & {"first_strike", "double_strike"})
    return "first_strike" not in keywords or "double_strike" in keywords


@dataclass(frozen=True, slots=True)
class BlockKeywords:
    """The keywords of a one-on-one block's two creatures, as
    :func:`step_deaths` reads them."""

    attacker: Iterable[str] = frozenset()
    blocker: Iterable[str] = frozenset()


def step_deaths(
    attacker: tuple[int, int, int], blocker: tuple[int, int, int],
    keywords: BlockKeywords = BlockKeywords(), step: str = "regular",
) -> tuple[bool, bool]:
    """``(attacker dies, blocker dies)`` in one damage step of a one-on-one block.

    Each creature is ``(power, toughness, damage already marked)``, ``keywords``
    their keywords. Rules 510 and
    702.7, 702.4, 702.2 and 702.12: a first or double striker deals damage in the
    first-strike step, a creature without first strike or with double strike in the
    regular step, both creatures striking in a step at once. Lethal damage is marked
    damage at least the toughness, or any damage from a deathtouch source;
    indestructible survives both, but not toughness 0 or less (704.5f).
    """
    power = {"a": attacker[0], "b": blocker[0]}
    toughness = {"a": attacker[1], "b": blocker[1]}
    marked = {"a": attacker[2], "b": blocker[2]}
    held = {"a": set(keywords.attacker), "b": set(keywords.blocker)}
    deathtouched = {"a": False, "b": False}
    other = {"a": "b", "b": "a"}

    for side in [s for s in ("a", "b") if strikes_in_step(held[s], step) and power[s] > 0]:
        marked[other[side]] += power[side]
        if "deathtouch" in held[side]:
            deathtouched[other[side]] = True
    dead = {side: _dies(toughness[side], marked[side], deathtouched[side],
                        "indestructible" in held[side])
            for side in ("a", "b")}
    return dead["a"], dead["b"]


def _dies(toughness: int, marked: int, deathtouched: bool, indestructible: bool) -> bool:
    """704.5f: toughness 0 or less dies whatever it has; 704.5g and 702.2b:
    lethal or deathtouch damage kills what is not indestructible."""
    return toughness <= 0 or (not indestructible and (marked >= toughness or deathtouched))


# ── protection (702.16) ──────────────────────────────────────────────────

COLOUR_WORDS = {"W": "white", "U": "blue", "B": "black", "R": "red", "G": "green"}

#: The card types protection can name, lower-case as a record spells them.
CARD_TYPES = ("artifact", "battle", "creature", "enchantment", "instant", "kindred",
              "land", "planeswalker", "sorcery")


class Characteristics(NamedTuple):
    """What a quality is tested against: one object's characteristics, colours
    as the snapshot spells them (``W``, ``U``, ...), the rest lower-case."""

    types: frozenset[str]
    subtypes: frozenset[str]
    colours: frozenset[str]
    mana_value: int
    #: The object has changeling: every creature type (702.73a), which no
    #: subtype list of Forge's spells out.
    changeling: bool = False

    @classmethod
    def of(cls, entity: EntityState) -> Characteristics:
        made = cls.made(entity.types, entity.subtypes, entity.colors, entity.mana_value)
        return made.with_changeling(has_changeling(entity))

    @classmethod
    def made(cls, types: Iterable[str] = (), subtypes: Iterable[str] = (),
             colours: Iterable[str] = (), mana_value: int = 0) -> Characteristics:
        """Characteristics from their parts, without changeling
        (:meth:`with_changeling` adds it)."""
        return cls(
            frozenset(t.lower() for t in types), frozenset(s.lower() for s in subtypes),
            frozenset(c for c in colours if c in COLOUR_WORDS), int(mana_value),
        )

    def with_changeling(self, changeling: bool) -> Characteristics:
        """These characteristics, with or without changeling."""
        return self._replace(changeling=bool(changeling))


def has_changeling(entity: EntityState) -> bool:
    """Whether the entity has changeling, granted or on one of its lines."""
    if "changeling" in entity.granted_temporary.keywords:
        return True
    return any(line.keyword == "changeling" for line in entity.lines)


class QualityKind(StrEnum):
    """What a protection quality names (702.16a)."""

    COLOUR = "colour"
    CARD_TYPE = "card_type"
    CREATURE_TYPE = "creature_type"
    CONDITION = "condition"
    EVERYTHING = "everything"  # 702.16j


@dataclass(frozen=True, slots=True)
class _Condition:
    """A condition Forge's cards print as a protection quality."""

    #: Forge's keyword fragment for it.
    fragment: str
    holds: Callable[[Characteristics], bool]


#: Conditions by the description Forge prints after the filter.
CONDITIONS: dict[str, _Condition] = {
    "multicolored": _Condition(
        "Protection:Card.MultiColor:multicolored", lambda c: len(c.colours) >= 2),
    "monocolored": _Condition(
        "Protection:Card.MonoColor:monocolored", lambda c: len(c.colours) == 1),
    "mana value 3 or less": _Condition(
        "Protection:Card.cmcLE3:mana value 3 or less", lambda c: c.mana_value <= 3),
    "mana value 3 or greater": _Condition(
        "Protection:Card.cmcGE3:mana value 3 or greater", lambda c: c.mana_value >= 3),
    #: 702.16h: one protection per colour, so any coloured object.
    "each color": _Condition("Protection from each color", lambda c: bool(c.colours)),
}
#: Forge's subtypes, the creature types a keyword list's quality may name.
_SUBTYPES = frozenset(SUBTYPES)
_CONDITION_BY_FRAGMENT = {
    condition.fragment.lower(): name for name, condition in CONDITIONS.items()
}


@dataclass(frozen=True, slots=True)
class ProtectionQuality:
    """One "protection from [quality]" (702.16a), as Forge writes the keyword.

    ``value`` is a colour word (``red``), a card type or a creature type in a
    record's lower-case spelling (``artifact``, ``goblin``), a
    :data:`CONDITIONS` name, or empty for everything.
    """

    kind: QualityKind
    value: str = ""

    @classmethod
    def everything(cls) -> ProtectionQuality:
        return cls(QualityKind.EVERYTHING)

    @property
    def fragment(self) -> str:
        """The keyword line's fragment Forge writes for this protection:
        ``Protection from red``, ``Protection:Artifact``, ``Protection:Goblin``,
        ``Protection:Card.MultiColor:multicolored``, ``Protection from everything``."""
        if self.kind is QualityKind.EVERYTHING:
            return "Protection from everything"
        if self.kind is QualityKind.COLOUR:
            return f"Protection from {self.value}"
        if self.kind is QualityKind.CONDITION:
            return CONDITIONS[self.value].fragment
        return "Protection:" + "-".join(part[:1].upper() + part[1:]
                                        for part in self.value.split("-"))

    @property
    def keyword(self) -> str:
        """The keyword in the package's spelling, as a snapshot's keyword list
        carries it: ``protection_from_red``, ``protection_from_everything``,
        ``protection_from_goblin`` for Forge's ``Protection:Goblin``."""
        return normalize_keyword(self.fragment)

    def covers(self, obj: Characteristics) -> bool:
        """Whether ``obj`` has this quality."""
        if self.kind is QualityKind.EVERYTHING:
            return True
        if self.kind is QualityKind.COLOUR:
            return any(COLOUR_WORDS[c] == self.value for c in obj.colours)
        if self.kind is QualityKind.CARD_TYPE:
            return self.value in obj.types
        if self.kind is QualityKind.CREATURE_TYPE:
            return self.value in obj.subtypes or obj.changeling
        return CONDITIONS[self.value].holds(obj)

    @classmethod
    def parse(cls, fragment: str) -> ProtectionQuality | None:
        """The quality a protection keyword line's fragment names; None for any
        other line, and for a quality outside the five kinds (a player, a chosen
        colour, a filter no :data:`CONDITIONS` entry reads)."""
        text = fragment.strip()
        lowered = text.lower()
        if lowered in _CONDITION_BY_FRAGMENT:
            return cls(QualityKind.CONDITION, _CONDITION_BY_FRAGMENT[lowered])
        if lowered.startswith("protection from "):
            return cls.of_keyword(normalize_keyword(text))
        if not lowered.startswith("protection:"):
            return None
        value = lowered.split(":")[1].strip()
        kind = _filter_kind(value)
        return None if kind is None else cls(kind, value)

    @classmethod
    def of_keyword(cls, keyword: str) -> ProtectionQuality | None:
        """The quality a keyword list's ``protection_from_<x>`` names: a colour,
        a card type, a creature type (one of Forge's subtypes), a condition or
        everything; None for bare ``protection``, whose quality only its line's
        fragment states, and for a quality none of those reads."""
        if not keyword.startswith(PROTECTION_FROM):
            return None
        value = keyword.removeprefix(PROTECTION_FROM).replace("_", " ")
        if value == "everything":
            return cls.everything()
        if value in COLOUR_WORDS.values():
            return cls(QualityKind.COLOUR, value)
        if value in CONDITIONS:
            return cls(QualityKind.CONDITION, value)
        if value in CARD_TYPES:
            return cls(QualityKind.CARD_TYPE, value)
        if value in _SUBTYPES:
            return cls(QualityKind.CREATURE_TYPE, value)
        return None


def _filter_kind(value: str) -> QualityKind | None:
    """What a ``Protection:<filter>`` fragment's lower-case filter names: a
    colour, a card type, else a creature type; None for a filter naming no
    quality of the five kinds (:func:`_names_no_quality`)."""
    if _names_no_quality(value):
        return None
    if value in COLOUR_WORDS.values():
        return QualityKind.COLOUR
    if value in CARD_TYPES:
        return QualityKind.CARD_TYPE
    return QualityKind.CREATURE_TYPE


def _names_no_quality(value: str) -> bool:
    """An empty filter, a property filter (``card.chosencolor``), everything
    (``true``) or a player."""
    return not value or "." in value or value in ("true", "player")


def _covered(qualities: Iterable[ProtectionQuality], obj: Characteristics) -> bool:
    return any(quality.covers(obj) for quality in qualities)


def may_target(qualities: Iterable[ProtectionQuality], source: Characteristics) -> bool:
    """Whether a spell, or an ability from ``source``, may target what has these
    protections (702.16b, j)."""
    return not _covered(qualities, source)


def may_enchant(qualities: Iterable[ProtectionQuality], aura: Characteristics) -> bool:
    """Whether ``aura`` may enchant what has these protections: an Aura spell
    targeting it is countered on resolution for want of a legal target, and an
    Aura already attached is put into its owner's graveyard (702.16c, j)."""
    return not _covered(qualities, aura)


def damage_prevented(qualities: Iterable[ProtectionQuality], source: Characteristics) -> bool:
    """Whether damage ``source`` would deal to what has these protections is
    prevented (702.16e, j)."""
    return _covered(qualities, source)


# ── blocking ─────────────────────────────────────────────────────────────


class BlockingRestriction(StrEnum):
    """A rule that can keep one creature from blocking another (509.1b):
    an evasion of the attacker's, or what the blocker itself must be."""

    UNBLOCKABLE = "unblockable"
    FLYING = "flying"  # 702.9b, 702.17b
    HORSEMANSHIP = "horsemanship"  # 702.31b
    FEAR = "fear"  # 702.36b
    INTIMIDATE = "intimidate"  # 702.13b
    SHADOW = "shadow"  # 702.28b, both ways
    PROTECTION = "protection"  # 702.16f, j
    SKULK = "skulk"  # 702.118b: no creature with greater power


class Combatant(NamedTuple):
    """One creature as the blocking rules read it: its keywords, its
    characteristics (types, creature types, colours, mana value, changeling:
    every creature type, 702.73a) and its power where it has one."""

    keywords: frozenset[str]
    characteristics: Characteristics
    power: int | None = None

    @classmethod
    def of(cls, entity: EntityState, keywords: Iterable[str] | None = None) -> Combatant:
        """``entity`` with ``keywords`` (its granted ones by default; a caller
        that resolves printed keywords too passes them)."""
        found = frozenset(entity.granted_temporary.keywords if keywords is None else keywords)
        characteristics = Characteristics.of(entity)
        if "changeling" in found:
            characteristics = characteristics._replace(changeling=True)
        return cls(found, characteristics, entity.power if entity.pt is not None else None)

    @classmethod
    def made(cls, keywords: Iterable[str] = (), colours: Iterable[str] = (),
             types: Iterable[str] = (), subtypes: Iterable[str] = ()) -> Combatant:
        """A combatant from its parts, for a case no entity states, with mana
        value 0 and no power: a case needing either builds the tuple itself."""
        found = frozenset(keywords)
        made = Characteristics.made(types, subtypes, colours)
        return cls(found, made.with_changeling("changeling" in found))


def blocking_restrictions(
    attacker: Combatant, blocker: Combatant, *,
    attacker_protections: Iterable[ProtectionQuality] = (),
) -> frozenset[BlockingRestriction]:
    """Every :class:`BlockingRestriction` that keeps ``blocker`` from blocking
    ``attacker``, as far as these keywords decide; empty where it may block.
    :func:`may_block` is its emptiness. The keyword list states a protection
    from a colour, from each colour or from everything; ``attacker_protections``
    adds the ones only a keyword line's fragment states (a card type, a
    creature type, a condition)."""
    found = {restriction for restriction, keeps_out in _BLOCKING_RULES
             if keeps_out(attacker, blocker)}
    qualities = _keyword_qualities(attacker.keywords)
    if _covered((*qualities, *attacker_protections), blocker.characteristics):
        found.add(BlockingRestriction.PROTECTION)
    return frozenset(found)


def _keyword_qualities(keywords: Iterable[str]) -> list[ProtectionQuality]:
    """The protection qualities a keyword list states, in keyword order."""
    return [q for q in map(ProtectionQuality.of_keyword, sorted(keywords)) if q is not None]


def _unblockable(attacker: Combatant, blocker: Combatant) -> bool:
    return "unblockable" in attacker.keywords


def _flying_blocks(attacker: Combatant, blocker: Combatant) -> bool:
    return "flying" in attacker.keywords and not blocker.keywords & {"flying", "reach"}


def _horsemanship_blocks(attacker: Combatant, blocker: Combatant) -> bool:
    return "horsemanship" in attacker.keywords and "horsemanship" not in blocker.keywords


def _fear_blocks(attacker: Combatant, blocker: Combatant) -> bool:
    held = blocker.characteristics
    return "fear" in attacker.keywords and "artifact" not in held.types and (
        "B" not in held.colours)


def _intimidate_blocks(attacker: Combatant, blocker: Combatant) -> bool:
    held = blocker.characteristics
    return ("intimidate" in attacker.keywords and "artifact" not in held.types
            and not attacker.characteristics.colours & held.colours)


def _shadow_blocks(attacker: Combatant, blocker: Combatant) -> bool:
    return ("shadow" in attacker.keywords) != ("shadow" in blocker.keywords)


def _skulk_blocks(attacker: Combatant, blocker: Combatant) -> bool:
    return ("skulk" in attacker.keywords and attacker.power is not None
            and blocker.power is not None and blocker.power > attacker.power)


#: Each restriction but protection, with whether it keeps a blocker out, in
#: the order :func:`blocking_restrictions` tests them.
_BLOCKING_RULES: tuple[tuple[BlockingRestriction, Callable[[Combatant, Combatant], bool]],
                       ...] = (
    (BlockingRestriction.UNBLOCKABLE, _unblockable),
    (BlockingRestriction.FLYING, _flying_blocks),
    (BlockingRestriction.HORSEMANSHIP, _horsemanship_blocks),
    (BlockingRestriction.FEAR, _fear_blocks),
    (BlockingRestriction.INTIMIDATE, _intimidate_blocks),
    (BlockingRestriction.SHADOW, _shadow_blocks),
    (BlockingRestriction.SKULK, _skulk_blocks),
)


def may_block(attacker: Combatant, blocker: Combatant, *,
              attacker_protections: Iterable[ProtectionQuality] = ()) -> bool:
    """Whether one creature may block another, as far as these keywords decide:
    ``blocking_restrictions`` finds none.

    Rules 702.9 (flying, reach), 702.31 (horsemanship), 702.36 (fear), 702.13
    (intimidate), 702.28 (shadow), 702.118 (skulk), 702.16f and j (protection)
    and "can't be blocked"."""
    return not blocking_restrictions(attacker, blocker, attacker_protections=attacker_protections)


def landwalk_lands(selector: str, land: EntityState) -> bool | None:
    """Whether ``land`` is a land a landwalk's Forge selector names (702.14c):
    ``Swamp``, ``Desert``, ``Land.nonBasic``, ``Land.Legendary``,
    ``Swamp.Snow``; None for one read off the game rather than the card
    (``ChosenType``, ``YourBasic``)."""
    base, *properties = selector.split(".")
    if base in ("ChosenType", "YourBasic"):
        return None
    if "land" not in {t.lower() for t in land.types}:
        return False
    supertypes = {s.lower() for s in land.supertypes}
    if not _names_land_type(base.lower(), land):
        return False
    return all(_has_property(prop.lower(), supertypes) for prop in properties)


def _names_land_type(base: str, land: EntityState) -> bool:
    """Whether a selector's lower-case base, ``land`` or a land type, names ``land``."""
    return base == "land" or base in {s.lower() for s in land.subtypes}


def _has_property(wanted: str, supertypes: Collection[str]) -> bool:
    """Whether a land with these lower-case supertypes has a selector's
    lower-case property: ``nonbasic``, or a supertype (``snow``, ``legendary``)."""
    if wanted == "nonbasic":
        return "basic" not in supertypes
    return wanted in supertypes


# ── targeting (702.11, 702.16b, 702.18) ──────────────────────────────────


class TargetingRestriction(StrEnum):
    """A rule that can keep an object or player from being a legal target."""

    HEXPROOF = "hexproof"  # 702.11b: an opponent's spells and abilities
    HEXPROOF_FROM = "hexproof_from"  # 702.11d: an opponent's, of the quality
    SHROUD = "shroud"  # 702.18a: anyone's
    PROTECTION = "protection"  # 702.16b, j: of the quality


def targeting_restrictions(
    target_keywords: Iterable[str], *, source: Characteristics, opponents: bool,
) -> frozenset[TargetingRestriction]:
    """Every :class:`TargetingRestriction` that keeps what carries
    ``target_keywords`` from being targeted by a spell or ability of
    ``source``, ``opponents`` when the source's controller is an opponent of
    the target's; empty where it may be targeted as far as these keywords
    decide. The twin of :func:`blocking_restrictions`; :func:`may_target`
    reads protection alone."""
    keywords = set(target_keywords)
    found: set[TargetingRestriction] = set()
    if "shroud" in keywords:
        found.add(TargetingRestriction.SHROUD)
    if opponents and "hexproof" in keywords:
        found.add(TargetingRestriction.HEXPROOF)
    if opponents and _covered(_hexproof_qualities(keywords), source):
        found.add(TargetingRestriction.HEXPROOF_FROM)
    if not may_target(_keyword_qualities(keywords), source):
        found.add(TargetingRestriction.PROTECTION)
    return frozenset(found)


def _hexproof_qualities(keywords: Iterable[str]) -> list[ProtectionQuality]:
    """The qualities a keyword list's ``hexproof_from_<x>`` keywords name (702.11d)."""
    found = [ProtectionQuality.of_keyword(PROTECTION_FROM + k.removeprefix(HEXPROOF_FROM))
             for k in keywords if k.startswith(HEXPROOF_FROM)]
    return [q for q in found if q is not None]


# ── turn structure (500.1, 307.1, 702.61) ────────────────────────────────

#: A turn's phases and steps in order (500.1), spelled as a record's
#: ``global.phase`` spells them.
PHASE_ORDER: tuple[str, ...] = (
    "untap", "upkeep", "draw", "main1", "combat_begin", "combat_declare_attackers",
    "combat_declare_blockers", "combat_first_strike_damage", "combat_damage",
    "combat_end", "main2", "end_of_turn", "cleanup",
)
#: Forge's script name of each (``PhaseType.nameForScripts``).
_SCRIPT_PHASES: dict[str, str] = dict(zip((
    "untap", "upkeep", "draw", "main1", "begincombat", "declare attackers",
    "declare blockers", "first strike damage", "combat damage", "endcombat",
    "main2", "end of turn", "cleanup",
), PHASE_ORDER))
MAIN_PHASES: frozenset[str] = frozenset({"main1", "main2"})


def _phase(name: str) -> str | None:
    name = name.strip().lower()
    if name in _SCRIPT_PHASES:
        return _SCRIPT_PHASES[name]
    return name if name in PHASE_ORDER else None


def forge_phases(value: str | None) -> frozenset[str] | None:
    """The phases and steps a script's phase list names (``ActivationPhases$``,
    ``ConditionPhases$``, a static's ``Phases$``), as Forge's
    ``PhaseType.parseRange`` reads it: a comma-separated list of names, ``Main``
    for both main phases, ``A->B`` the steps from A to B inclusive, ``A->`` from
    A to cleanup. None for a value naming no phase Forge knows, or a range
    running backwards."""
    if not value or not value.strip():
        return None
    out: set[str] = set()
    for part in value.split(","):
        phases = _part_phases(part)
        if phases is None:
            return None
        out.update(phases)
    return frozenset(out)


def _part_phases(part: str) -> Iterable[str] | None:
    """The phases one comma-separated part of a phase list names; None for a
    part naming no phase Forge knows."""
    if "->" in part:
        return _range_phases(part)
    if part.strip() == "Main":
        return MAIN_PHASES
    phase = _phase(part)
    return None if phase is None else (phase,)


def _range_phases(part: str) -> tuple[str, ...] | None:
    """The steps ``A->B`` runs over, ``A->`` to cleanup; None for an end
    naming no phase, or a range running backwards."""
    start, _, end = part.partition("->")
    first, last = _phase(start), _phase(end) if end.strip() else "cleanup"
    if first is None or last is None:
        return None
    i, j = PHASE_ORDER.index(first), PHASE_ORDER.index(last)
    return None if i > j else PHASE_ORDER[i : j + 1]


def sorcery_timing(*, own_turn: bool, phase: str, stack_size: int) -> bool:
    """Whether a sorcery-speed play's timing allows it: during its controller's
    own main phase while the stack is empty (307.1, 117.1a)."""
    return own_turn and phase in MAIN_PHASES and stack_size == 0


def split_second_forbids(*, split_second_on_stack: bool, land_play: bool,
                         special_action: bool = False) -> bool:
    """Whether split second forbids a candidate: a spell with split second is
    on the stack and the candidate is a spell or a non-mana ability (702.61a;
    mana abilities are never candidates). A land play and the other special
    actions (turning a face-down permanent face up, suspending a card) stay
    legal (702.61b); the caller says which candidate is one."""
    return split_second_on_stack and not land_play and not special_action

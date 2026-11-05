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

from effects.domain.keyword_names import PROTECTION_FROM, normalize_keyword
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


def step_deaths(
    attacker: tuple[int, int, int], blocker: tuple[int, int, int],
    attacker_keywords: Iterable[str] = frozenset(),
    blocker_keywords: Iterable[str] = frozenset(), step: str = "regular",
) -> tuple[bool, bool]:
    """``(attacker dies, blocker dies)`` in one damage step of a one-on-one block.

    Each creature is ``(power, toughness, damage already marked)``. Rules 510 and
    702.7, 702.4, 702.2 and 702.12: a first or double striker deals damage in the
    first-strike step, a creature without first strike or with double strike in the
    regular step, both creatures striking in a step at once. Lethal damage is marked
    damage at least the toughness, or any damage from a deathtouch source;
    indestructible survives both.
    """
    power = {"a": attacker[0], "b": blocker[0]}
    toughness = {"a": attacker[1], "b": blocker[1]}
    marked = {"a": attacker[2], "b": blocker[2]}
    keywords = {"a": set(attacker_keywords), "b": set(blocker_keywords)}
    deathtouched = {"a": False, "b": False}
    other = {"a": "b", "b": "a"}

    for side in [s for s in ("a", "b") if strikes_in_step(keywords[s], step) and power[s] > 0]:
        marked[other[side]] += power[side]
        if "deathtouch" in keywords[side]:
            deathtouched[other[side]] = True
    dead = {side: "indestructible" not in keywords[side]
            and (marked[side] >= toughness[side] or deathtouched[side])
            for side in ("a", "b")}
    return dead["a"], dead["b"]


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

    @classmethod
    def of(cls, entity: EntityState) -> Characteristics:
        return cls.made(entity.types, entity.subtypes, entity.colors, entity.mana_value)

    @classmethod
    def made(cls, types: Iterable[str] = (), subtypes: Iterable[str] = (),
             colours: Iterable[str] = (), mana_value: int = 0) -> Characteristics:
        return cls(
            frozenset(t.lower() for t in types), frozenset(s.lower() for s in subtypes),
            frozenset(c for c in colours if c in COLOUR_WORDS), int(mana_value),
        )


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
        carries it: ``protection_from_red``, ``protection_from_everything``, or
        bare ``protection`` where Forge writes the quality after a colon."""
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
            return self.value in obj.subtypes
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
        if not value or "." in value or value in ("true", "player"):
            return None
        if value in COLOUR_WORDS.values():
            return cls(QualityKind.COLOUR, value)
        if value in CARD_TYPES:
            return cls(QualityKind.CARD_TYPE, value)
        return cls(QualityKind.CREATURE_TYPE, value)

    @classmethod
    def of_keyword(cls, keyword: str) -> ProtectionQuality | None:
        """The quality a keyword list's ``protection_from_<x>`` names; None for
        bare ``protection``, whose quality only its line's fragment states."""
        if not keyword.startswith(PROTECTION_FROM):
            return None
        value = keyword.removeprefix(PROTECTION_FROM).replace("_", " ")
        if value == "everything":
            return cls.everything()
        if value in COLOUR_WORDS.values():
            return cls(QualityKind.COLOUR, value)
        if value in CONDITIONS:
            return cls(QualityKind.CONDITION, value)
        return None


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


def may_block(attacker_keywords: Iterable[str], attacker_colours: Iterable[str],
              blocker_keywords: Iterable[str], blocker_colours: Iterable[str],
              blocker_types: Collection[str], *,
              attacker_protections: Iterable[ProtectionQuality] = (),
              blocker_subtypes: Collection[str] = (), blocker_mana_value: int = 0) -> bool:
    """Whether one creature may block another, as far as these keywords decide.

    Rules 702.9 (flying, reach), 702.31 (horsemanship), 702.36 (fear), 702.13
    (intimidate), 702.28 (shadow), 702.16f and j (protection) and "can't be
    blocked". The keyword list states a protection from a colour, from each
    colour or from everything; ``attacker_protections`` adds the ones only a
    keyword line's fragment states (a card type, a creature type, a condition).
    """
    a, b = set(attacker_keywords), set(blocker_keywords)
    attacker_colours, blocker_colours = set(attacker_colours), set(blocker_colours)
    if "unblockable" in a:
        return False
    if "flying" in a and not b & {"flying", "reach"}:
        return False
    if "horsemanship" in a and "horsemanship" not in b:
        return False
    if "fear" in a and "artifact" not in blocker_types and "B" not in blocker_colours:
        return False
    if ("intimidate" in a and "artifact" not in blocker_types
            and not attacker_colours & blocker_colours):
        return False
    if ("shadow" in a) != ("shadow" in b):
        return False
    qualities = [q for q in map(ProtectionQuality.of_keyword, sorted(a)) if q is not None]
    blocker = Characteristics.made(
        blocker_types, blocker_subtypes, blocker_colours, blocker_mana_value)
    return not _covered((*qualities, *attacker_protections), blocker)


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
        if "->" in part:
            start, _, end = part.partition("->")
            first, last = _phase(start), _phase(end) if end.strip() else "cleanup"
            if first is None or last is None:
                return None
            i, j = PHASE_ORDER.index(first), PHASE_ORDER.index(last)
            if i > j:
                return None
            out.update(PHASE_ORDER[i : j + 1])
        elif part.strip() == "Main":
            out.update(MAIN_PHASES)
        else:
            phase = _phase(part)
            if phase is None:
                return None
            out.add(phase)
    return frozenset(out)


def sorcery_timing(*, own_turn: bool, phase: str, stack_size: int) -> bool:
    """Whether a sorcery-speed play's timing allows it: during its controller's
    own main phase while the stack is empty (307.1, 117.1a)."""
    return own_turn and phase in MAIN_PHASES and stack_size == 0


def split_second_forbids(*, split_second_on_stack: bool, land_play: bool) -> bool:
    """Whether split second forbids a candidate: a spell with split second is
    on the stack and the candidate is a spell or a non-mana ability, which
    every candidate but a land play is (702.61a; mana abilities are never
    candidates)."""
    return split_second_on_stack and not land_play

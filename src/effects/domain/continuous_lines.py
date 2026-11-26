"""What one continuous contribution changes on each characteristic line.

A continuous record's board is the board without its static (collection
internals § Continuous records): the entity's keywords, types and colours there
are what it held before the static applied. A line the static sets outright (a
lone ``=``) or a class of it the static removes whole (``-all-card-types``,
``-all-creature-types`` and the rest of Forge's ``RemoveType``) displaces what
the entity held there. That is what it lost, less what the static gives back,
and what the static gives that the entity already held is no gain: Witness
Protection on a creature gains it no creature type, Sinister Strength's ``= B``
on a black creature no black.

A card type lost takes the subtypes that go with it (rule 205.3d: an object
has no subtype of a type it lacks): a creature made a Treasure loses its
creature types, a reconfigured Equipment Spider its Spider.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass

from effects.domain.card_features import keywords_of
from effects.domain.effect_head_layout import HeadField
from effects.domain.forge_vocabularies import SUBTYPE_SECTIONS, SUBTYPES
from effects.domain.input_vocabularies import CARD_TYPES, SUPERTYPES, vocabulary_slot
from effects.domain.keyword_names import changed_keyword
from effects.domain.output_vocabularies import KEYWORDS_CHANGED, SUBTYPES_CHANGED
from effects.domain.records import ChannelChange, Contribution
from effects.domain.state_snapshot import COLORS, EntityState

_LAND_SECTIONS = frozenset({"BasicTypes", "LandTypes"})
_CREATURE_SECTIONS = frozenset({"CreatureTypes"})

#: The ``TypeLists.txt`` sections whose subtypes each card type carries; a
#: kindred card carries creature types (rule 308.3).
_SECTIONS_OF_TYPE: dict[str, frozenset[str]] = {
    "land": _LAND_SECTIONS,
    "creature": _CREATURE_SECTIONS,
    "kindred": _CREATURE_SECTIONS,
    "artifact": frozenset({"ArtifactTypes"}),
    "enchantment": frozenset({"EnchantmentTypes"}),
    "instant": frozenset({"SpellTypes"}),
    "sorcery": frozenset({"SpellTypes"}),
    "planeswalker": frozenset({"WalkerTypes"}),
    "battle": frozenset({"BattleTypes"}),
}

#: The subtype sections each whole-class removal clears (``LayerTokens.everyTypeOf``).
_WIPED_SECTIONS: dict[str, frozenset[str]] = {
    "all-land-types": _LAND_SECTIONS,
    "all-creature-types": _CREATURE_SECTIONS,
    "all-artifact-types": frozenset({"ArtifactTypes"}),
    "all-enchantment-types": frozenset({"EnchantmentTypes"}),
}
_ALL_CARD_TYPES = "all-card-types"
_ALL_SUPERTYPES = "all-super-types"
_ALL_SUBTYPES = "all-sub-types"
#: The changeling grant: every creature type, which no subtype column can state.
EVERY_CREATURE_TYPE = "all-creature-types"
_WHOLE_CLASS = "all-"

_SECTION_NAMES = tuple(name for name, _ in SUBTYPE_SECTIONS)
_SECTION_ENDS = tuple(end for _, end in SUBTYPE_SECTIONS)


def subtype_section(subtype: str) -> str | None:
    """The ``TypeLists.txt`` section ``subtype`` is listed in; None for one Forge does not list."""
    row = vocabulary_slot(subtype, SUBTYPES)
    return None if row is None else _SECTION_NAMES[bisect_right(_SECTION_ENDS, row)]


@dataclass(frozen=True, slots=True)
class LineChange:
    """One line's gained and lost values, with the two fields and the closed
    list they are flagged over."""

    gained_field: str
    lost_field: str
    vocabulary: tuple[str, ...]
    gained: frozenset[str] = frozenset()
    lost: frozenset[str] = frozenset()
    #: The static displaces what the line held, so every value it lost is
    #: named in ``lost``, and one not named is not lost.
    replaces: bool = False
    #: A replacing line on an entity the board lacks: what it displaced
    #: cannot be named.
    lost_unknown: bool = False
    gained_unknown: bool = False


@dataclass(frozen=True, slots=True)
class _Line:
    """One line as the contribution and the board read it."""

    gained: frozenset[str]
    #: The values the contribution removes by name; a whole class is no name.
    removed: frozenset[str]
    #: The entity's own values on the board; None where the board lacks it.
    held: frozenset[str] | None
    replaces: bool

    @classmethod
    def of(cls, change: ChannelChange, held, *, replaces: bool) -> _Line:
        removed = frozenset(value for value in change.lost if not value.startswith(_WHOLE_CLASS))
        return cls(frozenset(change.gained), removed, held, replaces)

    def displaced(self, *, outright: bool) -> frozenset[str]:
        return self.held if outright and self.held is not None else frozenset()

    def settle(self, fields: tuple[str, str, tuple[str, ...]], displaced: frozenset[str],
               *, gained_unknown: bool = False) -> LineChange:
        """The line's change: what it took less what it gives back, and what it
        gives that the entity did not already hold where it sets the line."""
        held = self.held
        gained = self.gained - held if self.replaces and held is not None else self.gained
        return LineChange(
            *fields, gained=gained, lost=self.removed | (displaced - self.gained),
            replaces=self.replaces, lost_unknown=self.replaces and held is None,
            gained_unknown=gained_unknown,
        )


def _held(entity: EntityState | None, values) -> frozenset[str] | None:
    return None if entity is None else frozenset(values(entity))


def _keyword_line(contribution: Contribution, entity: EntityState | None) -> LineChange:
    """A lone ``=`` is "loses all abilities" (``RemoveAllAbilities$``): every
    keyword the entity held is lost, but those the static grants it again."""
    change = contribution.keyword_changes
    spelled = ChannelChange(
        gained=tuple(map(changed_keyword, change.gained)),
        lost=tuple(map(changed_keyword, change.lost)),
        sets_outright=change.sets_outright,
    )
    held = _held(entity, lambda e: map(changed_keyword, keywords_of(e)))
    line = _Line.of(spelled, held, replaces=change.sets_outright)
    fields = (HeadField.KEYWORDS_GAINED, HeadField.KEYWORDS_LOST, KEYWORDS_CHANGED)
    return line.settle(fields, line.displaced(outright=change.sets_outright))


def _colour_line(contribution: Contribution, entity: EntityState | None) -> LineChange:
    change = contribution.color_changes
    line = _Line.of(change, _held(entity, lambda e: e.colors), replaces=change.sets_outright)
    fields = (HeadField.COLORS_GAINED, HeadField.COLORS_LOST, COLORS)
    return line.settle(fields, line.displaced(outright=change.sets_outright))


def _class_line(change: ChannelChange, held, whole: str) -> tuple[_Line, frozenset[str]]:
    """A line a ``=`` or its own whole-class token displaces outright."""
    outright = change.sets_outright or whole in change.lost
    line = _Line.of(change, held, replaces=outright)
    return line, line.displaced(outright=outright)


def _type_lines(contribution: Contribution, entity: EntityState | None) -> tuple[LineChange, ...]:
    """The card types, the subtypes and the supertypes: one flat token list,
    each line reading the tokens it lists."""
    change = contribution.type_changes
    types, displaced = _class_line(change, _held(entity, lambda e: e.types), _ALL_CARD_TYPES)
    card_types = types.settle((HeadField.TYPES_GAINED, HeadField.TYPES_LOST, CARD_TYPES), displaced)
    supertypes, displaced = _class_line(
        change, _held(entity, lambda e: e.supertypes), _ALL_SUPERTYPES)
    return (
        card_types,
        _subtype_line(change, entity, _types_after(types, card_types)),
        supertypes.settle(
            (HeadField.SUPERTYPES_GAINED, HeadField.SUPERTYPES_LOST, SUPERTYPES), displaced),
    )


def _types_after(types: _Line, change: LineChange) -> frozenset[str] | None:
    """The card types the entity has with the static, as far as the record
    says; None where it loses none, so no subtype goes with one."""
    if not change.lost & frozenset(CARD_TYPES):
        return None
    return ((types.held or frozenset()) - change.lost) | (change.gained & frozenset(CARD_TYPES))


def _subtype_line(change: ChannelChange, entity: EntityState | None,
                  types_after: frozenset[str] | None) -> LineChange:
    wiped = frozenset().union(*(_WIPED_SECTIONS.get(token, ()) for token in change.lost))
    outright = change.sets_outright or _ALL_SUBTYPES in change.lost
    line = _Line.of(change, _held(entity, lambda e: e.subtypes), replaces=outright or bool(wiped))
    displaced = line.displaced(outright=outright) | frozenset(
        subtype for subtype in line.held or ()
        if subtype_section(subtype) in wiped or _orphaned(subtype, types_after)
    )
    fields = (HeadField.SUBTYPES_GAINED, HeadField.SUBTYPES_LOST, SUBTYPES_CHANGED)
    return line.settle(fields, displaced, gained_unknown=EVERY_CREATURE_TYPE in change.gained)


def _orphaned(subtype: str, types_after: frozenset[str] | None) -> bool:
    """Whether a card type was lost and none left carries ``subtype``'s
    section; a subtype of no card type's section (a plane, a dungeon) never is."""
    if types_after is None:
        return False
    section = subtype_section(subtype)
    owners = [card_type for card_type, sections in _SECTIONS_OF_TYPE.items() if section in sections]
    return bool(owners) and not types_after.intersection(owners)


def line_changes(contribution: Contribution, entity: EntityState | None) -> tuple[LineChange, ...]:
    """Every line ``contribution`` changes on ``entity``, its board state
    without the static (None where the board lacks it): keywords, card types,
    subtypes, supertypes, colours."""
    return (
        _keyword_line(contribution, entity),
        *_type_lines(contribution, entity),
        _colour_line(contribution, entity),
    )


def abilities_changed(contribution: Contribution) -> tuple[int, bool]:
    """How many abilities a contribution grants, and whether it takes any away:
    each ``abilities`` token is one line's key, a ``-`` before one taken."""
    tokens = contribution.abilities or ()
    taken = sum(1 for token in tokens if token.startswith("-"))
    return len(tokens) - taken, taken > 0

"""The pre-event game state every effect record carries.

Stored as data and tensorized at training time by
:mod:`effects.domain.effect_head_input`, so a representation change is a code
change rather than a re-collection. Controllers are absolute player ids — mine
and opponent derive from the record's ``actor_player`` — and an absent
inclusion tier means uncollected, not empty.

One :class:`EntityState` reads both entity shapes. A gen-2 card slot lists
its keys in ``printed`` and ``granted_attached`` and its power and toughness
in ``pt``; a gen-3 one lists ``lines`` (:class:`CardLine`), ``base_pt`` and
``pt_boosts``. ``EntityState.__post_init__`` and
:attr:`EntityState.ability_keys` hold the one mapping from a gen-3 entity onto
what the gen-2 readers read, so every reader works on both shapes.

Every other gen-3 member (a card's facts, a player's, the global block's
command-zone objects, ``refs.random``, ``links``) is a field an earlier record
reads at its default. Two markers say which shape was on the wire: an
entity's ``lines``, and the snapshot's ``links`` for everything else.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import IntEnum, StrEnum

from effects.domain.event_schema import Event
from effects.domain.keyword_names import normalize_keyword
from effects.domain.provenance import ProvenanceKey


class InclusionTier(IntEnum):
    """What a snapshot was allowed to include, in application order.

    Ordered: a corpus never holds tier 3 without tier 2.
    """

    #: Every entity-valued ref, in whatever zone it sits, with that zone recorded.
    REFERENCED = 1
    #: Global state, battlefield entities, and command-zone effect cards.
    CORE = 2
    #: Stack objects that no ref already pulled in.
    UNREFERENCED_STACK = 3
    #: Hand and graveyard contents that no ref already pulled in.
    UNREFERENCED_HAND_GRAVEYARD = 4


COLORS: tuple[str, ...] = ("W", "U", "B", "R", "G", "C")


class Zone(StrEnum):
    """The zones an entity's ``zone`` names, in the collector's spelling.

    An entity keeps its zone as the string the record carries, so a zone this
    enum predates still reads; the enum is what code compares it against.
    """

    BATTLEFIELD = "battlefield"
    STACK = "stack"
    HAND = "hand"
    GRAVEYARD = "graveyard"
    LIBRARY = "library"
    EXILE = "exile"
    COMMAND = "command"


class CombatSubstep(StrEnum):
    """Which combat-damage step a ``combat`` record's snapshot was taken in.

    The snapshot keeps the string the record carries; the enum is what code
    compares it against.
    """

    FIRST_STRIKE = "first_strike"
    REGULAR = "regular"


class DayNight(StrEnum):
    """A gen-3 snapshot's ``day_night``, in the collector's spelling; the
    snapshot keeps the string, None when the game is neither."""

    DAY = "day"
    NIGHT = "night"


class EffectDuration(StrEnum):
    """How long a command-zone object lasts, ``Duration$`` bucketed as the
    collector buckets it; an object keeps the string its record carries."""

    END_OF_TURN = "end_of_turn"
    UNTIL_LEAVES = "until_leaves"
    PERMANENT = "permanent"


@dataclass(frozen=True, slots=True)
class CommandZoneObject:
    """One gen-3 floating effect or pending delayed trigger.

    ``id`` is an effect card's ``E<card id>``, a layer-table timestamp's
    ``F<timestamp>`` or a delayed trigger's ``D<trigger id>``; links name it
    by that id.
    """

    id: str
    #: The key of the ability that made it; empty where none did (the
    #: monarch, the initiative, the city's blessing).
    source_lines: tuple[ProvenanceKey, ...] = ()
    #: The cards its statics apply to (a delayed trigger lists none).
    affected: tuple[str, ...] = ()
    #: Its remembered cards and players.
    remembered: tuple[str, ...] = ()
    duration: str = EffectDuration.PERMANENT
    #: What a counting-down prevention shield has left, else None.
    shield_left: int | None = None


@dataclass(frozen=True, slots=True)
class Dungeon:
    """A dungeon in a command zone: its owner, its entity ref and its room."""

    player: str
    dungeon: str
    room: str | None = None


@dataclass(frozen=True, slots=True)
class GlobalState:
    """Turn structure and the parts of the board that belong to nobody.

    The last four fields are gen-3's (FR-024); a gen-1 or gen-2 snapshot reads
    them as their defaults, and whether they were written is the snapshot's
    shape (:attr:`StateSnapshot.links`), not theirs.
    """

    turn: int
    phase: str
    active: str
    priority: str | None
    stack_size: int
    combat_substep: str | None = None
    #: Command-zone emblems, as the provenance key of the line that made each.
    emblems: tuple[ProvenanceKey, ...] = ()
    #: ``day`` or ``night`` (:class:`DayNight`); None while the game is neither.
    day_night: str | None = None
    #: Effect cards in a command zone (no emblem, no dungeon), then the
    #: timestamped layer rows a noted clause drew, ascending.
    floating_effects: tuple[CommandZoneObject, ...] = ()
    #: Pending delayed triggers, each once; empty on an unpatched Forge.
    delayed_triggers: tuple[CommandZoneObject, ...] = ()
    dungeons: tuple[Dungeon, ...] = ()


@dataclass(frozen=True, slots=True)
class TurnCounts:
    """A gen-3 player's counts for the turn. ``permanents_to_graveyard``
    counts cards put into this player's graveyard from the battlefield this
    turn; ``dungeons_completed`` and ``descended`` are whole-game counts, as
    Forge keeps them."""

    cards_drawn: int = 0
    life_gained: int = 0
    life_lost: int = 0
    damage_taken: int = 0
    permanents_to_graveyard: int = 0
    spells_cast_last_turn: int = 0
    dungeons_completed: int = 0
    descended: int = 0


@dataclass(frozen=True, slots=True)
class PlayerState:
    """One player's resources.

    ``floating_mana`` and ``untapped_production`` are keyed by the symbols in
    :data:`COLORS`; ``this_turn`` holds the per-turn counters (creatures died,
    spells cast, lands played) that trigger conditions read. The fields from
    ``speed`` on are gen-3's, defaults on an earlier record.
    """

    id: str
    life: int
    hand: int
    library: int
    graveyard: int
    poison: int = 0
    energy: int = 0
    this_turn: dict[str, int] = field(default_factory=dict)
    floating_mana: dict[str, int] = field(default_factory=dict)
    untapped_production: dict[str, int] = field(default_factory=dict)
    speed: int = 0
    city_blessing: bool = False
    rad: int = 0
    monarch: bool = False
    initiative: bool = False
    #: None on a gen-1 or gen-2 player, which carries no counts.
    turn_counts: TurnCounts | None = None


@dataclass(frozen=True, slots=True)
class PowerToughness:
    """Power and toughness decomposed into the three sources that set it.

    Kept separate because a boost and a counter are different effects with
    the same total. A gen-2 entity writes the three as ``pt`` (``base`` the
    printed base); a gen-3 one as ``base_pt`` (the base a layer-7b setting
    effect leaves, Ensoul Artifact's 5/5) and ``pt_boosts`` ``{temporary,
    counters}``, read into ``base``, ``boosts`` and ``counters``. On both the
    three sum to the net power and toughness.
    """

    base: tuple[int, int]
    boosts: tuple[int, int] = (0, 0)
    counters: tuple[int, int] = (0, 0)

    @property
    def total(self) -> tuple[int, int]:
        return (
            self.base[0] + self.boosts[0] + self.counters[0],
            self.base[1] + self.boosts[1] + self.counters[1],
        )


@dataclass(frozen=True, slots=True)
class CombatStatus:
    """An entity's place in the current combat, if any."""

    attacking: str | None = None
    blocking: tuple[str, ...] = ()
    blocked_by: tuple[str, ...] = ()
    #: True once this attacker is blocked, even if every blocker later leaves.
    became_blocked: bool = False


@dataclass(frozen=True, slots=True)
class GrantedTemporary:
    """What the timestamped change tables report for one entity.

    Provenance keys where a grant resolves to a printed line, bare keywords
    where it does not ("gains flying until end of turn"). Separate from the
    ability tokens because temporary grants ride the overlay rather than the
    tokens, and gate 2 strips a keyword from whichever carries it.

    ``keywords`` is every keyword the card has now, printed ones included, on
    gen-2 and gen-3 alike: the collector writes the card's whole keyword list.
    ``abilities`` is the spell and activated abilities a layer-6 change
    granted.
    """

    keywords: tuple[str, ...] = ()
    abilities: tuple[ProvenanceKey, ...] = ()


def normalize_counter_type(raw: str) -> str:
    """Forge's spelling of a counter type, in the one this package reads.

    Forge prints ``"+1/+1"`` where :data:`COUNTER_TYPES` spells ``P1P1``;
    without this the commonest counters fall through to
    ``counters_delta_other``. Forge's naming rule read backwards, so rarer
    shapes (``+1/+0``) need no table.
    """
    return (
        raw.strip()
        .replace("+", "P")
        .replace("-", "M")
        .replace("/", "")
        .upper()
    )


class LineKind(StrEnum):
    """What a gen-3 card line is, in the collector's spelling.

    A line keeps its kind as the string the record carries, as an entity keeps
    its zone; the enum is what code compares it against.
    """

    SPELL = "spell"
    ACTIVATED = "activated"
    TRIGGER = "trigger"
    STATIC = "static"
    REPLACEMENT = "replacement"
    KEYWORD = "keyword"


@dataclass(frozen=True, slots=True)
class CardLine:
    """One line of a gen-3 card slot: a trait the card carries now.

    Read off Forge's live trait lists, so a creature that lost all its
    abilities lists none and an anthem's grant is listed on the creature it
    reaches. A face not up (a face-down card's real face, a double-faced
    card's other face) lists its printed lines with ``active = False``.
    """

    #: Zero or one provenance key: none where the line resolves to no printed
    #: line, such as a keyword a resolved effect granted (read by its
    #: ``fragment``). A granted keyword's key names the static that granted it.
    keys: tuple[ProvenanceKey, ...]
    kind: str
    #: The line is not printed on this card: a static's, an attachment's, an
    #: effect's, an emblem's or a splice's.
    granted: bool
    #: Forge's answer to whether a static's condition holds; False on every
    #: line of a face not up; None on any other line.
    active: bool | None = None
    #: Keyword lines of a card cast with them: how many times the spell paid
    #: the line's cost (1 for an alternative cost or kicker, the count for
    #: multikicker).
    cast_with: int | None = None
    #: The ``Choices$`` positions a modal ability on the stack chose, in chain order.
    chosen: tuple[int, ...] | None = None
    #: A keyword line's own fragment as Forge writes it (``Ward:2``,
    #: ``Protection:Goblin``).
    fragment: str | None = None
    #: A static of the face up that Forge applies as a continuous effect: the
    #: permanents it applies to. None on every other line, and on every line
    #: of a ``continuous`` record's board, whose answer it would be.
    affected: tuple[str, ...] | None = None

    @property
    def in_force(self) -> bool:
        """Whether the line applies now: neither an unmet static nor a line
        of a face not up."""
        return self.active is not False

    @property
    def keyword(self) -> str | None:
        """A keyword line's keyword in the package's spelling (``ward``,
        ``protection``, ``first_strike``); None on any other line."""
        if self.kind != LineKind.KEYWORD or not self.fragment:
            return None
        return normalize_keyword(self.fragment)


def printed_keys_of(lines: tuple[CardLine, ...]) -> tuple[ProvenanceKey, ...]:
    """A gen-3 entity's ``printed``: the keys of the lines printed on it.

    Every line not ``granted``, in force or not, each key once in line order.
    The keys name the card's own script, which is what ``printed``'s readers
    look up: the host of a continuous record, a card's script file.
    """
    return _unique_keys(line for line in lines if not line.granted)


def _unique_keys(lines) -> tuple[ProvenanceKey, ...]:
    return tuple(dict.fromkeys(key for line in lines for key in line.keys))


class FaceState(StrEnum):
    """Which face of a gen-3 card is up, in the collector's spelling; a split
    card, a prepared spell and a specialized card read ``front``."""

    FRONT = "front"
    BACK = "back"
    FLIPPED = "flipped"
    MELDED = "melded"
    FACE_DOWN = "face_down"


class EntityRole(StrEnum):
    """Why a gen-3 entity beyond the four inclusion tiers is in the snapshot."""

    #: A library's top card, or a library card an event names.
    LIBRARY_CARD = "library_card"
    #: A card exiled with a permanent of the tiers.
    EXILED_WITH = "exiled_with"
    #: A card merged into a mutated permanent.
    MUTATE_COMPONENT = "mutate_component"
    #: The pending event's subject as it last existed, under id ``L<card id>``.
    LAST_KNOWN = "last_known"


@dataclass(frozen=True, slots=True)
class RealFace:
    """The front a face-down card would turn up as."""

    name: str
    types: tuple[str, ...] = ()
    subtypes: tuple[str, ...] = ()
    colors: tuple[str, ...] = ()
    mana_value: int = 0
    #: Printed power and toughness; None on a non-creature.
    base_pt: tuple[int, int] | None = None


@dataclass(frozen=True, slots=True)
class CardChoices:
    """What a gen-3 card remembers choosing, each member None until set:
    Cavern of Souls's type, Gods Willing's colour, a named card. A modal
    spell's modes are its line's :attr:`CardLine.chosen`, not here."""

    colors: tuple[str, ...] | None = None
    type: str | None = None
    type2: str | None = None
    number: int | None = None
    #: A player ref.
    player: str | None = None
    #: Card refs.
    cards: tuple[str, ...] | None = None
    names: tuple[str, ...] | None = None
    direction: str | None = None
    mode: str | None = None
    even_odd: str | None = None
    sector: str | None = None
    stored_rolls: tuple[int, ...] | None = None


@dataclass(frozen=True, slots=True)
class CostsPaid:
    """What paid for the spell a gen-3 card is or was cast as."""

    #: Mana spent per symbol of :data:`COLORS`.
    mana: dict[str, int] = field(default_factory=dict)
    #: The zone it was cast from, None when Forge does not say.
    cast_from: str | None = None
    #: Some mana spent carried a rider: can't be countered, adds a counter or
    #: a keyword, triggers when spent.
    mana_rider: bool = False


@dataclass(frozen=True, slots=True)
class CardHistory:
    """What a gen-3 card did this turn."""

    entered_this_turn: bool = False
    attacked_this_turn: bool = False
    blocked_this_turn: bool = False
    loyalty_activated: bool = False


@dataclass(frozen=True, slots=True)
class StackExtras:
    """Fields an entity carries only while it is a stack object."""

    targets: tuple[str, ...] = ()
    #: Announced amount per target, e.g. divided damage.
    per_target_amounts: dict[str, int] = field(default_factory=dict)
    #: "Up to N" counts the caster announced.
    up_to_counts: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EntityState:
    """One card, token, stack object, or command-zone effect in the snapshot.

    Characteristics are computed (post-layer), never printed; a
    ``continuous`` record's snapshot omits the acting static's own
    contributions.
    """

    id: str
    name: str
    zone: str
    controller: str
    face: int = 0
    copy_source: str | None = None
    token_script_id: str | None = None

    types: tuple[str, ...] = ()
    subtypes: tuple[str, ...] = ()
    supertypes: tuple[str, ...] = ()
    colors: tuple[str, ...] = ()
    mana_value: int = 0
    pt: PowerToughness | None = None

    tapped: bool = False
    sick: bool = False
    damage: int = 0
    counters: dict[str, int] = field(default_factory=dict)
    combat: CombatStatus | None = None
    attached_to: str | None = None
    face_down: bool = False

    #: Gen-2 only: the spell abilities of the cards attached to this one. A
    #: gen-3 entity lists its grants among its ``lines`` and reads ``()``.
    granted_attached: tuple[ProvenanceKey, ...] = ()
    granted_temporary: GrantedTemporary = field(default_factory=GrantedTemporary)

    #: The entity's own printed lines, as keys into its sidecar. On a gen-3
    #: entity derived from ``lines`` (:func:`printed_keys_of`), replacing
    #: whatever was passed, so an edit of the lines carries through.
    printed: tuple[ProvenanceKey, ...] = ()

    stack_extras: StackExtras | None = None

    #: Gen-3: every line the card carries now (FR-021). None on a gen-1 or
    #: gen-2 entity, which wrote ``printed`` instead; this is what tells the
    #: two shapes apart, in memory and on the wire.
    lines: tuple[CardLine, ...] | None = None

    # The card facts of a gen-3 entity (FR-024): their defaults on gen-1 and
    # gen-2, which wrote none, and where a gen-3 card has none to state.

    #: A player ref; control changes leave it alone.
    owner: str | None = None
    #: :class:`FaceState`.
    face_state: str | None = None
    #: Symbol -> count over the non-zero symbols (``generic``, ``W``, ``W/U``,
    #: ``G/P``, ``X``, ...); ``{}`` for {0}; None for no mana cost.
    mana_cost: dict[str, int] | None = None
    #: The face up's printed value; None when it prints none or prints X.
    printed_loyalty: int | None = None
    printed_defense: int | None = None
    #: Face-down cards only.
    real_face: RealFace | None = None
    #: Sorted designations (``goaded``, ``monstrous``, ``phased_out``, ...).
    designations: tuple[str, ...] = ()
    #: The X the spell this card is or was cast as paid; None when not cast
    #: or not with X.
    x_paid: int | None = None
    #: Gen-3 always writes one, empty when nothing was chosen.
    chosen: CardChoices | None = None
    #: None when the card was not cast.
    costs_paid: CostsPaid | None = None
    history: CardHistory | None = None
    #: Rank in its zone, 0 first: battlefield by timestamp, stack and library
    #: from the top, graveyard, hand and exile in Forge's zone order; None in
    #: the command zone or outside a zone.
    order_rank: int | None = None
    #: :class:`EntityRole` on an entity beyond the four tiers, else None.
    role: str | None = None

    def __post_init__(self) -> None:
        if self.lines is not None:
            object.__setattr__(self, "printed", printed_keys_of(self.lines))
            object.__setattr__(self, "granted_attached", ())

    @property
    def ability_keys(self) -> tuple[ProvenanceKey, ...]:
        """The entity's ability tokens.

        Gen-2: the printed and attachment-granted lines. Gen-3: the key of
        every line in force (:attr:`CardLine.in_force`), printed or granted,
        each once in line order, so an unmet static and a face not up give no
        token.

        Temporary grants are not among them on either: they ride the overlay
        (FR-073), so a gen-3 key ``granted_temporary.abilities`` also lists is
        left to the overlay.
        """
        if self.lines is None:
            return (*self.printed, *self.granted_attached)
        overlay = set(self.granted_temporary.abilities)
        in_force = _unique_keys(line for line in self.lines if line.in_force)
        return tuple(key for key in in_force if key not in overlay)

    @property
    def line_keys(self) -> tuple[ProvenanceKey, ...]:
        """Every key the entity's lines name, in force or not: gen-2's printed
        and attached keys, or every gen-3 line's."""
        if self.lines is None:
            return (*self.printed, *self.granted_attached)
        return _unique_keys(self.lines)

    @property
    def keyword_lines(self) -> tuple[CardLine, ...]:
        """A gen-3 entity's keyword lines; none on gen-2, whose keywords are
        ``granted_temporary.keywords`` alone."""
        return tuple(line for line in self.lines or () if line.kind == LineKind.KEYWORD)

    @property
    def power(self) -> int:
        """Total power; zero for an entity that has none."""
        return self.pt.total[0] if self.pt else 0

    @property
    def toughness(self) -> int:
        """Total toughness; zero for an entity that has none."""
        return self.pt.total[1] if self.pt else 0

    @property
    def remaining_toughness(self) -> int:
        """Total toughness less the damage already marked on it."""
        return self.toughness - self.damage

    @property
    def in_tiers(self) -> bool:
        """Whether the entity is one of the four inclusion tiers' rather than
        a gen-3 extra (:class:`EntityRole`)."""
        return self.role is None

    @property
    def is_creature(self) -> bool:
        return "creature" in self.types

    @property
    def on_battlefield(self) -> bool:
        return self.zone == Zone.BATTLEFIELD

    def with_total_toughness(self, toughness: int) -> EntityState:
        """This entity at total toughness ``toughness`` with no marked damage.

        The base toughness absorbs the change, so the boosts and the counters
        keep their own channels; damage is cleared so the toughness is what an
        ability faces.
        """
        pt = self.pt or PowerToughness(base=(0, 0))
        base = toughness - pt.boosts[1] - pt.counters[1]
        return replace(self, pt=replace(pt, base=(pt.base[0], base)), damage=0)


@dataclass(frozen=True, slots=True)
class Refs:
    """What the acting ability points at, and what was chosen for it."""

    targets: tuple[str, ...] = ()
    source: str | None = None
    modes: tuple[str, ...] = ()
    x: int | None = None
    #: Resolution-time engine choices, keyed by the choice's name in the script.
    choices: dict = field(default_factory=dict)
    #: Gen-3, a resolution's effect half only: the dice it rolled and the
    #: coins it flipped. None on every other refs block.
    random: RandomResults | None = None


@dataclass(frozen=True, slots=True)
class RandomResults:
    """The die results and coin flips of one resolution."""

    dice: tuple[int, ...] = ()
    coin_flips_won: int = 0
    coin_flips_lost: int = 0


@dataclass(frozen=True, slots=True)
class LineRef:
    """A link end that is one line of a card: the ability that made the link."""

    entity: str
    key: ProvenanceKey


#: A link end: an entity or player ref (``E``, ``P``, ``L``, ``F``, ``D``) or a line.
LinkEnd = str | LineRef


@dataclass(frozen=True, slots=True)
class Link:
    """One gen-3 relation between two snapshot objects, written once in its
    family's forward direction (``relation_types.FAMILIES``); the reverse is
    the reader's to take. An end the snapshot carries no entity for is still
    named."""

    from_: LinkEnd
    to: LinkEnd
    #: A family name of ``relation_types.FAMILIES``.
    type: str
    #: ``targets`` only: the target's position across the stack object's
    #: chain, the root's targets first.
    index: int | None = None

    def line_keys(self) -> tuple[ProvenanceKey, ...]:
        """The keys of the ends that are lines."""
        return tuple(end.key for end in (self.from_, self.to) if isinstance(end, LineRef))


@dataclass(frozen=True, slots=True)
class StateSnapshot:
    """The board as it stood immediately before the record's event.

    ``global_`` is the JSON field ``global``.
    """

    global_: GlobalState
    players: tuple[PlayerState, ...]
    entities: tuple[EntityState, ...]
    refs: Refs = field(default_factory=Refs)
    #: ``rewrite`` and ``trigger`` records only: the incoming event being handled.
    pending_event: Event | None = None
    #: Which tiers the collector wrote; check before reading an empty list as
    #: an empty zone. Never inferred, so a snapshot naming none claims none.
    tiers: frozenset[InclusionTier] = frozenset()
    #: Gen-3: the links between the snapshot's objects (FR-026), possibly
    #: empty. None on a gen-1 or gen-2 snapshot, which is what tells the
    #: shapes apart: a gen-3 snapshot also writes the gen-3 members of its
    #: global block, its players and its refs, an earlier one none of them.
    links: tuple[Link, ...] | None = None

    def entity(self, entity_id: str) -> EntityState | None:
        for candidate in self.entities:
            if candidate.id == entity_id:
                return candidate
        return None

"""Gate 2's table: the eight damage-step keywords, one row each.

Each row says which output fields the keyword moves, which way, and when each
is observable at all. Applicability is per effect rather than per row, because
a keyword rarely moves all its fields on one combat: infect's creature-side and
player-side fields are exclusive by construction.

A combat record covers one damage step, so first strike and double strike are
visible only in the first-strike step's record; in the regular step the board
already carries the first hit. Every other row reads damage a creature deals
in the record's own step (rule 510.4): a creature without first or double
strike deals none in the first-strike step, and one with first strike alone
none in the regular step (``rule_outcomes.strikes_in_step``). A record whose
step is unknown is read as if every creature struck in it. Damage reaches a
creature only where no protection of its prevents it (702.16e), and a
blocked attacker whose blockers all left deals none unless it has trample,
which then assigns it all to the player (509.1h, 510.1c, 702.19e). A creature
fighting one creature deals it all its damage; one fighting several splits it
as the record's own assignment says (510.1c-d). A
phased-out creature and a last-known copy take no part in a combat, and a
combat whose own changes end the game is no observation at all: Forge
destroys no creature once the game is over (decision D13). A row's
``qualifies`` is derived from its effects, so the counter that decides
"under-sampled" and the scorer that decides "agrees" read one population
(FR-122). Evasion keywords are absent:
their signal is in playability records.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum

from effects.domain.effect_head_layout import HeadField, ZoneOutcome, counter_field
from effects.domain.keyword_names import with_implied
from effects.domain.provenance import ProvenanceKey, SidecarLookup
from effects.domain.records import RecordKind
from effects.domain.rule_detectors.sba_board import game_ends
from effects.domain.rule_outcomes import (
    Characteristics,
    ProtectionQuality,
    damage_prevented,
    strikes_in_step,
)
from effects.domain.state_snapshot import CombatSubstep, EntityState, StateSnapshot


class Subject(StrEnum):
    """Whose output field the keyword moves."""

    CARRIER = "carrier"
    #: The creatures the carrier's damage is dealt to.
    OPPONENT = "opponent"
    #: The player who controls the carrier.
    CONTROLLER = "controller"
    DEFENDING_PLAYER = "defending_player"


_STRIKES_FIRST = frozenset({"first_strike", "double_strike"})
_COUNTER_DAMAGE = frozenset({"wither", "infect"})
#: Where wither's and infect's damage to a creature lands.
_M1M1_COUNTERS = counter_field("M1M1")

#: ``EntityState -> the keywords it carries``, in this table's spelling.
KeywordsOf = Callable[[EntityState], "set[str]"]

#: ``source id -> {target ref: damage}``: the step's damage assignment as the
#: players made it, before prevention (``CombatPayload.assignment_choices``).
Assignment = Mapping[str, Mapping[str, int]]


# ── who is fighting whom ────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class CombatBoard:
    """One combat record as the table reads it: its snapshot, the keywords of
    every creature in its combat, each resolved once, and the step's damage
    assignment."""

    state: StateSnapshot
    #: ``entity id -> its keywords``, for every creature in the combat.
    keywords: Mapping[str, frozenset[str]]
    assignment: Assignment

    @classmethod
    def of(cls, record, keywords_of: KeywordsOf) -> CombatBoard:
        """``record``'s board, every combatant's keywords read through ``keywords_of``."""
        state = record.state
        return cls(
            state,
            {entity.id: frozenset(keywords_of(entity))
             for entity in state.entities if _in_combat(entity)},
            record.payload.assignment_choices,
        )

    def keywords_of(self, entity: EntityState) -> frozenset[str]:
        return self.keywords.get(entity.id, frozenset())

    def opponents_of(self, entity: EntityState) -> tuple[EntityState, ...]:
        """The creatures ``entity`` trades damage with: its blockers, or the
        attackers it blocks. A phased-out creature is treated as though it does
        not exist (702.26b)."""
        combat = entity.combat
        if combat is None:
            return ()
        ids = combat.blocked_by if combat.attacking is not None else combat.blocking
        return tuple(opponent for opponent in (self.state.entity(i) for i in ids)
                     if opponent is not None and opponent.on_battlefield)

    def assigned(self, source: EntityState, target: EntityState) -> int:
        """The combat damage ``source`` assigns ``target`` in this step.

        A creature fighting ``target`` alone assigns it all its damage (510.1c,
        510.1d), an upper bound for a trampler, which may assign the excess to
        the player (702.19b). One fighting several splits it by its
        controller's choice, which only the record's assignment says.
        """
        if tuple(o.id for o in self.opponents_of(source)) == (target.id,):
            return source.power
        return self.assignment.get(source.id, {}).get(target.id, 0)

    def defender_exists(self, attacker: EntityState) -> bool:
        """Whether what ``attacker`` attacks is still there: a player of the
        board, or a planeswalker or battle on the battlefield. One that left
        leaves the placeholder ``E-1``, which is dealt no damage (506.4c)."""
        defender = defender_ref(attacker)
        if any(player.id == defender for player in self.state.players):
            return True
        entity = self.state.entity(defender) if defender else None
        return entity is not None and entity.on_battlefield

    def participant(self, carrier: EntityState) -> CombatParticipant | None:
        """The carrier's place in the combat, or None if it has none."""
        if not _attacks_or_blocks(carrier):
            return None
        attacking = carrier.combat.attacking is not None
        return CombatParticipant(
            board=self,
            carrier=carrier,
            opponents=self.opponents_of(carrier),
            defending_player=defending_player(self.state, carrier) if attacking else None,
            combat_substep=self.state.global_.combat_substep,
        )


def _attacks_or_blocks(entity: EntityState) -> bool:
    combat = entity.combat
    return combat is not None and (combat.attacking is not None or bool(combat.blocking))


def _in_combat(entity: EntityState) -> bool:
    """A last-known copy (no zone) or a phased-out creature is in no combat."""
    return entity.combat is not None and entity.on_battlefield


@dataclass(frozen=True, slots=True)
class CombatParticipant:
    """One creature's place in a combat, and the table's predicates on it.

    ``opponents`` are the creatures the carrier trades damage with; an
    unblocked attacker has none. ``defending_player`` exists for an attacker
    only: a blocker's damage never lands on a player. ``combat_substep`` is
    the damage step the record covers. Every keyword a predicate reads comes
    from ``board``, which resolved each combatant's once.
    """

    board: CombatBoard = field(compare=False, repr=False)
    carrier: EntityState
    opponents: tuple[EntityState, ...]
    defending_player: str | None
    combat_substep: str | None = None

    # ── the combat's shape ──

    @property
    def is_attacking(self) -> bool:
        combat = self.carrier.combat
        return bool(combat and combat.attacking is not None)

    @property
    def is_unblocked(self) -> bool:
        """An attacker no creature blocked: one whose blockers all left
        combat is still blocked, and deals no combat damage without trample
        (509.1h, 510.1c)."""
        combat = self.carrier.combat
        return bool(combat and combat.attacking is not None and not combat.blocked_by
                    and not combat.became_blocked)

    @property
    def is_blockerless(self) -> bool:
        """A blocked attacker every creature blocking it has left (509.1h)."""
        combat = self.carrier.combat
        return bool(combat and combat.attacking is not None and combat.became_blocked
                    and not self.opponents)

    @property
    def has_opponents(self) -> bool:
        return bool(self.opponents)

    @property
    def deals_damage(self) -> bool:
        """Checked by every damage-dealing row: removing lifelink from a 0/3 wall
        moves nothing, and counting it would be a coin flip against the keyword."""
        return self.carrier.power >= 1

    @property
    def lone_opponent(self) -> EntityState | None:
        """The one creature the carrier trades damage with, or None.

        With several, the attacker's damage assignment decides who takes what
        (510.1c), so a "does this hit kill" rule cannot be answered for the
        carrier's hit. What a sum over the opponents reads holds either way:
        trample sums its blockers' toughness, which no split changes, and
        indestructible sums what each opponent assigned the carrier
        (:meth:`CombatBoard.assigned`).
        """
        return self.opponents[0] if len(self.opponents) == 1 else None

    @property
    def dealt_opponents(self) -> tuple[EntityState, ...]:
        """The opponents the carrier's damage is dealt to: each it assigns
        any (all of it to a lone one, 510.1c-d; the record's assignment with
        several) whose protection does not prevent it (702.16e)."""
        return tuple(
            opponent for opponent in self.opponents
            if self.board.assigned(self.carrier, opponent) >= 1
            and self.reaches(self.carrier, opponent))

    def subjects(self, subject: Subject) -> tuple[str, ...]:
        """The entities or players whose output field an effect on ``subject``
        moves: an ``OPPONENT`` effect is read at each creature the carrier's
        damage is dealt to, and a combat is one observation however many."""
        match subject:
            case Subject.CARRIER:
                found = (self.carrier.id,)
            case Subject.OPPONENT:
                found = tuple(opponent.id for opponent in self.dealt_opponents)
            case Subject.CONTROLLER:
                found = (self.carrier.controller,)
            case Subject.DEFENDING_PLAYER:
                found = (self.defending_player,)
            case _:
                raise ValueError(f"unhandled subject {subject}")
        return tuple(ref for ref in found if ref is not None)

    # ── what the predicates read ──

    def keywords_of(self, entity: EntityState) -> frozenset[str]:
        return self.board.keywords_of(entity)

    def carries(self, keyword: str) -> bool:
        return keyword in self.board.keywords_of(self.carrier)

    def strikes_now(self, creature: EntityState) -> bool:
        """Whether ``creature`` deals its combat damage in the record's step: it
        has power and the step is its own (rules 510.4, 702.7b, 702.4b); an
        unknown step lets every creature strike."""
        if creature.power < 1:
            return False
        step = self.combat_substep
        return step is None or strikes_in_step(self.keywords_of(creature), step)

    @property
    def carrier_strikes(self) -> bool:
        return self.strikes_now(self.carrier)

    def reaches(self, source: EntityState, target: EntityState) -> bool:
        """Whether damage ``source`` deals ``target`` is not prevented by a
        protection of ``target``'s (702.16e)."""
        protections = tuple(
            quality for keyword in self.keywords_of(target) if keyword.startswith("protection")
            if (quality := ProtectionQuality.of_keyword(keyword)) is not None)
        return not damage_prevented(protections, Characteristics.of(source))

    def hitters(self, target: EntityState) -> list[EntityState]:
        """Every creature fighting ``target`` whose damage is dealt to it in
        this step: it strikes now, assigns ``target`` some, and no protection
        of ``target``'s prevents it."""
        return [
            source for source in self.board.opponents_of(target)
            if self.strikes_now(source) and self.board.assigned(source, target) >= 1
            and self.reaches(source, target)
        ]

    # ── when each effect is observable ──

    def strikes_first_because_of(self, keyword: str) -> bool:
        """The carrier deals damage in this first-strike step because of ``keyword``.

        The record is the first-strike step's, the carrier has power, and no other
        keyword of its own puts it in this step anyway (stripping first strike
        from a double striker moves nothing; 702.7b, 702.4b). The opponent's
        keywords do not matter: either way the carrier's hit lands here only
        with the keyword.
        """
        others = _STRIKES_FIRST - {keyword}
        return (self.combat_substep == CombatSubstep.FIRST_STRIKE and self.deals_damage
                and not others & self.keywords_of(self.carrier))

    def marks_damage_on_its_lone_opponent(self) -> bool:
        """The carrier fights one creature, its damage reaches it (702.16e), and
        it is marked as damage: wither's and infect's is dealt as -1/-1 counters
        instead (702.80a, 702.90a)."""
        opponent = self.lone_opponent
        return (opponent is not None and not _COUNTER_DAMAGE & self.keywords_of(self.carrier)
                and self.reaches(self.carrier, opponent))

    def kills_outright(self) -> bool:
        """One hit from the carrier would kill the one creature it is fighting,
        deathtouch making one point lethal (702.2b); an indestructible creature
        dies to neither (702.12b)."""
        opponent = self.lone_opponent
        if opponent is None or "indestructible" in self.keywords_of(opponent):
            return False
        if not self.reaches(self.carrier, opponent):
            return False
        power = self.carrier.power
        if self.carries("deathtouch"):
            return power >= 1
        return power >= opponent.remaining_toughness

    def costs_the_defending_player_life(self) -> bool:
        """An unblocked attacker's damage is life loss, unless it has infect,
        whose damage to a player is poison counters (702.90b)."""
        return self.is_unblocked and not self.carries("infect")

    def lone_opponent_survives_a_normal_hit(self) -> bool:
        """The carrier strikes and reaches its one opponent, which would live
        through the step's damage without deathtouch and die with it (702.2b).

        Not an indestructible one (702.12b), and not one the step kills anyway:
        every other creature dealing it damage counts, each blocker of an
        attacker dealing it all its damage (510.1c), and another deathtouch
        hitter kills it whatever the carrier carries.
        """
        opponent = self.lone_opponent
        if opponent is None or not self.carrier_strikes:
            return False
        if "indestructible" in self.keywords_of(opponent):
            return False
        return self.reaches(self.carrier, opponent) and self._survives_the_rest(opponent)

    def _survives_the_rest(self, opponent: EntityState) -> bool:
        """``opponent`` lives through the carrier's hit without deathtouch and
        through every other hit it is dealt in the step."""
        others = [h for h in self.hitters(opponent) if h.id != self.carrier.id]
        if any("deathtouch" in self.keywords_of(h) for h in others):
            return False
        incoming = sum(self.board.assigned(h, opponent) for h in others)
        return opponent.remaining_toughness > self.carrier.power + incoming

    def damages_a_creature(self) -> bool:
        """The carrier strikes, and its damage is dealt to a creature it fights
        (:attr:`dealt_opponents`), where an ``OPPONENT`` effect is read."""
        return self.carrier_strikes and bool(self.dealt_opponents)

    def damages_anything(self) -> bool:
        """The carrier's damage reaches a creature it fights or what it attacks
        (an unblocked attacker, or a trampler whose blockers left, 702.19e),
        if that is still there (506.4c)."""
        if not self.carrier_strikes:
            return False
        if self._hits_what_it_attacks():
            return True
        return any(self.reaches(self.carrier, opponent) for opponent in self.opponents)

    def _hits_what_it_attacks(self) -> bool:
        runs_through = self.is_unblocked or (self.is_blockerless and self.carries("trample"))
        return runs_through and self.board.defender_exists(self.carrier)

    def damages_the_defending_player(self) -> bool:
        return self.carrier_strikes and self.is_unblocked

    def runs_over_its_blockers(self) -> bool:
        """A blocked attacker striking in this step with more power than its
        blockers can soak, or whose blockers all left, which assigns all its
        damage to the player (702.19e)."""
        if not self.carrier_strikes:
            return False
        if self.is_blockerless:
            return True
        if not self.is_attacking or not self.has_opponents:
            return False
        # With deathtouch one damage is lethal damage (702.2c, 702.19b).
        deadly = self.carries("deathtouch")
        soaked = sum(
            min(1, max(0, blocker.remaining_toughness)) if deadly
            else max(0, blocker.remaining_toughness)
            for blocker in self.opponents
        )
        return soaked < self.carrier.power

    def tramples_life_away(self) -> bool:
        """Trample's excess is life loss, unless the trampler has infect,
        whose damage to a player is poison counters (702.90b)."""
        return not self.carries("infect") and self.runs_over_its_blockers()

    def would_be_destroyed(self) -> bool:
        """The damage dealt to the carrier in this step would otherwise be
        lethal (702.12b): enough of it, summed over what each opponent
        striking in the step assigned it (:meth:`CombatBoard.assigned`), or
        any of it from a deathtouch creature."""
        hitters = self.hitters(self.carrier)
        if not hitters or any(_COUNTER_DAMAGE & self.keywords_of(h) for h in hitters):
            # -1/-1 counters kill an indestructible creature without destroying
            # it, so the keyword saves nothing the record could show.
            return False
        if any("deathtouch" in self.keywords_of(h) for h in hitters):
            return True
        incoming = sum(self.board.assigned(h, self.carrier) for h in hitters)
        return incoming >= self.carrier.remaining_toughness


def combat_participant(
    record, carrier: EntityState, keywords_of: KeywordsOf,
) -> CombatParticipant | None:
    """The carrier's place in ``record``'s combat, or None if it has none; a
    caller reading several carriers of one record builds its
    :class:`CombatBoard` once instead."""
    return CombatBoard.of(record, keywords_of).participant(carrier)


def defender_ref(attacker: EntityState) -> str | None:
    """The ref of what ``attacker`` attacks, or None when it is not attacking:
    its ``CombatStatus.attacking``, a player's ``P<id>``, a planeswalker's or
    battle's ``E<id>``, or ``E-1`` for the placeholder a departed one leaves
    (506.4c), which no entity carries."""
    return attacker.combat.attacking if attacker.combat else None


def defending_player(
    state: StateSnapshot, carrier: EntityState,
) -> str | None:
    """Which player the carrier is attacking: the defender's ref when it
    names one of the board's players. An attack on a planeswalker or battle
    has no defending player, since no player's life or poison can move.
    """
    defender = defender_ref(carrier)
    return defender if any(p.id == defender for p in state.players) else None


# ── the table ───────────────────────────────────────────────────────────


#: Whether one effect is observable on one combat.
Applies = Callable[[CombatParticipant], bool]


def _always(_participant: CombatParticipant) -> bool:
    return True


@dataclass(frozen=True, slots=True)
class KeywordEffect:
    """One field the keyword moves, which way, and when that is visible."""

    field: str
    subject: Subject
    #: Sign of the change when the keyword is **present** versus absent.
    direction: int
    #: For a categorical field, the class whose probability should move.
    category: str | None = None
    applies: Applies = _always

    def __post_init__(self) -> None:
        if self.direction not in (-1, 1):
            raise ValueError(
                f"direction must be -1 or 1, got {self.direction} for "
                f"{self.field}"
            )

    @property
    def label(self) -> str:
        """How this effect is named in a per-field breakdown, prefixed by its
        subject because one field can move on either side of a combat."""
        name = self.field if self.category is None else (
            f"{self.field}/{self.category}"
        )
        return f"{self.subject.value}.{name}"

    def observable(self, participant: CombatParticipant) -> bool:
        """Whether this combat can show this effect at all: the one definition
        both the counter and the scorer read."""
        return self.applies(participant) and bool(participant.subjects(self.subject))


@dataclass(frozen=True, slots=True)
class DamageStepKeyword:
    """One row of gate 2's table."""

    keyword: str
    effects: tuple[KeywordEffect, ...]

    @property
    def fields(self) -> tuple[str, ...]:
        """The head fields this row reads, each named once."""
        return tuple(dict.fromkeys(effect.field for effect in self.effects))

    def qualifies(self, participant: CombatParticipant) -> bool:
        """Whether this combat is an observation of this keyword, derived from
        the effects so no counted combat is one none of them can be read on."""
        return any(effect.observable(participant) for effect in self.effects)


def _strikes_first_and(keyword: str, holds: Applies) -> Applies:
    """The carrier strikes first because of ``keyword``, and ``holds`` of its hit."""

    def applies(participant: CombatParticipant) -> bool:
        return participant.strikes_first_because_of(keyword) and holds(participant)

    return applies


def _first_strike_step_effects(keyword: str) -> tuple[KeywordEffect, ...]:
    """What first strike and double strike change inside one combat record.

    The same three effects for both: in the first-strike step's record the two
    keywords do the same thing, and neither row reads the regular step.
    """
    return (
        KeywordEffect(
            HeadField.DAMAGE_TAKEN, Subject.OPPONENT, direction=1,
            applies=_strikes_first_and(
                keyword, CombatParticipant.marks_damage_on_its_lone_opponent),
        ),
        KeywordEffect(
            HeadField.ZONE_OUTCOME, Subject.OPPONENT, direction=1,
            category=ZoneOutcome.TO_GRAVEYARD,
            applies=_strikes_first_and(keyword, CombatParticipant.kills_outright),
        ),
        KeywordEffect(
            HeadField.LIFE_DELTA, Subject.DEFENDING_PLAYER, direction=-1,
            applies=_strikes_first_and(
                keyword, CombatParticipant.costs_the_defending_player_life),
        ),
    )


def _counter_damage_effects() -> tuple[KeywordEffect, ...]:
    """Wither's and infect's creature half (702.80a, 702.90a): -1/-1 counters
    instead of marked damage, read at each creature the damage is dealt to."""
    return (
        KeywordEffect(
            _M1M1_COUNTERS, Subject.OPPONENT, direction=1,
            applies=CombatParticipant.damages_a_creature,
        ),
        KeywordEffect(
            HeadField.DAMAGE_TAKEN, Subject.OPPONENT, direction=-1,
            applies=CombatParticipant.damages_a_creature,
        ),
    )


DAMAGE_STEP_KEYWORDS: tuple[DamageStepKeyword, ...] = (
    DamageStepKeyword("first_strike", _first_strike_step_effects("first_strike")),
    DamageStepKeyword("double_strike", _first_strike_step_effects("double_strike")),
    DamageStepKeyword("deathtouch", (
        KeywordEffect(
            HeadField.ZONE_OUTCOME, Subject.OPPONENT, direction=1,
            category=ZoneOutcome.TO_GRAVEYARD,
            applies=CombatParticipant.lone_opponent_survives_a_normal_hit,
        ),
    )),
    DamageStepKeyword("lifelink", (
        KeywordEffect(
            HeadField.LIFE_DELTA, Subject.CONTROLLER, direction=1,
            applies=CombatParticipant.damages_anything,
        ),
    )),
    DamageStepKeyword("trample", (
        KeywordEffect(
            HeadField.LIFE_DELTA, Subject.DEFENDING_PLAYER, direction=-1,
            applies=CombatParticipant.tramples_life_away,
        ),
    )),
    DamageStepKeyword("indestructible", (
        KeywordEffect(
            HeadField.ZONE_OUTCOME, Subject.CARRIER, direction=-1,
            category=ZoneOutcome.TO_GRAVEYARD,
            applies=CombatParticipant.would_be_destroyed,
        ),
    )),
    DamageStepKeyword("wither", _counter_damage_effects()),
    DamageStepKeyword("infect", (
        *_counter_damage_effects(),
        KeywordEffect(
            HeadField.POISON_DELTA, Subject.DEFENDING_PLAYER, direction=1,
            applies=CombatParticipant.damages_the_defending_player,
        ),
        # Poison instead of life loss: life moves up with infect.
        KeywordEffect(
            HeadField.LIFE_DELTA, Subject.DEFENDING_PLAYER, direction=1,
            applies=CombatParticipant.damages_the_defending_player,
        ),
    )),
)

KEYWORDS_BY_NAME: dict[str, DamageStepKeyword] = {
    row.keyword: row for row in DAMAGE_STEP_KEYWORDS
}

#: Gate 2's thresholds (FR-119).
MIN_QUALIFYING_RECORDS = 200
MIN_DIRECTION_AGREEMENT = 0.70


# ── who is observed ─────────────────────────────────────────────────────
#
# Here rather than beside gate 2, which imports torch, because build-corpus's
# survey workers use it too (FR-045).


class KeywordResolver:
    """An entity's keywords, from both channels, memoized by provenance key.

    Printed and attachment-granted keywords arrive as ability tokens; only an
    until-end-of-turn grant is a bare string in the overlay, so reading the
    overlay alone misses the common case. A keyword another implies is added
    (``keyword_names.with_implied``): a creature with trample over
    planeswalkers alone is a trampler.
    """

    def __init__(self, sidecars: SidecarLookup | None = None) -> None:
        self._sidecars = sidecars
        self._by_key: dict[ProvenanceKey, str | None] = {}

    def _keyword_for(self, key: ProvenanceKey) -> str | None:
        if key not in self._by_key:
            line = self._sidecars.line_or_none(key)
            self._by_key[key] = line.bare_keyword if line is not None else None
        return self._by_key[key]

    def keywords_of(self, entity) -> set[str]:
        # Already normalized by ``record_codec`` at parse time.
        found = set(entity.granted_temporary.keywords)
        if self._sidecars is not None:
            for key in entity.ability_keys:
                keyword = self._keyword_for(key)
                if keyword is not None:
                    found.add(keyword)
        # Trample over planeswalkers is trample too (702.19c).
        return with_implied(found)


def qualifying_observations(
    record, rows: tuple[DamageStepKeyword, ...], resolver: KeywordResolver,
) -> list[tuple[DamageStepKeyword, CombatParticipant]]:
    """Every ``(row, carrier)`` in one combat record the gate can observe, by
    the row's own ``qualifies``, in a combat that does not end the game; each
    combatant's keywords are resolved once (:class:`CombatBoard`)."""
    if not _shows_its_combat(record):
        return []
    board = CombatBoard.of(record, resolver.keywords_of)
    wanted = {row.keyword for row in rows}
    found: list[tuple[DamageStepKeyword, CombatParticipant]] = []
    for entity in record.state.entities:
        participant = _carrier(board, entity, wanted)
        if participant is not None:
            # Table order, not set order, which follows the hash seed.
            found.extend((row, participant) for row in rows
                         if row.keyword in board.keywords_of(entity) and row.qualifies(participant))
    return found


def _carrier(board: CombatBoard, entity: EntityState, wanted: set[str]) -> CombatParticipant | None:
    """``entity``'s place in the combat when it carries a ``wanted`` keyword."""
    if not _in_combat(entity) or not board.keywords_of(entity) & wanted:
        return None
    return board.participant(entity)


def _shows_its_combat(record) -> bool:
    """Whether the record is a combat whose changes do not end the game
    (``sba_board.game_ends``, the test ``sba_zone_move`` reads). Forge's
    state-based check returns at the game's end before it destroys any
    creature, so a combat that ends the game shows no creature death and is
    no observation, nor is a probe of one (decision D13)."""
    return record.kind is RecordKind.COMBAT and not game_ends(record)

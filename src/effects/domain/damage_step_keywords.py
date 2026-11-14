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
which then assigns it all to the player (509.1h, 510.1c, 702.19e). A
phased-out creature and a last-known copy take no part in a combat. A row's
``qualifies`` is derived from its effects, so the counter that decides
"under-sampled" and the scorer that decides "agrees" read one population
(FR-122). Evasion keywords are absent:
their signal is in playability records.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from effects.domain.effect_head_layout import HeadField, ZoneOutcome, counter_field
from effects.domain.keyword_names import with_implied
from effects.domain.provenance import ProvenanceKey, SidecarLookup
from effects.domain.records import RecordKind
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
    #: The creature the carrier is fighting.
    OPPONENT = "opponent"
    #: The player who controls the carrier.
    CONTROLLER = "controller"
    DEFENDING_PLAYER = "defending_player"


# ── who is fighting whom ────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class CombatParticipant:
    """One creature's place in a combat, as the table's predicates read it.

    ``opponents`` are the creatures the carrier trades damage with; an
    unblocked attacker has none. ``defending_player`` exists for an attacker
    only: a blocker's damage never lands on a player. ``combat_substep`` is
    the damage step the record covers.
    """

    carrier: EntityState
    opponents: tuple[EntityState, ...]
    defending_player: str | None
    combat_substep: str | None = None

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

        With several, Forge's damage assignment decides who takes what, so a
        "does this hit kill" rule cannot be answered; trample and indestructible
        read every opponent instead, since a sum does not depend on the split.
        """
        return self.opponents[0] if len(self.opponents) == 1 else None

    def subject(self, subject: Subject) -> str | None:
        """The entity or player whose output field an effect on ``subject`` moves.

        An ``OPPONENT`` effect is read at the first creature only, so a
        three-blocker combat is one observation rather than three.
        """
        match subject:
            case Subject.CARRIER:
                return self.carrier.id
            case Subject.OPPONENT:
                return self.opponents[0].id if self.opponents else None
            case Subject.CONTROLLER:
                return self.carrier.controller
            case Subject.DEFENDING_PLAYER:
                return self.defending_player
        raise ValueError(f"unhandled subject {subject}")


#: ``EntityState -> the keywords it carries``, in this table's spelling.
KeywordsOf = Callable[[EntityState], "set[str]"]

#: Whether one effect is observable on one combat.
Applies = Callable[[CombatParticipant, KeywordsOf], bool]


def combat_participant(
    state: StateSnapshot, carrier: EntityState,
) -> CombatParticipant | None:
    """The carrier's place in the snapshot's combat, or None if it has none."""
    combat = carrier.combat
    if combat is None:
        return None
    if combat.attacking is not None:
        opponent_ids: tuple[str, ...] = combat.blocked_by
    elif combat.blocking:
        opponent_ids = combat.blocking
    else:
        return None
    # A phased-out creature is treated as though it does not exist (702.26b).
    opponents = tuple(
        entity for entity in (state.entity(i) for i in opponent_ids)
        if entity is not None and entity.on_battlefield
    )
    return CombatParticipant(
        carrier=carrier,
        opponents=opponents,
        defending_player=(
            defending_player(state, carrier)
            if combat.attacking is not None else None
        ),
        combat_substep=state.global_.combat_substep,
    )


#: Forge renders a card defender (a planeswalker or a battle) as ``Name (id)``,
#: and a player defender by the player's name, which carries no such suffix.
_CARD_DEFENDER = re.compile(r" \(\d+\)$")

#: Forge's rendering of an attack whose defender is gone.
_NO_DEFENDER = "<Nothing>"


def defending_player(
    state: StateSnapshot, carrier: EntityState,
) -> str | None:
    """Which player the carrier is attacking.

    ``CombatStatus.attacking`` holds Forge's rendering of the defender object,
    not a player id, so the id is recovered from the board: the one player who
    is not the attacker's controller. An attack on a planeswalker or battle
    has no defending player, since no player's life or poison can move.
    """
    attacking = carrier.combat.attacking if carrier.combat else None
    if attacking in (None, _NO_DEFENDER) or _CARD_DEFENDER.search(attacking):
        return None
    others = [p.id for p in state.players if p.id != carrier.controller]
    return others[0] if len(others) == 1 else None


# ── what the predicates read ────────────────────────────────────────────


_STRIKES_FIRST = frozenset({"first_strike", "double_strike"})
_COUNTER_DAMAGE = frozenset({"wither", "infect"})
#: Where wither's and infect's damage to a creature lands.
_M1M1_COUNTERS = counter_field("M1M1")


def _kills_outright(
    participant: CombatParticipant, keywords_of: KeywordsOf,
) -> bool:
    """One hit from the carrier would kill the creature it is fighting,
    deathtouch making one point lethal; an indestructible creature dies to
    neither (702.12b)."""
    opponent = participant.lone_opponent
    if opponent is None or "indestructible" in keywords_of(opponent):
        return False
    if not _reaches(participant.carrier, opponent, keywords_of):
        return False
    power = participant.carrier.power
    if "deathtouch" in keywords_of(participant.carrier):
        return power >= 1
    return power >= opponent.remaining_toughness


def _survives_a_normal_hit(participant: CombatParticipant) -> bool:
    """The opponent would live through one hit, counting toughness alone.

    Blind to deathtouch on purpose: it is the deathtouch row's own question,
    and that row's carrier always has deathtouch.
    """
    opponent = participant.lone_opponent
    return opponent is not None and (
        opponent.remaining_toughness > participant.carrier.power
    )


# ── when each effect is observable ──────────────────────────────────────


def _always(_participant: CombatParticipant, _keywords_of: KeywordsOf) -> bool:
    return True


def _strikes_now(
    participant: CombatParticipant, creature: EntityState, keywords_of: KeywordsOf,
) -> bool:
    """Whether ``creature`` deals its combat damage in the record's step: it has
    power and the step is its own (rules 510.4, 702.7b, 702.4b); an unknown
    step lets every creature strike."""
    if creature.power < 1:
        return False
    step = participant.combat_substep
    return step is None or strikes_in_step(keywords_of(creature), step)


def _carrier_strikes(participant: CombatParticipant, keywords_of: KeywordsOf) -> bool:
    return _strikes_now(participant, participant.carrier, keywords_of)


def _protections(entity: EntityState, keywords_of: KeywordsOf) -> tuple[ProtectionQuality, ...]:
    """The protections ``entity``'s keywords state."""
    return tuple(quality for keyword in keywords_of(entity) if keyword.startswith("protection")
                 if (quality := ProtectionQuality.of_keyword(keyword)) is not None)


def _reaches(source: EntityState, target: EntityState, keywords_of: KeywordsOf) -> bool:
    """Whether damage ``source`` deals ``target`` is not prevented by a
    protection of ``target``'s (702.16e)."""
    return not damage_prevented(_protections(target, keywords_of), Characteristics.of(source))


def _hits_its_opponent(participant: CombatParticipant, keywords_of: KeywordsOf) -> bool:
    """The carrier strikes, and its hit reaches the creature an ``OPPONENT``
    effect is read at, the first it fights."""
    return (_carrier_strikes(participant, keywords_of) and participant.has_opponents
            and _reaches(participant.carrier, participant.opponents[0], keywords_of))


def _strikes_first_because_of(keyword: str) -> Applies:
    """The carrier deals damage in this first-strike step because of ``keyword``.

    The record is the first-strike step's, the carrier has power, and no other
    keyword of its own puts it in this step anyway (stripping first strike
    from a double striker moves nothing). The opponent's keywords do not
    matter: either way the carrier's hit lands here only with the keyword.
    """
    others = _STRIKES_FIRST - {keyword}

    def strikes_first(
        participant: CombatParticipant, keywords_of: KeywordsOf,
    ) -> bool:
        return (
            participant.combat_substep == CombatSubstep.FIRST_STRIKE
            and participant.deals_damage
            and not (others & keywords_of(participant.carrier))
        )

    return strikes_first


def _strikes_first_and(keyword: str, holds: Applies) -> Applies:
    """The carrier strikes first because of ``keyword``, and ``holds`` of its hit."""
    strikes_first = _strikes_first_because_of(keyword)

    def applies(participant: CombatParticipant, keywords_of: KeywordsOf) -> bool:
        return strikes_first(participant, keywords_of) and holds(participant, keywords_of)

    return applies


def _an_opponent_survives_a_normal_hit(
    participant: CombatParticipant, keywords_of: KeywordsOf,
) -> bool:
    """The carrier strikes, and its one opponent would survive the hit without
    deathtouch and die with it: not an indestructible one (702.12b)."""
    opponent = participant.lone_opponent
    return (
        _carrier_strikes(participant, keywords_of)
        and _survives_a_normal_hit(participant)
        and opponent is not None and "indestructible" not in keywords_of(opponent)
        and _reaches(participant.carrier, opponent, keywords_of)
    )


def _damages_a_creature(
    participant: CombatParticipant, keywords_of: KeywordsOf,
) -> bool:
    return _hits_its_opponent(participant, keywords_of)


def _damages_anything(
    participant: CombatParticipant, keywords_of: KeywordsOf,
) -> bool:
    """The carrier's damage reaches a creature it fights or the defending
    player (an unblocked attacker, or a trampler whose blockers left)."""
    if not _carrier_strikes(participant, keywords_of):
        return False
    if participant.is_unblocked or (
            participant.is_blockerless and "trample" in keywords_of(participant.carrier)):
        return True
    return any(_reaches(participant.carrier, opponent, keywords_of)
               for opponent in participant.opponents)


def _damages_the_defending_player(
    participant: CombatParticipant, keywords_of: KeywordsOf,
) -> bool:
    return _carrier_strikes(participant, keywords_of) and participant.is_unblocked


def _runs_over_its_blockers(
    participant: CombatParticipant, keywords_of: KeywordsOf,
) -> bool:
    """A blocked attacker striking in this step with more power than its
    blockers can soak, or whose blockers all left, which assigns all its
    damage to the player (702.19e)."""
    if not _carrier_strikes(participant, keywords_of):
        return False
    if participant.is_blockerless:
        return True
    if not participant.is_attacking or not participant.has_opponents:
        return False
    # With deathtouch one damage is lethal damage (702.2c, 702.19b).
    deadly = "deathtouch" in keywords_of(participant.carrier)
    soaked = sum(
        min(1, max(0, blocker.remaining_toughness)) if deadly
        else max(0, blocker.remaining_toughness)
        for blocker in participant.opponents
    )
    return soaked < participant.carrier.power


def _would_be_destroyed(
    participant: CombatParticipant, keywords_of: KeywordsOf,
) -> bool:
    """The damage coming in this step would otherwise be lethal: enough of it,
    summed over every opponent striking in the step, or any of it from a
    deathtouch creature."""
    if not participant.has_opponents:
        return False
    hitters = [o for o in participant.opponents if _strikes_now(participant, o, keywords_of)
               and _reaches(o, participant.carrier, keywords_of)]
    if not hitters:
        return False
    if any(_COUNTER_DAMAGE & keywords_of(opponent) for opponent in hitters):
        # -1/-1 counters kill an indestructible creature without destroying
        # it, so the keyword saves nothing the record could show.
        return False
    if any("deathtouch" in keywords_of(opponent) for opponent in hitters):
        return True
    incoming = sum(opponent.power for opponent in hitters)
    return incoming >= participant.carrier.remaining_toughness


# ── the table ───────────────────────────────────────────────────────────


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

    def observable(
        self, participant: CombatParticipant, keywords_of: KeywordsOf,
    ) -> bool:
        """Whether this combat can show this effect at all: the one definition
        both the counter and the scorer read."""
        return (
            self.applies(participant, keywords_of)
            and participant.subject(self.subject) is not None
        )


@dataclass(frozen=True, slots=True)
class DamageStepKeyword:
    """One row of gate 2's table."""

    keyword: str
    #: The effects' ``applies`` predicates restated in words, for a reader.
    qualifies_when: str
    effects: tuple[KeywordEffect, ...]

    @property
    def fields(self) -> tuple[str, ...]:
        """The head fields this row reads, each named once."""
        return tuple(dict.fromkeys(effect.field for effect in self.effects))

    def qualifies(
        self, participant: CombatParticipant, keywords_of: KeywordsOf,
    ) -> bool:
        """Whether this combat is an observation of this keyword, derived from
        the effects so no counted combat is one none of them can be read on."""
        return any(
            effect.observable(participant, keywords_of)
            for effect in self.effects
        )


def _first_strike_step_effects(keyword: str) -> tuple[KeywordEffect, ...]:
    """What first strike and double strike change inside one combat record.

    The same three effects for both: in the first-strike step's record the two
    keywords do the same thing, and neither row reads the regular step.
    """
    return (
        KeywordEffect(
            HeadField.DAMAGE_TAKEN, Subject.OPPONENT, direction=1,
            applies=_strikes_first_and(
                keyword, lambda participant, _: participant.lone_opponent is not None,
            ),
        ),
        KeywordEffect(
            HeadField.ZONE_OUTCOME, Subject.OPPONENT, direction=1, category=ZoneOutcome.DIED,
            applies=_strikes_first_and(keyword, _kills_outright),
        ),
        KeywordEffect(
            HeadField.LIFE_DELTA, Subject.DEFENDING_PLAYER, direction=-1,
            applies=_strikes_first_and(
                keyword, lambda participant, _: participant.is_unblocked,
            ),
        ),
    )


_FIRST_STRIKE_STEP_QUALIFIES_WHEN = (
    "the record is the first-strike damage step's, the carrier deals combat "
    "damage and no other keyword of its own would make it strike first, and "
    "it is either fighting exactly one creature or an unblocked attacker"
)


DAMAGE_STEP_KEYWORDS: tuple[DamageStepKeyword, ...] = (
    DamageStepKeyword(
        keyword="first_strike",
        qualifies_when=_FIRST_STRIKE_STEP_QUALIFIES_WHEN,
        effects=_first_strike_step_effects("first_strike"),
    ),
    DamageStepKeyword(
        keyword="double_strike",
        qualifies_when=_FIRST_STRIKE_STEP_QUALIFIES_WHEN,
        effects=_first_strike_step_effects("double_strike"),
    ),
    DamageStepKeyword(
        keyword="deathtouch",
        qualifies_when=(
            "the carrier deals combat damage in the record's step to exactly one "
            "creature, not an indestructible one, and that creature would "
            "survive that much damage normally"
        ),
        effects=(
            KeywordEffect(
                HeadField.ZONE_OUTCOME, Subject.OPPONENT, direction=1, category=ZoneOutcome.DIED,
                applies=_an_opponent_survives_a_normal_hit,
            ),
        ),
    ),
    DamageStepKeyword(
        keyword="lifelink",
        qualifies_when=(
            "the carrier deals combat damage in the record's step to a creature or player"
        ),
        effects=(
            KeywordEffect(
                HeadField.LIFE_DELTA, Subject.CONTROLLER, direction=1,
                applies=_damages_anything,
            ),
        ),
    ),
    DamageStepKeyword(
        keyword="trample",
        qualifies_when=(
            "the carrier strikes in the record's step and is blocked by creatures "
            "with less remaining toughness between them than the carrier's power, "
            "each needing one damage where the carrier has deathtouch"
        ),
        effects=(
            KeywordEffect(
                HeadField.LIFE_DELTA, Subject.DEFENDING_PLAYER, direction=-1,
                applies=_runs_over_its_blockers,
            ),
        ),
    ),
    DamageStepKeyword(
        keyword="indestructible",
        qualifies_when=(
            "the carrier takes combat damage in the record's step at least equal "
            "to its remaining toughness, or any of it from a creature with "
            "deathtouch, and none of it from a creature with wither or infect"
        ),
        effects=(
            KeywordEffect(
                HeadField.ZONE_OUTCOME, Subject.CARRIER, direction=-1, category=ZoneOutcome.DIED,
                applies=_would_be_destroyed,
            ),
        ),
    ),
    DamageStepKeyword(
        keyword="wither",
        qualifies_when="the carrier deals combat damage in the record's step to a creature",
        effects=(
            KeywordEffect(
                _M1M1_COUNTERS, Subject.OPPONENT, direction=1,
                applies=_damages_a_creature,
            ),
            KeywordEffect(
                HeadField.DAMAGE_TAKEN, Subject.OPPONENT, direction=-1,
                applies=_damages_a_creature,
            ),
        ),
    ),
    DamageStepKeyword(
        keyword="infect",
        qualifies_when=(
            "the carrier deals combat damage in the record's step to a creature or player"
        ),
        effects=(
            KeywordEffect(
                _M1M1_COUNTERS, Subject.OPPONENT, direction=1,
                applies=_damages_a_creature,
            ),
            KeywordEffect(
                HeadField.DAMAGE_TAKEN, Subject.OPPONENT, direction=-1,
                applies=_damages_a_creature,
            ),
            KeywordEffect(
                HeadField.POISON_DELTA, Subject.DEFENDING_PLAYER, direction=1,
                applies=_damages_the_defending_player,
            ),
            # Poison instead of life loss: life moves up with infect.
            KeywordEffect(
                HeadField.LIFE_DELTA, Subject.DEFENDING_PLAYER, direction=1,
                applies=_damages_the_defending_player,
            ),
        ),
    ),
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
    the row's own ``qualifies``."""
    if record.kind is not RecordKind.COMBAT:
        return []
    found: list[tuple[DamageStepKeyword, CombatParticipant]] = []
    wanted = {row.keyword for row in rows}
    keywords_of = _memoized(resolver.keywords_of)
    for entity in record.state.entities:
        # A last-known copy (no zone) or a phased-out creature is in no combat.
        if entity.combat is None or not entity.on_battlefield:
            continue
        carried = keywords_of(entity) & wanted
        if not carried:
            continue
        participant = combat_participant(record.state, entity)
        if participant is None:
            continue
        # Table order, not set order, which follows the hash seed.
        for row in rows:
            if row.keyword not in carried:
                continue
            if row.qualifies(participant, keywords_of):
                found.append((row, participant))
    return found


def _memoized(keywords_of: KeywordsOf) -> KeywordsOf:
    """``keywords_of`` read once per entity of one record.

    Every row's predicates ask for the carrier's and its opponents' keywords
    again, about twenty times a carrier. Keyed by the entity object, which the
    record keeps alive for as long as the memo lives.
    """
    memo: dict[int, set[str]] = {}

    def memoized(entity: EntityState) -> set[str]:
        found = memo.get(id(entity))
        if found is None:
            found = memo[id(entity)] = keywords_of(entity)
        return found

    return memoized

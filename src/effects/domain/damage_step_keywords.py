"""Gate 2's table: the eight damage-step keywords, one row each.

Gate 2 asks a narrow question with a clean answer: when a keyword is removed
from a combat participant's model input, does the prediction move the way the
rules say it should? Each row below carries what makes that checkable — which
output fields the keyword touches, which way each moves when it is present, and
**when each one is observable at all**.

Applicability is per effect, not per row, because a keyword rarely moves all of
its fields on the same combat. Infect's creature-side fields and its
player-side fields are exclusive by construction — a blocked attacker damages a
creature and no player, an unblocked one the reverse — so a row-level
"qualifies" would have made one half of every infect observation a guaranteed
disagreement.

A combat record covers **one damage step**, so what first strike and double
strike change is visible in one record only: the first-strike step's. There the
carrier deals its damage with the keyword and none without it, while a creature
with neither keyword deals none either way — so the carrier's own damage taken
cannot move, and what both rows read is what the carrier does to the creature
or player it is fighting. In the regular step a first striker has already
struck, and the board the record starts from already carries its hit, so
neither row reads a regular-step record at all.

The row's own ``qualifies`` is therefore **derived** rather than written: a
combat qualifies for a keyword exactly when at least one of its effects applies
and has a subject on the board. That is what keeps the counter that decides
"under-sampled" and the scorer that decides "agrees" describing one population —
not by two rules kept in step, but by one rule (FR-122). ``qualifies_when``
restates it in prose for a reader.

The perturbation is **model-side**, not a game fork: the keyword is removed from
the entity's ability token where it is printed or attachment-granted, and from
the overlay's temporarily-granted-keywords channel where it is not. That is what
lets gate 2 run at stage one, before any fork machinery exists — and its
per-keyword verdict is what decides whether that machinery gets built at all.

These are the eight keywords whose whole effect lands inside a damage step, so a
combat record contains both the cause and the consequence. Evasion keywords are
deliberately absent: their signal is in playability records, which stage one
does not collect.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from effects.domain.keyword_names import normalize_keyword
from effects.domain.records import RecordKind
from effects.domain.state_snapshot import EntityState, StateSnapshot


class Subject(StrEnum):
    """Whose output field the keyword moves."""

    #: The creature carrying the keyword.
    CARRIER = "carrier"
    #: The creature it is fighting.
    OPPONENT = "opponent"
    #: The player who controls the carrier.
    CONTROLLER = "controller"
    #: The player being attacked.
    DEFENDING_PLAYER = "defending_player"


# ── who is fighting whom ────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class CombatParticipant:
    """One creature's place in a combat, as the table's predicates read it.

    ``opponents`` are the creatures the carrier trades damage with — an
    attacker's blockers, or the attackers a blocker is blocking. An unblocked
    attacker has none, which is the difference between "deals damage to a
    creature" (wither) and "deals damage" (lifelink).

    ``defending_player`` is the player the **attack** is aimed at, and so exists
    for an attacker only. A blocker's combat damage lands on the attacker it
    blocks and never on a player, so it has none — reading its own controller
    there would hand trample and infect a player whose life and poison their
    carrier cannot touch.

    ``combat_substep`` is the damage step the record covers — ``first_strike``
    or ``regular`` — because a record holds one step's events, and the
    first-strike rows are observable in one of the two only.
    """

    carrier: EntityState
    opponents: tuple[EntityState, ...]
    controller: str
    defending_player: str | None
    combat_substep: str | None = None


#: ``EntityState -> the keywords it carries``, spelled the way this table does.
#: Passed in rather than read here because resolving a printed keyword means
#: joining a provenance key to its sidecar line, which is infrastructure.
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
    opponents = tuple(
        entity for entity in (state.entity(i) for i in opponent_ids)
        if entity is not None
    )
    return CombatParticipant(
        carrier=carrier,
        opponents=opponents,
        controller=carrier.controller,
        defending_player=(
            _defending_player(state, carrier)
            if combat.attacking is not None else None
        ),
        combat_substep=state.global_.combat_substep,
    )


def _defending_player(
    state: StateSnapshot, carrier: EntityState,
) -> str | None:
    """Which player the carrier is attacking.

    ``CombatStatus.attacking`` holds Forge's rendering of the *defender object*
    rather than a player id — a player's display name, or a planeswalker's or a
    battle's — so the id is recovered from the board instead: in a two-player
    game the attack is aimed at the one player who is not the attacker's
    controller.

    An attack whose defender is an **entity on the board** has no defending
    player at all. Trample over a planeswalker's blockers spills onto the
    planeswalker and not onto its controller's life total, and infect poisons
    nobody, so reading the controller there would score both against a life
    total the carrier never touched.
    """
    attacking = carrier.combat.attacking if carrier.combat else None
    for player in state.players:
        if player.id == attacking:
            return player.id
    if any(entity.name == attacking for entity in state.entities):
        return None
    others = [p.id for p in state.players if p.id != carrier.controller]
    return others[0] if len(others) == 1 else None


# ── the arithmetic the predicates read ──────────────────────────────────


def _power(entity: EntityState) -> int:
    return entity.pt.total[0] if entity.pt else 0


def _remaining_toughness(entity: EntityState) -> int:
    """Toughness less the damage already marked: what it takes to kill it."""
    return (entity.pt.total[1] if entity.pt else 0) - entity.damage


def _has_opponents(participant: CombatParticipant) -> bool:
    return bool(participant.opponents)


def _is_unblocked_attacker(participant: CombatParticipant) -> bool:
    combat = participant.carrier.combat
    return bool(combat and combat.attacking is not None and not combat.blocked_by)


def _deals_damage(participant: CombatParticipant) -> bool:
    """A creature with no power deals no damage, whatever it is in combat with.

    The first thing every damage-dealing row has to check: a 0/3 wall with
    lifelink gains its controller nothing, so removing lifelink from it should
    move nothing and counting it would only be a coin flip against the keyword.
    """
    return _power(participant.carrier) >= 1


_STRIKES_FIRST = frozenset({"first_strike", "double_strike"})
#: Keywords whose damage lands as -1/-1 counters rather than as damage.
_COUNTER_DAMAGE = frozenset({"wither", "infect"})


def _lone_opponent(participant: CombatParticipant) -> EntityState | None:
    """The one creature the carrier trades damage with, or None.

    Every rule that turns on "does this hit kill" needs it. With two blockers
    Forge's own damage assignment decides who takes what, and a table that
    guessed would qualify combats the keyword provably does not change — a 3/3
    first striker against two 2/2s kills whichever Forge sends the third point
    to, or neither. Trample and indestructible read every opponent instead,
    because a sum does not depend on the split.
    """
    opponents = participant.opponents
    return opponents[0] if len(opponents) == 1 else None


def _kills_outright(
    participant: CombatParticipant, keywords_of: KeywordsOf,
) -> bool:
    """One hit from the carrier would kill the creature it is fighting.

    The carrier's own deathtouch is what makes this more than an arithmetic
    comparison: with it, one point is lethal to anything, so a 1/1 first striker
    kills a 6/6 in the first-strike step.
    """
    opponent = _lone_opponent(participant)
    if opponent is None:
        return False
    power = _power(participant.carrier)
    if "deathtouch" in keywords_of(participant.carrier):
        return power >= 1
    return power >= _remaining_toughness(opponent)


def _survives_a_normal_hit(participant: CombatParticipant) -> bool:
    """The opponent would live through one hit, counting toughness alone.

    Deliberately blind to the carrier's deathtouch: this is what the deathtouch
    row itself asks — "would this creature have survived *normally*" — and a
    version that read the carrier's deathtouch would make that row, whose
    carrier always has deathtouch, never qualify.
    """
    opponent = _lone_opponent(participant)
    return opponent is not None and (
        _remaining_toughness(opponent) > _power(participant.carrier)
    )


# ── when each effect is observable ──────────────────────────────────────


def _always(_participant: CombatParticipant, _keywords_of: KeywordsOf) -> bool:
    return True


def _strikes_first_because_of(keyword: str) -> Applies:
    """The carrier deals damage in this first-strike step because of ``keyword``.

    Three things have to hold. The record is the first-strike step's: in the
    regular step a first striker has already struck and the board carries its
    hit, so nothing it does there turns on the keyword. The carrier has power.
    And no **other** keyword of its own puts it in this step anyway — stripping
    first strike from a creature that also has double strike leaves it
    striking first, and moves nothing.

    What the opponent carries does not enter into it. In this step a creature
    with neither keyword deals no damage, and one with either deals its own
    whatever the carrier does, so in both cases the carrier's hit lands in this
    step with the keyword and not without it.
    """
    others = _STRIKES_FIRST - {keyword}

    def strikes_first(
        participant: CombatParticipant, keywords_of: KeywordsOf,
    ) -> bool:
        return (
            participant.combat_substep == "first_strike"
            and _deals_damage(participant)
            and not (others & keywords_of(participant.carrier))
        )

    return strikes_first


def _first_hit_lands_on_a_creature(keyword: str) -> Applies:
    """The carrier's first-step hit lands on the one creature it is fighting.

    One creature only: with several, Forge's own damage assignment decides who
    takes what, and the opponent effect is read at the first of them.
    """
    strikes_first = _strikes_first_because_of(keyword)

    def applies(participant: CombatParticipant, keywords_of: KeywordsOf) -> bool:
        return (
            strikes_first(participant, keywords_of)
            and _lone_opponent(participant) is not None
        )

    return applies


def _first_hit_kills(keyword: str) -> Applies:
    """...and is lethal to it, by remaining toughness or by deathtouch.

    A hit that does not kill still marks damage, which the damage effect reads,
    but only a lethal one moves the opponent's death in this step.
    """
    strikes_first = _strikes_first_because_of(keyword)

    def applies(participant: CombatParticipant, keywords_of: KeywordsOf) -> bool:
        return (
            strikes_first(participant, keywords_of)
            and _kills_outright(participant, keywords_of)
        )

    return applies


def _first_hit_lands_on_the_defending_player(keyword: str) -> Applies:
    """An unblocked carrier's first-step hit lands on the defending player."""
    strikes_first = _strikes_first_because_of(keyword)

    def applies(participant: CombatParticipant, keywords_of: KeywordsOf) -> bool:
        return (
            strikes_first(participant, keywords_of)
            and _is_unblocked_attacker(participant)
        )

    return applies


def _an_opponent_survives_a_normal_hit(
    participant: CombatParticipant, _keywords_of: KeywordsOf,
) -> bool:
    return _deals_damage(participant) and _survives_a_normal_hit(participant)


def _damages_a_creature(
    participant: CombatParticipant, _keywords_of: KeywordsOf,
) -> bool:
    return _deals_damage(participant) and _has_opponents(participant)


def _damages_anything(
    participant: CombatParticipant, _keywords_of: KeywordsOf,
) -> bool:
    return _deals_damage(participant) and (
        _has_opponents(participant) or _is_unblocked_attacker(participant)
    )


def _damages_the_defending_player(
    participant: CombatParticipant, _keywords_of: KeywordsOf,
) -> bool:
    return _deals_damage(participant) and _is_unblocked_attacker(participant)


def _runs_over_its_blockers(
    participant: CombatParticipant, _keywords_of: KeywordsOf,
) -> bool:
    """A blocked attacker with more power than its blockers can soak."""
    combat = participant.carrier.combat
    if not combat or combat.attacking is None or not participant.opponents:
        return False
    soaked = sum(
        max(0, _remaining_toughness(blocker))
        for blocker in participant.opponents
    )
    return soaked < _power(participant.carrier)


def _would_be_destroyed(
    participant: CombatParticipant, keywords_of: KeywordsOf,
) -> bool:
    """The damage coming in would otherwise be lethal.

    Two ways for it to be. Enough of it — summed over every opponent, since a
    sum is the same however Forge splits the carrier's own damage back — or any
    of it from a creature with deathtouch, which destroys whatever it damages.
    """
    if not _has_opponents(participant):
        return False
    hitters = [o for o in participant.opponents if _power(o) >= 1]
    if any(_COUNTER_DAMAGE & keywords_of(opponent) for opponent in hitters):
        # Wither and infect deal their damage as -1/-1 counters, which shrink
        # an indestructible creature to death without destroying it, so the
        # keyword saves nothing the record could show.
        return False
    if any("deathtouch" in keywords_of(opponent) for opponent in hitters):
        return True
    incoming = sum(_power(opponent) for opponent in participant.opponents)
    return incoming >= _remaining_toughness(participant.carrier)


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
    #: When this effect is observable on a given combat. The default is
    #: "always", which only the subject's existence then narrows.
    applies: Applies = _always

    def __post_init__(self) -> None:
        if self.direction not in (-1, 1):
            raise ValueError(
                f"direction must be -1 or 1, got {self.direction} for "
                f"{self.field}"
            )

    @property
    def label(self) -> str:
        """How this effect is named in a per-field breakdown.

        Prefixed by its subject because one field can be moved on either side
        of a combat: a breakdown that called both the carrier's and the
        opponent's ``damage_taken`` by the field name alone could not say which
        one it was reporting.
        """
        name = self.field if self.category is None else (
            f"{self.field}/{self.category}"
        )
        return f"{self.subject.value}.{name}"

    def observable(
        self, participant: CombatParticipant, keywords_of: KeywordsOf,
    ) -> bool:
        """Whether this combat can show this effect at all.

        The one definition of it: the row's ``qualifies`` counts effects through
        this, and the scorer skips them through this, so the population counted
        and the population scored cannot come apart.
        """
        return (
            self.applies(participant, keywords_of)
            and subject_id(self, participant) is not None
        )


@dataclass(frozen=True, slots=True)
class DamageStepKeyword:
    """One row of gate 2's table."""

    keyword: str
    #: What makes a combat a qualifying observation, in words. A restatement of
    #: what the effects' ``applies`` predicates already say, for a reader.
    qualifies_when: str
    effects: tuple[KeywordEffect, ...]

    @property
    def fields(self) -> tuple[str, ...]:
        """The head fields this row reads, each named once."""
        return tuple(dict.fromkeys(effect.field for effect in self.effects))

    def qualifies(
        self, participant: CombatParticipant, keywords_of: KeywordsOf,
    ) -> bool:
        """Whether this combat is an observation of this keyword.

        Derived from the effects rather than written beside them: a row that
        qualified a combat none of its effects could be read on would count
        toward the 200-record threshold and then score as a disagreement every
        time, which is a canary that sings whatever the model does.
        """
        return any(
            effect.observable(participant, keywords_of)
            for effect in self.effects
        )


def _first_strike_step_effects(keyword: str) -> tuple[KeywordEffect, ...]:
    """What first strike and double strike change inside one combat record.

    The same three effects for both rows, because within the first-strike
    step's record the two keywords do the same thing: the carrier deals its
    damage in this step with the keyword and none without it. Double strike's
    second hit is the regular step's, which neither row reads — that record
    starts from a board already carrying the first hit, whose damage and deaths
    the snapshot shows rather than the payload.
    """
    return (
        # Its hit lands on the creature it is fighting.
        KeywordEffect(
            "damage_taken", Subject.OPPONENT, direction=1,
            applies=_first_hit_lands_on_a_creature(keyword),
        ),
        # A lethal one kills it in this step.
        KeywordEffect(
            "zone_outcome", Subject.OPPONENT, direction=1, category="died",
            applies=_first_hit_kills(keyword),
        ),
        # Unblocked, the defending player loses the life in this step.
        KeywordEffect(
            "life_delta", Subject.DEFENDING_PLAYER, direction=-1,
            applies=_first_hit_lands_on_the_defending_player(keyword),
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
            "the carrier deals combat damage to exactly one creature, and that "
            "creature would survive that much damage normally"
        ),
        effects=(
            KeywordEffect(
                "zone_outcome", Subject.OPPONENT, direction=1, category="died",
                applies=_an_opponent_survives_a_normal_hit,
            ),
        ),
    ),
    DamageStepKeyword(
        keyword="lifelink",
        qualifies_when="the carrier deals combat damage to a creature or player",
        effects=(
            KeywordEffect(
                "life_delta", Subject.CONTROLLER, direction=1,
                applies=_damages_anything,
            ),
        ),
    ),
    DamageStepKeyword(
        keyword="trample",
        qualifies_when=(
            "the carrier is blocked by creatures with less remaining toughness "
            "between them than the carrier's power"
        ),
        effects=(
            # The excess runs over: the defender loses more life.
            KeywordEffect(
                "life_delta", Subject.DEFENDING_PLAYER, direction=-1,
                applies=_runs_over_its_blockers,
            ),
        ),
    ),
    DamageStepKeyword(
        keyword="indestructible",
        qualifies_when=(
            "the carrier takes combat damage at least equal to its remaining "
            "toughness, or any of it from a creature with deathtouch, and none "
            "of it from a creature with wither or infect"
        ),
        effects=(
            KeywordEffect(
                "zone_outcome", Subject.CARRIER, direction=-1, category="died",
                applies=_would_be_destroyed,
            ),
        ),
    ),
    DamageStepKeyword(
        keyword="wither",
        qualifies_when="the carrier deals combat damage to a creature",
        effects=(
            # Damage arrives as counters instead of as damage.
            KeywordEffect(
                "counters_delta_m1m1", Subject.OPPONENT, direction=1,
                applies=_damages_a_creature,
            ),
            KeywordEffect(
                "damage_taken", Subject.OPPONENT, direction=-1,
                applies=_damages_a_creature,
            ),
        ),
    ),
    DamageStepKeyword(
        keyword="infect",
        qualifies_when="the carrier deals combat damage to a creature or player",
        effects=(
            KeywordEffect(
                "counters_delta_m1m1", Subject.OPPONENT, direction=1,
                applies=_damages_a_creature,
            ),
            KeywordEffect(
                "damage_taken", Subject.OPPONENT, direction=-1,
                applies=_damages_a_creature,
            ),
            KeywordEffect(
                "poison_delta", Subject.DEFENDING_PLAYER, direction=1,
                applies=_damages_the_defending_player,
            ),
            # The player takes poison *instead of* life loss, so its life change
            # is less negative with infect than without it.
            KeywordEffect(
                "life_delta", Subject.DEFENDING_PLAYER, direction=1,
                applies=_damages_the_defending_player,
            ),
        ),
    ),
)

KEYWORDS_BY_NAME: dict[str, DamageStepKeyword] = {
    row.keyword: row for row in DAMAGE_STEP_KEYWORDS
}

#: Gate 2's thresholds (FR-119). Both are contract.
MIN_QUALIFYING_RECORDS = 200
MIN_DIRECTION_AGREEMENT = 0.70


def subject_id(
    effect: KeywordEffect, participant: CombatParticipant,
) -> str | None:
    """The entity or player whose output field this effect moves.

    An ``OPPONENT`` effect is read at the **first** creature the carrier is
    fighting rather than at all of them. One observation per (record, carrier,
    keyword) is what makes "qualifying records" a count of records; reading
    every blocker would turn a three-blocker combat into three observations and
    weight it triple, for a rule that is the same rule each time.
    """
    match effect.subject:
        case Subject.CARRIER:
            return participant.carrier.id
        case Subject.OPPONENT:
            return participant.opponents[0].id if participant.opponents else None
        case Subject.CONTROLLER:
            return participant.controller
        case Subject.DEFENDING_PLAYER:
            return participant.defending_player
    raise ValueError(f"unhandled subject {effect.subject}")


def keyword_of_line(line) -> str | None:
    """The keyword a sidecar line *is*, or None if the line is not one.

    A printed keyword converts to a ``static`` line whose script text is the
    keyword's display name — White Knight's is exactly ``First Strike``. Read
    in the gate's spelling, so ``first_strike`` matches.

    It lives here, beside the table, rather than in the evaluator because the
    batcher needs it too: gate 2's perturbation drops an entity's ability slot
    when that slot's line *is* the keyword being stripped, and an application
    module importing the evaluator to ask would be a cycle.
    """
    script = getattr(line, "script_text", None)
    if not script or "$" in script or "|" in script:
        return None
    return normalize_keyword(script)


# ── who is observed ─────────────────────────────────────────────────────
#
# Shared by gate 2 and by build-corpus's game-disjoint placement (FR-045),
# which is why it lives here rather than beside the gate: the gate imports
# torch at module top, and a corpus survey worker must not.


class KeywordResolver:
    """An entity's keywords, from both channels, memoized by provenance key.

    Both channels have to be looked at. An entity's printed and
    attachment-granted keywords reach the model as ability tokens, and only a
    keyword granted until end of turn appears as a bare string in the overlay —
    so reading the overlay alone sees the rare case and misses every creature
    that printed the keyword, which is the common one.

    The memo is what makes that affordable: a corpus repeats the same few
    thousand cards across millions of combat records, and resolving each key
    once turns the walk into a dict hit.
    """

    def __init__(self, sidecars=None) -> None:
        self._sidecars = sidecars
        self._by_key: dict[object, str | None] = {}

    def _keyword_for(self, key) -> str | None:
        if key not in self._by_key:
            keyword = None
            try:
                line = self._sidecars.line_for(key)
            except (KeyError, FileNotFoundError):
                line = None
            if line is not None:
                keyword = keyword_of_line(line)
            self._by_key[key] = keyword
        return self._by_key[key]

    def keywords_of(self, entity) -> set[str]:
        # Already in this package's spelling: ``record_io`` runs Forge's
        # keyword strings through ``normalize_keyword`` at parse time.
        found = set(entity.granted_temporary.keywords)
        if self._sidecars is None:
            return found
        for key in (*entity.printed, *entity.granted_attached):
            keyword = self._keyword_for(key)
            if keyword is not None:
                found.add(keyword)
        return found


def qualifying_observations(
    record, rows: tuple[DamageStepKeyword, ...], resolver: KeywordResolver,
) -> list[tuple[DamageStepKeyword, CombatParticipant]]:
    """Every ``(row, carrier)`` in one combat record the gate can observe.

    The whole condition is the row's own ``qualifies`` — at least one effect
    that applies and has a subject on the board — and it is a board fact rather
    than a model fact, so the counter and the scorer reach the same answer by
    asking the same question rather than by two rules kept in step.
    """
    if record.kind is not RecordKind.COMBAT:
        return []
    found: list[tuple[DamageStepKeyword, CombatParticipant]] = []
    wanted = {row.keyword for row in rows}
    for entity in record.state.entities:
        if entity.combat is None:
            continue
        carried = resolver.keywords_of(entity) & wanted
        if not carried:
            continue
        participant = combat_participant(record.state, entity)
        if participant is None:
            continue
        # In table order, not set-iteration order: which observations share a
        # batch would otherwise depend on the hash seed.
        for row in rows:
            if row.keyword not in carried:
                continue
            if row.qualifies(participant, resolver.keywords_of):
                found.append((row, participant))
    return found

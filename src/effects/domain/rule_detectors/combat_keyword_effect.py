"""Detector family ``combat_keyword_effect``: a combat keyword outside gate 2's table.

Gate 2's eight damage-step keywords are ``keyword_combat``'s, and their table
stays what it is, since it is how the generations compare; this family reads
the other keywords whose effect a damage step's record shows. Each keyword
fixes the situation it judges and what tells its two sides apart, read off a
real combat (a damage-step probe strips only gate 2's keywords, so a probe
shows nothing here):

- ``toxic``: an attacker carrying it, with power, attacking a player, in the
  damage step it deals its damage in (510.4: one without first or double
  strike deals none in the first-strike step, one with first strike alone none
  in the regular step; so for ``poisonous`` and
  ``trample_over_planeswalkers``). It is
  on its ``fires`` side when its combat damage reaches that player and on its
  near miss when none does (blocked, prevented); the outcome entity is that
  player, observed by a ``poison_change`` at them in the step's own record
  (unreadable on the near miss when another source's combat damage also
  reached them, since a poison change names no creature).
- ``poisonous``: the same situation and sides; the poison comes from a
  trigger whose resolution is another record, so the outcome is unreadable.
- ``vigilance``: an attacking creature carrying the keyword, ``fires``, with
  no near miss: an attacker without it says nothing of vigilance and is the
  ``fires`` side of ``fires_when: absent`` (508.1f: attacking taps it), whose
  near miss is an attacker carrying it; the outcome entity is the attacker,
  observed untapped.
- ``trample_over_planeswalkers``: an attacker with power attacking a
  planeswalker it is not blocked from (``defender: planeswalker``, its power
  above the planeswalker's loyalty, so there is excess to assign) or whose
  attacked planeswalker is gone (``defender: gone``), ``fires`` for one
  carrying the keyword and the near miss for one without; the outcome entity
  is the planeswalker's controller, observed by the attacker's combat damage
  reaching a player.
- ``absorb``: a carrier some combat damage was aimed at in the step (dealt or
  prevented at it); ``fires`` only, since no probe strips absorb; the outcome
  entity is the carrier, observed by a ``damage_prevented`` at it.
- ``first_strike``: a carrier with first strike and without double strike in
  the regular damage step, with power and something to hit (unblocked
  attacking a player, or fighting a creature). A creature that had first
  strike when the first-strike step began dealt its damage there, so one
  dealing damage in the regular step gained it afterwards (702.7c), its
  ``fires`` side; one dealing none is the near miss. The outcome entities are
  what it hits, or would; the side is read off the outcome, so it is never
  reviewed.

A lost or gained double strike between the two steps (702.4c-d), or a first
strike lost after the first-strike step, cannot be told from one record: no
record says what a creature dealt in the other step.

Parameters:

- ``keyword`` (required): one of the six above;
- ``fires_when``: ``carried`` (default) or ``absent``, for ``vigilance`` and
  ``trample_over_planeswalkers``, whose sides are carriers against
  non-carriers: ``absent`` puts the creatures without the keyword on the
  ``fires`` side, its outcome being the keyword's effect not happening (an
  attacker tapped by attacking, 508.1f; one dealing no damage once its
  planeswalker is gone, 702.19f);
- ``defender``: ``planeswalker`` (default) or ``gone``, with
  ``trample_over_planeswalkers`` only;
- ``tapped``: true, with ``vigilance`` only: the carriers tapped after the
  declaration, which stay attacking (506.4b); ``fires`` only, observed by
  the carrier dealing combat damage, the outcome entities what it damaged.

The condition entity is always the carrier, or the creature without the
keyword on the sides that judge one. Keywords are read as ``keyword_combat``
reads them (``KeywordResolver``).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from effects.domain.damage_step_keywords import (
    CombatBoard,
    CombatParticipant,
    defender_ref,
)
from effects.domain.event_schema import EventType, damage_source
from effects.domain.provenance import SidecarLookup
from effects.domain.records import EffectRecord, RecordKind
from effects.domain.rule_catalogue import Polarity
from effects.domain.rule_detectors import EventlessFamily, FamilyMatch, Matcher, register
from effects.domain.rule_detectors import declared_values as dv
from effects.domain.rule_detectors.keyword_reading import KeywordReading
from effects.domain.rule_outcomes import strikes_in_step
from effects.domain.state_snapshot import (
    CombatSubstep,
    EntityState,
    StateSnapshot,
    normalize_counter_type,
)

_CARRIED = "carried"
_ABSENT = "absent"
_PLANESWALKER = "planeswalker"
_GONE = "gone"
_LOYALTY = normalize_counter_type("LOYALTY")


@dataclass(frozen=True, slots=True)
class _Finding:
    """One creature's side of the rule in one record."""

    carrier: str
    #: The keyword's effect side: True where the situation's ``fires`` side
    #: holds before ``fires_when`` flips it.
    effect_side: bool
    outcome: tuple[str, ...]
    #: Whether the keyword's effect happened; None where unreadable.
    effect_seen: bool | None


@dataclass(frozen=True, slots=True)
class _Combat:
    """One combat record as the keyword shapes read it."""

    record: EffectRecord
    keywords_of: Callable[[EntityState], set[str]]

    @property
    def state(self) -> StateSnapshot:
        return self.record.state

    def combatants(self) -> Iterator[tuple[EntityState, CombatParticipant]]:
        board = CombatBoard.of(self.record, self.keywords_of)
        for entity in self.state.entities:
            if entity.combat is None:
                continue
            participant = board.participant(entity)
            if participant is not None:
                yield entity, participant

    def attackers(self) -> Iterator[tuple[EntityState, CombatParticipant]]:
        return ((e, p) for e, p in self.combatants() if p.is_attacking)

    def events(self, kind: EventType):
        return (event for event in self.record.payload.events if event.type is kind)

    def strikes(self, entity: EntityState) -> bool:
        """Whether ``entity`` deals combat damage in this record's step (510.4,
        702.7b, 702.4b): a creature striking in the other step shows neither
        side of a rule about its damage here."""
        step = self.state.global_.combat_substep or CombatSubstep.REGULAR
        return strikes_in_step(self.keywords_of(entity), str(step))

    def damage_from(self, source: str) -> tuple[str, ...]:
        """What ``source`` dealt combat damage to in this step."""
        return tuple(dict.fromkeys(
            subject for event in self.events(EventType.DAMAGE_DEALT)
            if event.params.get("combat") and damage_source(event) == source
            for subject in event.subjects))

    def damage_to(self, subject: str) -> tuple[str, ...]:
        """Every source whose combat damage reached ``subject`` in this step."""
        return tuple(dict.fromkeys(
            damage_source(event) for event in self.events(EventType.DAMAGE_DEALT)
            if event.params.get("combat") and subject in event.subjects))


def _players(found: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(e for e in found if e.startswith("P"))


# ── the declaration ─────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class _Declared:
    """What a keyword shape reads off its declaration, read once."""

    keyword: str
    #: ``fires_when: absent``: the creatures without the keyword on the ``fires`` side.
    flip: bool
    defender: str
    tapped: bool

    @classmethod
    def of(cls, params: Mapping[str, Any]) -> _Declared:
        return cls(params.get("keyword"), params.get("fires_when") == _ABSENT,
                   params.get("defender", _PLANESWALKER), bool(params.get("tapped")))


# ── the keyword shapes ──────────────────────────────────────────────────


def _poison(combat: _Combat, keyword: str, readable: bool) -> Iterator[_Finding]:
    for entity, player in _poisoning(combat, keyword):
        hit = _players(combat.damage_from(entity.id))
        seen = _poison_seen(combat, entity, player, hit) if readable else None
        yield _Finding(entity.id, bool(hit), hit or (player,), seen)


def _poisoning(combat: _Combat, keyword: str) -> Iterator[tuple[EntityState, str]]:
    """Each attacker that would poison the player it attacks, with that player."""
    for entity, participant in combat.attackers():
        player = participant.defending_player
        if player is not None and _deals_poison(combat, entity, keyword):
            yield entity, player


def _deals_poison(combat: _Combat, entity: EntityState, keyword: str) -> bool:
    """With power, carrying the keyword and striking in this step."""
    if not entity.power >= 1:
        return False
    return keyword in combat.keywords_of(entity) and combat.strikes(entity)


def _poison_seen(combat: _Combat, entity: EntityState, player: str,
                 hit: tuple[str, ...]) -> bool | None:
    """Whether a poison change reached ``player``; None where ``entity`` hit
    nothing and another source's combat damage reached them."""
    if not (hit or set(combat.damage_to(player)) <= {entity.id}):
        return None
    return any(player in event.subjects
               for event in combat.events(EventType.POISON_CHANGE)
               if event.params.get("delta", 0) > 0)


def _toxic(combat: _Combat, declared: _Declared) -> Iterator[_Finding]:
    return _poison(combat, declared.keyword, readable=True)


def _poisonous(combat: _Combat, declared: _Declared) -> Iterator[_Finding]:
    return _poison(combat, declared.keyword, readable=False)


def _vigilance(combat: _Combat, declared: _Declared) -> Iterator[_Finding]:
    if declared.tapped:
        return _tapped_carriers(combat, declared.keyword)
    return _vigilant(combat, declared)


def _vigilant(combat: _Combat, declared: _Declared) -> Iterator[_Finding]:
    # An attacker without vigilance is the fires side of attacking tapping it
    # (508.1f), and says nothing of vigilance: no near miss of the keyword.
    judged = (True, False) if declared.flip else (True,)
    for entity, _participant in combat.attackers():
        carried = declared.keyword in combat.keywords_of(entity)
        if carried in judged:
            yield _Finding(entity.id, carried, (entity.id,), not entity.tapped)


def _tapped_carriers(combat: _Combat, keyword: str) -> Iterator[_Finding]:
    """The carriers tapped after the declaration, observed dealing damage (506.4b)."""
    for entity, _participant in combat.attackers():
        if keyword in combat.keywords_of(entity) and entity.tapped:
            hit = combat.damage_from(entity.id)
            yield _Finding(entity.id, True, hit,
                           bool(hit) if entity.power >= 1 else None)


def _trample_over_planeswalkers(combat: _Combat, declared: _Declared) -> Iterator[_Finding]:
    reach = _REACHED[declared.defender]
    for entity, participant in _striking_attackers(combat):
        outcome = reach(combat, entity, participant)
        if outcome is not None:
            yield _Finding(entity.id, declared.keyword in combat.keywords_of(entity), outcome,
                           bool(_players(combat.damage_from(entity.id))))


def _striking_attackers(combat: _Combat) -> Iterator[tuple[EntityState, CombatParticipant]]:
    """The attackers with power dealing combat damage in this step."""
    for entity, participant in combat.attackers():
        if entity.power >= 1 and combat.strikes(entity):
            yield entity, participant


def _over_planeswalker(combat: _Combat, entity: EntityState,
                       participant: CombatParticipant) -> tuple[str, ...] | None:
    """``defender: planeswalker``: the controller of the planeswalker the
    attacker is unblocked against with excess to assign; None for any other
    attacker."""
    attacked = _attacked_card(combat.state, entity)
    if attacked is None or not _excess_over(attacked, entity, participant):
        return None
    return _outcome_player(attacked.controller)


def _excess_over(attacked: EntityState, entity: EntityState,
                 participant: CombatParticipant) -> bool:
    if "planeswalker" not in {t.lower() for t in attacked.types}:
        return False
    return participant.is_unblocked and entity.power > _loyalty(attacked)


def _past_gone(combat: _Combat, entity: EntityState,
               participant: CombatParticipant) -> tuple[str, ...] | None:
    """``defender: gone``: the other player, where the attacked permanent left."""
    if not _defender_gone(combat.state, entity):
        return None
    return _outcome_player(_other_player(combat.state, entity))


def _outcome_player(player: str | None) -> tuple[str, ...]:
    return (player,) if player else ()


_Reach = Callable[[_Combat, EntityState, CombatParticipant], "tuple[str, ...] | None"]

#: ``trample_over_planeswalkers``' ``defender``: where an attacker's excess goes.
_REACHED: dict[str, _Reach] = {_PLANESWALKER: _over_planeswalker, _GONE: _past_gone}


def _absorb(combat: _Combat, declared: _Declared) -> Iterator[_Finding]:
    for entity, _participant in combat.combatants():
        if declared.keyword not in combat.keywords_of(entity):
            continue
        prevented = any(entity.id in event.subjects
                        for event in combat.events(EventType.DAMAGE_PREVENTED))
        if prevented or combat.damage_to(entity.id):
            yield _Finding(entity.id, True, (entity.id,), prevented)


def _first_strike(combat: _Combat, declared: _Declared) -> Iterator[_Finding]:
    if combat.state.global_.combat_substep != CombatSubstep.REGULAR:
        return
    for entity, participant in combat.combatants():
        aimed = (_aimed(participant) if _first_strike_alone(combat, entity, declared.keyword)
                 else ())
        if aimed:
            hit = combat.damage_from(entity.id)
            yield _Finding(entity.id, bool(hit), hit or aimed, bool(hit))


def _first_strike_alone(combat: _Combat, entity: EntityState, keyword: str) -> bool:
    """Carrying first strike without double strike, with power."""
    carried = combat.keywords_of(entity)
    if keyword not in carried or "double_strike" in carried:
        return False
    return entity.power >= 1


def _aimed(participant: CombatParticipant) -> tuple[str, ...]:
    """What the creature would hit: the creatures it fights, else the player
    an unblocked attacker attacks."""
    fought = tuple(o.id for o in participant.opponents)
    if fought:
        return fought
    player = participant.defending_player
    return (player,) if participant.is_unblocked and player else ()


_Shape = Callable[[_Combat, _Declared], Iterator[_Finding]]

_SHAPES: dict[str, _Shape] = {
    "toxic": _toxic,
    "poisonous": _poisonous,
    "vigilance": _vigilance,
    "trample_over_planeswalkers": _trample_over_planeswalkers,
    "absorb": _absorb,
    "first_strike": _first_strike,
}

#: The keywords whose two sides are carriers against non-carriers.
_FLIPPABLE = frozenset({"vigilance", "trample_over_planeswalkers"})


# ── who is attacked ─────────────────────────────────────────────────────


def _attacked_card(state: StateSnapshot, attacker: EntityState) -> EntityState | None:
    """The permanent ``attacker`` attacks: the entity its defender's ref
    names, None for a player or a permanent the board does not hold."""
    defender = defender_ref(attacker)
    return state.entity(defender) if defender else None


def _defender_gone(state: StateSnapshot, attacker: EntityState) -> bool:
    """Whether the permanent ``attacker`` attacked has left: its defender's
    ref names no player and no card on the battlefield (the placeholder
    ``E-1`` a departed planeswalker or battle leaves, 506.4c, among them)."""
    defender = defender_ref(attacker)
    if not defender or any(p.id == defender for p in state.players):
        return False
    card = state.entity(defender)
    return card is None or card.zone != "battlefield"


def _other_player(state: StateSnapshot, attacker: EntityState) -> str | None:
    others = [p.id for p in state.players if p.id != attacker.controller]
    return others[0] if len(others) == 1 else None


def _loyalty(card: EntityState) -> int:
    return sum(n for name, n in card.counters.items() if normalize_counter_type(name) == _LOYALTY)


# ── the sides ───────────────────────────────────────────────────────────


def _side_match(found: list[_Finding], flip: bool) -> FamilyMatch | None:
    """The ``fires`` side where any finding is on it, else the near miss."""
    for polarity, fires in ((Polarity.FIRES, True), (Polarity.DOES_NOT_FIRE, False)):
        side = [f for f in found if (f.effect_side != flip) is fires]
        if side:
            return _side(polarity, side, flip)
    return None


def _side(polarity: Polarity, side: list[_Finding], flip: bool) -> FamilyMatch:
    seen = [f.effect_seen != flip for f in side if f.effect_seen is not None]
    return FamilyMatch(
        polarity, _observed(seen, polarity is Polarity.FIRES),
        tuple(dict.fromkeys(f.carrier for f in side)),
        tuple(dict.fromkeys(e for f in side for e in f.outcome)))


def _observed(seen: list[bool], fires: bool) -> bool | None:
    """Every creature of the side must show the rule's outcome for a
    ``fires``; any one showing it on the near miss is a disagreement."""
    if not seen:
        return None
    return all(seen) if fires else any(seen)


def _by_entity_id(keywords_of: Callable[[EntityState], set[str]]
                  ) -> Callable[[EntityState], set[str]]:
    """``keywords_of`` read once per entity id of one record."""
    memo: dict[str, set[str]] = {}

    def memoized(entity: EntityState) -> set[str]:
        if entity.id not in memo:
            memo[entity.id] = keywords_of(entity)
        return memo[entity.id]
    return memoized


@dataclass(frozen=True, slots=True, eq=False)
class CombatKeywordMatcher:
    """A compiled ``combat_keyword_effect`` declaration."""

    shape: _Shape
    declared: _Declared
    keywords: KeywordReading = field(default_factory=KeywordReading)

    @classmethod
    def of(cls, params: Mapping[str, Any]) -> CombatKeywordMatcher:
        return cls(_SHAPES[params.get("keyword")], _Declared.of(params))

    def __call__(self, record: EffectRecord, lines: SidecarLookup | None) -> FamilyMatch | None:
        if record.kind is not RecordKind.COMBAT or record.is_probe:
            return None
        combat = _Combat(record, _by_entity_id(self.keywords.of(lines).keywords_of))
        return _side_match(list(self.shape(combat, self.declared)), self.declared.flip)


# ── the family ──────────────────────────────────────────────────────────


def _check_keyword(keyword: Any) -> None:
    if keyword not in _SHAPES:
        raise ValueError(
            f"combat_keyword_effect: {keyword!r} is not one of its keywords; "
            f"known: {sorted(_SHAPES)}")


def _fires_when_applies(fires_when: Any, keyword: str) -> bool:
    if fires_when == _ABSENT:
        return keyword in _FLIPPABLE
    return fires_when == _CARRIED


def _defender_applies(params: Mapping[str, Any], keyword: str) -> bool:
    if "defender" not in params:
        return True
    return keyword == "trample_over_planeswalkers" and params["defender"] in (_PLANESWALKER, _GONE)


@dataclass(frozen=True, slots=True)
class CombatKeywordEffect(EventlessFamily):
    id: str = "combat_keyword_effect"
    kinds: frozenset[str] = frozenset({"combat"})
    params: frozenset[str] = frozenset({"keyword", "fires_when", "defender", "tapped"})
    general: frozenset[str] = frozenset({"keyword", "fires_when"})

    def needs(self, params: Mapping[str, Any]) -> frozenset[str]:
        return frozenset()

    def validate(self, params: Mapping[str, Any]) -> None:
        keyword = params.get("keyword")
        _check_keyword(keyword)
        fires_when = params.get("fires_when", _CARRIED)
        if not _fires_when_applies(fires_when, keyword):
            raise ValueError(f"combat_keyword_effect: fires_when {fires_when!r} "
                             f"does not apply to {keyword!r}")
        if not _defender_applies(params, keyword):
            raise ValueError(f"combat_keyword_effect: defender {params['defender']!r} "
                             f"does not apply to {keyword!r}")
        dv.check(self.id, params, {"tapped": dv.boolean})
        if "tapped" in params and keyword != "vigilance":
            raise ValueError(f"combat_keyword_effect: tapped does not apply to {keyword!r}")

    def compile(self, params: Mapping[str, Any]) -> Matcher:
        return CombatKeywordMatcher.of(params)

    def match(self, record: EffectRecord, lines: SidecarLookup | None,
              params: Mapping[str, Any]) -> FamilyMatch | None:
        return self.compile(params)(record, lines)


FAMILY = register(CombatKeywordEffect())

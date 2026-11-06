"""Detector family ``combat_keyword_effect``: a combat keyword outside gate 2's table.

Gate 2's eight damage-step keywords are ``keyword_combat``'s, and their table
stays what it is, since it is how the generations compare; this family reads
the other keywords whose effect a damage step's record shows. Each keyword
fixes the situation it judges and what tells its two sides apart, read off a
real combat (a damage-step probe strips only gate 2's keywords, so a probe
shows nothing here):

- ``toxic``: an attacker carrying it, with power, attacking a player. It is
  on its ``fires`` side when its combat damage reaches that player and on its
  near miss when none does (blocked, prevented); the outcome entity is that
  player, observed by a ``poison_change`` at them in the step's own record
  (unreadable on the near miss when another source's combat damage also
  reached them, since a poison change names no creature).
- ``poisonous``: the same situation and sides; the poison comes from a
  trigger whose resolution is another record, so the outcome is unreadable.
- ``vigilance``: every attacking creature, ``fires`` for one carrying the
  keyword and the near miss for one without; the outcome entity is the
  attacker, observed untapped.
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
reads them (``KeywordResolver``), on gen-2 records as well.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from effects.domain.damage_step_keywords import (
    CombatParticipant,
    KeywordResolver,
    combat_participant,
)
from effects.domain.event_schema import EventType
from effects.domain.provenance import SidecarLookup
from effects.domain.records import EffectRecord, RecordKind
from effects.domain.rule_catalogue import Polarity
from effects.domain.rule_detectors import FamilyMatch, register
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
        for entity in self.state.entities:
            if entity.combat is None:
                continue
            participant = combat_participant(self.state, entity)
            if participant is not None:
                yield entity, participant

    def attackers(self) -> Iterator[tuple[EntityState, CombatParticipant]]:
        return ((e, p) for e, p in self.combatants() if p.is_attacking)

    def events(self, kind: EventType):
        return (event for event in self.record.payload.events if event.type is kind)

    def damage_from(self, source: str) -> tuple[str, ...]:
        """What ``source`` dealt combat damage to in this step."""
        return tuple(dict.fromkeys(
            subject for event in self.events(EventType.DAMAGE_DEALT)
            if event.params.get("combat") and _source(event) == source
            for subject in event.subjects))

    def damage_to(self, subject: str) -> tuple[str, ...]:
        """Every source whose combat damage reached ``subject`` in this step."""
        return tuple(dict.fromkeys(
            _source(event) for event in self.events(EventType.DAMAGE_DEALT)
            if event.params.get("combat") and subject in event.subjects))


def _source(event) -> str | None:
    return event.params.get("source") or event.params.get("cause")


def _players(found: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(e for e in found if e.startswith("P"))


# ── the keyword shapes ──────────────────────────────────────────────────


def _poison(combat: _Combat, params: Mapping[str, Any], keyword: str,
            readable: bool) -> Iterator[_Finding]:
    for entity, participant in combat.attackers():
        player = participant.defending_player
        if player is None or not entity.power >= 1 or keyword not in combat.keywords_of(entity):
            continue
        hit = _players(combat.damage_from(entity.id))
        seen = None
        if readable and (hit or set(combat.damage_to(player)) <= {entity.id}):
            seen = any(player in event.subjects
                       for event in combat.events(EventType.POISON_CHANGE)
                       if event.params.get("delta", 0) > 0)
        yield _Finding(entity.id, bool(hit), hit or (player,), seen)


def _toxic(combat, params, keyword):
    return _poison(combat, params, keyword, readable=True)


def _poisonous(combat, params, keyword):
    return _poison(combat, params, keyword, readable=False)


def _vigilance(combat: _Combat, params: Mapping[str, Any], keyword: str) -> Iterator[_Finding]:
    tapped_only = bool(params.get("tapped"))
    for entity, _participant in combat.attackers():
        carried = keyword in combat.keywords_of(entity)
        if not tapped_only:
            yield _Finding(entity.id, carried, (entity.id,), not entity.tapped)
        elif carried and entity.tapped:
            hit = combat.damage_from(entity.id)
            yield _Finding(entity.id, True, hit,
                           bool(hit) if entity.power >= 1 else None)


def _trample_over_planeswalkers(combat: _Combat, params: Mapping[str, Any],
                                keyword: str) -> Iterator[_Finding]:
    defender = params.get("defender", _PLANESWALKER)
    for entity, participant in combat.attackers():
        if entity.power < 1:
            continue
        attacked = _attacked_card(combat.state, entity)
        if defender == _PLANESWALKER:
            if (attacked is None or "planeswalker" not in {t.lower() for t in attacked.types}
                    or not participant.is_unblocked or entity.power <= _loyalty(attacked)):
                continue
            player = attacked.controller
        else:
            if not _defender_gone(combat.state, entity):
                continue
            player = _other_player(combat.state, entity)
        outcome = (player,) if player else ()
        yield _Finding(entity.id, keyword in combat.keywords_of(entity), outcome,
                       bool(_players(combat.damage_from(entity.id))))


def _absorb(combat: _Combat, params: Mapping[str, Any], keyword: str) -> Iterator[_Finding]:
    for entity, _participant in combat.combatants():
        if keyword not in combat.keywords_of(entity):
            continue
        prevented = any(entity.id in event.subjects
                        for event in combat.events(EventType.DAMAGE_PREVENTED))
        if prevented or combat.damage_to(entity.id):
            yield _Finding(entity.id, True, (entity.id,), prevented)


def _first_strike(combat: _Combat, params: Mapping[str, Any],
                  keyword: str) -> Iterator[_Finding]:
    if combat.state.global_.combat_substep != CombatSubstep.REGULAR:
        return
    for entity, participant in combat.combatants():
        carried = combat.keywords_of(entity)
        if keyword not in carried or "double_strike" in carried or entity.power < 1:
            continue
        aimed = tuple(o.id for o in participant.opponents) or (
            (participant.defending_player,) if participant.is_unblocked
            and participant.defending_player else ())
        if not aimed:
            continue
        hit = combat.damage_from(entity.id)
        yield _Finding(entity.id, bool(hit), hit or aimed, bool(hit))


_Shape = Callable[[_Combat, Mapping[str, Any], str], Iterator[_Finding]]

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
    """The permanent ``attacker`` attacks, from Forge's ``Name (id)`` rendering."""
    attacking = attacker.combat.attacking if attacker.combat else None
    if not attacking or not attacking.endswith(")") or " (" not in attacking:
        return None
    card_id = attacking.rsplit(" (", 1)[1][:-1]
    return state.entity(f"E{card_id}") if card_id.isdigit() else None


def _defender_gone(state: StateSnapshot, attacker: EntityState) -> bool:
    """Whether the permanent ``attacker`` attacked has left: Forge renders
    the defender as ``<Nothing>``, or names a card no longer on the battlefield."""
    attacking = attacker.combat.attacking if attacker.combat else None
    if attacking == "<Nothing>":
        return True
    if not attacking or not attacking.endswith(")") or " (" not in attacking:
        return False
    card = _attacked_card(state, attacker)
    return card is None or card.zone != "battlefield"


def _other_player(state: StateSnapshot, attacker: EntityState) -> str | None:
    others = [p.id for p in state.players if p.id != attacker.controller]
    return others[0] if len(others) == 1 else None


def _loyalty(card: EntityState) -> int:
    return sum(n for name, n in card.counters.items() if normalize_counter_type(name) == _LOYALTY)


# ── the family ──────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class CombatKeywordEffect:
    id: str = "combat_keyword_effect"
    params: frozenset[str] = frozenset({"keyword", "fires_when", "defender", "tapped"})
    general: frozenset[str] = frozenset({"keyword", "fires_when"})

    def needs(self, params: Mapping[str, Any]) -> frozenset[str]:
        return frozenset()

    def match(self, record: EffectRecord, lines: SidecarLookup | None,
              params: Mapping[str, Any]) -> FamilyMatch | None:
        keyword = params.get("keyword")
        shape = _SHAPES.get(keyword)
        if shape is None:
            raise ValueError(
                f"combat_keyword_effect: {keyword!r} is not one of its keywords; "
                f"known: {sorted(_SHAPES)}")
        fires_when = params.get("fires_when", _CARRIED)
        if fires_when not in (_CARRIED, _ABSENT) or (
                fires_when == _ABSENT and keyword not in _FLIPPABLE):
            raise ValueError(f"combat_keyword_effect: fires_when {fires_when!r} "
                             f"does not apply to {keyword!r}")
        if "defender" in params and (keyword != "trample_over_planeswalkers"
                                     or params["defender"] not in (_PLANESWALKER, _GONE)):
            raise ValueError(f"combat_keyword_effect: defender {params['defender']!r} "
                             f"does not apply to {keyword!r}")
        if "tapped" in params and keyword != "vigilance":
            raise ValueError(f"combat_keyword_effect: tapped does not apply to {keyword!r}")
        if record.kind is not RecordKind.COMBAT or record.is_probe:
            return None
        resolver = KeywordResolver(lines)
        memo: dict[str, set[str]] = {}

        def keywords_of(entity: EntityState) -> set[str]:
            if entity.id not in memo:
                memo[entity.id] = resolver.keywords_of(entity)
            return memo[entity.id]

        flip = fires_when == _ABSENT
        found = list(shape(_Combat(record, keywords_of), params, keyword))
        for polarity, fires in ((Polarity.FIRES, True), (Polarity.DOES_NOT_FIRE, False)):
            side = [f for f in found if (f.effect_side != flip) is fires]
            if not side:
                continue
            seen = [f.effect_seen != flip for f in side if f.effect_seen is not None]
            # Every creature of the side must show the rule's outcome for a
            # ``fires``; any one showing it on the near miss is a disagreement.
            observed = (all(seen) if fires else any(seen)) if seen else None
            return FamilyMatch(
                polarity, observed,
                tuple(dict.fromkeys(f.carrier for f in side)),
                tuple(dict.fromkeys(e for f in side for e in f.outcome)))
        return None


FAMILY = register(CombatKeywordEffect())

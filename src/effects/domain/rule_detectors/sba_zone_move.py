"""Detector family ``sba_zone_move``: a permanent put into the graveyard by the state-based check.

Rule 704.5 lists the state-based actions that move a permanent from the
battlefield to the graveyard. The record files every such move made in the
check after a resolution in that resolution's effect half, as a
``zone_change`` attributed ``unresolved``. A combat damage step's record
holds its own check the same way (its deaths, attributed ``unresolved``), and the
family reads it too; a damage-step probe, whose snapshot still shows the
keyword it stripped, it leaves out. Indestructible is read off the snapshot
and off the keywords the record itself grants.

Parameter ``check`` names which action, each read off the board the record
shows (the snapshot before the resolution, plus the resolution's own events):

=====================  ======================================================
``toughness_zero``     704.5f: a creature the resolution lowered, or one a
                       spell's or activation's board already shows at
                       toughness 0, whose toughness after the P/T changes
                       is 0 or less; the near miss is one it lowered that
                       keeps 1 or more
``lethal_damage``      704.5g: a creature the resolution dealt damage to, by
                       whether the damage marked reaches its toughness; damage
                       from a wither or infect source marks none (-1/-1
                       counters, 704.5f's) and is left out
``deathtouch_damage``  704.5h: a creature dealt non-lethal damage by a source
                       with deathtouch; the near miss is non-lethal damage
                       from a source without it
``loyalty_zero``       704.5i: a planeswalker the resolution removed loyalty
                       counters from, by whether none remain
``defense_zero``       704.5v: a battle the resolution removed defense
                       counters from, by whether none remain
``legend_rule``        704.5j: two or more legendary permanents with one name
                       under one controller, one of them touched by the
                       record (a pair already standing, as under Mirror
                       Gallery, shows no check); the near miss is a legendary
                       permanent entering with no such twin
``world_rule``         704.5k: two or more world permanents; the near miss is
                       a world permanent entering alone
``aura_illegal``       704.5m: an Aura on the battlefield whose host left it
                       in the record, or gained protection from one of the
                       Aura's colours; the near miss is an Aura whose host the
                       record touched and which stayed legal
=====================  ======================================================

Two parameters narrow a check: ``marked_before`` (``lethal_damage`` only)
keeps the creatures the board already shows damage marked on, the damage
accumulating over the turn (120.6); ``aura_cause`` (``aura_illegal`` only),
``left`` or ``protection``, keeps the Auras whose host left, or gained a
protection, in the record.

The polarity is read from the board, ``observed`` from the record's own move,
so a record where Forge disagrees with the rule is the one
:attr:`RuleTag.needs_review` flags. Left out of every check, because the
record cannot say which rule moved them:

- a permanent the record shows no change of the check's kind for: a death
  alone names no cause, since a sacrifice, a death trigger's damage and a
  static's -X/-X kill are attributed ``unresolved`` too, and none writes a
  P/T change for a creature the check removed;
- a permanent the resolution moved itself (an event attributed to its chain);
- an Aura that is the record's source or target: its move is the ability's
  own (a sacrifice, a destroy), which may be attributed ``unresolved``; a
  bestowed Aura, which becomes a creature when its host leaves (702.103e);
  and an Aura whose host stops being a creature or gains a protection other
  than from a colour, against which the record does not say what the Aura
  may enchant;
- in ``toughness_zero``'s near miss, a creature whose marked damage the
  lowered toughness makes lethal: 704.5g's;
- an indestructible creature in the damage checks, which destroy, and in
  ``lethal_damage`` a creature dealt non-lethal damage by a deathtouch
  source, which is ``deathtouch_damage``'s;
- in the legend and world rules, a lone entering permanent at toughness 0,
  whose death is 704.5f's;
- a creature that stops being a creature in the record: its death is another
  rule's.

The record's own move is one from the battlefield to the graveyard, or to
exile where a replacement sends the dying permanent there instead
(Disintegrate, Rest in Peace), attributed ``unresolved``, and not followed by
its return to the battlefield (a flicker).

Two records show no check at all: the game-end record, which holds the
game's outcome alone on a board read after it, and a record whose changes end the
game (a player at 0 life or 10 poison, or a game-end event): Forge stops the
check at the game's end, so nothing else of it happens.

Prior art: ``lethality_counterfactuals.qualify``, which reads the same facts
off a damage or shrink resolution; the rules are ``rule_outcomes``'.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from effects.domain.provenance import SidecarLookup
from effects.domain.records import EffectRecord, Moment, RecordKind
from effects.domain.rule_detectors import (
    EventlessFamily,
    FamilyMatch,
    Matcher,
    first_side,
    register,
)
from effects.domain.rule_detectors import declared_values as dv
from effects.domain.rule_detectors.keyword_reading import KeywordReading
from effects.domain.rule_detectors.sba_board import Board, Touched, game_ends, touched_by
from effects.domain.rule_outcomes import damage_lethal
from effects.domain.state_snapshot import EntityState, normalize_counter_type

CHECKS = frozenset({
    "toughness_zero", "lethal_damage", "deathtouch_damage", "loyalty_zero", "defense_zero",
    "legend_rule", "world_rule", "aura_illegal",
})
_COUNTER_OF = {"loyalty_zero": ("LOYALTY", "planeswalker"), "defense_zero": ("DEFENSE", "battle")}


def _is(entity: EntityState, card_type: str) -> bool:
    return card_type in {t.lower() for t in entity.types}


def _is_creature(entity: EntityState) -> bool:
    """A creature whose power and toughness the snapshot reads."""
    return entity.pt is not None and _is(entity, "creature")


def _counter(entity: EntityState, name: str) -> int:
    return sum(n for counter, n in entity.counters.items()
               if normalize_counter_type(counter) == name)


def _spell_or_activation(record: EffectRecord) -> bool:
    """Whether the acting line is a spell's or an activated ability's, whose
    board is read before it resolves; a trigger's, a keyword's (persist,
    undying) may read its dead host by last known information."""
    return bool(record.ability) and all(key.trait_kind == "spell" for key in record.ability)


#: A colour's protection keyword, as the record spells it, to the colour code.
_PROTECTION_FROM = {f"protection from {name}": code for name, code in (
    ("white", "W"), ("blue", "U"), ("black", "B"), ("red", "R"), ("green", "G"))}


def _protection_side(aura: EntityState, protections: list[str]) -> bool | None:
    """Whether a protection the host gains makes the Aura illegal on it: a
    colour's against the Aura's colours; None for any other protection, whose
    quality the record does not spell out."""
    if not protections:
        return False
    found = False
    for protection in protections:
        colour = _PROTECTION_FROM.get(protection)
        if colour is None:
            return None
        found = found or colour in aura.colors
    return found


_DEATHTOUCH = frozenset({"deathtouch"})
#: Damage from a source with one of these is dealt as -1/-1 counters and
#: marks none (702.80a, 702.90b).
_COUNTER_DAMAGE = frozenset({"wither", "infect"})


@dataclass(slots=True)
class _Found:
    """The permanents a check judged on each side, and those the record moved."""

    fires: list[str] = field(default_factory=list)
    spared: list[str] = field(default_factory=list)
    observed: list[str] = field(default_factory=list)

    def add(self, entity_id: str, side: bool, died: bool) -> None:
        (self.fires if side else self.spared).append(entity_id)
        if died:
            self.observed.append(entity_id)


_Check = Callable[[Board], _Found]


# ── the creature checks: 704.5f, g and h ────────────────────────────────


@dataclass(frozen=True, slots=True)
class _Creature:
    """A creature the record touched, at its toughness after the P/T changes."""

    id: str
    touched: Touched
    entity: EntityState
    toughness: int


def _creatures(board: Board) -> Iterator[_Creature]:
    """The creatures the record touched that it neither moved itself nor
    turned into a noncreature."""
    for entity_id, t in board.touched.items():
        entity = board.entity(entity_id)
        if entity is not None and _judged_creature(entity, t):
            yield _Creature(entity_id, t, entity, entity.toughness + t.toughness_delta)


def _judged_creature(entity: EntityState, t: Touched) -> bool:
    if t.left_creatures or t.moved_by_effect:
        return False
    return _is_creature(entity)


def _toughness_zero(board: Board, c: _Creature) -> bool | None:
    if c.touched.damage_dealt or not _lowered_or_at_zero(board, c):
        return None
    if c.toughness <= 0:
        return True
    # Marked damage now lethal is 704.5g's, not a near miss.
    return None if c.toughness <= c.entity.damage else False


def _lowered_or_at_zero(board: Board, c: _Creature) -> bool:
    """The record lowers its toughness, or the board already stands at 0 (a
    cost removed the counter a characteristic-defining toughness counts)."""
    if c.touched.lowered:
        return True
    return c.entity.on_battlefield and c.toughness <= 0 and _spell_or_activation(board.record)


def _destroyable(board: Board, c: _Creature) -> bool:
    """Dealt damage, and not indestructible: the damage checks destroy."""
    if not c.touched.damage_dealt:
        return False
    return "indestructible" not in board.keywords(c.entity) | c.touched.gained


def _lethal(c: _Creature) -> bool:
    return damage_lethal(c.toughness, c.entity.damage, c.touched.damage_dealt)


def _lethal_damage(board: Board, c: _Creature) -> bool | None:
    if not _destroyable(board, c) or board.source_has(c.touched, _COUNTER_DAMAGE):
        return None  # -1/-1 counters, no marked damage: 704.5f's
    side = _lethal(c)
    if not side and board.source_has(c.touched, _DEATHTOUCH):
        return None  # 704.5h's, not a near miss of 704.5g
    return side


def _deathtouch_damage(board: Board, c: _Creature) -> bool | None:
    """Non-lethal damage, by whether a source has deathtouch."""
    if not _destroyable(board, c) or _lethal(c):
        return None
    return board.source_has(c.touched, _DEATHTOUCH)


_CreatureSide = Callable[[Board, _Creature], "bool | None"]

#: Each creature check's side of one creature; None where it judges none.
_CREATURE_SIDES: dict[str, _CreatureSide] = {
    "toughness_zero": _toughness_zero,
    "lethal_damage": _lethal_damage,
    "deathtouch_damage": _deathtouch_damage,
}


@dataclass(frozen=True, slots=True)
class _CreatureCheck:
    side_of: _CreatureSide
    #: Damage accumulating over the turn (120.6): only a creature the board
    #: shows damage already marked on.
    marked_before: bool

    def __call__(self, board: Board) -> _Found:
        found = _Found()
        for c in _creatures(board):
            side = self.side_of(board, c)
            if side is not None and self._kept(c):
                found.add(c.id, side, c.touched.died)
        return found

    def _kept(self, c: _Creature) -> bool:
        return not self.marked_before or c.entity.damage > 0


# ── the counter checks: 704.5i and v ────────────────────────────────────


@dataclass(frozen=True, slots=True)
class _CounterCheck:
    counter: str
    card_type: str

    def __call__(self, board: Board) -> _Found:
        found = _Found()
        for entity_id, t in board.touched.items():
            side = self._side(board.entity(entity_id), t)
            if side is not None:
                found.add(entity_id, side, t.died)
        return found

    def _side(self, entity: EntityState | None, t: Touched) -> bool | None:
        """Whether the counters the record removed leave none; None for a
        permanent of another type, moved by the record, or losing none."""
        if entity is None or not self._judged(entity, t):
            return None
        delta = t.counter_delta.get(self.counter, 0)
        if delta >= 0:
            return None
        return _counter(entity, self.counter) + delta <= 0

    def _judged(self, entity: EntityState, t: Touched) -> bool:
        return _is(entity, self.card_type) and not t.moved_by_effect


# ── the uniqueness checks: 704.5j and k ─────────────────────────────────


@dataclass(frozen=True, slots=True)
class _UniquenessCheck:
    """The legend and world rules: who shares the battlefield once the record's moves are made."""

    supertype: str
    #: The legend rule groups by name and controller; the world rule groups all.
    by_name: bool

    def __call__(self, board: Board) -> _Found:
        found = _Found()
        for members in self._groups(board).values():
            side = _group_side(board, members)
            if side is not None:
                for member in members:
                    found.add(member, side, board.touch(member).died)
        return found

    def _groups(self, board: Board) -> dict[tuple, list[str]]:
        """The permanents present once the record's moves are made, by group."""
        groups: dict[tuple, list[str]] = {}
        for entity in board.record.state.entities:
            if self._present(board, entity):
                groups.setdefault(self._group(entity), []).append(entity.id)
        return groups

    def _present(self, board: Board, entity: EntityState) -> bool:
        if self.supertype not in {s.lower() for s in entity.supertypes}:
            return False
        t = board.touch(entity.id)
        return (entity.zone == "battlefield" or t.entered) and not (t.left and not t.died)

    def _group(self, entity: EntityState) -> tuple:
        return (entity.name, entity.controller) if self.by_name else ()


def _group_side(board: Board, members: list[str]) -> bool | None:
    """A pair or more is the ``fires`` side where the record touched one of
    them (a pair it did not make, as under Mirror Gallery, or one the check
    already let stand, shows no check); a lone permanent entering is the near
    miss, unless it dies entering at toughness 0, by 704.5f."""
    if len(members) >= 2:
        return True if any(m in board.touched for m in members) else None
    if not any(board.touch(m).entered for m in members):
        return None
    died = [m for m in members if board.touch(m).died]
    return None if any(_zero_toughness_on_entry(board, m) for m in died) else False


def _zero_toughness_on_entry(board: Board, entity_id: str) -> bool:
    """Whether an entering creature's readable toughness is 0 or less: its
    death is 704.5f's, whatever else it shows."""
    entity = board.entity(entity_id)
    if entity is None or not _is_creature(entity):
        return False
    return entity.toughness + board.touched[entity_id].toughness_delta <= 0


# ── the Aura check: 704.5m ──────────────────────────────────────────────


#: Why an Aura's host stops being legal: it left, or it gained protection.
AURA_CAUSES = frozenset({"left", "protection"})


def _any_cause(host: Touched, protected: bool) -> bool:
    return True


def _left_cause(host: Touched, protected: bool) -> bool:
    return host.left or not protected


def _protection_cause(host: Touched, protected: bool) -> bool:
    if host.left:
        return False
    return any(k.startswith("protection") for k in host.gained)


#: ``aura_cause``: which Auras the declaration keeps.
_CAUSE_KEEPS: dict[str | None, Callable[[Touched, bool], bool]] = {
    None: _any_cause, "left": _left_cause, "protection": _protection_cause}


@dataclass(frozen=True, slots=True)
class _AuraCheck:
    keeps: Callable[[Touched, bool], bool]

    def __call__(self, board: Board) -> _Found:
        found = _Found()
        for aura, host in _judged_auras(board):
            side = self._side(aura, host)
            if side is not None:
                found.add(aura.id, side, board.touch(aura.id).died)
        return found

    def _side(self, aura: EntityState, host: Touched) -> bool | None:
        protected = _protection_side(
            aura, [k for k in host.gained if k.startswith("protection")])
        if protected is None or host.left_creatures:
            # What the Aura may enchant is not on the record, so a host
            # changing types or gaining an unnamed protection says nothing of
            # its legality.
            return None
        if not self.keeps(host, protected):
            return None
        return host.left or protected


def _judged_auras(board: Board) -> Iterator[tuple[EntityState, Touched]]:
    """Each Aura on the battlefield whose host the record touched, the
    record's own source and targets, Auras it moved and bestowed ones left out."""
    acted_on = {board.record.state.refs.source, *board.record.state.refs.as_cast}
    for aura in board.record.state.entities:
        if not _attached_aura(aura) or not _judged_aura(board, aura, acted_on):
            continue
        host = board.touched.get(aura.attached_to)
        if host is not None:
            yield aura, host


def _attached_aura(entity: EntityState) -> bool:
    if "aura" not in {s.lower() for s in entity.subtypes}:
        return False
    return bool(entity.attached_to) and entity.on_battlefield


def _judged_aura(board: Board, aura: EntityState, acted_on: set[str | None]) -> bool:
    if aura.id in acted_on or board.touch(aura.id).moved_by_effect:
        return False
    return "bestow" not in board.keywords(aura)


# ── the family ──────────────────────────────────────────────────────────


def _compiled_check(params: Mapping[str, Any]) -> _Check:
    check = params.get("check")
    if check in _CREATURE_SIDES:
        return _CreatureCheck(_CREATURE_SIDES[check], bool(params.get("marked_before")))
    if check in _COUNTER_OF:
        return _CounterCheck(*_COUNTER_OF[check])
    if check == "aura_illegal":
        return _AuraCheck(_CAUSE_KEEPS[params.get("aura_cause")])
    legend = check == "legend_rule"
    return _UniquenessCheck("legendary" if legend else "world", legend)


def _shows_a_check(record: EffectRecord) -> bool:
    """A resolution's effect half or a real combat, neither the game-end
    record nor one whose changes end the game."""
    if not _checked_kind(record) or record.is_game_end_record:
        return False
    return not game_ends(record)


def _checked_kind(record: EffectRecord) -> bool:
    if record.kind is RecordKind.RESOLUTION:
        return record.moment is Moment.RESOLUTION
    return record.kind is RecordKind.COMBAT and not record.is_probe


@dataclass(frozen=True, slots=True, eq=False)
class ZoneMoveMatcher:
    """A compiled ``sba_zone_move`` declaration."""

    check: _Check
    #: ``aura_illegal``: an Aura's host is a condition entity too.
    aura: bool
    keywords: KeywordReading = field(default_factory=KeywordReading)

    @classmethod
    def of(cls, params: Mapping[str, Any]) -> ZoneMoveMatcher:
        return cls(_compiled_check(params), params.get("check") == "aura_illegal")

    def __call__(self, record: EffectRecord, lines: SidecarLookup | None) -> FamilyMatch | None:
        if not _shows_a_check(record):
            return None
        board = Board(record, touched_by(record), self.keywords.of(lines).keywords_of)
        found = self.check(board)
        side = first_side(found.fires, found.spared)
        if side is None:
            return None
        polarity, judged = side
        return FamilyMatch(polarity, any(e in found.observed for e in judged),
                           self._condition(record, judged), judged)

    def _condition(self, record: EffectRecord, judged: tuple[str, ...]) -> tuple[str, ...]:
        """What the check reads: the permanent it judges, and for an Aura the
        host whose departure or protection makes it illegal. The outcome, the
        move to the graveyard or its absence, falls on the judged permanent."""
        if not self.aura:
            return tuple(judged)
        hosts = [aura.attached_to for aura in map(record.state.entity, judged)
                 if aura is not None and aura.attached_to]
        return tuple(dict.fromkeys((*judged, *hosts)))


def _aura_cause_applies(params: Mapping[str, Any], check: Any) -> bool:
    if "aura_cause" not in params:
        return True
    return check == "aura_illegal" and params["aura_cause"] in AURA_CAUSES


@dataclass(frozen=True, slots=True)
class SbaZoneMove(EventlessFamily):
    id: str = "sba_zone_move"
    kinds: frozenset[str] = frozenset({"combat", "resolution"})
    params: frozenset[str] = frozenset({"check", "marked_before", "aura_cause"})
    general: frozenset[str] = frozenset({"check"})

    def needs(self, params: Mapping[str, Any]) -> frozenset[str]:
        return frozenset()

    def validate(self, params: Mapping[str, Any]) -> None:
        check = params.get("check")
        if check not in CHECKS:
            raise ValueError(f"sba_zone_move: unknown check {check!r}; known: {sorted(CHECKS)}")
        dv.check(self.id, params, {"marked_before": dv.boolean})
        if "marked_before" in params and check != "lethal_damage":
            raise ValueError("sba_zone_move: marked_before reads lethal_damage")
        if not _aura_cause_applies(params, check):
            raise ValueError(f"sba_zone_move: aura_cause is one of {sorted(AURA_CAUSES)}, "
                             "with aura_illegal")

    def compile(self, params: Mapping[str, Any]) -> Matcher:
        return ZoneMoveMatcher.of(params)

    def match(self, record: EffectRecord, lines: SidecarLookup | None,
              params: Mapping[str, Any]) -> FamilyMatch | None:
        return self.compile(params)(record, lines)


FAMILY = register(SbaZoneMove())

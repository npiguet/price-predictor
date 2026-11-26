"""Detector family ``keyword_combat``: a damage-step keyword in a combat, or stripped from one.

The ``fires`` side is a real combat in which a creature carries the keyword
and gate 2's table says the keyword's effect is observable for it (it deals
damage, a lone opponent survives a normal hit …); the near miss is a
damage-step probe fork that re-ran that combat with the keyword stripped from
its carrier, named by the probe's ``probed_keyword`` and ``probed_entity``.

Parameter ``keyword``: one of the eight damage-step keywords of gate 2's table.

The condition entity is the carrier; the outcome entities are where gate 2's
table says the keyword's effects fall (``CombatParticipant.subjects``): the
creatures its damage is dealt to, the defending player, the carrier's controller
(lifelink), or the carrier itself (indestructible). On a ``fires`` combat
they are the subjects of the effects observable there; on a probe fork, the
subjects of every effect of the row, since the stripped keyword's effect is
what did not happen there.

Wraps ``damage_step_keywords.qualifying_observations`` rather than restating
when a keyword is observable, so the detector and gate 2 cannot disagree.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from effects.domain.damage_step_keywords import (
    DAMAGE_STEP_KEYWORDS,
    CombatParticipant,
    DamageStepKeyword,
    combat_participant,
    qualifying_observations,
)
from effects.domain.provenance import SidecarLookup
from effects.domain.records import EffectRecord, RecordKind
from effects.domain.rule_catalogue import Polarity
from effects.domain.rule_detectors import EventlessFamily, FamilyMatch, Matcher, register
from effects.domain.rule_detectors.keyword_outcomes import effect_happened
from effects.domain.rule_detectors.keyword_reading import KeywordReading
from effects.domain.rule_detectors.sba_board import game_ends

_ROWS = {row.keyword: row for row in DAMAGE_STEP_KEYWORDS}


@dataclass(frozen=True, slots=True, eq=False)
class KeywordCombatMatcher:
    """A compiled ``keyword_combat`` declaration."""

    keyword: str
    row: DamageStepKeyword
    keywords: KeywordReading = field(default_factory=KeywordReading)

    @classmethod
    def of(cls, params: Mapping[str, Any]) -> KeywordCombatMatcher:
        keyword = params.get("keyword")
        return cls(keyword, _ROWS[keyword])

    def __call__(self, record: EffectRecord, lines: SidecarLookup | None) -> FamilyMatch | None:
        if not _read_combat(record):
            return None
        if record.is_probe:
            return self._stripped(record, lines)
        return self._carried(record, lines)

    def _stripped(self, record: EffectRecord, lines: SidecarLookup | None) -> FamilyMatch | None:
        """The near miss: the probe fork that stripped the keyword from its carrier."""
        if record.payload.probed_keyword != self.keyword:
            return None
        entity = record.payload.probed_entity
        carrier = record.state.entity(entity) if entity else None
        participant = (combat_participant(record, carrier, self.keywords.of(lines).keywords_of)
                       if carrier is not None else None)
        outcome = (_outcome(self.row, participant, observable=False)
                   if participant is not None else ())
        return FamilyMatch(Polarity.DOES_NOT_FIRE, False,
                           (entity,) if entity else (), outcome)

    def _carried(self, record: EffectRecord, lines: SidecarLookup | None) -> FamilyMatch | None:
        """The ``fires`` side: a real combat with an observable carrier."""
        row = self.row
        resolver = self.keywords.of(lines)
        found = qualifying_observations(record, (row,), resolver)
        if not found:
            return None
        carriers = tuple(dict.fromkeys(p.carrier.id for _, p in found))
        outcome = tuple(dict.fromkeys(
            e for _, p in found for e in _outcome(row, p, observable=True)))
        return FamilyMatch(Polarity.FIRES, _happened(record, row, found), carriers, outcome)


@dataclass(frozen=True, slots=True)
class KeywordCombat(EventlessFamily):
    id: str = "keyword_combat"
    kinds: frozenset[str] = frozenset({"combat"})
    params: frozenset[str] = frozenset({"keyword"})
    general: frozenset[str] = frozenset({"keyword"})

    def needs(self, params: Mapping[str, Any]) -> frozenset[str]:
        return frozenset()

    def validate(self, params: Mapping[str, Any]) -> None:
        keyword = params.get("keyword")
        if keyword not in _ROWS:
            raise ValueError(
                f"keyword_combat: {keyword!r} is not a damage-step keyword; known: {sorted(_ROWS)}")

    def compile(self, params: Mapping[str, Any]) -> Matcher:
        return KeywordCombatMatcher.of(params)

    def match(self, record: EffectRecord, lines: SidecarLookup | None,
              params: Mapping[str, Any]) -> FamilyMatch | None:
        return self.compile(params)(record, lines)


def _read_combat(record: EffectRecord) -> bool:
    """A combat, real or probe, whose changes do not end the game
    (``sba_board.game_ends``): Forge destroys no creature in a combat that
    ends the game, so neither side shows what the keyword did (decision D13)."""
    return record.kind is RecordKind.COMBAT and not game_ends(record)


def _happened(record: EffectRecord, row: DamageStepKeyword, found) -> bool:
    """Whether the record shows an observable effect of the row happening to
    a subject of it, for any carrier: what ``observed`` says of a ``fires`` combat."""
    events = record.payload.listed_events()
    return any(
        effect_happened(effect, subject, events)
        for _, participant in found
        for effect in row.effects if effect.observable(participant)
        for subject in participant.subjects(effect.subject))


def _outcome(row: DamageStepKeyword, participant: CombatParticipant, *,
             observable: bool) -> tuple[str, ...]:
    """Where the row's effects fall for one carrier: the subjects of the
    effects observable on this combat, or of every effect when not
    ``observable`` (a probe fork, whose carrier lost the keyword the
    predicates read)."""
    return tuple(dict.fromkeys(
        subject for effect in row.effects
        if not observable or effect.observable(participant)
        for subject in participant.subjects(effect.subject)))


FAMILY = register(KeywordCombat())

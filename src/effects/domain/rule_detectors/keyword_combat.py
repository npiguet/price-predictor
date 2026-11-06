"""Detector family ``keyword_combat``: a damage-step keyword in a combat, or stripped from one.

The ``fires`` side is a real combat in which a creature carries the keyword
and gate 2's table says the keyword's effect is observable for it (it deals
damage, a lone opponent survives a normal hit …); the near miss is a
damage-step probe fork that re-ran that combat with the keyword stripped from
its carrier, named by the probe's ``probed_keyword`` and ``probed_entity``.

Parameter ``keyword``: one of the eight damage-step keywords of gate 2's table.

The condition entity is the carrier; the outcome entities are where gate 2's
table says the keyword's effects fall (``KeywordEffect.subject``): the
creature it fights, the defending player, the carrier's controller
(lifelink), or the carrier itself (indestructible). On a ``fires`` combat
they are the subjects of the effects observable there; on a probe fork, the
subjects of every effect of the row, since the stripped keyword's effect is
what did not happen there.

Wraps ``damage_step_keywords.qualifying_observations`` rather than restating
when a keyword is observable, so the detector and gate 2 cannot disagree.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from effects.domain.damage_step_keywords import (
    DAMAGE_STEP_KEYWORDS,
    CombatParticipant,
    DamageStepKeyword,
    KeywordResolver,
    KeywordsOf,
    combat_participant,
    qualifying_observations,
)
from effects.domain.provenance import SidecarLookup
from effects.domain.records import EffectRecord, RecordKind
from effects.domain.rule_catalogue import Polarity
from effects.domain.rule_detectors import FamilyMatch, register

_ROWS = {row.keyword: row for row in DAMAGE_STEP_KEYWORDS}


@dataclass(frozen=True, slots=True)
class KeywordCombat:
    id: str = "keyword_combat"
    params: frozenset[str] = frozenset({"keyword"})
    general: frozenset[str] = frozenset({"keyword"})

    def needs(self, params: Mapping[str, Any]) -> frozenset[str]:
        return frozenset()

    def match(self, record: EffectRecord, lines: SidecarLookup | None,
              params: Mapping[str, Any]) -> FamilyMatch | None:
        keyword = params.get("keyword")
        if keyword not in _ROWS:
            raise ValueError(
                f"keyword_combat: {keyword!r} is not a damage-step keyword; known: {sorted(_ROWS)}")
        if record.kind is not RecordKind.COMBAT:
            return None
        row = _ROWS[keyword]
        if record.is_probe:
            if record.payload.probed_keyword != keyword:
                return None
            entity = record.payload.probed_entity
            carrier = record.state.entity(entity) if entity else None
            participant = (combat_participant(record.state, carrier)
                           if carrier is not None else None)
            outcome = _outcome(row, participant, None) if participant is not None else ()
            return FamilyMatch(Polarity.DOES_NOT_FIRE, False,
                               (entity,) if entity else (), outcome)
        resolver = KeywordResolver(lines)
        found = qualifying_observations(record, (row,), resolver)
        if not found:
            return None
        carriers = tuple(dict.fromkeys(p.carrier.id for _, p in found))
        outcome = tuple(dict.fromkeys(
            e for _, p in found for e in _outcome(row, p, resolver.keywords_of)))
        return FamilyMatch(Polarity.FIRES, True, carriers, outcome)


def _outcome(row: DamageStepKeyword, participant: CombatParticipant,
             keywords_of: KeywordsOf | None) -> tuple[str, ...]:
    """Where the row's effects fall for one carrier: the subjects of the
    effects observable on this combat, or of every effect with a subject
    when ``keywords_of`` is None (a probe fork, whose carrier lost the
    keyword the predicates read)."""
    return tuple(dict.fromkeys(
        subject for effect in row.effects
        if keywords_of is None or effect.observable(participant, keywords_of)
        if (subject := participant.subject(effect.subject)) is not None))


FAMILY = register(KeywordCombat())

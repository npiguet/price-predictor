"""Detector family ``keyword_combat``: a damage-step keyword in a combat, or stripped from one.

The ``fires`` side is a real combat in which a creature carries the keyword
and gate 2's table says the keyword's effect is observable for it (it deals
damage, a lone opponent survives a normal hit …); the near miss is a
damage-step probe fork that re-ran that combat with the keyword stripped from
its carrier, named by the probe's ``probed_keyword`` and ``probed_entity``.

Parameter ``keyword``: one of the eight damage-step keywords of gate 2's table.

Wraps ``damage_step_keywords.qualifying_observations`` rather than restating
when a keyword is observable, so the detector and gate 2 cannot disagree.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from effects.domain.damage_step_keywords import (
    DAMAGE_STEP_KEYWORDS,
    KeywordResolver,
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
        if record.is_probe:
            if record.payload.probed_keyword != keyword:
                return None
            entity = record.payload.probed_entity
            return FamilyMatch(Polarity.DOES_NOT_FIRE, False, (entity,) if entity else ())
        carriers = tuple(participant.carrier.id for _, participant in qualifying_observations(
            record, (_ROWS[keyword],), KeywordResolver(lines)))
        if not carriers:
            return None
        return FamilyMatch(Polarity.FIRES, True, carriers)


FAMILY = register(KeywordCombat())

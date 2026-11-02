"""The breadth check: how much of its family's pool a rule detector tags (T036).

A declaration names a family and parameters, some stating the family's
condition and the rule's subject (``DetectorFamily.general``: the event, the
check, the verdict bit, the API or keyword the rule is about) and the rest
picking one situation of it (an event filter, a script parameter, a trigger's
zones or selector, a qualifier). Per declared scenario, the check counts the
records whose ``fires`` side the declaration finds (``tagged``) and those the
family finds with the situation's parameters dropped (``pool``). A detector
tagging a small share of a large pool is flagged for review: it is what a
declaration copied from one scenario's specifics looks like, though a rule
about a rare situation of a common subject looks the same, so a flag is a
question for the reviewer rather than a verdict.

Prior art: :func:`effects.domain.rule_detectors.tag_record`, whose per-kind
index and generation rule this follows scenario by scenario.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from effects.domain.provenance import SidecarLookup
from effects.domain.records import EffectRecord
from effects.domain.rule_catalogue import DetectorDeclaration, Polarity
from effects.domain.rule_detectors import DetectorFamily, DetectorIndex

POOL = "pool"
TAGGED = "tagged"


def pool_params(declaration: DetectorDeclaration, family: DetectorFamily) -> dict[str, Any]:
    """The declaration's parameters with the rule-specific ones dropped."""
    return {name: value for name, value in declaration.params.items()
            if name in family.general}


def _fires(family: DetectorFamily, record: EffectRecord, lines: SidecarLookup | None,
           params: Mapping[str, Any]) -> bool:
    found = family.match(record, lines, params)
    return found is not None and found.polarity is Polarity.FIRES


def breadth_counts(record: EffectRecord, index: DetectorIndex,
                   lines: SidecarLookup | None = None) -> Counter:
    """``(scenario, POOL | TAGGED) -> 1`` for each declared scenario of the
    record's kind whose pool, or whose own detector, finds a ``fires`` side.

    A scenario whose detector needs a field the record's generation lacks is
    skipped, as :func:`tag_record` skips it.
    """
    found: Counter = Counter()
    gen3 = record.generation == 3
    for scenario, declaration, family in index.by_kind.get(record.kind.value, ()):
        if not gen3 and (declaration.needs or family.needs(declaration.params)):
            continue
        general = pool_params(declaration, family)
        if gen3 or not family.needs(general):
            if _fires(family, record, lines, general):
                found[(scenario.id, POOL)] += 1
        if _fires(family, record, lines, declaration.params):
            found[(scenario.id, TAGGED)] += 1
    return found


@dataclass(frozen=True, slots=True)
class BreadthRow:
    scenario: str
    rules: tuple[str, ...]
    dropped: tuple[str, ...]
    pool: int
    tagged: int

    @property
    def share(self) -> float:
        return self.tagged / self.pool if self.pool else 0.0

    def flagged(self, min_pool: int, max_share: float) -> bool:
        """A small share of a large pool, where the declaration dropped anything."""
        return bool(self.dropped) and self.pool >= min_pool and self.share < max_share

    def to_json(self) -> dict[str, Any]:
        return {"scenario": self.scenario, "rules": list(self.rules),
                "dropped": list(self.dropped), "pool": self.pool, "tagged": self.tagged,
                "share": round(self.share, 6)}


def breadth_rows(index: DetectorIndex, counts: Counter) -> list[BreadthRow]:
    """One row per declared scenario, in index order."""
    rows: list[BreadthRow] = []
    for entries in index.by_kind.values():
        for scenario, declaration, family in entries:
            dropped = tuple(sorted(set(declaration.params) - family.general))
            rows.append(BreadthRow(
                scenario.id, scenario.rules, dropped,
                counts.get((scenario.id, POOL), 0), counts.get((scenario.id, TAGGED), 0)))
    return rows


def flagged_rows(rows: Iterable[BreadthRow], min_pool: int,
                 max_share: float) -> list[BreadthRow]:
    """The flagged rows, the largest pool first."""
    return sorted((row for row in rows if row.flagged(min_pool, max_share)),
                  key=lambda row: (-row.pool, row.scenario))

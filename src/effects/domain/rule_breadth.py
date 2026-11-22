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
index this follows scenario by scenario.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from effects.domain.provenance import SidecarLookup
from effects.domain.records import EffectRecord, is_game_end
from effects.domain.rule_catalogue import DetectorDeclaration, Polarity
from effects.domain.rule_detectors import DetectorFamily, DetectorIndex, Matcher

POOL = "pool"
TAGGED = "tagged"


def pool_params(declaration: DetectorDeclaration, family: DetectorFamily) -> dict[str, Any]:
    """The declaration's parameters with the rule-specific ones dropped."""
    return {name: value for name, value in declaration.params.items()
            if name in family.general}


def _fires(matcher: Matcher, record: EffectRecord, lines: SidecarLookup | None) -> bool:
    found = matcher(record, lines)
    return found is not None and found.polarity is Polarity.FIRES


@dataclass(frozen=True, slots=True)
class _Scenario:
    """One declared scenario's matchers, each compiled once: its pool's (the
    declaration's general parameters alone) and its own."""

    id: str
    pool: Matcher
    tagged: Matcher


@dataclass(frozen=True, slots=True)
class BreadthPlan:
    """Every declared scenario's matchers by record kind, compiled once."""

    by_kind: Mapping[str, tuple[_Scenario, ...]]

    @classmethod
    def of(cls, index: DetectorIndex) -> BreadthPlan:
        return cls({kind: tuple(
            _Scenario(scenario.id, family.compile(pool_params(declaration, family)),
                      family.compile(declaration.params))
            for scenario, declaration, family in entries)
            for kind, entries in index.by_kind.items()})

    def counts(self, record: EffectRecord, lines: SidecarLookup | None = None) -> Counter:
        """``(scenario, POOL | TAGGED) -> 1`` for each declared scenario of the
        record's kind whose pool, or whose own detector, finds a ``fires`` side."""
        found: Counter = Counter()
        if is_game_end(record):
            return found  # no detector reads the game-end marker, as tag_record says
        for scenario in self.by_kind.get(record.kind.value, ()):
            if _fires(scenario.pool, record, lines):
                found[(scenario.id, POOL)] += 1
            if _fires(scenario.tagged, record, lines):
                found[(scenario.id, TAGGED)] += 1
        return found


def breadth_counts(record: EffectRecord, index: DetectorIndex,
                   lines: SidecarLookup | None = None) -> Counter:
    """:meth:`BreadthPlan.counts` of a plan built for this one record; a
    caller reading many records builds the plan once."""
    return BreadthPlan.of(index).counts(record, lines)


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

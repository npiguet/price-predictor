"""Rule detectors: which rules of the game one effect record shows (FR-006, research R1).

A detector is a family of record shapes, parameterized by a rule's general
condition: a zone move inside the state-based check by which check, a trigger
by its ``Mode$``, a playability verdict by its bit. A scenario's ``detector``
declaration names one family and its parameters, so 1,763 scenarios share a
handful of functions. The declaration describes the rule as it shows in any
game; a family never takes a card name or anything else one scenario's board
fixes, or its detector would match that board and nothing else.

:func:`tag_record` runs every declared scenario of the record's kind and
merges what several scenarios of one rule say into one :class:`RuleTag` per
rule. The tags feed ``rule-counts``, the rule budgets and the rule mask.

Prior art: ``effect_families.effect_family`` (one record to one label, read off
its acting line), ``damage_step_keywords.qualifying_observations`` and
``lethality_counterfactuals.qualify`` (a record qualifying for a rule by its
board and events).

Torch-free: the ``build-corpus`` survey workers tag records.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from effects.domain.provenance import SidecarLookup
from effects.domain.records import EffectRecord
from effects.domain.rule_catalogue import (
    DetectorDeclaration,
    Polarity,
    Scenario,
    ScenarioCatalogue,
)


@dataclass(frozen=True, slots=True)
class RuleTag:
    """One rule a record shows: which side of it, and whether its outcome happened.

    ``polarity`` comes from the board, the variant situation the record
    matches; ``observed`` from the record's outcome. A tag whose two disagree
    goes on the review list (FR-004).
    """

    rule: str
    polarity: Polarity
    observed: bool
    entities: tuple[str, ...]
    scenario: str

    @property
    def needs_review(self) -> bool:
        return self.polarity is not Polarity.OTHER and self.observed != (
            self.polarity is Polarity.FIRES)


@dataclass(frozen=True, slots=True)
class FamilyMatch:
    """What a family finds in one record for one scenario, before the rule is attached."""

    polarity: Polarity
    observed: bool
    entities: tuple[str, ...] = ()


class DetectorFamily(Protocol):
    """One record shape a rule's outcome can take."""

    #: The family id a declaration names.
    id: str
    #: The parameters a declaration may pass: the rule's general conditions.
    params: frozenset[str]

    def needs(self, params: Mapping[str, Any]) -> frozenset[str]:
        """The gen-3 fields the family reads under ``params``; empty on gen-2 records."""
        ...

    def match(self, record: EffectRecord, lines: SidecarLookup | None,
              params: Mapping[str, Any]) -> FamilyMatch | None:
        """The record's side of the rule, or None where the record does not show it."""
        ...


_REGISTRY: dict[str, DetectorFamily] = {}


def register(family: DetectorFamily) -> DetectorFamily:
    """Add ``family`` to the registry under its id; a second family of one id is refused."""
    if family.id in _REGISTRY:
        raise ValueError(f"detector family {family.id!r} is registered twice")
    _REGISTRY[family.id] = family
    return family


def families() -> Mapping[str, DetectorFamily]:
    """Every registered family by id, the family modules imported."""
    _load_families()
    return dict(_REGISTRY)


#: One module per family, each registering its family on import.
FAMILY_MODULES: tuple[str, ...] = (
    "sba_zone_move", "trigger_mode", "keyword_combat", "verdict_bit",
    "continuous_contribution", "rewrite_event", "legality_membership",
    "outcome_event", "resolution_outcome", "sba_game_loss",
)


def _load_families() -> None:
    for name in FAMILY_MODULES:
        importlib.import_module(f"{__name__}.{name}")


@dataclass(frozen=True, slots=True)
class TagResult:
    """The rules one record shows, and the rules it cannot speak for."""

    tags: tuple[RuleTag, ...] = ()
    #: Rules a declared scenario of the record's kind needs a field for that
    #: the record's generation lacks (FR-007): not countable, never zero.
    not_countable: frozenset[str] = frozenset()
    #: Each tag's per-scenario findings that disagree with their polarity.
    review: tuple[RuleTag, ...] = ()


@dataclass(frozen=True, slots=True)
class DetectorIndex:
    """The catalogue's declared scenarios by record kind, built once per command.

    Running every scenario on every record would cost 1,763 calls a record;
    a record is only ever matched against the scenarios of its own kind.
    """

    by_kind: Mapping[str, tuple[tuple[Scenario, DetectorDeclaration, DetectorFamily], ...]]
    declared_rules: frozenset[str] = field(default_factory=frozenset)

    @classmethod
    def of(cls, catalogue: ScenarioCatalogue) -> DetectorIndex:
        registry = families()
        by_kind: dict[str, list] = {}
        rules: set[str] = set()
        for scenario in catalogue.scenarios:
            declaration = scenario.detector
            if not isinstance(declaration, DetectorDeclaration):
                continue
            family = registry.get(declaration.family)
            if family is None:
                raise KeyError(
                    f"{scenario.id}: detector family {declaration.family!r} is not registered")
            by_kind.setdefault(declaration.record_kind, []).append(
                (scenario, declaration, family))
            rules.update(scenario.rules)
        return cls({kind: tuple(found) for kind, found in by_kind.items()}, frozenset(rules))


_ORDER = {Polarity.FIRES: 0, Polarity.DOES_NOT_FIRE: 1, Polarity.OTHER: 2}


def _merge(rule: str, found: Iterable[RuleTag]) -> RuleTag:
    """Several scenarios of one rule on one record: ``fires`` if any says so,
    else ``does_not_fire``, else ``other``; entities the union (research R1)."""
    found = list(found)
    best = min(found, key=lambda tag: _ORDER[tag.polarity])
    entities = tuple(dict.fromkeys(e for tag in found for e in tag.entities))
    observed = any(tag.observed for tag in found if tag.polarity is best.polarity)
    return RuleTag(rule, best.polarity, observed, entities, best.scenario)


def tag_record(record: EffectRecord, index: DetectorIndex,
               lines: SidecarLookup | None = None) -> TagResult:
    """Every rule ``record`` shows, one tag per rule id."""
    gen3 = record.generation == 3
    per_rule: dict[str, list[RuleTag]] = {}
    not_countable: set[str] = set()
    for scenario, declaration, family in index.by_kind.get(record.kind.value, ()):
        if not gen3 and (declaration.needs or family.needs(declaration.params)):
            not_countable.update(scenario.rules)
            continue
        found = family.match(record, lines, declaration.params)
        if found is None:
            continue
        for rule in scenario.rules:
            per_rule.setdefault(rule, []).append(RuleTag(
                rule, found.polarity, found.observed, found.entities, scenario.id))
    tags = tuple(_merge(rule, found) for rule, found in sorted(per_rule.items()))
    review = tuple(tag for found in per_rule.values() for tag in found if tag.needs_review)
    return TagResult(tags, frozenset(not_countable - set(per_rule)), review)

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
import json
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any, Protocol

from effects.domain.cost_targets import is_cost_half, paid_entities
from effects.domain.event_schema import EventType
from effects.domain.provenance import SidecarLookup
from effects.domain.records import EffectRecord, RecordOrigin, is_game_end
from effects.domain.rule_catalogue import (
    DetectorDeclaration,
    Polarity,
    SameAs,
    Scenario,
    ScenarioCatalogue,
)
from effects.domain.rule_detectors.acting_line import apis_of
from effects.domain.rule_detectors.emittable_events import carries


@dataclass(frozen=True, slots=True)
class RuleTag:
    """One rule a record shows: which side of it, and whether its outcome happened.

    ``polarity`` comes from the board, the variant situation the record
    matches; ``observed`` from the record's outcome, None where the record
    cannot show it (the collector files it in another record). A tag whose
    two disagree goes on the review list (FR-004); an unreadable outcome
    disagrees with nothing.

    The tag names entities in two roles (research R1): ``condition``, what
    the rule's condition reads (the keyword's carrier, the acting source,
    the object a check judges), and ``outcome``, where the rule's outcome
    falls (the creature destroyed, the player gaining life, the entities a
    target legality is written at). One entity can hold both roles. A tag
    whose outcome is an absence (an ``absent`` scenario, a scoped
    ``does_not_fire`` with no event) names as its outcome where the outcome
    would have fallen: the acting clause's targets, else its controller.
    """

    rule: str
    polarity: Polarity
    observed: bool | None
    condition: tuple[str, ...]
    outcome: tuple[str, ...]
    scenario: str

    @property
    def entities(self) -> tuple[str, ...]:
        """Both roles, condition first, each entity once: where the hard
        rule mask trains the affected gate."""
        return tuple(dict.fromkeys((*self.condition, *self.outcome)))

    @property
    def needs_review(self) -> bool:
        return (self.polarity is not Polarity.OTHER and self.observed is not None
                and self.observed != (self.polarity is Polarity.FIRES))


class RuleSource(StrEnum):
    """Where a tagged record came from, which decides what it can be trusted
    for (FR-008): a fork of a live game's moment, aimed collection, or a
    natural record of a game with or without a random seat.

    ``fork`` is a probe's, an intervention's or a protection fork's record
    (:data:`FORK_SOURCE_ORIGINS`): it re-records a moment its live game
    recorded already, a combat once per probe, so the counts a decision
    reads leave it out and a moment counts once. A directed fork's record is
    ``aimed``, a record of its own.
    """

    RANDOM_SEAT = "random_seat"
    FORGE_AI = "forge_ai"
    AIMED = "aimed"
    FORK = "fork"

    @classmethod
    def of(cls, *, fork: bool, aimed: bool, random_seat_game: bool) -> RuleSource:
        """``fork`` whatever else holds (a coverage run's probe carries its
        game's ``aimed_at`` too); else ``aimed`` whatever the game; else by
        whether the game had a random seat."""
        if fork:
            return cls.FORK
        if aimed:
            return cls.AIMED
        return cls.RANDOM_SEAT if random_seat_game else cls.FORGE_AI


#: The origins counted under :attr:`RuleSource.FORK`.
FORK_SOURCE_ORIGINS: frozenset[RecordOrigin] = frozenset({
    RecordOrigin.PROBE, RecordOrigin.FORCED, RecordOrigin.PROTECTION_FORK})


def counts_as_fork(record: EffectRecord) -> bool:
    """Whether ``record`` counts under :attr:`RuleSource.FORK`."""
    return record.origin in FORK_SOURCE_ORIGINS


@dataclass(frozen=True, slots=True)
class FamilyMatch:
    """What a family finds in one record for one scenario, before the rule is attached.

    ``observed`` is None where the record cannot show the outcome at all.
    """

    polarity: Polarity
    observed: bool | None
    #: What the rule's condition reads (:class:`RuleTag`'s two roles).
    condition: tuple[str, ...] = ()
    #: Where the rule's outcome falls.
    outcome: tuple[str, ...] = ()

    @property
    def entities(self) -> tuple[str, ...]:
        """Both roles, condition first, each entity once."""
        return tuple(dict.fromkeys((*self.condition, *self.outcome)))


#: A declaration compiled by its family: the declaration's side of its rule in
#: one record (read with the sidecar lines), or None where the record does not
#: show it.
Matcher = Callable[[EffectRecord, SidecarLookup | None], FamilyMatch | None]


def first_side(fires: Sequence[str], spared: Sequence[str]
               ) -> tuple[Polarity, tuple[str, ...]] | None:
    """The side a family judging entities one by one reports: the entities it
    judged on the ``fires`` side where there are any, else those the rule's
    condition spared; None where it judged none."""
    for polarity, judged in ((Polarity.FIRES, fires), (Polarity.DOES_NOT_FIRE, spared)):
        if judged:
            return polarity, tuple(judged)
    return None


class DetectorFamily(Protocol):
    """One record shape a rule's outcome can take."""

    #: The family id a declaration names.
    id: str
    #: The parameters a declaration may pass: the rule's general conditions.
    params: frozenset[str]
    #: Those that state the family's condition and the rule's subject (the
    #: event, the check, the verdict bit; the API, the keyword); the rest pick
    #: one situation of it (an event filter, a script parameter, a selector),
    #: and the breadth check (``rule_breadth``) drops them to measure the pool
    #: a rule's detector is cut from.
    general: frozenset[str]
    #: The record kinds (``RecordKind`` values) the family reads: a declaration
    #: on another kind could never match, so ``DetectorIndex.of`` refuses it.
    kinds: frozenset[str]

    def events(self, params: Mapping[str, Any]) -> frozenset[EventType]:
        """The event types the declaration asks its record kind for (none for
        a family that reads no event list): one the kind never lists makes
        the declaration untaggable (``emittable_events``)."""
        ...

    def reads_acting_line(self, params: Mapping[str, Any]) -> bool:
        """Whether the declaration only speaks for records acting through a line."""
        ...

    def needs(self, params: Mapping[str, Any]) -> frozenset[str]:
        """The gen-3 fields the family reads under ``params``."""
        ...

    def validate(self, params: Mapping[str, Any]) -> None:
        """Raise ``ValueError`` on a value the family cannot read (an unknown
        check, list, bit, event type or filter), so a misspelt declaration
        stops the command that builds the index rather than a record in the
        middle of a run; ``match`` takes the same values."""
        ...

    def compile(self, params: Mapping[str, Any]) -> Matcher:
        """The declaration's matcher, built once by :class:`DetectorIndex`:
        whatever the family reads off ``params`` alone is read here, not on
        every record. ``params`` passed ``validate``."""
        ...

    def match(self, record: EffectRecord, lines: SidecarLookup | None,
              params: Mapping[str, Any]) -> FamilyMatch | None:
        """The record's side of the rule, or None where the record does not
        show it: what ``compile(params)`` finds in it."""
        ...


class EventlessFamily:
    """What a family that reads no event list and no acting line inherits:
    the protocol's ``events`` and ``reads_acting_line`` answered for it."""

    __slots__ = ()

    def events(self, params: Mapping[str, Any]) -> frozenset[EventType]:
        return frozenset()

    def reads_acting_line(self, params: Mapping[str, Any]) -> bool:
        return False


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
    "outcome_event", "resolution_outcome", "sba_game_loss", "combat_keyword_effect",
)


def _load_families() -> None:
    for name in FAMILY_MODULES:
        importlib.import_module(f"{__name__}.{name}")


@dataclass(frozen=True, slots=True)
class TagResult:
    """The rules one record shows, and the rules it cannot speak for."""

    tags: tuple[RuleTag, ...] = ()
    #: Rules a declared scenario of the record's kind needs a field for that
    #: the record lacks (FR-007): not countable, never zero. A gen-3 record
    #: carries every field a declaration needs, so :func:`tag_record` names none.
    not_countable: frozenset[str] = frozenset()
    #: Each tag's per-scenario findings that disagree with their polarity.
    review: tuple[RuleTag, ...] = ()


class DeclarationError(ValueError):
    """A catalogue declaration no family can read, naming its scenario."""


@dataclass(frozen=True, slots=True)
class Duplicate:
    """A scenario declared ``same_as`` others, and the rules it is counted under."""

    rules: tuple[str, ...]
    same_as: tuple[str, ...]
    counted_under: tuple[str, ...]


#: One declared scenario of the index: the scenario, its declaration and the
#: family reading it.
Entry = tuple[Scenario, DetectorDeclaration, DetectorFamily]
_UNRUN = object()


def _declaration_key(family: DetectorFamily, declaration: DetectorDeclaration) -> tuple[int, str]:
    """What makes two declarations of one record kind identical: the family
    reading them and the parameters, written out in one order."""
    return id(family), json.dumps(dict(declaration.params), sort_keys=True)


@dataclass(frozen=True, slots=True)
class KindPlan:
    """One record kind's declared scenarios with their compiled matchers.

    Identical declarations (one family, the same parameters: two scenarios of
    one rule stating one fact) share one matcher, which runs once a record
    and is fanned out to each of them.
    """

    matchers: tuple[Matcher, ...] = ()
    #: Each scenario in index order, with the position of its matcher.
    scenarios: tuple[tuple[Scenario, int], ...] = ()

    @classmethod
    def of(cls, entries: Iterable[Entry]) -> KindPlan:
        slots: dict[tuple[int, str], int] = {}
        matchers: list[Matcher] = []
        placed: list[tuple[Scenario, int]] = []
        for scenario, declaration, family in entries:
            key = _declaration_key(family, declaration)
            if key not in slots:
                slots[key] = len(matchers)
                matchers.append(family.compile(declaration.params))
            placed.append((scenario, slots[key]))
        return cls(tuple(matchers), tuple(placed))

    def found(self, record: EffectRecord,
              lines: SidecarLookup | None) -> Iterator[tuple[Scenario, FamilyMatch]]:
        """Each scenario of which ``record`` shows a side, in index order."""
        results: list[Any] = [_UNRUN] * len(self.matchers)
        for scenario, slot in self.scenarios:
            if results[slot] is _UNRUN:
                results[slot] = self.matchers[slot](record, lines)
            if results[slot] is not None:
                yield scenario, results[slot]


_NO_PLAN = KindPlan()


@dataclass(frozen=True, slots=True)
class DetectorIndex:
    """The catalogue's declared scenarios by record kind, built once per command.

    Running every scenario on every record would cost 1,763 calls a record;
    a record is only ever matched against the scenarios of its own kind, by
    the matchers ``plans`` compiles once from ``by_kind``.

    Building it checks every declaration: a family that is not registered, a
    parameter the family does not declare, a record kind it does not read, or a
    value it cannot read raises
    :class:`DeclarationError` naming the scenario. A declaration asking its
    record kind for an event that kind never lists (``emittable_events``) is
    set aside in ``untaggable`` with its reason, since it would read every
    record of the kind as the event's absence; a ``same_as`` scenario goes to
    ``duplicates``. Neither tags. A ``same_as`` naming a scenario of one of its
    own rules raises: a duplicate is counted under other rules, and two
    scenarios of one rule stating one fact each declare it.
    """

    by_kind: Mapping[str, tuple[Entry, ...]]
    #: The rules some taggable declaration speaks for.
    declared_rules: frozenset[str] = field(default_factory=frozenset)
    #: Scenario id -> why its declaration can tag no record.
    untaggable: Mapping[str, str] = field(default_factory=dict)
    #: Scenario id -> the scenario it states the same fact as.
    duplicates: Mapping[str, Duplicate] = field(default_factory=dict)
    #: Scenario id -> the rules it lists, for every untaggable scenario.
    untaggable_rules: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    #: Record kind -> its scenarios' compiled matchers, built from ``by_kind``.
    plans: Mapping[str, KindPlan] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "plans", {
            kind: KindPlan.of(entries) for kind, entries in self.by_kind.items()})

    def plan(self, kind: str) -> KindPlan:
        """The record kind's plan; an empty one for a kind no scenario reads."""
        return self.plans.get(kind, _NO_PLAN)

    @classmethod
    def of(cls, catalogue: ScenarioCatalogue) -> DetectorIndex:
        indexing = _Indexing(families())
        for scenario in catalogue.scenarios:
            indexing.read(scenario)
        scenarios = {s.id: s for s in catalogue.scenarios}
        duplicates = {s.id: _duplicate(s, scenarios) for s in indexing.same_as}
        return cls({kind: tuple(found) for kind, found in indexing.by_kind.items()},
                   frozenset(indexing.rules), indexing.untaggable, duplicates,
                   indexing.untaggable_rules)


class _Indexing:
    """What :meth:`DetectorIndex.of` gathers while it reads the scenarios in
    catalogue order."""

    def __init__(self, registry: Mapping[str, DetectorFamily]) -> None:
        self.registry = registry
        self.by_kind: dict[str, list[Entry]] = {}
        self.rules: set[str] = set()
        self.untaggable: dict[str, str] = {}
        self.untaggable_rules: dict[str, tuple[str, ...]] = {}
        self.same_as: list[Scenario] = []

    def read(self, scenario: Scenario) -> None:
        declaration = scenario.detector
        if isinstance(declaration, SameAs):
            self.same_as.append(scenario)
        elif isinstance(declaration, DetectorDeclaration):
            self._declare(scenario, declaration)

    def _declare(self, scenario: Scenario, declaration: DetectorDeclaration) -> None:
        family = self.registry.get(declaration.family)
        if family is None:
            raise DeclarationError(
                f"{scenario.id}: detector family {declaration.family!r} is not registered")
        _check(scenario, declaration, family)
        reason = _untaggable(declaration, family)
        if reason is not None:
            self.untaggable[scenario.id] = reason
            self.untaggable_rules[scenario.id] = scenario.rules
            return
        self.by_kind.setdefault(declaration.record_kind, []).append(
            (scenario, declaration, family))
        self.rules.update(scenario.rules)


def _duplicate(scenario: Scenario, scenarios: Mapping[str, Scenario]) -> Duplicate:
    """A ``same_as`` scenario and the rules of the scenarios it names."""
    named = scenario.detector.scenarios
    under = [rule for one in named for rule in _counted_under(scenario, one, scenarios)]
    return Duplicate(scenario.rules, named, tuple(dict.fromkeys(under)))


def _counted_under(scenario: Scenario, named: str,
                   scenarios: Mapping[str, Scenario]) -> tuple[str, ...]:
    """The rules of the scenario ``named``, which declares a detector and
    states none of ``scenario``'s own rules."""
    target = scenarios.get(named)
    if target is None or not isinstance(target.detector, DetectorDeclaration):
        raise DeclarationError(
            f"{scenario.id}: same_as {named!r} names no scenario declaring a detector")
    if set(target.rules) & set(scenario.rules):
        raise DeclarationError(
            f"{scenario.id}: same_as {named!r} names a scenario of its own rule; "
            "a duplicate is counted under other rules, so declare the detector")
    return target.rules


def _check(scenario: Scenario, declaration: DetectorDeclaration,
           family: DetectorFamily) -> None:
    if declaration.record_kind not in family.kinds:
        raise DeclarationError(
            f"{scenario.id}: {family.id} reads {sorted(family.kinds)} records, "
            f"not {declaration.record_kind!r}")
    undeclared = set(declaration.params) - family.params
    if undeclared:
        raise DeclarationError(f"{scenario.id}: {family.id} has no parameter {sorted(undeclared)}")
    try:
        family.validate(declaration.params)
    except (KeyError, ValueError) as error:
        raise DeclarationError(f"{scenario.id}: {error}") from error


def _untaggable(declaration: DetectorDeclaration, family: DetectorFamily) -> str | None:
    """Why ``declaration`` can tag nothing, or None: an event type it reads
    that its record kind never lists."""
    events, kind = family.events(declaration.params), declaration.record_kind
    missing = sorted(e.value for e in events if not carries(kind, e))
    if missing:
        return f"a {kind} record never lists {', '.join(missing)} (emittable_events)"
    if not family.reads_acting_line(declaration.params):
        return None
    api = declaration.params.get("api")
    apis = apis_of(api) if api is not None else (None,)
    filed = sorted(e.value for e in events
                   if not any(carries(kind, e, acting_line=True, api=one) for one in apis))
    if filed:
        return (f"{', '.join(filed)} is filed in the game-end record, never in a "
                "resolution acting through a line other than one ending the game "
                "outright (WinsGame, LosesGame; emittable_events)")
    return None


_ORDER = {Polarity.FIRES: 0, Polarity.DOES_NOT_FIRE: 1, Polarity.OTHER: 2}


def _merge(rule: str, found: Iterable[RuleTag]) -> RuleTag:
    """Several scenarios of one rule on one record: ``fires`` if any says so,
    else ``does_not_fire``, else ``other``; each role the union of that role
    over the scenarios (research R1); observed where any readable one of the
    chosen side says so, None where none of them is readable."""
    found = list(found)
    best = min(found, key=lambda tag: _ORDER[tag.polarity])
    condition = tuple(dict.fromkeys(e for tag in found for e in tag.condition))
    outcome = tuple(dict.fromkeys(e for tag in found for e in tag.outcome))
    readable = [tag.observed for tag in found
                if tag.polarity is best.polarity and tag.observed is not None]
    observed = any(readable) if readable else None
    return RuleTag(rule, best.polarity, observed, condition, outcome, best.scenario)


def tag_record(record: EffectRecord, index: DetectorIndex,
               lines: SidecarLookup | None = None) -> TagResult:
    """Every rule ``record`` shows, one tag per rule id; none on the game-end
    marker, which holds no event and changes nothing, so no declaration of
    its kind (an absence's least of all) may read it."""
    if is_game_end(record):
        return TagResult()
    per_rule: dict[str, list[RuleTag]] = {}
    for scenario, found in index.plan(record.kind.value).found(record, lines):
        for rule in scenario.rules:
            per_rule.setdefault(rule, []).append(RuleTag(
                rule, found.polarity, found.observed, found.condition, found.outcome,
                scenario.id))
    tags = _paid_outcomes(record, tuple(
        _merge(rule, found) for rule, found in sorted(per_rule.items())))
    review = tuple(tag for found in per_rule.values() for tag in found if tag.needs_review)
    return TagResult(tags, review=review)


def _paid_outcomes(record: EffectRecord, tags: tuple[RuleTag, ...]) -> tuple[RuleTag, ...]:
    """``tags``, on a cost half each naming among its outcome entities the
    entities the payment trains at (``cost_targets.paid_entities``): what a cost
    half's targets place is what it paid, so a rule read off one reads its outputs there."""
    if not tags or not is_cost_half(record):
        return tags
    paid = paid_entities(record)
    return tuple(replace(tag, outcome=tuple(dict.fromkeys((*tag.outcome, *paid))))
                 for tag in tags)

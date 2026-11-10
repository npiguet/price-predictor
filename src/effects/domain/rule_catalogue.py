"""The Comprehensive Rules catalogue's vocabulary: what a scenario, a detector
declaration, a departure and a heads line may say.

``rules/`` is a tree of JSON sidecars kept by hand and by agent passes; the
validator under ``scripts/rules_catalogue.py`` checks them, and the corpus
code reads them through ``effects.infrastructure.rules_catalogue_io``. Both
need one spelling of every enum, so the spellings live here, in the package,
where ``scripts/`` can import them and ``effects`` never has to import a script.

Torch-free: ``build-corpus`` reads the map snapshot through this module.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from effects.domain.effect_head_layout import GLOBAL_LAYOUT, VERDICT_LAYOUT, HeadField
from effects.domain.fork_operation import ForkOperation
from effects.domain.records import RecordKind

# ── triage and scenario enums ────────────────────────────────────────────


class Category(StrEnum):
    """What kind of rule a triage row says a rule is."""

    MECHANICAL = "mechanical"
    LEGALITY = "legality"
    DECISION = "decision"
    DEFINITIONAL = "definitional"
    OUT_OF_SCOPE = "out_of_scope"


class Observability(StrEnum):
    """How directly a record shows the rule's outcome."""

    DIRECT = "direct"
    INDIRECT = "indirect"
    NEEDS_NEW_FIELDS = "needs_new_fields"
    NONE = "none"


class Polarity(StrEnum):
    """Which side of a rule a scenario variant, or a tagged record, is on."""

    FIRES = "fires"
    DOES_NOT_FIRE = "does_not_fire"
    OTHER = "other"


class MapVerdict(StrEnum):
    """A heads line's verdict on a rule: whether the model trains its outcome."""

    TRAINED = "trained"
    RECORDED = "recorded"
    MISSING = "missing"
    PLANNED = "planned"


class InputStatus(StrEnum):
    """Whether the model reads every input a rule's outcome depends on."""

    COMPLETE = "complete"
    INCOMPLETE = "incomplete"


class DepartureStatus(StrEnum):
    """How far a review has got with a disagreement between a scenario and Forge."""

    SUSPECTED = "suspected"
    CONFIRMED = "confirmed"


class DepartureResolution(StrEnum):
    """What was done about a confirmed departure."""

    UNPATCHED = "unpatched"
    PATCHED = "patched"
    ACCEPTED = "accepted"


CATEGORIES: frozenset[str] = frozenset(Category)
OBSERVABILITY: frozenset[str] = frozenset(Observability)
#: Every record kind a scenario may name; ``RecordKind`` is the record reader's own list.
RECORD_KINDS: frozenset[str] = frozenset(RecordKind)
SCENARIO_CATEGORIES: frozenset[str] = frozenset({Category.MECHANICAL, Category.LEGALITY})
POLARITIES: frozenset[str] = frozenset(Polarity)
OUTPUT_STATUS: frozenset[str] = frozenset(MapVerdict)
INPUT_STATUS: frozenset[str] = frozenset(InputStatus)
FORK_OPERATIONS: frozenset[str] = frozenset(ForkOperation)
#: The card-level operations a variant also states as a Forge script change
#: (``script_edit``), which the steered variant round applies (FR-058f).
SCRIPT_EDIT_OPERATIONS: frozenset[ForkOperation] = frozenset({
    ForkOperation.PT, ForkOperation.KEYWORD, ForkOperation.COLOUR_TYPE,
    ForkOperation.COUNTERS, ForkOperation.REPLACE_CARD,
})

# ── the model's output names, as a heads line spells them ────────────────

#: Generated output families: one field per counter type and per mana colour.
HEAD_OUTPUT_PREFIXES: tuple[str, ...] = ("counters_delta_", "mana_delta_")
#: A heads output that names something the model lacks: a payload field only
#: (``record:``), a proposed output (``new:``) or one a later generation plans.
OUTPUT_TAGS: tuple[str, ...] = ("record:", "new:", "planned:")
#: The same three roles for a missing input.
INPUT_TAGS: tuple[str, ...] = ("snapshot:", "new:", "planned:")

#: Outputs a heads line names that are neither a per-entity field nor a
#: verdict or global head field: the gate and the created-objects head.
_NON_ENTITY_OUTPUTS: tuple[str, ...] = ("gate", "created_objects")


def head_outputs() -> frozenset[str]:
    """Every untagged output name a heads line may use, from the layout itself.

    Derived rather than listed, so an output the layout gains is nameable at
    once and one it drops stops validating. The generated counter fields are
    matched by :data:`HEAD_OUTPUT_PREFIXES`, so the catch-all counter field is
    not listed by name.
    """
    entity = {f.value for f in HeadField if f is not HeadField.COUNTERS_DELTA_OTHER}
    record_heads = {str(spec.name) for spec in (*VERDICT_LAYOUT.fields, *GLOBAL_LAYOUT.fields)}
    return frozenset(entity | record_heads | set(_NON_ENTITY_OUTPUTS))


def is_model_output(name: str) -> bool:
    """Whether ``name`` is an output the model has, untagged."""
    return name in head_outputs() or name.startswith(HEAD_OUTPUT_PREFIXES)


# ── value objects read from the sidecars ─────────────────────────────────


@dataclass(frozen=True, slots=True)
class DetectorDeclaration:
    """How a scenario's outcome is found in a record (FR-002, research R1).

    ``family`` names a registered detector family and ``params`` its arguments;
    ``needs`` lists the gen-3 fields the detector reads, empty when it runs on
    gen-2 records too.
    """

    family: str
    record_kind: str
    outcome: str
    near_miss: str
    params: Mapping[str, Any] = field(default_factory=dict)
    needs: tuple[str, ...] = ()

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> DetectorDeclaration:
        return cls(
            family=data["family"],
            record_kind=data["record_kind"],
            outcome=data.get("outcome", ""),
            near_miss=data.get("near_miss", ""),
            params=dict(data.get("params") or {}),
            needs=tuple(data.get("needs") or ()),
        )


@dataclass(frozen=True, slots=True)
class NoDetector:
    """A scenario whose outcome no record field or event shows, and why."""

    reason: str


@dataclass(frozen=True, slots=True)
class Departure:
    """A recorded disagreement between a scenario and what Forge does (FR-005)."""

    status: DepartureStatus
    resolution: DepartureResolution | None = None
    forge_behaviour: str = ""
    note: str = ""

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> Departure:
        resolution = data.get("resolution")
        return cls(
            status=DepartureStatus(data["status"]),
            resolution=DepartureResolution(resolution) if resolution else None,
            forge_behaviour=data.get("forge_behaviour", ""),
            note=data.get("note", ""),
        )

    @property
    def blocks_rule_tests(self) -> bool:
        """A confirmed, unpatched departure keeps its rule out of rule tests (FR-071)."""
        return (self.status is DepartureStatus.CONFIRMED
                and self.resolution is DepartureResolution.UNPATCHED)


@dataclass(frozen=True, slots=True)
class Unclassifiable:
    """A non-base variant no fork operation can produce, and why."""

    reason: str


@dataclass(frozen=True, slots=True)
class ScriptEdit:
    """A variant's one fact as a change to one card's Forge script (FR-058f).

    ``card`` is the card as Forge's ``Name:`` line spells it; ``remove`` the
    lines of its script the change takes out, compared without surrounding
    spaces; ``add`` the lines it puts in. ``note`` says what the edit stands
    for where the script cannot say it alone, such as a replacement card the
    base card's script is edited toward.
    """

    card: str
    remove: tuple[str, ...] = ()
    add: tuple[str, ...] = ()
    note: str = ""

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> ScriptEdit:
        return cls(
            card=data["card"],
            remove=tuple(data.get("remove") or ()),
            add=tuple(data.get("add") or ()),
            note=data.get("note") or "",
        )

    def applied(self, lines: list[str]) -> list[str]:
        """``lines``, a script of :attr:`card`, with the edit made: each
        removed line's first match taken out, the added lines appended."""
        kept = list(lines)
        for removed in self.remove:
            stripped = [line.strip() for line in kept]
            if removed.strip() not in stripped:
                raise ValueError(f"{self.card}: no script line {removed!r} to remove")
            del kept[stripped.index(removed.strip())]
        return kept + list(self.add)


@dataclass(frozen=True, slots=True)
class ScenarioVariant:
    """One variant of a scenario: the base, which is the first, or a one-fact change to it."""

    label: str
    polarity: Polarity
    delta: str
    expected: str
    operation: ForkOperation | Unclassifiable | None = None
    is_base: bool = False
    script_edit: ScriptEdit | None = None


@dataclass(frozen=True, slots=True)
class Scenario:
    """A minimal-pair scenario of the catalogue, as the corpus code reads it.

    ``detector`` is None while a scenario has no declaration yet, which the
    validator refuses; readers treat it as a scenario that tags nothing.
    """

    id: str
    unit: str
    rules: tuple[str, ...]
    record_kind: str
    variants: tuple[ScenarioVariant, ...]
    detector: DetectorDeclaration | NoDetector | None = None
    departure: Departure | None = None
    #: Every card the scenario names, as Forge's ``Name:`` line spells it.
    cards: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ScenarioCatalogue:
    """Every scenario of the catalogue, with an index by rule id."""

    scenarios: tuple[Scenario, ...]

    def by_rule(self) -> dict[str, tuple[Scenario, ...]]:
        index: dict[str, list[Scenario]] = {}
        for scenario in self.scenarios:
            for rule in scenario.rules:
                index.setdefault(rule, []).append(scenario)
        return {rule: tuple(found) for rule, found in index.items()}

    def rules(self) -> frozenset[str]:
        return frozenset(rule for scenario in self.scenarios for rule in scenario.rules)


@dataclass(frozen=True, slots=True)
class RuleMapEntry:
    """One rule's heads line as ``build-corpus`` snapshots it (research R4).

    ``outputs`` holds only the untagged outputs the model has.
    """

    rule: str
    verdict: MapVerdict
    outputs: tuple[str, ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {"rule": self.rule, "verdict": str(self.verdict), "outputs": list(self.outputs)}

"""Freeze the selection set of one curated corpus (contracts/selection-set.md;
FR-068a, FR-068b, FR-070–FR-073; research E1–E6).

Run once per curated corpus, before any model exists, from the records and
their labels alone:

1. Read the manifest's halves, rule-map snapshot and budgets. Placement's
   halves are used unchanged and the freeze assigns no game to a half. A half
   holding fewer than ``min_games_per_half`` games refuses the freeze, naming
   stratum and half, before a record is read (E5).
2. Read both validation strata. Rule tests, probes and grids read each
   distinct record once; the deviance sample reads every copy (FR-068a).

   - Rule-test records: the catalogue's detectors' tags on rule-test-eligible
     records (``stratum_placement.rule_test_eligible``: not aimed, not a fork
     of any kind, not interventional, not synthetic, not polarity ``other``,
     not on the review list), of rules the map marks ``trained`` and no
     scenario of which carries a confirmed, unpatched departure (FR-071,
     FR-072). An accepted departure keeps its rule: its scenario's expected
     outcome already reads Forge's.
   - Probe items: what a :class:`ProbeItemSource` labels in each shard's
     distinct records, synthetic protection copies left out.
   - Grid records: per :class:`GridSource`, one qualifying record a game, the
     one of smallest keyed hash under the corpus seed, so grids qualifying
     the same records choose the same one; then, across games, the fewest of
     them by smallest keyed hash that bring the grid's worst-case standard
     error to ``grid_target_error``, at most ``grid_record_cap``
     (``selection_set.grid_records_needed``), since each grid record is read
     once per cell in every epoch (research E2, T160).
   - The deviance sample: every copy of the half, then the
     ``deviance_cap`` of smallest keyed hash, each copy weighed by its
     ``curation.class_weight`` (FR-059a) in its standard errors.
3. Balance with ``r`` by a seeded draw: each rule's larger polarity to r times
   its smaller, each probe type to r times the smallest probe type that
   meets the item target (E3). The grids are edit grids, whose cells all edit
   the same records, so no cell needs balancing.
4. Each item's worst-case standard error (E4, ``score_scales``); an item
   above ``max_item_error`` is left out and named, and every trained rule
   whose detector finds no test record in a half is named untested (FR-073),
   as is, in every half, one whose tagged records carry none of the outputs
   its map entry names (the manifest's ``effective_outputs`` empty, FR-060a).
   For a rule whose effective outputs are all per-entity, read at the tag's
   outcome entities, only the rows whose record carries one of those outputs
   at a named entity count toward n and the standard error, and a rule with
   a polarity holding none is left out as ``no outcome entity`` (finding 7.8);
   a rule with a verdict, global or
   created-objects output keeps every row, read at the record.
   A deviance item is an output term of any shipped head — a per-entity
   field or the gate, a verdict or global field, a created-objects slot
   column or the overflow flag (``selection_set.deviance_item``), the terms
   the loss denominators divide by — and its standard error is computed
   under both rule-mask modes, from a :class:`DevianceLosses` reader:
   ``√(Σ_g S_g²) / Σ_g S_g``, ``S_g`` summing over game g's copies the
   constant's loss times the copy's mask weight times its class weight,
   which is ``1/√n_eff`` for Kish's effective number of games
   ``n_eff = (Σ_g S_g)² / Σ_g S_g²``.
5. Every component's ``√(Σ SEᵢ²) / m`` at most ``max_component_error`` (both
   modes for the deviance component), and every grid cell holding records;
   otherwise the freeze refuses, naming stratum, half and component.
6. Write the set.

The thresholds are a :class:`FreezeThresholds`, E5's values its defaults, so
a test can pass small ones; no operator flag sets them.

Torch-free: the readers that need torch (:class:`DevianceLosses`) are passed in.
"""

from __future__ import annotations

import heapq
import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from effects.application.stratum_records import (
    StratumReader,
    shard_copies,
    stratum_copies,
)
from effects.domain.corpus_manifest import CorpusManifest
from effects.domain.grid_edits import GridQualification
from effects.domain.provenance import SidecarLookup
from effects.domain.records import EffectRecord
from effects.domain.rule_catalogue import Polarity, ScenarioCatalogue
from effects.domain.rule_detectors import DetectorIndex, RuleTag, tag_record
from effects.domain.rule_output_carriage import carried_outputs, read_at_entities_only
from effects.domain.rule_output_targets import (
    Indicators,
    RecordIndicators,
    informative_rows,
    reads_output,
)
from effects.domain.score_scales import (
    clustered_auc_standard_error,
    component_standard_error,
    grid_standard_error,
    loss_ratio_standard_error,
)
from effects.domain.seeded_draw import keyed_hash
from effects.domain.selection_set import (
    AMOUNT,
    BALANCE_RATIO,
    BINARY,
    DEVIANCE,
    DEVIANCE_SAMPLE_CAP,
    GRID_RECORD_CAP,
    GRID_TARGET_ERROR,
    GRIDS,
    NO_OUTCOME_ENTITY,
    NO_READABLE_OUTPUT,
    PROBES,
    RULE_TESTS,
    UNTESTED,
    UNTESTED_REASONS,
    BalanceDraw,
    FrozenGrid,
    GridCell,
    LeftOut,
    ProbeRecord,
    SampleRecord,
    SelectionSet,
    SelectionSetError,
    StandardError,
    TaggedRecord,
    balanced,
    error_value,
    grid_records_needed,
    item_key,
    selection_set_path,
    write_selection_set,
)
from effects.domain.strata import VALIDATION_STRATA, Half
from effects.domain.stratum_placement import (
    MAX_COMPONENT_ERROR,
    MAX_ITEM_ERROR,
    MIN_GAMES_PER_HALF,
    TEST_POLARITIES,
    rule_test_eligible,
)

#: The rule-mask modes an output field's standard error is computed under
#: (``rule_mask.RuleMask``'s values; FR-061).
MASK_MODES: tuple[str, ...] = ("hard", "soft")


# ── the readers the freeze is handed ─────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ProbeItem:
    """One probe item a :class:`ProbeItemSource` labels: its probe type, how
    the type is scored (``binary`` by AUC, ``amount`` by R²), the record, the
    entity it is read at (None at ``[ACT]``) and its label."""

    probe_type: str
    kind: str
    record_id: str
    entity: str | None
    label: float


class ProbeItemSource(Protocol):
    """Labels the rung-3 probe items of one shard's distinct records (a game
    never spans a shard, so a game-wide join such as the trigger's sees every
    record it needs)."""

    def items(self, records: Sequence[EffectRecord]) -> Iterable[ProbeItem]: ...


class GridSource(Protocol):
    """One edit grid: its name, its cells, and which records it edits
    (:class:`~effects.domain.grid_edits.GridQualification`); the shipped
    grids are ``grid_edits.grid_sources``."""

    name: str
    cells: tuple[str, ...]

    def qualify(self, record: EffectRecord) -> GridQualification | None: ...


class DevianceLosses(Protocol):
    """The constant prediction's loss on each record of a deviance sample.

    Returns ``item -> mode -> losses``, one loss per record in the order
    given, each the item's constant-prediction loss on the record weighted as
    the run's ``--rule-mask`` mode weighs that copy (research E4); an item is
    any shipped head's output term, named by ``selection_set.deviance_item``,
    and one the sample never supervises may be absent.
    """

    def losses(
        self, records: Iterable[EffectRecord],
    ) -> Mapping[str, Mapping[str, Sequence[float]]]: ...


@dataclass(frozen=True)
class RuleTestRules:
    """Which rules may have rule tests, and the detectors that find them."""

    #: ``record -> its rule tags`` (``tag_record`` over the catalogue's index).
    tag: Callable[[EffectRecord], Sequence[RuleTag]]
    #: Rules a declared detector can tag.
    declared: frozenset[str]
    #: The rules the manifest's map marks ``trained`` (FR-054, FR-071).
    trained: frozenset[str]
    #: Rules a scenario of which carries a confirmed, unpatched departure.
    blocked: frozenset[str] = frozenset()
    #: Trained rules whose tagged records carry no output of their map
    #: entry (the manifest's ``effective_outputs`` empty): nothing to read.
    unreadable: frozenset[str] = frozenset()
    #: Rules whose effective outputs are all per-entity
    #: (``rule_output_carriage.read_at_entities_only``), each with those
    #: outputs: a row whose record carries none of them at its outcome
    #: entities is read nowhere (the scorer's ``untargeted``), so only rows
    #: carrying one are tests.
    entity_read: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    #: Each mapped rule's effective outputs (the manifest's
    #: ``effective_rule_outputs``): a rule held here is frozen with the
    #: outputs alone whose targets settle a direction over its drawn rows
    #: (``rule_output_targets``, finding 12.9), as the control score reads it.
    outputs: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    @classmethod
    def of(cls, catalogue: ScenarioCatalogue, manifest: CorpusManifest,
           lines: SidecarLookup | None = None) -> RuleTestRules:
        index = DetectorIndex.of(catalogue)
        return cls(
            tag=lambda record: tag_record(record, index, lines).tags,
            declared=index.declared_rules,
            trained=manifest.budgeted_rules,
            blocked=blocked_rules(catalogue),
            unreadable=frozenset(
                rule for rule, outputs in manifest.effective_rule_outputs.items()
                if not outputs),
            entity_read={rule: tuple(outputs)
                         for rule, outputs in manifest.effective_rule_outputs.items()
                         if read_at_entities_only(outputs)},
            outputs=manifest.effective_rule_outputs,
        )

    @property
    def testable(self) -> frozenset[str]:
        """Rules a detector can find tests for: declared, trained, not
        blocked, with an output left to read."""
        return (self.declared & self.trained) - self.blocked - self.unreadable

    @property
    def unreadable_testable(self) -> frozenset[str]:
        """Rules testable but for having no output left to read."""
        return ((self.declared & self.trained) - self.blocked) & self.unreadable

    def tests_of(self, record: EffectRecord) -> list[tuple[str, Polarity, tuple[str, ...]]]:
        """``(rule, polarity, outcome entities)`` for each rule-test tag of
        ``record``: a rule test reads its per-entity outputs where the rule's
        outcome falls (research R1)."""
        testable = self.testable
        return [
            (tag.rule, tag.polarity, self._read_at(record, tag)) for tag in self.tag(record)
            if tag.rule in testable and rule_test_eligible(record, tag)
        ]

    def _read_at(self, record: EffectRecord, tag: RuleTag) -> tuple[str, ...]:
        """The tag's outcome entities; none for an entity-read rule whose
        record carries no effective output of the rule at them."""
        outputs = self.entity_read.get(tag.rule)
        if outputs is None or carried_outputs(record, tag.outcome, outputs):
            return tag.outcome
        return ()


def blocked_rules(catalogue: ScenarioCatalogue) -> frozenset[str]:
    """Rules a scenario of which carries a confirmed departure neither patched
    nor accepted (FR-005, FR-071)."""
    return frozenset(
        rule for scenario in catalogue.scenarios
        if scenario.departure is not None and scenario.departure.blocks_rule_tests
        for rule in scenario.rules
    )


# ── refusals ─────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class FreezeMiss:
    """One reason the freeze refuses: a half too small, a component above its
    precision target, or a grid with empty cells (FR-068a, FR-068b)."""

    stratum: str
    half: str
    component: str
    detail: str

    def describe(self) -> str:
        return f"{self.stratum} {self.half}, {self.component}: {self.detail}"


class SelectionSetRefused(RuntimeError):
    """The freeze refused; ``misses`` names each stratum, half and component,
    and ``selection`` is the set as far as it was assembled (None when a half
    was too small to read), so its left-out items can still be named."""

    def __init__(self, misses: Sequence[FreezeMiss],
                 selection: SelectionSet | None = None) -> None:
        self.misses = tuple(misses)
        self.selection = selection
        super().__init__("\n".join(miss.describe() for miss in self.misses))


def left_out_lines(selection: SelectionSet, *, limit: int = 12) -> list[str]:
    """One line per half and component naming what it left out, the first
    ``limit`` items by name and the rest counted; untested rules are counted
    rather than listed, as most rules are untested in a small half. The set
    itself names every one."""
    return [_left_out_line(f"{stratum} {half}, {component}", items, limit)
            for stratum, by_half in sorted(selection.left_out.items())
            for half, by_component in sorted(by_half.items())
            for component, items in sorted(by_component.items())]


def _left_out_line(where: str, items: Sequence[LeftOut], limit: int) -> str:
    untested = sum(1 for item in items if item.reason in UNTESTED_REASONS)
    named = [f"{item.item} ({item.reason})" for item in items
             if item.reason not in UNTESTED_REASONS]
    if len(named) > limit:
        named = [*named[:limit], f"{len(named) - limit} more"]
    parts = named + ([f"{untested} rule(s) untested"] if untested else [])
    return f"{where} left out: " + "; ".join(parts)


def small_halves(manifest: CorpusManifest, minimum: int) -> list[FreezeMiss]:
    """Every half of the manifest holding fewer than ``minimum`` games (E5)."""
    misses = []
    for stratum, by_half in sorted(manifest.halves.items()):
        for half in Half:
            games = len(by_half.get(str(half), ()))
            if games < minimum:
                misses.append(FreezeMiss(
                    stratum, str(half), "games",
                    f"{games} game(s), fewer than the {minimum} a half needs"))
    if not manifest.halves:
        misses.append(FreezeMiss("every stratum", "every half", "games",
                                 "the manifest records no halves (a gen-3 build writes them)"))
    return misses



# ── the freeze ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FreezeThresholds:
    """Research E5's thresholds and E3's r, handed to the freeze and to
    :func:`run`; never flags."""

    r: int = BALANCE_RATIO
    min_games_per_half: int = MIN_GAMES_PER_HALF
    max_item_error: float = MAX_ITEM_ERROR
    max_component_error: float = MAX_COMPONENT_ERROR
    deviance_cap: int = DEVIANCE_SAMPLE_CAP
    grid_target_error: float = GRID_TARGET_ERROR
    grid_record_cap: int = GRID_RECORD_CAP


@dataclass(frozen=True)
class FreezePorts:
    """The rules and readers the freeze is handed: which rules have tests,
    the probe items' labeller, the edit grids and the deviance sample's loss
    reader. A component given no reader (None, or no grid) holds no item."""

    rules: RuleTestRules
    probes: ProbeItemSource | None
    grids: Sequence[GridSource]
    deviance: DevianceLosses | None


@dataclass
class _Half:
    """What pass one gathers for one half."""

    rule_tests: dict[str, dict[Polarity, list[TaggedRecord]]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(list)))
    probes: dict[str, list[ProbeRecord]] = field(default_factory=lambda: defaultdict(list))
    #: ``grid -> game -> (hash, record, qualification)``: one record a game.
    grids: dict[str, dict[str, tuple]] = field(default_factory=lambda: defaultdict(dict))
    #: A bounded max-heap (negated hashes) of the deviance sample's copies.
    sample: list[tuple[int, str, str, int]] = field(default_factory=list)
    #: ``(rule, row) -> output -> its indicators on the row``, for each rule
    #: :attr:`RuleTestRules.outputs` holds.
    rule_targets: dict[tuple[str, TaggedRecord], dict[str, Indicators | None]] = field(
        default_factory=dict)


@dataclass(frozen=True, slots=True)
class _HalfId:
    """One half of one stratum, as pass two names it."""

    stratum: str
    half: str

    def key(self, component: str, item: str) -> str:
        return item_key(self.stratum, self.half, component, item)


def freeze_selection_set(
    manifest: CorpusManifest,
    strata: Mapping[str, StratumReader],
    *,
    ports: FreezePorts,
    thresholds: FreezeThresholds = FreezeThresholds(),
) -> SelectionSet:
    """The corpus's selection set, unsealed; raises :class:`SelectionSetRefused`
    naming every miss. ``strata`` reads each validation stratum the manifest
    halves."""
    small = small_halves(manifest, thresholds.min_games_per_half)
    if small:
        raise SelectionSetRefused(small)
    gathering = _Gathering(manifest, ports, cap=thresholds.deviance_cap)
    for stratum in sorted(manifest.halves):
        gathering.read(strata[stratum], stratum)
    out = _Assembly(manifest, thresholds, gathering.probe_kinds, ports.rules)
    for where, found in gathering.in_half_order():
        out.half(where, found, strata[where.stratum], ports)
    if out.misses:
        raise SelectionSetRefused(out.misses, out.selection())
    return out.selection()


class _Gathering:
    """Pass one over each stratum: rule tests, probe items and grid records
    over distinct records, the deviance sample over every copy."""

    def __init__(self, manifest: CorpusManifest, ports: FreezePorts, *, cap: int) -> None:
        self.ports = ports
        self.seed = manifest.seed
        self.cap = cap
        self.half_of = {
            game: (stratum, half)
            for stratum, by_half in manifest.halves.items()
            for half, games in by_half.items() for game in games
        }
        self.gathered: dict[tuple[str, str], _Half] = {
            (stratum, half): _Half()
            for stratum, by_half in manifest.halves.items() for half in by_half
        }
        self.probe_kinds: dict[str, str] = {}

    def in_half_order(self) -> list[tuple[_HalfId, _Half]]:
        """Each half's gathering, by stratum, then selection before reporting."""
        order = {str(half): i for i, half in enumerate(Half)}
        return [(_HalfId(stratum, half), found) for (stratum, half), found in sorted(
            self.gathered.items(), key=lambda item: (item[0][0], order.get(item[0][1], 2)))]

    def read(self, read: StratumReader, stratum: str) -> None:
        """Pass one over ``stratum``, shard by shard: a game never spans a
        shard, so the probes label each shard's distinct records together."""
        for shard in shard_copies(read):
            distinct = [record for record, copy in shard if self._hold(record, copy, stratum)]
            self._probe_items(distinct)

    def _hold(self, record: EffectRecord, copy: int, stratum: str) -> bool:
        """Hold ``record`` where its half of ``stratum`` reads it: every copy
        in the deviance sample, a first occurrence in the rule tests and,
        unless synthetic, in the grids; whether the probes read it too."""
        placed = self.half_of.get(record.game_id)
        if placed is None or placed[0] != stratum:
            return False
        found = self.gathered[placed]
        self._sample(found.sample, record, copy)
        if copy:
            return False
        _hold_rule_tests(found, self.ports.rules, record)
        if record.synthetic:
            return False
        self._hold_grids(found, record)
        return True

    def _sample(self, heap: list, record: EffectRecord, copy: int) -> None:
        """Keep the ``cap`` copies of smallest keyed hash, as a max-heap."""
        if self.cap <= 0:
            return
        order = keyed_hash(f"{record.record_id}#{copy}", self.seed)
        entry = (-order, record.record_id, record.game_id, copy)
        if len(heap) < self.cap:
            heapq.heappush(heap, entry)
        elif order < -heap[0][0]:
            heapq.heapreplace(heap, entry)

    def _hold_grids(self, found: _Half, record: EffectRecord) -> None:
        """``record`` as its game's record of each grid it qualifies, where its
        keyed hash is the game's smallest so far (the first read on a tie)."""
        for grid in self.ports.grids:
            qualified = grid.qualify(record)
            if qualified is None:
                continue
            # Keyed by the record alone, so grids qualifying the same
            # records (the combat grid's variants) choose the same ones.
            order = keyed_hash(record.record_id, self.seed)
            held = found.grids[grid.name].get(record.game_id)
            if held is None or order < held[0]:
                found.grids[grid.name][record.game_id] = (order, record.record_id, qualified)

    def _probe_items(self, distinct: list[EffectRecord]) -> None:
        """The probe items of one shard's distinct records, each held in its
        game's half."""
        probes = self.ports.probes
        if probes is None or not distinct:
            return
        game_of = {record.record_id: record.game_id for record in distinct}
        for item in probes.items(distinct):
            game = game_of.get(item.record_id)
            if game is not None:
                self._hold_probe(item, game)

    def _hold_probe(self, item: ProbeItem, game: str) -> None:
        known = self.probe_kinds.setdefault(item.probe_type, item.kind)
        if known != item.kind:
            raise ValueError(f"probe type {item.probe_type} is labelled both "
                             f"{known} and {item.kind}")
        self.gathered[self.half_of[game]].probes[item.probe_type].append(
            ProbeRecord(item.record_id, game, item.entity, float(item.label)))


def _hold_rule_tests(found: _Half, rules: RuleTestRules, record: EffectRecord) -> None:
    """``record``'s rule-test rows, each with its rule's outputs' indicators
    where ``rules`` holds the rule's outputs."""
    indicators = RecordIndicators(record)
    for rule, polarity, entities in rules.tests_of(record):
        row = TaggedRecord(record.record_id, record.game_id, tuple(entities))
        found.rule_tests[rule][polarity].append(row)
        if rule in rules.outputs:
            found.rule_targets[(rule, row)] = {
                output: indicators.of(output, row.entities)
                for output in rules.outputs[rule] if reads_output(output)}


def _games_error(positives: Sequence, negatives: Sequence) -> float:
    """The clustered AUC standard error of rows carrying a ``game_id`` (E4)."""
    return clustered_auc_standard_error([row.game_id for row in positives],
                                        [row.game_id for row in negatives])


class _Assembly:
    """Pass two: balance, standard errors, left-out items and the checks."""

    def __init__(self, manifest: CorpusManifest, limits: FreezeThresholds,
                 probe_kinds: Mapping[str, str], rules: RuleTestRules) -> None:
        self.manifest = manifest
        self.limits = limits
        self.seed = manifest.seed
        self.probe_kinds = dict(probe_kinds)
        self.testable = sorted(rules.testable)
        self.unreadable = sorted(rules.unreadable_testable)
        self.entity_read = rules.entity_read
        self.misses: list[FreezeMiss] = []
        self.errors: dict[str, StandardError] = {}
        self.left_out: dict[str, dict[str, dict[str, list[LeftOut]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list)))
        self.frozen_rule_tests: dict = defaultdict(dict)
        self.frozen_probes: dict = defaultdict(dict)
        self.frozen_grids: dict = defaultdict(dict)
        self.frozen_samples: dict = defaultdict(dict)

    def half(self, where: _HalfId, found: _Half, read: StratumReader,
             ports: FreezePorts) -> None:
        """Pass two over one half: rule tests, probes, grids, deviance."""
        self.rule_tests(where, found)
        self.probes(where, found)
        self.grids(where, found, ports.grids)
        self.deviance(where, found, read, ports.deviance)

    def _leave_out(self, where: _HalfId, component: str, item: str, reason: str) -> None:
        self.left_out[where.stratum][where.half][component].append(LeftOut(item, reason))

    def _leave_out_all(self, where: _HalfId, component: str,
                       items: Iterable[tuple[str, str]]) -> None:
        self.left_out[where.stratum][where.half][component] += [LeftOut(*item) for item in items]

    def _enters(self, where: _HalfId, component: str, item: str, error: float) -> bool:
        """Record ``item``'s standard error; whether it meets the item target."""
        self.errors[where.key(component, item)] = error_value(error)
        if error <= self.limits.max_item_error:
            return True
        self._leave_out(where, component, item, _above(error, self.limits))
        return False

    def _check(self, where: _HalfId, component: str, errors: Sequence[float],
               *, mode: str | None = None) -> None:
        """Refuse a component above its precision target (FR-068b)."""
        combined = component_standard_error(errors)
        if combined <= self.limits.max_component_error:
            return
        under = f" under the {mode} rule mask" if mode else ""
        detail = ("no item meets its precision target" + under if not errors else
                  f"standard error {combined:.4f} over {len(errors)} item(s){under}, "
                  f"above {self.limits.max_component_error}")
        self.misses.append(FreezeMiss(where.stratum, where.half, component, detail))

    # ── rule tests ──

    def rule_tests(self, where: _HalfId, found: _Half) -> None:
        errors, kept = [], {}
        for rule in self.unreadable:
            self._leave_out(where, RULE_TESTS, rule, NO_READABLE_OUTPUT)
        for rule in self.testable:
            tested = self._rule_test(where, rule, found)
            if tested is None:
                continue
            drawn, error = tested
            if self._enters(where, RULE_TESTS, rule, error):
                errors.append(error)
                kept[rule] = {str(p): tuple(rows) for p, rows in drawn.items()}
        self.frozen_rule_tests[where.stratum][where.half] = kept
        self._check(where, RULE_TESTS, errors)

    def _rule_test(self, where: _HalfId, rule: str, found: _Half,
                   ) -> tuple[dict[Polarity, list[TaggedRecord]], float] | None:
        """A rule's drawn rows and their standard error, counting the rows a
        kept output reads; None, the rule named in ``left_out``, when it is
        left out before its error, and each output left out named too."""
        sides = found.rule_tests.get(rule)
        if not sides or not any(sides.values()):
            self._leave_out(where, RULE_TESTS, rule, UNTESTED)
            return None
        rows, reason = self._test_rows(rule, sides)
        if reason is None:
            drawn = balanced(rows, self.limits.r, draw=BalanceDraw(
                self.seed, f"rule-test|{where.stratum}|{where.half}|{rule}",
                lambda row: row.record_id))
            read, left = informative_rows(rule, drawn, found.rule_targets)
            self._leave_out_all(where, RULE_TESTS, left)
            if read is not None:
                return drawn, _games_error(read[Polarity.FIRES], read[Polarity.DOES_NOT_FIRE])
        else:
            self._leave_out(where, RULE_TESTS, rule, reason)
        self.errors[where.key(RULE_TESTS, rule)] = None
        return None

    def _test_rows(self, rule: str, sides: Mapping[Polarity, list[TaggedRecord]],
                   ) -> tuple[dict[Polarity, list[TaggedRecord]], str | None]:
        """A rule's test rows per polarity, or why it is left out: a polarity
        with no row, or, for a rule read at entities only, none naming an
        outcome entity, the only rows such a rule's test reads (finding 7.8)."""
        rows = {polarity: list(sides.get(polarity, ())) for polarity in TEST_POLARITIES}
        counts = {polarity: len(found) for polarity, found in rows.items()}
        if min(counts.values()) == 0:
            return {}, "one polarity only: " + ", ".join(
                f"{n} {polarity}" for polarity, n in counts.items())
        if rule not in self.entity_read:
            return rows, None
        return _rows_naming_an_entity(rows, counts)

    # ── probes ──

    def _probe_error(self, probe_type: str, rows: Sequence[ProbeRecord]) -> float:
        if self.probe_kinds[probe_type] == BINARY:
            return _games_error([row for row in rows if row.label >= 0.5],
                                [row for row in rows if row.label < 0.5])
        if not rows:
            return math.inf
        mean = sum(row.label for row in rows) / len(rows)
        return loss_ratio_standard_error(
            [row.game_id for row in rows], [(row.label - mean) ** 2 for row in rows])

    def probes(self, where: _HalfId, found: _Half) -> None:
        """Probe types meeting the item target, balanced against the smallest of
        them, and checked again after the draw."""
        drawn = balanced(self._probe_candidates(where, found), self.limits.r, draw=BalanceDraw(
            self.seed, f"probe|{where.stratum}|{where.half}",
            lambda row: f"{row.record_id}|{row.entity}"))
        errors, kept = [], {}
        for probe_type, rows in sorted(drawn.items()):
            error = self._probe_error(probe_type, rows)
            if self._enters(where, PROBES, probe_type, error):
                errors.append(error)
                kept[probe_type] = tuple(rows)
        self.frozen_probes[where.stratum][where.half] = kept
        self._check(where, PROBES, errors)

    def _probe_candidates(self, where: _HalfId, found: _Half) -> dict[str, list[ProbeRecord]]:
        """Each probe type's distinct rows where they meet the item target
        before the draw; a type that does not is left out and named."""
        candidates = {}
        for probe_type in sorted(found.probes):
            rows = _distinct(found.probes[probe_type])
            error = self._probe_error(probe_type, rows)
            if error <= self.limits.max_item_error:
                candidates[probe_type] = rows
            else:
                self._enters(where, PROBES, probe_type, error)
        return candidates

    # ── grids ──

    def grids(self, where: _HalfId, found: _Half, sources: Sequence[GridSource]) -> None:
        errors, kept = [], {}
        for grid in sources:
            frozen = self._grid(where, grid, found.grids.get(grid.name, {}))
            if frozen is not None:
                errors.append(frozen[1])
                kept[grid.name] = frozen[0]
        self.frozen_grids[where.stratum][where.half] = kept
        self._check(where, GRIDS, errors)

    def _grid(self, where: _HalfId, grid: GridSource,
              qualified: Mapping[str, tuple]) -> tuple[FrozenGrid, float] | None:
        """The grid frozen over the records it keeps, with its standard error;
        None when every cell is empty (a miss) or the grid is left out."""
        if not qualified:
            self.misses.append(FreezeMiss(where.stratum, where.half, GRIDS,
                                          f"grid {grid.name}: every cell is empty"))
            return None
        chosen = _grid_records(qualified, len(grid.cells), self.limits)
        records = tuple(TaggedRecord(record_id, game, q.entities)
                        for game, record_id, q in chosen)
        answers = _answers([q for *_, q in chosen], len(grid.cells))
        error = grid_standard_error(answers)
        if not self._enters(where, GRIDS, grid.name, error):
            return None
        return FrozenGrid(
            tuple(GridCell(cell, answer) for cell, answer in zip(grid.cells, answers)),
            records), error

    # ── deviance ──

    def deviance(self, where: _HalfId, found: _Half, read: StratumReader,
                 losses: DevianceLosses | None) -> None:
        chosen = sorted(((record_id, game, copy) for _, record_id, game, copy in found.sample),
                        key=lambda row: (row[1], row[0], row[2]))
        self.frozen_samples[where.stratum][where.half] = tuple(SampleRecord(*row) for row in chosen)
        if losses is None:
            self.misses.append(FreezeMiss(where.stratum, where.half, DEVIANCE,
                                          "no reader of the constant-prediction losses"))
            return
        wanted = {(record_id, copy) for record_id, _, copy in chosen}
        games: list[str] = []
        by_field = losses.losses(_streamed(read, wanted, games))
        for mode in MASK_MODES:
            self._deviance_mode(where, by_field, games, mode)

    def _deviance_mode(self, where: _HalfId,
                       by_field: Mapping[str, Mapping[str, Sequence[float]]],
                       games: Sequence[str], mode: str) -> None:
        """Each deviance item's standard error under the ``mode`` rule mask,
        and the component's check."""
        errors = []
        for name in sorted(by_field):
            error = _loss_ratio_error(games, by_field[name].get(mode))
            held = self.errors.setdefault(where.key(DEVIANCE, name), {})
            held[mode] = error_value(error)
            if error <= self.limits.max_item_error:
                errors.append(error)
            else:
                self._leave_out(where, DEVIANCE, name,
                                f"{_above(error, self.limits)} under the {mode} rule mask")
        self._check(where, DEVIANCE, errors, mode=mode)

    def selection(self) -> SelectionSet:
        return SelectionSet(
            corpus_digest=self.manifest.digest(),
            r=self.limits.r,
            halves={s: {h: tuple(sorted(g)) for h, g in by.items()}
                    for s, by in self.manifest.halves.items()},
            probes=_plain(self.frozen_probes),
            probe_kinds=self.probe_kinds,
            grids=_plain(self.frozen_grids),
            rule_tests=_plain(self.frozen_rule_tests),
            deviance_samples=_plain(self.frozen_samples),
            standard_errors=dict(self.errors),
            left_out={s: {h: {c: tuple(rows) for c, rows in by_c.items()}
                          for h, by_c in by_h.items()}
                      for s, by_h in self.left_out.items()},
            settings={
                "seed": self.seed,
                "min_games_per_half": self.limits.min_games_per_half,
                "max_item_error": self.limits.max_item_error,
                "max_component_error": self.limits.max_component_error,
                "deviance_cap": self.limits.deviance_cap,
                "grid_target_error": self.limits.grid_target_error,
                "grid_record_cap": self.limits.grid_record_cap,
            },
        )


def _rows_naming_an_entity(
    rows: Mapping[Polarity, list[TaggedRecord]], counts: Mapping[Polarity, int],
) -> tuple[dict[Polarity, list[TaggedRecord]], str | None]:
    """A rule read at entities only: its rows naming an outcome entity, or,
    when a polarity has none, why the rule is left out."""
    named = {polarity: [row for row in found if row.entities]
             for polarity, found in rows.items()}
    if all(named.values()):
        return named, None
    return {}, f"{NO_OUTCOME_ENTITY}: " + ", ".join(
        f"{len(named[polarity])} of {n} {polarity}" for polarity, n in counts.items()
    ) + " name one"


def _grid_records(qualified: Mapping[str, tuple], cells: int,
                  limits: FreezeThresholds) -> list[tuple[str, str, GridQualification]]:
    """``(game, record, qualification)`` of the records a grid keeps, by game:
    sized to the grid's precision target over every qualifying game, then
    judged on the records kept (FR-068b)."""
    by_hash = sorted((order, game, record_id, q)
                     for game, (order, record_id, q) in qualified.items())
    needed = grid_records_needed(
        _answers([q for *_, q in by_hash], cells),
        target=limits.grid_target_error, cap=limits.grid_record_cap)
    return sorted((game, record_id, q) for _, game, record_id, q in by_hash[:max(needed, 1)])


def _answers(qualifications: Sequence[GridQualification],
             cells: int) -> list[tuple[float, ...]]:
    """Each cell's rule outcome for each of a grid's records, in their order."""
    return [tuple(q.outcomes[c] for q in qualifications) for c in range(cells)]


def _loss_ratio_error(games: Sequence[str], per_record: Sequence[float] | None) -> float:
    """A deviance item's standard error; infinite where the item has no losses."""
    return math.inf if per_record is None else loss_ratio_standard_error(games, per_record)


def _above(error: float, limits: FreezeThresholds) -> str:
    if math.isinf(error):
        return "no standard error: its labels cannot bound one"
    return f"standard error {error:.4f} above {limits.max_item_error}"


def _distinct(rows: Sequence[ProbeRecord]) -> list[ProbeRecord]:
    """One row per (record, entity): a probe reads each distinct record once."""
    seen: set[tuple[str, str | None]] = set()
    out = []
    for row in rows:
        if (row.record_id, row.entity) not in seen:
            seen.add((row.record_id, row.entity))
            out.append(row)
    return out


def _plain(nested: Mapping) -> dict:
    return {stratum: dict(by_half) for stratum, by_half in nested.items()}


def _copies(read: StratumReader, wanted: set[tuple[str, int]]) -> Iterator[EffectRecord]:
    """The stratum's copies named in ``wanted``, in reading order."""
    return (record for record, copy in stratum_copies(read) if (record.record_id, copy) in wanted)


def _streamed(read: StratumReader, wanted: set[tuple[str, int]],
              games: list[str]) -> Iterator[EffectRecord]:
    """The copies named in ``wanted``, each one's game appended to ``games``
    as it is read: streamed rather than held, since a sample is up to 16,384
    parsed records."""
    for record in _copies(read, wanted):
        games.append(record.game_id)
        yield record


# ── the command ──────────────────────────────────────────────────────────


def stratum_reader(shards: Sequence[Path]) -> StratumReader:
    """A stratum's listed ``shards`` in the order given, each read lazily
    (``curated_shards.require_listed_shards`` names them: no directory is
    walked, so a stale shard is never read)."""
    from effects.infrastructure.shard_io import read_shard

    def read() -> Iterable[Iterable[EffectRecord]]:
        return (read_shard(path) for path in shards)

    return read


@dataclass(frozen=True)
class FreezeRunPorts:
    """What :func:`run` freezes a corpus with: the rules catalogue and the
    sidecars its detectors read, the probe and grid readers, the builder of
    the deviance loss reader from the manifest, and where its lines go."""

    rules_dir: Path = Path("rules")
    lines: SidecarLookup | None = None
    probes: ProbeItemSource | None = None
    grids: Sequence[GridSource] = ()
    deviance: Callable[[CorpusManifest], DevianceLosses] | None = None
    echo: Callable[[str], None] = print

    def freeze_ports(self, manifest: CorpusManifest) -> FreezePorts:
        """The freeze's ports over ``manifest``, the deviance reader built
        last; called once every half holds its games."""
        from effects.infrastructure.rules_catalogue_io import load_scenarios

        return FreezePorts(
            rules=RuleTestRules.of(load_scenarios(self.rules_dir), manifest, self.lines),
            probes=self.probes, grids=self.grids,
            deviance=None if self.deviance is None else self.deviance(manifest),
        )


def run(
    corpus: Path,
    *,
    reports_dir: Path,
    ports: FreezeRunPorts = FreezeRunPorts(),
    thresholds: FreezeThresholds = FreezeThresholds(),
) -> int:
    """Freeze ``corpus``'s selection set into ``reports_dir``: 0 once written,
    1 after naming every miss. ``ports.deviance`` builds the loss reader from
    the manifest, and is called only once every half holds its games."""
    from effects.infrastructure.corpus_store import CorpusStore

    store = CorpusStore(Path(corpus))
    manifest = store.load()
    try:
        selection = _freeze_corpus(store, manifest, ports, thresholds)
    except SelectionSetRefused as refused:
        _echo_refusal(ports.echo, corpus, refused)
        return 1
    return _write(corpus, selection_set_path(reports_dir, manifest.digest()), selection,
                  ports.echo)


def _freeze_corpus(store, manifest: CorpusManifest, ports: FreezeRunPorts,
                   thresholds: FreezeThresholds) -> SelectionSet:
    """The corpus's set over the validation strata's listed shards, once every
    half holds its games and the shards on disk are the listed ones."""
    from effects.infrastructure.curated_shards import require_listed_shards

    small = small_halves(manifest, thresholds.min_games_per_half)
    if small:
        raise SelectionSetRefused(small)
    listed = require_listed_shards(store, manifest)
    return freeze_selection_set(
        manifest,
        {str(stratum): stratum_reader(listed[str(stratum)]) for stratum in VALIDATION_STRATA},
        ports=ports.freeze_ports(manifest), thresholds=thresholds,
    )


def _echo_refusal(echo: Callable[[str], None], corpus: Path,
                  refused: SelectionSetRefused) -> None:
    echo(f"selection set not frozen for {corpus}:")
    for miss in refused.misses:
        echo(f"  {miss.describe()}")
    lines = [] if refused.selection is None else left_out_lines(refused.selection)
    for line in lines:
        echo(f"  {line}")


def _write(corpus: Path, path: Path, selection: SelectionSet,
           echo: Callable[[str], None]) -> int:
    try:
        written = write_selection_set(path, selection)
    except SelectionSetError as refused:
        echo(f"selection set not written for {corpus}: {refused}")
        return 1
    echo(_froze_line(selection, written))
    return 0


def _froze_line(selection: SelectionSet, path: Path) -> str:
    rule_tests = sum(len(rules) for by_half in selection.rule_tests.values()
                     for rules in by_half.values())
    probes = sum(len(p) for b in selection.probes.values() for p in b.values())
    grids = sum(len(g) for b in selection.grids.values() for g in b.values())
    return (f"froze {rule_tests} rule test(s), {probes} probe type(s), {grids} grid(s) "
            f"over {len(selection.halves)} strata -> {path}")


__all__ = [
    "AMOUNT", "BINARY", "MASK_MODES", "DevianceLosses", "FreezeMiss", "FreezePorts",
    "FreezeRunPorts", "FreezeThresholds", "GridQualification", "GridSource", "ProbeItem",
    "ProbeItemSource", "RuleTestRules", "SelectionSetRefused", "blocked_rules",
    "freeze_selection_set", "left_out_lines", "run", "small_halves", "stratum_reader",
]

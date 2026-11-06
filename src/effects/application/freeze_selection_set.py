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
     ``deviance_cap`` of smallest keyed hash.
3. Balance with ``r`` by a seeded draw: each rule's larger polarity to r times
   its smaller, each probe type to r times the smallest probe type that
   meets the item target (E3). The grids are edit grids, whose cells all edit
   the same records, so no cell needs balancing.
4. Each item's worst-case standard error (E4, ``score_scales``); an item
   above ``max_item_error`` is left out and named, and every trained rule
   whose detector finds no test record in a half is named untested (FR-073),
   as is, in every half, one whose tagged records carry none of the outputs
   its map entry names (the manifest's ``effective_outputs`` empty, FR-060a).
   A deviance item is an output term of any shipped head — a per-entity
   field or the gate, a verdict or global field, a created-objects slot
   column or the overflow flag (``selection_set.deviance_item``), the terms
   the loss denominators divide by — and its standard error is computed
   under both rule-mask modes, from a :class:`DevianceLosses` reader.
5. Every component's ``√(Σ SEᵢ²) / m`` at most ``max_component_error`` (both
   modes for the deviance component), and every grid cell holding records;
   otherwise the freeze refuses, naming stratum, half and component.
6. Write the set.

The thresholds are keyword arguments with E5's values as defaults, so a test
can pass small ones; no operator flag sets them.

Torch-free: the readers that need torch (:class:`DevianceLosses`) are passed in.
"""

from __future__ import annotations

import heapq
import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from effects.domain.corpus_manifest import CorpusManifest
from effects.domain.grid_edits import GridQualification
from effects.domain.provenance import SidecarLookup
from effects.domain.records import EffectRecord
from effects.domain.rule_catalogue import Polarity, ScenarioCatalogue
from effects.domain.rule_detectors import DetectorIndex, RuleTag, tag_record
from effects.domain.score_scales import (
    auc_standard_error,
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
    NO_READABLE_OUTPUT,
    PROBES,
    RULE_TESTS,
    UNTESTED,
    UNTESTED_REASONS,
    FrozenGrid,
    GridCell,
    LeftOut,
    ProbeRecord,
    SampleRecord,
    SelectionSet,
    StandardError,
    TaggedRecord,
    balanced,
    error_value,
    grid_records_needed,
    item_key,
)
from effects.domain.strata import Half
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
            (tag.rule, tag.polarity, tag.outcome) for tag in self.tag(record)
            if tag.rule in testable and rule_test_eligible(record, tag)
        ]


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
    lines = []
    for stratum, by_half in sorted(selection.left_out.items()):
        for half, by_component in sorted(by_half.items()):
            for component, items in sorted(by_component.items()):
                untested = sum(1 for item in items if item.reason in UNTESTED_REASONS)
                named = [f"{item.item} ({item.reason})" for item in items
                         if item.reason not in UNTESTED_REASONS]
                if len(named) > limit:
                    named = [*named[:limit], f"{len(named) - limit} more"]
                parts = named + ([f"{untested} rule(s) untested"] if untested else [])
                lines.append(f"{stratum} {half}, {component} left out: " + "; ".join(parts))
    return lines


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

#: Reads one stratum: a fresh iterable of its shards in path order, each an
#: iterable of its records in line order.
StratumReader = Callable[[], Iterable[Iterable[EffectRecord]]]


@dataclass(frozen=True)
class FreezeThresholds:
    """Research E5's thresholds and E3's r; keyword arguments, never flags."""

    r: int = BALANCE_RATIO
    min_games_per_half: int = MIN_GAMES_PER_HALF
    max_item_error: float = MAX_ITEM_ERROR
    max_component_error: float = MAX_COMPONENT_ERROR
    deviance_cap: int = DEVIANCE_SAMPLE_CAP
    grid_target_error: float = GRID_TARGET_ERROR
    grid_record_cap: int = GRID_RECORD_CAP


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


def freeze_selection_set(
    manifest: CorpusManifest,
    strata: Mapping[str, StratumReader],
    *,
    rules: RuleTestRules,
    probes: ProbeItemSource | None,
    grids: Sequence[GridSource],
    deviance: DevianceLosses | None,
    r: int = BALANCE_RATIO,
    min_games_per_half: int = MIN_GAMES_PER_HALF,
    max_item_error: float = MAX_ITEM_ERROR,
    max_component_error: float = MAX_COMPONENT_ERROR,
    deviance_cap: int = DEVIANCE_SAMPLE_CAP,
    grid_target_error: float = GRID_TARGET_ERROR,
    grid_record_cap: int = GRID_RECORD_CAP,
) -> SelectionSet:
    """The corpus's selection set, unsealed; raises :class:`SelectionSetRefused`
    naming every miss. ``strata`` reads each validation stratum the manifest
    halves."""
    limits = FreezeThresholds(r, min_games_per_half, max_item_error, max_component_error,
                              deviance_cap, grid_target_error, grid_record_cap)
    small = small_halves(manifest, limits.min_games_per_half)
    if small:
        raise SelectionSetRefused(small)
    seed = manifest.seed
    half_of = {
        game: (stratum, half)
        for stratum, by_half in manifest.halves.items()
        for half, games in by_half.items() for game in games
    }
    gathered: dict[tuple[str, str], _Half] = {
        (stratum, half): _Half()
        for stratum, by_half in manifest.halves.items() for half in by_half
    }
    probe_kinds: dict[str, str] = {}
    for stratum in sorted(manifest.halves):
        _gather(strata[stratum], stratum, half_of, gathered, rules=rules, probes=probes,
                grids=grids, probe_kinds=probe_kinds, seed=seed, cap=limits.deviance_cap)
    out = _Assembly(manifest, limits, seed, probe_kinds, rules)
    order = {str(half): i for i, half in enumerate(Half)}
    for (stratum, half), found in sorted(
            gathered.items(), key=lambda item: (item[0][0], order.get(item[0][1], 2))):
        out.rule_tests(stratum, half, found)
        out.probes(stratum, half, found)
        out.grids(stratum, half, found, grids)
        out.deviance(stratum, half, found, strata[stratum], deviance)
    if out.misses:
        raise SelectionSetRefused(out.misses, out.selection())
    return out.selection()


def _gather(
    read: StratumReader, stratum: str, half_of: Mapping[str, tuple[str, str]],
    gathered: dict[tuple[str, str], _Half], *, rules: RuleTestRules,
    probes: ProbeItemSource | None, grids: Sequence[GridSource],
    probe_kinds: dict[str, str], seed: int, cap: int,
) -> None:
    """Pass one over a stratum: rule tests, probe items and grid records over
    distinct records, the deviance sample over every copy."""
    occurrences: dict[str, int] = defaultdict(int)
    for shard in read():
        distinct: list[EffectRecord] = []
        for record in shard:
            placed = half_of.get(record.game_id)
            copy = occurrences[record.record_id]
            occurrences[record.record_id] += 1
            if placed is None or placed[0] != stratum:
                continue
            found = gathered[placed]
            _sample(found.sample, record, copy, seed=seed, cap=cap)
            if copy:
                continue
            for rule, polarity, entities in rules.tests_of(record):
                found.rule_tests[rule][polarity].append(
                    TaggedRecord(record.record_id, record.game_id, tuple(entities)))
            if record.synthetic:
                continue
            distinct.append(record)
            for grid in grids:
                qualified = grid.qualify(record)
                if qualified is None:
                    continue
                # Keyed by the record alone, so grids qualifying the same
                # records (the combat grid's variants) choose the same ones.
                order = keyed_hash(record.record_id, seed)
                held = found.grids[grid.name].get(record.game_id)
                if held is None or order < held[0]:
                    found.grids[grid.name][record.game_id] = (order, record.record_id, qualified)
        if probes is not None and distinct:
            game_of = {record.record_id: record.game_id for record in distinct}
            for item in probes.items(distinct):
                game = game_of.get(item.record_id)
                if game is None:
                    continue
                known = probe_kinds.setdefault(item.probe_type, item.kind)
                if known != item.kind:
                    raise ValueError(f"probe type {item.probe_type} is labelled both "
                                     f"{known} and {item.kind}")
                gathered[half_of[game]].probes[item.probe_type].append(
                    ProbeRecord(item.record_id, game, item.entity, float(item.label)))


def _sample(heap: list, record: EffectRecord, copy: int, *, seed: int, cap: int) -> None:
    """Keep the ``cap`` copies of smallest keyed hash, as a max-heap."""
    if cap <= 0:
        return
    order = keyed_hash(f"{record.record_id}#{copy}", seed)
    entry = (-order, record.record_id, record.game_id, copy)
    if len(heap) < cap:
        heapq.heappush(heap, entry)
    elif order < -heap[0][0]:
        heapq.heapreplace(heap, entry)


class _Assembly:
    """Pass two: balance, standard errors, left-out items and the checks."""

    def __init__(self, manifest: CorpusManifest, limits: FreezeThresholds, seed: int,
                 probe_kinds: Mapping[str, str], rules: RuleTestRules) -> None:
        self.manifest = manifest
        self.limits = limits
        self.seed = seed
        self.probe_kinds = dict(probe_kinds)
        self.testable = sorted(rules.testable)
        self.unreadable = sorted(rules.unreadable_testable)
        self.misses: list[FreezeMiss] = []
        self.errors: dict[str, StandardError] = {}
        self.left_out: dict[str, dict[str, dict[str, list[LeftOut]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list)))
        self.frozen_rule_tests: dict = defaultdict(dict)
        self.frozen_probes: dict = defaultdict(dict)
        self.frozen_grids: dict = defaultdict(dict)
        self.frozen_samples: dict = defaultdict(dict)

    def _leave_out(self, stratum: str, half: str, component: str, item: str,
                   reason: str) -> None:
        self.left_out[stratum][half][component].append(LeftOut(item, reason))

    def _enters(self, stratum: str, half: str, component: str, item: str,
                error: float) -> bool:
        """Record ``item``'s standard error; whether it meets the item target."""
        self.errors[item_key(stratum, half, component, item)] = error_value(error)
        if error <= self.limits.max_item_error:
            return True
        self._leave_out(stratum, half, component, item, _above(error, self.limits))
        return False

    def _check(self, stratum: str, half: str, component: str, errors: Sequence[float],
               *, mode: str | None = None) -> None:
        """Refuse a component above its precision target (FR-068b)."""
        combined = component_standard_error(errors)
        if combined <= self.limits.max_component_error:
            return
        under = f" under the {mode} rule mask" if mode else ""
        detail = ("no item meets its precision target" + under if not errors else
                  f"standard error {combined:.4f} over {len(errors)} item(s){under}, "
                  f"above {self.limits.max_component_error}")
        self.misses.append(FreezeMiss(stratum, half, component, detail))

    # ── rule tests ──

    def rule_tests(self, stratum: str, half: str, found: _Half) -> None:
        errors, kept = [], {}
        for rule in self.unreadable:
            self._leave_out(stratum, half, RULE_TESTS, rule, NO_READABLE_OUTPUT)
        for rule in self.testable:
            sides = found.rule_tests.get(rule)
            if not sides or not any(sides.values()):
                self._leave_out(stratum, half, RULE_TESTS, rule, UNTESTED)
                continue
            counts = {polarity: len(sides.get(polarity, ())) for polarity in TEST_POLARITIES}
            if min(counts.values()) == 0:
                self.errors[item_key(stratum, half, RULE_TESTS, rule)] = None
                self._leave_out(stratum, half, RULE_TESTS, rule, "one polarity only: " + ", ".join(
                    f"{n} {polarity}" for polarity, n in counts.items()))
                continue
            drawn = balanced({p: sides.get(p, []) for p in TEST_POLARITIES}, self.limits.r,
                             seed=self.seed, salt=f"rule-test|{stratum}|{half}|{rule}",
                             key=lambda row: row.record_id)
            error = auc_standard_error(len(drawn[Polarity.FIRES]),
                                       len(drawn[Polarity.DOES_NOT_FIRE]))
            if self._enters(stratum, half, RULE_TESTS, rule, error):
                errors.append(error)
                kept[rule] = {str(p): tuple(rows) for p, rows in drawn.items()}
        self.frozen_rule_tests[stratum][half] = kept
        self._check(stratum, half, RULE_TESTS, errors)

    # ── probes ──

    def _probe_error(self, probe_type: str, rows: Sequence[ProbeRecord]) -> float:
        if self.probe_kinds[probe_type] == BINARY:
            positives = sum(1 for row in rows if row.label >= 0.5)
            return auc_standard_error(positives, len(rows) - positives)
        if not rows:
            return math.inf
        mean = sum(row.label for row in rows) / len(rows)
        return loss_ratio_standard_error(
            [row.game_id for row in rows], [(row.label - mean) ** 2 for row in rows])

    def probes(self, stratum: str, half: str, found: _Half) -> None:
        """Probe types meeting the item target, balanced against the smallest of
        them, and checked again after the draw."""
        candidates = {}
        for probe_type in sorted(found.probes):
            rows = _distinct(found.probes[probe_type])
            error = self._probe_error(probe_type, rows)
            if error <= self.limits.max_item_error:
                candidates[probe_type] = rows
            else:
                self._enters(stratum, half, PROBES, probe_type, error)
        drawn = balanced(candidates, self.limits.r, seed=self.seed,
                         salt=f"probe|{stratum}|{half}",
                         key=lambda row: f"{row.record_id}|{row.entity}")
        errors, kept = [], {}
        for probe_type, rows in sorted(drawn.items()):
            error = self._probe_error(probe_type, rows)
            if self._enters(stratum, half, PROBES, probe_type, error):
                errors.append(error)
                kept[probe_type] = tuple(rows)
        self.frozen_probes[stratum][half] = kept
        self._check(stratum, half, PROBES, errors)

    # ── grids ──

    def grids(self, stratum: str, half: str, found: _Half,
              sources: Sequence[GridSource]) -> None:
        errors, kept = [], {}
        for grid in sources:
            qualified = found.grids.get(grid.name, {})
            if not qualified:
                self.misses.append(FreezeMiss(stratum, half, GRIDS,
                                              f"grid {grid.name}: every cell is empty"))
                continue
            # Sized to the grid's precision target over every qualifying game,
            # then judged on the records kept (FR-068b).
            by_hash = sorted((order, game, record_id, q)
                             for game, (order, record_id, q) in qualified.items())
            needed = grid_records_needed(
                _mean_outcomes([q for *_, q in by_hash], len(grid.cells)),
                target=self.limits.grid_target_error, cap=self.limits.grid_record_cap)
            chosen = sorted((game, record_id, q)
                            for _, game, record_id, q in by_hash[:max(needed, 1)])
            records = tuple(TaggedRecord(record_id, game, q.entities)
                            for game, record_id, q in chosen)
            outcomes = _mean_outcomes([q for *_, q in chosen], len(grid.cells))
            error = grid_standard_error([len(chosen)] * len(grid.cells), outcomes)
            if not self._enters(stratum, half, GRIDS, grid.name, error):
                continue
            errors.append(error)
            kept[grid.name] = FrozenGrid(
                tuple(GridCell(cell, outcome) for cell, outcome in zip(grid.cells, outcomes)),
                records)
        self.frozen_grids[stratum][half] = kept
        self._check(stratum, half, GRIDS, errors)

    # ── deviance ──

    def deviance(self, stratum: str, half: str, found: _Half, read: StratumReader,
                 losses: DevianceLosses | None) -> None:
        chosen = sorted(((record_id, game, copy) for _, record_id, game, copy in found.sample),
                        key=lambda row: (row[1], row[0], row[2]))
        self.frozen_samples[stratum][half] = tuple(SampleRecord(*row) for row in chosen)
        if losses is None:
            self.misses.append(FreezeMiss(stratum, half, DEVIANCE,
                                          "no reader of the constant-prediction losses"))
            return
        wanted = {(record_id, copy) for record_id, _, copy in chosen}
        games: list[str] = []

        def streamed() -> Iterable[EffectRecord]:
            # Streamed rather than held: a sample is up to 16,384 parsed records.
            for record in _copies(read, wanted):
                games.append(record.game_id)
                yield record

        by_field = losses.losses(streamed())
        for mode in MASK_MODES:
            errors = []
            for name in sorted(by_field):
                per_record = by_field[name].get(mode)
                error = (math.inf if per_record is None
                         else loss_ratio_standard_error(games, per_record))
                key = item_key(stratum, half, DEVIANCE, name)
                held = self.errors.setdefault(key, {})
                held[mode] = error_value(error)
                if error <= self.limits.max_item_error:
                    errors.append(error)
                else:
                    self._leave_out(stratum, half, DEVIANCE, name,
                                    f"{_above(error, self.limits)} under the {mode} rule mask")
            self._check(stratum, half, DEVIANCE, errors, mode=mode)

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


def _mean_outcomes(qualifications: Sequence[GridQualification], cells: int) -> list[float]:
    """Each cell's rule outcome, averaged over a grid's records."""
    return [sum(q.outcomes[c] for q in qualifications) / len(qualifications)
            for c in range(cells)]


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


def _copies(read: StratumReader, wanted: set[tuple[str, int]]) -> Iterable[EffectRecord]:
    """The stratum's copies named in ``wanted``, in reading order."""
    occurrences: dict[str, int] = defaultdict(int)
    for shard in read():
        for record in shard:
            copy = occurrences[record.record_id]
            occurrences[record.record_id] += 1
            if (record.record_id, copy) in wanted:
                yield record


# ── the command ──────────────────────────────────────────────────────────


def stratum_reader(directory: Path) -> StratumReader:
    """A stratum directory's shards in path order, each read lazily."""
    from effects.infrastructure.shard_io import iter_shards, read_shard

    def read() -> Iterable[Iterable[EffectRecord]]:
        return (read_shard(path) for path in iter_shards(directory))

    return read


def run(
    corpus: Path,
    *,
    reports_dir: Path,
    rules_dir: Path = Path("rules"),
    lines: SidecarLookup | None = None,
    probes: ProbeItemSource | None = None,
    grids: Sequence[GridSource] = (),
    deviance: Callable[[CorpusManifest], DevianceLosses] | None = None,
    echo: Callable[[str], None] = print,
    **thresholds,
) -> int:
    """Freeze ``corpus``'s selection set into ``reports_dir``: 0 once written,
    1 after naming every miss. ``deviance`` builds the loss reader from the
    manifest, and is called only once every half holds its games."""
    from effects.domain.selection_set import selection_set_path, write_selection_set
    from effects.domain.strata import VALIDATION_STRATA
    from effects.infrastructure.corpus_store import CorpusStore
    from effects.infrastructure.rules_catalogue_io import load_scenarios

    store = CorpusStore(Path(corpus))
    manifest = store.load()
    try:
        small = small_halves(manifest, thresholds.get("min_games_per_half", MIN_GAMES_PER_HALF))
        if small:
            raise SelectionSetRefused(small)
        selection = freeze_selection_set(
            manifest,
            {str(stratum): stratum_reader(store.output_dir(stratum))
             for stratum in VALIDATION_STRATA},
            rules=RuleTestRules.of(load_scenarios(rules_dir), manifest, lines),
            probes=probes, grids=grids,
            deviance=None if deviance is None else deviance(manifest),
            **thresholds,
        )
    except SelectionSetRefused as refused:
        echo(f"selection set not frozen for {corpus}:")
        for miss in refused.misses:
            echo(f"  {miss.describe()}")
        if refused.selection is not None:
            for line in left_out_lines(refused.selection):
                echo(f"  {line}")
        return 1
    path = write_selection_set(selection_set_path(reports_dir, manifest.digest()), selection)
    rule_tests = sum(len(rules) for by_half in selection.rule_tests.values()
                     for rules in by_half.values())
    echo(f"froze {rule_tests} rule test(s), "
         f"{sum(len(p) for b in selection.probes.values() for p in b.values())} probe type(s), "
         f"{sum(len(g) for b in selection.grids.values() for g in b.values())} grid(s) "
         f"over {len(selection.halves)} strata -> {path}")
    return 0


__all__ = [
    "AMOUNT", "BINARY", "MASK_MODES", "DevianceLosses", "FreezeMiss", "FreezeThresholds",
    "GridQualification", "GridSource", "ProbeItem", "ProbeItemSource", "RuleTestRules",
    "SelectionSetRefused", "blocked_rules", "freeze_selection_set", "left_out_lines", "run",
    "small_halves",
    "stratum_reader",
]

"""The write pass of ``build-corpus``: every record to its stratum, by the plan.

A worker re-reads one raw shard through the survey's :class:`WorkerContext`
and writes one part per output stratum, which ``build()`` repacks; every number
the manifest reports about what was written is tallied here.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from effects.application.corpus_survey import ReadConfig, RefusalTally, WorkerContext
from effects.application.parallel_map import run_parallel
from effects.domain.corpus_curation import CopyPlan
from effects.domain.corpus_plan import Cell, GameStrata, Keys
from effects.domain.record_quality import has_unattributed_events
from effects.domain.records import (
    EffectRecord,
    LegalityDecision,
    RecordKind,
    SeatPolicy,
    acting_keys,
)
from effects.domain.sampling import SamplingClass
from effects.domain.strata import OUTPUT_STRATA, Stratum
from effects.domain.tally_merge import merge_into


@dataclass(frozen=True, slots=True)
class WritePlan:
    """What the write pass applies, decided once in the main process."""

    #: One part per source shard goes here, for ``build()`` to repack.
    parts_dirs: dict[Stratum, str]
    strata: GameStrata
    plans: dict[tuple[Cell, Keys], CopyPlan]
    keyless_plans: dict[Cell, CopyPlan]
    #: Places a legality record in its rarest mode's family exactly as the
    #: survey's supply was folded.
    mode_counts: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class WriteConfig:
    read: ReadConfig
    plan: WritePlan


# ── the tallies ─────────────────────────────────────────────────────────


@dataclass(slots=True)
class StratumTally:
    """Records written to each stratum and the acting keys behind them.

    The dropped strata are counted too (FR-145), and ``keys`` folds into each
    output's unique-ability-text count (FR-143).
    """

    records: Counter[Stratum] = field(default_factory=Counter)
    keys: dict[Stratum, set[Keys]] = field(default_factory=dict)
    gate_one_texts: set[str] = field(default_factory=set)

    def note(self, stratum: Stratum, keys: Keys | None, count: int = 1) -> None:
        self.records[stratum] += count
        if keys:
            self.keys.setdefault(stratum, set()).add(keys)


@dataclass(slots=True)
class TrainingTally:
    """Per sampling class: records kept by the rules, written, dropped by the plan."""

    #: In every stratum.
    read: Counter[SamplingClass] = field(default_factory=Counter)
    #: Copies included.
    kept: Counter[SamplingClass] = field(default_factory=Counter)
    dropped_by_cap: Counter[SamplingClass] = field(default_factory=Counter)
    kept_keys: dict[SamplingClass, set[Keys]] = field(default_factory=dict)


@dataclass(slots=True)
class FamilyTally:
    """What the training output holds per family, signature, policy and decision (FR-053)."""

    #: Written counts copies, distinct does not: the difference is repeats.
    written: Counter[tuple[SamplingClass, str]] = field(default_factory=Counter)
    distinct: Counter[tuple[SamplingClass, str]] = field(default_factory=Counter)
    signatures: Counter[tuple[SamplingClass, str, str]] = field(default_factory=Counter)
    policy: Counter[tuple[SamplingClass, SeatPolicy]] = field(default_factory=Counter)
    #: ``(legality subkind, real | what-if | unknown)``.
    legality: Counter[tuple[str, str]] = field(default_factory=Counter)

    def note(
        self, record: EffectRecord, klass: SamplingClass, cell: Cell, copies: int,
    ) -> None:
        label = cell.label
        self.written[klass, label] += copies
        self.distinct[klass, label] += 1
        self.signatures[klass, label, cell.signature] += copies
        self.policy[klass, SeatPolicy.of(record)] += copies
        if cell.half:
            self.legality[record.subkind.value, LegalityDecision.of(record).value] += copies


@dataclass(slots=True)
class WriteResult:
    """Everything one shard's write pass, or the whole pass, wrote and refused."""

    refusals: RefusalTally = field(default_factory=RefusalTally)
    strata: StratumTally = field(default_factory=StratumTally)
    training: TrainingTally = field(default_factory=TrainingTally)
    families: FamilyTally = field(default_factory=FamilyTally)
    #: Kept records whose events name no producing clause (FR-148), watched
    #: rather than refused.
    unattributed: int = 0


# ── the pass ────────────────────────────────────────────────────────────

_WRITE: tuple[WritePlan, WorkerContext] | None = None

_NO_COPIES = CopyPlan()


def init_write_worker(config: WriteConfig) -> None:
    global _WRITE
    _WRITE = (config.plan, WorkerContext(config.read))


def output_name(relative: str) -> str:
    """A flat, **injective** part name for a source in a subdirectory.

    Injective because ``write_shard`` truncates, so a collision is a silent
    overwrite; escaping ``_`` first makes it so, where a plain ``/ -> __``
    would map ``a_/b`` and ``a/_b`` alike.
    """
    return relative.replace("_", "_u").replace("/", "_s")


def write_shard_pass(relative: str) -> WriteResult:
    """Filter one shard into its output parts. Module-level so it pickles."""
    assert _WRITE is not None, "init_write_worker was not run"
    plan, context = _WRITE
    return write_shard_with(plan, context, relative)


def write_shard_with(plan: WritePlan, context: WorkerContext, relative: str) -> WriteResult:
    writer = _ShardWriter(plan, context)
    for record in context.records(relative):
        writer.observe(record)
    return writer.finish(output_name(relative))


class _ShardWriter:
    """One shard's write pass in progress: every record routed to its outputs
    and tallied, then each output written as one part."""

    def __init__(self, plan: WritePlan, context: WorkerContext) -> None:
        self.plan = plan
        self.context = context
        self.out = WriteResult()
        self.outputs: dict[Stratum, list] = {stratum: [] for stratum in OUTPUT_STRATA}

    def observe(self, record: EffectRecord) -> None:
        """Route one record to its outputs and tally it."""
        klass = self.context.admit(record, self.out.refusals)
        if klass is None:
            return
        self.out.training.read[klass] += 1
        if has_unattributed_events(record):
            self.out.unattributed += 1
        keys = acting_keys(record) or None
        stratum = self.plan.strata.stratum_of(record.game_id)
        if stratum is Stratum.DROPPED_HELD_OUT:
            self.out.strata.note(stratum, None)
        elif stratum is Stratum.TRAINING:
            self._train(record, klass, keys)
        else:
            self._hold(record, stratum, keys)

    def _emit(self, stratum: Stratum, record: EffectRecord, keys: Keys | None) -> None:
        self.outputs[stratum].append(record)
        self.out.strata.note(stratum, keys)

    def _hold(self, record: EffectRecord, stratum: Stratum, keys: Keys | None) -> None:
        """A validation game's record, and gate 1's copy of a card-disjoint
        resolution acting through a held-out text."""
        self._emit(stratum, record, keys)
        if stratum is Stratum.CARD_DISJOINT and record.kind is RecordKind.RESOLUTION:
            text = self.context.held_out_text(keys)
            if text is not None:
                self._emit(Stratum.GATE_ONE, record, keys)
                self.out.strata.gate_one_texts.add(text)

    def _train(self, record: EffectRecord, klass: SamplingClass, keys: Keys | None) -> None:
        """A training game's record, written as many times as its cell's plan
        draws for it."""
        out = self.out
        if self.context.held_out_text(keys) is not None:
            # A token or variant printing a held-out text: the rest of the game
            # still trains, this record would train on the measured text.
            out.strata.note(Stratum.DROPPED_HELD_OUT_TEXT, None)
            return
        cell = self.context.cell_of(record, self.plan.mode_counts)
        copies = self._copy_plan(cell, keys).copies(self.context.hash_of(record))
        if copies <= 0:
            out.training.dropped_by_cap[klass] += 1
            return
        self.outputs[Stratum.TRAINING].extend([record] * copies)
        out.strata.note(Stratum.TRAINING, keys, copies)
        out.training.kept[klass] += copies
        if keys:
            out.training.kept_keys.setdefault(klass, set()).add(keys)
        out.families.note(record, klass, cell, copies)

    def _copy_plan(self, cell: Cell, keys: Keys | None) -> CopyPlan:
        """The plan of the record's ``(cell, keys)`` group, else its cell's."""
        copy_plan = self.plan.plans.get((cell, keys)) if keys else None
        if copy_plan is None:
            copy_plan = self.plan.keyless_plans.get(cell, _NO_COPIES)
        return copy_plan

    def finish(self, name: str) -> WriteResult:
        """Write each non-empty output as part ``name`` of its stratum."""
        from effects.infrastructure.record_io import write_shard

        for stratum, records in self.outputs.items():
            if records:
                write_shard(Path(self.plan.parts_dirs[stratum]) / name, records)
        return self.out


def run_write_pass(
    names: list[str], *, config: WriteConfig, workers: int | None,
    progress_every: int = 50,
) -> WriteResult:
    """Write every shard's output parts, in parallel, reporting progress."""
    total = WriteResult()
    for part in run_parallel(
        names, write_shard_pass, initializer=init_write_worker, initargs=(config,),
        workers=workers, label="Wrote", progress_every=progress_every,
    ):
        merge_into(total, part)
    return total

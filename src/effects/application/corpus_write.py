"""The write pass of ``build-corpus``: every record to its stratum, by the plan.

A worker re-reads one raw shard through the same :class:`WorkerContext` the
survey used, routes each record by its game (:class:`GameStrata`) and writes
one part per output stratum; ``build()`` repacks the parts afterwards. Every
number the manifest reports about what was written is tallied here, in four
value objects that each merge on their own.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from effects.application.corpus_survey import (
    ReadConfig,
    RefusalTally,
    WorkerContext,
    run_parallel,
)
from effects.domain.corpus_curation import CopyPlan
from effects.domain.corpus_plan import Cell, GameStrata, Keys
from effects.domain.record_quality import has_unattributed_events
from effects.domain.records import EffectRecord, LegalityDecision, RecordKind, acting_keys
from effects.domain.sampling import sampling_class
from effects.domain.strata import OUTPUT_STRATA, Stratum
from effects.domain.token_key_remap import RemapCounts


@dataclass(frozen=True, slots=True)
class WritePlan:
    """What the write pass applies, decided once in the main process."""

    #: A **parts** directory per output: a worker writes one part per source
    #: shard, and ``build()`` repacks the parts into uniform shards.
    parts_dirs: dict[Stratum, str]
    strata: GameStrata
    plans: dict[tuple[Cell, Keys], CopyPlan]
    keyless_plans: dict[Cell, CopyPlan]
    #: Kept records per legality mode, which places a legality record in its
    #: rarest mode's family exactly as the survey's supply was folded.
    mode_counts: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class WriteConfig:
    read: ReadConfig
    plan: WritePlan


# ── the tallies ─────────────────────────────────────────────────────────


@dataclass(slots=True)
class StratumTally:
    """Records written to each stratum and the acting keys behind them.

    Also counts what reaches no output: held-out games the card-disjoint cap
    declined, and training records acting through a held-out text. FR-145
    wants the report per stratum, and a stratum nobody counted is a stratum
    nobody notices is empty. ``keys`` folds through the build's key-to-text
    map into each output's unique-ability-text count (FR-143).
    """

    records: Counter[Stratum] = field(default_factory=Counter)
    keys: dict[Stratum, set[Keys]] = field(default_factory=dict)
    #: The held-out texts the gate-one slice holds a record of.
    gate_one_texts: set[str] = field(default_factory=set)

    def note(self, stratum: Stratum, keys: Keys | None, count: int = 1) -> None:
        self.records[stratum] += count
        if keys:
            self.keys.setdefault(stratum, set()).add(keys)

    def merge(self, other: StratumTally) -> None:
        self.records.update(other.records)
        for stratum, keys in other.keys.items():
            self.keys.setdefault(stratum, set()).update(keys)
        self.gate_one_texts |= other.gate_one_texts


@dataclass(slots=True)
class TrainingTally:
    """Per sampling class: records kept by the rules, written, dropped by the plan."""

    #: Records the quality rules kept, in every stratum.
    read: Counter[str] = field(default_factory=Counter)
    #: Records written to the training output, copies included.
    kept: Counter[str] = field(default_factory=Counter)
    #: Training records a copy plan wrote zero times.
    dropped_by_cap: Counter[str] = field(default_factory=Counter)
    kept_keys: dict[str, set[Keys]] = field(default_factory=dict)

    def merge(self, other: TrainingTally) -> None:
        self.read.update(other.read)
        self.kept.update(other.kept)
        self.dropped_by_cap.update(other.dropped_by_cap)
        for klass, keys in other.kept_keys.items():
            self.kept_keys.setdefault(klass, set()).update(keys)


@dataclass(slots=True)
class FamilyTally:
    """What the training output holds per family, signature, policy and decision (FR-053)."""

    #: ``(class, family label) ->`` records written, copies included, and the
    #: distinct records behind them: the difference is the family's repeats.
    written: Counter[tuple[str, str]] = field(default_factory=Counter)
    distinct: Counter[tuple[str, str]] = field(default_factory=Counter)
    signatures: Counter[tuple[str, str, str]] = field(default_factory=Counter)
    #: ``(class, on_policy | off_policy)``.
    policy: Counter[tuple[str, str]] = field(default_factory=Counter)
    #: ``(legality subkind, real | what-if | unknown)``.
    legality: Counter[tuple[str, str]] = field(default_factory=Counter)

    def note(self, record: EffectRecord, klass: str, cell: Cell, copies: int) -> None:
        label = cell.label
        self.written[klass, label] += copies
        self.distinct[klass, label] += 1
        self.signatures[klass, label, cell.signature] += copies
        self.policy[klass, "off_policy" if record.random_seat else "on_policy"] += copies
        if cell.half:
            self.legality[record.subkind.value, LegalityDecision.of(record).value] += copies

    def merge(self, other: FamilyTally) -> None:
        self.written.update(other.written)
        self.distinct.update(other.distinct)
        self.signatures.update(other.signatures)
        self.policy.update(other.policy)
        self.legality.update(other.legality)


@dataclass(slots=True)
class WriteResult:
    """Everything one shard's write pass, or the whole pass, wrote and refused."""

    refusals: RefusalTally = field(default_factory=RefusalTally)
    strata: StratumTally = field(default_factory=StratumTally)
    training: TrainingTally = field(default_factory=TrainingTally)
    families: FamilyTally = field(default_factory=FamilyTally)
    #: Old token keys rewritten, and the ones refused, **on written records**
    #: (FR-151): a record this pass refuses or drops contributes neither.
    remap: RemapCounts = field(default_factory=RemapCounts)
    #: Kept records whose events name no producing clause (FR-148). Watched
    #: rather than refused: a clause the collector could not find on the
    #: acting chain does not make the outcome another ability's.
    unattributed: int = 0

    def merge(self, other: WriteResult) -> None:
        self.refusals.merge(other.refusals)
        self.strata.merge(other.strata)
        self.training.merge(other.training)
        self.families.merge(other.families)
        self.remap.merge(other.remap)
        self.unattributed += other.unattributed


# ── the pass ────────────────────────────────────────────────────────────

_WRITE: tuple[WritePlan, WorkerContext] | None = None

#: The plan of a cell the decision gave nothing: written zero times.
_NO_COPIES = CopyPlan()


def init_write_worker(config: WriteConfig) -> None:
    """Pool initializer: one context per worker process, built once."""
    global _WRITE
    _WRITE = (config.plan, WorkerContext(config.read))


def output_name(relative: str) -> str:
    """A flat, **injective** part name for a source that may sit in a subdirectory.

    ``depleted/run.0-a.jsonl.gz`` and ``full-strength/run.0-a.jsonl.gz`` are
    different shards and must not write to one file -- and neither may any
    other pair of distinct relative paths, because ``write_shard`` opens in
    truncate mode, so a collision is a silent overwrite with no stable answer,
    under ``workers > 1``, as to which source survives.

    A plain ``"/" -> "__"`` substitution is not injective: ``"a_/b"`` and
    ``"a/_b"`` both become ``"a___b"``. Escaping the escape character first
    makes it so: every ``_`` in the result opens a two-character escape --
    ``_u`` for an original ``_``, ``_s`` for an original ``/``.
    """
    return relative.replace("_", "_u").replace("/", "_s")


def write_shard_pass(relative: str) -> WriteResult:
    """Filter one shard into its output parts. Module-level so it pickles."""
    assert _WRITE is not None, "init_write_worker was not run"
    plan, context = _WRITE
    return write_shard_with(plan, context, relative)


def write_shard_with(plan: WritePlan, context: WorkerContext, relative: str) -> WriteResult:
    from effects.infrastructure.record_io import write_shard

    out = WriteResult()
    outputs: dict[Stratum, list] = {stratum: [] for stratum in OUTPUT_STRATA}
    # The remap's counts land here first and are folded into ``out.remap``
    # only for a record that reaches an output, so the manifest's figure is
    # what the written dataset carries rather than what the shards held
    # (FR-151).
    pending = RemapCounts()
    for record in context.records(relative, pending):
        if write_record(record, plan, context, out, outputs):
            out.remap.merge(pending)
        pending.reset()

    name = output_name(relative)
    # Only where there is something to write: a source shard contributes to
    # one stratum far more often than to every one.
    for stratum, records in outputs.items():
        if records:
            write_shard(Path(plan.parts_dirs[stratum]) / name, records)
    return out


def write_record(
    record: EffectRecord, plan: WritePlan, context: WorkerContext, out: WriteResult,
    outputs: dict[Stratum, list],
) -> bool:
    """Route one record to its outputs and tally it; whether anything was written."""
    klass = sampling_class(record)
    defect = context.refusal(record)
    if defect is not None:
        out.refusals.note(record, klass, defect)
        return False
    out.training.read[klass] += 1
    if has_unattributed_events(record):
        out.unattributed += 1

    keys = acting_keys(record) or None
    stratum = plan.strata.stratum_of(record.game_id)
    if stratum is Stratum.DROPPED_HELD_OUT:
        # Held out but not admitted to the stratum: dropped, never trained on.
        out.strata.note(stratum, None)
        return False
    if stratum is not Stratum.TRAINING:
        outputs[stratum].append(record)
        out.strata.note(stratum, keys)
        if stratum is Stratum.CARD_DISJOINT and record.kind is RecordKind.RESOLUTION:
            text = context.held_out_text(keys)
            if text is not None:
                outputs[Stratum.GATE_ONE].append(record)
                out.strata.note(Stratum.GATE_ONE, keys)
                out.strata.gate_one_texts.add(text)
        return True

    if context.held_out_text(keys) is not None:
        # A training game's record acting through a held-out text -- a token
        # or a variant printing it, or its template. The game carries no
        # held-out card, so the rest of it still trains; this record would
        # train on the very text the card-disjoint stratum measures.
        out.strata.note(Stratum.DROPPED_HELD_OUT_TEXT, None)
        return False
    cell = context.cell_of(record, plan.mode_counts)
    copy_plan = plan.plans.get((cell, keys)) if keys else None
    if copy_plan is None:
        copy_plan = plan.keyless_plans.get(cell, _NO_COPIES)
    copies = copy_plan.copies(context.hash_of(record))
    if copies <= 0:
        out.training.dropped_by_cap[klass] += 1
        return False
    outputs[Stratum.TRAINING].extend([record] * copies)
    out.strata.note(Stratum.TRAINING, keys, copies)
    out.training.kept[klass] += copies
    if keys:
        out.training.kept_keys.setdefault(klass, set()).add(keys)
    out.families.note(record, klass, cell, copies)
    return True


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
        total.merge(part)
    return total

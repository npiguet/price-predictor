"""The write pass of ``build-corpus``: every record to its stratum, by the plan.

A worker re-reads one raw shard through the survey's :class:`WorkerContext`
and writes one part per output stratum, which ``build()`` repacks; every number
the manifest reports about what was written is tallied here.

Every written copy carries its ``curation`` (research T1): a training record's
natural copies, the ones gen-2's budgets give it, then the copies the rule
budgets added, each carrying the rule it was drawn for and that rule's tag
entities; a validation record's one natural copy and its added copies, by the
same rule budgets planned over its stratum (FR-068a); and a record's
protection copies (FR-058), each a synthetic record of its own. ``turn_action``
records are written nowhere and counted by subkind (FR-059).
"""

from __future__ import annotations

import functools
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

from effects.application.corpus_survey import ReadConfig, RefusalTally, WorkerContext
from effects.application.parallel_map import run_parallel
from effects.domain.corpus_curation import CopyPlan
from effects.domain.corpus_manifest import SourceShard
from effects.domain.corpus_plan import Cell, GameStrata, Keys, planned_copies
from effects.domain.protection_copies import protection_copy, qualify
from effects.domain.record_quality import has_unattributed_events
from effects.domain.records import (
    Curation,
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
    parts_dirs: dict[Stratum, Path]
    strata: GameStrata
    plans: dict[tuple[Cell, Keys], CopyPlan]
    keyless_plans: dict[Cell, CopyPlan]
    #: Places a legality record in its rarest mode's family exactly as the
    #: survey's supply was folded.
    mode_counts: dict[str, int] = field(default_factory=dict)


#: ``game -> record hash -> drawn rule per added copy``: the rule budgets'
#: draws (``RuleBudgets.copies``), a record's protection copies under its
#: protection hash. A worker receives its own shard's games only.
CopyMap = Mapping[str, Mapping[int, tuple[str, ...]]]


@dataclass(frozen=True, slots=True)
class WriteConfig:
    read: ReadConfig
    plan: WritePlan


@dataclass(frozen=True, slots=True)
class WriteTask:
    """One raw shard to write, and the added copies of its games."""

    shard: SourceShard
    copies: CopyMap = field(default_factory=dict)


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

    def note(self, stratum: Stratum, keys: Keys, count: int = 1) -> None:
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
    #: Copies the rule budgets added, protection copies included, per stratum.
    added: Counter[Stratum] = field(default_factory=Counter)
    #: Protection copies written per drawn rule (FR-058).
    protection: Counter[str] = field(default_factory=Counter)
    #: ``turn_action`` records by subkind, written to no stratum (FR-059).
    turn_actions: Counter[str] = field(default_factory=Counter)


# ── the pass ────────────────────────────────────────────────────────────

class WriteWorker:
    """One process's write side: the plan, and the read context every shard
    it writes shares. Built once per process by :func:`run_parallel`."""

    def __init__(self, config: WriteConfig) -> None:
        self.plan = config.plan
        self.context = WorkerContext(config.read)

    def write(self, task: WriteTask) -> WriteResult:
        """Filter one shard into its output parts."""
        writer = _ShardWriter(self.plan, self.context, task.copies)
        for record in self.context.records(task.shard):
            writer.observe(record)
        return writer.finish(task.shard.part_name)


_NO_ADDED: Mapping[int, tuple[str, ...]] = {}


class _ShardWriter:
    """One shard's write pass in progress: every record routed to its outputs
    and tallied, then each output written as one part."""

    def __init__(self, plan: WritePlan, context: WorkerContext, copies: CopyMap) -> None:
        self.plan = plan
        self.context = context
        self.copies = copies
        self.out = WriteResult()
        self.outputs: dict[Stratum, list] = {stratum: [] for stratum in OUTPUT_STRATA}

    def observe(self, record: EffectRecord) -> None:
        """Route one record to its outputs and tally it."""
        if record.kind is RecordKind.TURN_ACTION:
            self.out.turn_actions[str(record.subkind)] += 1
            return
        klass = self.context.admit(record, self.out.refusals)
        if klass is None:
            return
        self.out.training.read[klass] += 1
        if has_unattributed_events(record):
            self.out.unattributed += 1
        keys = acting_keys(record)
        stratum = self.plan.strata.stratum_of(record.game_id)
        if stratum is Stratum.DROPPED_HELD_OUT:
            self.out.strata.note(stratum, ())
        elif stratum is Stratum.TRAINING:
            self._train(record, klass, keys)
        else:
            self._hold(record, stratum, keys)

    def _hold(self, record: EffectRecord, stratum: Stratum, keys: Keys) -> None:
        """A validation game's record, its added copies and protection copies,
        and gate 1's one copy of a card-disjoint resolution acting through a
        held-out text."""
        natural = self._write(stratum, record, keys, natural=1)[0]
        self._protection_copies(record, stratum, keys)
        if stratum is Stratum.CARD_DISJOINT and record.kind is RecordKind.RESOLUTION:
            text = self.context.held_out_text(keys)
            if text is not None:
                self.outputs[Stratum.GATE_ONE].append(natural)
                self.out.strata.note(Stratum.GATE_ONE, keys)
                self.out.strata.gate_one_texts.add(text)

    def _train(self, record: EffectRecord, klass: SamplingClass, keys: Keys) -> None:
        """A training game's record, written as many times as its cell's plan
        draws for it, then the copies the rule budgets added."""
        out = self.out
        if self.context.held_out_text(keys) is not None:
            # A token or variant printing a held-out text: the rest of the game
            # still trains, this record would train on the measured text.
            out.strata.note(Stratum.DROPPED_HELD_OUT_TEXT, ())
            return
        cell = self.context.cell_of(record, self.plan.mode_counts)
        natural = planned_copies(
            self.plan.plans, self.plan.keyless_plans, cell, keys, self.context.hash_of(record),
        )
        written = len(self._write(Stratum.TRAINING, record, keys, natural=natural))
        written += self._protection_copies(record, Stratum.TRAINING, keys)
        if not written:
            out.training.dropped_by_cap[klass] += 1
            return
        out.training.kept[klass] += written
        if keys:
            out.training.kept_keys.setdefault(klass, set()).add(keys)
        if natural > 0:
            # The families report gen-2's budgets: natural copies alone.
            out.families.note(record, klass, cell, natural)

    def _write(
        self, stratum: Stratum, record: EffectRecord, keys: Keys, *, natural: int,
    ) -> list[EffectRecord]:
        """``natural`` copies of ``record`` then its added copies, each with
        its curation; the copies written, natural first."""
        added = self.copies.get(record.game_id, _NO_ADDED).get(
            self.context.hash_of(record), ())
        total = natural + len(added)
        if total <= 0:
            return []
        drawn = tuple(sorted(set(added)))
        copies = [replace(record, curation=Curation(natural, total, None, drawn))] * natural
        if added:
            tags = {tag.rule: tag for tag in self.context.tags(record).tags}
            copies += [
                replace(record, curation=Curation(
                    natural, total, rule, drawn, *_roles(tags.get(rule))))
                for rule in added
            ]
            self.out.added[stratum] += len(added)
        self.outputs[stratum].extend(copies)
        self.out.strata.note(stratum, keys, total)
        return copies

    def _protection_copies(self, record: EffectRecord, stratum: Stratum, keys: Keys) -> int:
        """The protection copies the rule budgets drew for ``record`` (FR-058),
        each a synthetic record carrying its drawn rule; how many were written.

        The draws are seeded by the record's protection hash, so a rebuild
        writes the same copies."""
        value = self.context.protection_hash_of(record)
        rules = self.copies.get(record.game_id, _NO_ADDED).get(value, ())
        if not rules:
            return 0
        opportunities = qualify(record)
        rng = random.Random(value)
        drawn = tuple(sorted(set(rules)))
        written = 0
        for index, rule in enumerate(rules):
            made = protection_copy(record, opportunities, rule=rule, rng=rng, index=index)
            if made is None:
                continue
            self.outputs[stratum].append(replace(made.record, curation=Curation(
                0, len(rules), rule, drawn, made.condition_entities, made.outcome_entities)))
            self.out.protection[rule] += 1
            written += 1
        self.out.strata.note(stratum, keys, written)
        self.out.added[stratum] += written
        return written

    def finish(self, name: str) -> WriteResult:
        """Write each non-empty output as part ``name`` of its stratum."""
        from effects.infrastructure.shard_io import write_shard

        for stratum, records in self.outputs.items():
            if records:
                write_shard(self.plan.parts_dirs[stratum] / name, records)
        return self.out


def run_write_pass(
    tasks: Sequence[WriteTask], *, config: WriteConfig, workers: int | None,
    progress_every: int = 50,
) -> WriteResult:
    """Write every shard's output parts, in parallel, reporting progress."""
    total = WriteResult()
    for part in run_parallel(
        tasks, WriteWorker.write, context=functools.partial(WriteWorker, config),
        workers=workers, label="Wrote", progress_every=progress_every,
    ):
        merge_into(total, part)
    return total


def _roles(tag) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """A drawn rule's tag entities as the curation carries them: condition,
    then outcome; none where the record no longer shows the rule."""
    return (tag.condition, tag.outcome) if tag is not None else ((), ())

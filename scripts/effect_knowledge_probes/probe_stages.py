"""The stages of one probing run: reading the halves, the ladders, the sweeps and the ablations.

Nothing here does work at import time.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from effect_knowledge_probes import ablation, common, halves, labels, ladder, record_walk, sweeps
from effect_knowledge_probes import board as board_probe
from effect_knowledge_probes.line_labelling import RecordObservers
from effect_knowledge_probes.line_ladders import MIN_LADDER_ITEMS
from effect_knowledge_probes.method_c_probe import method_c
from effect_knowledge_probes.progress import progress
from effect_knowledge_probes.scorecard import Scorecard

HALF_CHOICES = {"both": halves.HALVES, "selection": (halves.SELECTION,),
                "reporting": (halves.REPORTING,)}


def selected_families(args) -> set[str]:
    return set(args.families or labels.FAMILIES)


def selected_targets(args) -> list:
    """The probe targets of the families the run asked for."""
    families = selected_families(args)
    return [spec for spec in labels.TARGETS if spec.family in families]


# ── reading the halves ──────────────────────────────────────────────────


class _HalfReader:
    """Reads one stratum's half of the selection set (``halves.read_half``)."""

    def __init__(self, args, context: common.ProbeContext, selection,
                 observers: RecordObservers) -> None:
        from effects.domain.damage_step_keywords import KeywordResolver

        sidecars = context.sidecars()
        self.args, self.context, self.selection = args, context, selection
        self.observe = observers.observe
        self.acting_text = common.acting_text_function(
            sidecars, common.manifest_surface(context.manifest))
        self.keywords_of = KeywordResolver(sidecars).keywords_of
        self.board_targets = frozenset(
            spec.name for spec in selected_targets(args) if spec.scope in ("act", "entity"))

    def read(self, stratum: str, half: str) -> halves.HalfRead:
        args = self.args
        settings = halves.ReadSettings(
            probe_types=self.board_targets, gen3_probes=not args.no_gen3_probes,
            sweeps=not args.no_sweeps and half == halves.REPORTING,
            limit=args.probe_records, per_game=args.probe_per_game)
        progress(f"reading {stratum}/{half}: "
                 f"{len(self.selection.halves[stratum].get(half, ()))} games")
        read = halves.read_half(self.context, self.selection, stratum, half, settings,
                                acting_text=self.acting_text, keywords_of=self.keywords_of,
                                observe=self.observe)
        progress(f"  {len(read.records)} records kept, {len(read.probe_items)} with "
                 f"probe items, {len(read.grids)} grids")
        return read


def read_halves(args, context: common.ProbeContext, selection,
                observers: RecordObservers) -> list[halves.HalfRead]:
    """Each stratum's halves, read once each (``halves.read_half``)."""
    reader = _HalfReader(args, context, selection, observers)
    reads = [reader.read(stratum, half)
             for stratum in sorted(selection.halves) for half in HALF_CHOICES[args.half]]
    if not any(read.records for read in reads):
        raise SystemExit("no record the selection set names was found in the corpus's "
                         "validation strata")
    return reads


def _scan_stratum(args, context: common.ProbeContext, stratum: str) -> dict[str, list]:
    progress(f"board subsets: scanning {stratum}")
    subsets = board_probe.collect(
        record_walk.records(
            [context.stratum_dir(stratum)],
            shards=args.board_shards or board_probe.SHARDS_PER_STRATUM),
        context.sidecars(),
        per_subset=args.board_records or board_probe.RECORDS_PER_SUBSET)
    progress("  " + ", ".join(f"{name} {len(rows)}" for name, rows in subsets.items()))
    return subsets


def board_subsets(args, context: common.ProbeContext) -> dict[str, dict[str, list]]:
    """Each validation stratum's board-abilities subsets (``board.py``), unless skipped."""
    if args.no_board:
        return {}
    return {stratum: _scan_stratum(args, context, stratum)
            for stratum in common.STRATA if context.stratum_dir(stratum).exists()}


# ── the ladders over the frozen probe items ─────────────────────────────


def _probe_scores(inputs: dict) -> dict[str, dict]:
    """Each frozen probe type's rung-3 score in one half, on the control
    score's scale too: ``2·AUC − 1`` for a yes/no type, R² as is."""
    from effects.domain.probe_readouts import item_score
    from effects.domain.score_scales import rescaled_auc

    out = {}
    for name, data in sorted(inputs.items()):
        value = float(item_score(data.kind, data.y, data.rung3))
        out[name] = {"kind": data.kind, "n": int(len(data.y)), "score": value,
                     "rescaled": rescaled_auc(value) if data.kind == labels.BINARY else value}
    return out


@dataclass
class _LadderRun:
    """The state of the per-half ladders: where results go, what ``[ACT]`` gathered."""

    scorecard: Scorecard
    probe_model: object
    per_layer: bool
    kinds: dict[str, str] = field(
        default_factory=lambda: {spec.name: spec.kind for spec in labels.TARGETS})
    act_hidden: dict[str, list] = field(default_factory=dict)

    def read(self, read: halves.HalfRead) -> None:
        chosen = read.probe_records()
        progress(f"{read.name}: extracting rung features over {len(chosen)} records")
        extracted = ladder.extract_features(
            self.probe_model, chosen, read.probe_items,
            ladder.ExtractionOptions(per_layer=self.per_layer, act_hidden=self.act_hidden))
        inputs = ladder.ladder_inputs(extracted, self.kinds)
        self.scorecard.half(read.stratum, read.half)["probes"] = _probe_scores(inputs)
        self._ladders(read, inputs)

    def _ladders(self, read: halves.HalfRead, inputs: dict) -> None:
        for position, (name, data) in enumerate(sorted(inputs.items()), start=1):
            if len(data.y) < MIN_LADDER_ITEMS:
                continue
            progress(f"{read.name}: ladder {position}/{len(inputs)} {name} (n={len(data.y)})")
            spec = labels.TARGETS_BY_NAME[name]
            entry = self.scorecard.target(spec.family, name, scope=spec.scope, kind=spec.kind)
            entry["strata"][read.name] = ladder.run_ladder(data)


def board_ladder(args, scorecard: Scorecard, probe_model,
                 reads: list[halves.HalfRead]) -> dict[str, list]:
    """Every frozen probe type's ladder, per stratum and half; returns
    ``[ACT]`` by text.

    The trunk's ``[ACT]`` output is gathered from the same forward pass, by
    acting text, for the line ladder's rung 2.
    """
    run = _LadderRun(scorecard, probe_model, args.per_layer)
    for read in reads:
        run.read(read)
    return run.act_hidden


# ── the sweeps ──────────────────────────────────────────────────────────


def _sweep_cases(read: halves.HalfRead):
    """``(label, sweep name, record, row)`` for every sweep row whose record was kept."""
    rows = ((name, row) for name, rows in read.sweep_rows.items() for row in rows)
    for name, row in rows:
        record = read.records.get(row["record_id"])
        if record is not None:
            yield (f"damage_{row['host']}" if name == "damage" else name), name, record, row


def run_sweeps(reads: list[halves.HalfRead], probe_model) -> dict:
    """Every board sweep over the reporting half's records: per-record curves
    and their mean, both strata pooled."""
    curves: dict[str, list] = defaultdict(list)
    for read in reads:
        for label, name, record, row in _sweep_cases(read):
            curves[label].append(sweeps.run_sweep(name, probe_model, record, row))
    return {label: {"mean": sweeps.summarize(curve), "records": curve}
            for label, curve in curves.items()}


# ── the ablations ───────────────────────────────────────────────────────


def validation_samples(context: common.ProbeContext, limit: int) -> list:
    """The first ``limit`` distinct records of each stratum's fixed validation sample."""
    from effects.infrastructure.shard_io import read_shard, require_generation

    samples = []
    for stratum in common.STRATA:
        path = Path(context.corpus) / "validation" / "samples" / f"{stratum}.jsonl.gz"
        if path.exists():
            require_generation([path])
            distinct = halves.distinct(read_shard(path))
            samples += [record for _, record in zip(range(limit), distinct)]
    return samples


@dataclass(frozen=True)
class AblationInputs:
    """What the ablations read: the validation samples, the ability cache's lines and
    the board subsets, and the corpus method C trains on."""

    samples: list
    line_cache: list
    board_subsets: dict
    corpus: Path


def _method_b(args, scorecard: Scorecard, probe_model, inputs: AblationInputs) -> None:
    stats = ablation.cache_statistics(inputs.line_cache)
    if not args.no_ablation and inputs.samples:
        progress(f"ablation over {len(inputs.samples)} records")
        scorecard.ablation = ablation.run_ablation(probe_model, inputs.samples, stats)
    if inputs.board_subsets:
        progress("board-abilities ablation")
        scorecard.board_abilities = board_probe.run(probe_model, inputs.board_subsets, stats)


def run_ablations(args, scorecard: Scorecard, probe_model, inputs: AblationInputs) -> None:
    """Method B over the validation samples and the board subsets, and method C.

    Both ablations draw their replacements for ``e`` from the ability cache,
    so a checkpoint whose cache was never encoded runs neither.
    """
    wanted = bool(inputs.board_subsets) or not args.no_ablation
    if wanted and not inputs.line_cache:
        progress(f"no ability cache under {args.abilities_root}: both ablations skipped")
    elif wanted:
        _method_b(args, scorecard, probe_model, inputs)
    if args.method_c and inputs.samples:
        progress("method C: a shallow trunk on frozen e")
        scorecard.method_c = method_c(args, probe_model, inputs.corpus, inputs.samples)

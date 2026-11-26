"""``evaluate-effect-model``: the three gates and the reported checks.

The orchestration only: each gate lives in its own module (``gate_one``,
``gate_two``, ``geometry_checks``), the UMAP picture in ``umap_check``, the
breakdowns in ``breakdowns``, the report lines of a half's control score in
``half_score_report``, and the reporting half's score in ``control_score``
(:func:`run_reporting_half`): the selected checkpoint scored once on the half
of each validation stratum its selection never read, through the scorer
training selected it with (FR-070), its rule tests per rule (FR-074).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from pathlib import Path

from effects.application.breakdown_table import BREAKDOWNS_CSV, write_breakdowns_csv
from effects.application.breakdowns import breakdown_checks
from effects.application.evaluation_context import (
    EvaluateEffectModelConfig,
    EvaluationContext,
    load_baselines,
    load_context,
)
from effects.application.gate_one import run_gate_one
from effects.application.gate_two import run_gate_two
from effects.application.geometry_checks import (
    NEIGHBOUR_QUERIES,
    cached_vectors,
    check_baseline_geometry,
    check_decodability,
    check_nearest_neighbours,
    check_ward,
    run_gate_three,
)
from effects.application.half_score_report import HalfReading, half_score_checks
from effects.application.record_scoring import score_records
from effects.application.umap_check import check_umap
from effects.application.unbuilt_checks import NOT_IMPLEMENTED_CHECKS, not_implemented
from effects.domain.baselines import Baseline
from effects.domain.evaluation_report import (
    CheckName,
    CheckResult,
    CheckStatus,
    EvaluationReport,
)
from effects.domain.records import distinct_records
from effects.domain.strata import VALIDATION_STRATA, Half
from effects.infrastructure.ability_cache_store import (
    AbilityCacheStore,
    SidecarIndex,
    cache_stamp,
    require_cache_stamp,
)
from effects.infrastructure.corpus_store import CorpusStore
from effects.infrastructure.shard_io import read_shard

logger = logging.getLogger(__name__)


def breakdown_records(context: EvaluationContext) -> dict[str, list]:
    """Both validation strata, as the curated dataset's fixed samples, each
    distinct record once (a sample repeats a record's row for every added
    copy, FR-068a)."""
    store = CorpusStore(context.corpus_path)
    out = {}
    for stratum in VALIDATION_STRATA:
        path = store.sample_path(stratum)
        out[stratum] = list(distinct_records(read_shard(path))) if path.exists() else []
    return out


def run_breakdowns(context: EvaluationContext) -> list[CheckResult]:
    """Score both validation strata once and report every breakdown."""
    strata = breakdown_records(context)
    everything = [record for rows in strata.values() for record in rows]
    if not everything:
        return [CheckResult(
            CheckName.BREAKDOWNS, CheckStatus.SKIPPED,
            "neither validation stratum holds a record to score",
        )]
    runnable = context.main_runnable(everything)
    manifest = context.manifest
    scored = {
        stratum: (
            score_records(
                rows, runnable, legality_mode_counts=manifest.legality_mode_counts,
            )
            if rows else []
        )
        for stratum, rows in strata.items()
    }
    games_by_text: dict[str, int] = dict(manifest.rarity)
    checks = breakdown_checks(
        scored, games_by_text=games_by_text,
        withheld_keyword=context.main.provenance.withheld_keyword,
    )
    table = context.config.output_dir / BREAKDOWNS_CSV
    rows = write_breakdowns_csv(table, checks)
    return [*checks, CheckResult(
        CheckName.BREAKDOWNS, CheckStatus.REPORTED,
        f"{rows} per-field figures of the lines above: {table}",
    )]


def attempt(
    name: str, produce: Callable[[], CheckResult | list[CheckResult]],
) -> list[CheckResult]:
    """A reported check's results; one that raises becomes a skip line naming
    it and the exception, its traceback going to the log, because a picture or
    a breakdown that cannot be drawn must not take the gate verdicts with it.
    A gate is never wrapped: its failure fails the run."""
    try:
        produced = produce()
    except Exception as error:
        logger.warning("the reported check %s raised", name, exc_info=True)
        return [CheckResult(name, CheckStatus.SKIPPED, f"{type(error).__name__}: {error}")]
    return produced if isinstance(produced, list) else [produced]


def run(
    config: EvaluateEffectModelConfig, report: EvaluationReport | None = None,
) -> EvaluationReport:
    """Load the checkpoints, run every check, and report. Each check lands in
    ``report`` (whose ``echo`` prints it) the moment it is made, the gates
    first."""
    context = load_context(config)
    variants = load_baselines(config, context.main)

    report = report if report is not None else EvaluationReport()
    sidecars = _sidecar_index(context)
    cached = _load_cache(context, sidecars, variants)
    report.add(run_gate_three(cached, config.abilities_root))
    report.add(_gate_one(context, variants))

    gate_two = run_gate_two(context)
    report.keyword_verdicts = gate_two.verdicts
    report.add(gate_two.check)

    logger.info("breakdowns: scoring both validation strata")
    report.extend(attempt(CheckName.BREAKDOWNS, lambda: run_breakdowns(context)))
    logger.info("the reporting half: scoring each validation stratum's")
    report.extend(attempt(CheckName.REPORTING_HALF, lambda: run_reporting_half(context)))
    report.extend(_geometry_checks(context, cached, sidecars, variants))
    report.extend(_unmeasured_checks(context.main))
    return report


def run_reporting_half(context: EvaluationContext) -> list[CheckResult]:
    """The reporting half of both validation strata, scored once through the
    control score's scorer under the checkpoint's own rule mask, stratum
    weight and deviance floor (FR-070); its rule tests per rule (FR-074).

    The selection set is the one frozen for the corpus, found by its digest
    under ``reports_dir``; without one the half is reported as skipped,
    naming why, since a report reports.
    """
    from effects.application.control_score import (
        ControlFlags,
        FrozenSelection,
        ScoredCheckpoint,
        score_half,
    )
    from effects.application.control_stratum import HalfScope
    from effects.application.freeze_selection_set import stratum_reader
    from effects.domain.selection_set import (
        SelectionSetError,
        load_selection_set,
        selection_set_path,
    )

    digest = context.manifest.digest()
    try:
        selection = load_selection_set(
            selection_set_path(context.config.reports_dir, digest), digest)
    except SelectionSetError as error:
        return [CheckResult(CheckName.REPORTING_HALF, CheckStatus.SKIPPED, str(error))]
    main = context.main
    note = ""
    if main.selection_set_digest and main.selection_set_digest != selection.digest:
        note = (f"the selection set was frozen again since training "
                f"({main.selection_set_digest[:12]}… then, {selection.digest[:12]}… now)")
        logger.warning("reporting half: %s", note)
    readers = {stratum: stratum_reader(context.shards[stratum])
               for stratum in selection.halves}
    loaded = context.main_model()
    half = str(Half.REPORTING)
    flags = main.scored_under()
    score = score_half(
        FrozenSelection(selection, readers),
        ScoredCheckpoint(loaded.encoder, loaded.model, loaded.batcher(),
                         context.manifest.effective_rule_outputs),
        ControlFlags(flags.mask, flags.stratum_weight, flags.deviance_floor),
        HalfScope(half),
    )
    games = {stratum: len(selection.halves[stratum].get(half, ())) for stratum in readers}
    return half_score_checks(
        score, HalfReading(half, games, selection_control=main.control_score, note=note))


def _sidecar_index(context: EvaluationContext):
    """The card and token trees' sidecars (FR-123), keyed on the checkpoint's
    surface once for every variant's cache."""
    from effects.infrastructure.vocabulary_file import vocabulary_surface

    return SidecarIndex(context.config.cards_folders,
                        surface=vocabulary_surface(context.tokenizer_files.vocab))


def _load_cache(
    context: EvaluationContext, sidecars, variants: dict,
    variant: Baseline = Baseline.FULL,
):
    """One variant's cache: a vector per unique text on the checkpoint's surface."""
    logger.info("loading the %s ability cache under %s", variant, context.config.abilities_root)
    return cached_vectors(sidecars, cache_store(context, variant, variants))


def cache_store(
    context: EvaluationContext, variant: Baseline, variants: dict,
) -> AbilityCacheStore:
    """``variant``'s cache, once its stamp names the checkpoint of that variant
    this run holds (finding 11.7): the main one for ``full``, a
    ``--variant-checkpoint`` for a baseline. The taxonomy cache is read without
    one, compared on everything but the checkpoint, since no weights enter it;
    any other baseline's needs its checkpoint."""
    return require_cache_stamp(
        context.config.abilities_root, lambda: _wanted_stamp(context, variant, variants),
        variant=variant,
    )


def _wanted_stamp(context: EvaluationContext, variant: Baseline, variants: dict):
    if variant in variants:
        return _stamp_of(context, context.config.baseline_checkpoints[variant], variants[variant])
    if variant not in (Baseline.FULL, Baseline.TAXONOMY):
        return None
    main = _stamp_of(context, context.config.checkpoint, context.main)
    return main if variant == Baseline.FULL else main.without_checkpoint()


def _stamp_of(context: EvaluationContext, path: Path, checkpoint):
    from effects.infrastructure.vocabulary_file import vocabulary_surface

    return cache_stamp(
        Path(path), checkpoint, cards_folders=context.config.cards_folders,
        surface=vocabulary_surface(context.tokenizer_files.vocab),
    )


def _gate_one(context: EvaluationContext, variants: dict) -> CheckResult:
    """Gate 1, which is defined against the identity baseline and needs it."""
    if Baseline.IDENTITY not in variants:
        return CheckResult(
            CheckName.GATE_1, CheckStatus.SKIPPED,
            "needs --variant-checkpoint identity=PATH: gate 1 is defined "
            "against that baseline",
        )
    return run_gate_one(context, variants[Baseline.IDENTITY])


def _geometry_checks(
    context: EvaluationContext, cached, sidecars, variants: dict,
) -> Iterator[CheckResult]:
    """The e-geometry checks, all reported, none gating, each yielded as it is
    made; one that raises is a skip line."""
    logger.info("reported checks: ward, neighbours, UMAP, decodability, baselines")
    steps = [
        (CheckName.WARD_CANARY, lambda: check_ward(cached)),
        (CheckName.NEAREST_NEIGHBOUR,
         lambda: check_nearest_neighbours(cached, NEIGHBOUR_QUERIES)),
        (CheckName.UMAP, lambda: check_umap(cached, context.config.output_dir)),
        (CheckName.DECODABILITY, lambda: check_decodability(
            cached, Path(context.config.win_rates),
            held_out_cards=context.main.provenance.held_out_cards)),
    ]
    for variant in (Baseline.NO_STATE, Baseline.TAXONOMY):
        steps.append((
            CheckName.baseline_comparison(variant),
            lambda variant=variant: check_baseline_geometry(
                cached, _load_cache(context, sidecars, variants, variant), variant),
        ))
    for name, step in steps:
        yield from attempt(name, step)


def _unmeasured_checks(main) -> list[CheckResult]:
    """The checks this run measured nothing for, each named with why."""
    checks = [not_implemented(name) for name in NOT_IMPLEMENTED_CHECKS]
    if not main.provenance.withheld_keyword:
        checks.append(CheckResult(
            CheckName.ZERO_SHOT_KEYWORD, CheckStatus.SKIPPED,
            "the checkpoint withheld no keyword; retrain with "
            "--withhold-keyword to give this check something to measure",
        ))
    return checks

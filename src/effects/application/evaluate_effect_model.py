"""``evaluate-effect-model``: the three gates and the reported checks.

The orchestration only: each gate lives in its own module (``gate_one``,
``gate_two``, ``geometry_checks``), the breakdowns in ``breakdowns``.
"""

from __future__ import annotations

import logging
from pathlib import Path

from effects.application.evaluation_context import (
    EvaluateEffectModelConfig,
    EvaluationContext,
    load_baselines,
    load_context,
)
from effects.domain.baselines import Baseline
from effects.domain.evaluation_report import (
    CheckName,
    CheckResult,
    CheckStatus,
    EvaluationReport,
)
from effects.domain.strata import VALIDATION_STRATA

logger = logging.getLogger(__name__)

#: What each unbuilt FR-116 check would measure. Never reported as a skip: the
#: corpus holds the records these read, so a skip would send an operator after
#: data that is already there.
NOT_IMPLEMENTED_CHECKS: dict[CheckName, str] = {
    CheckName.MATCHED_REAL_VS_FORK: (
        "it would compare each fork's observed outcome with the real record "
        "it mirrors"
    ),
    CheckName.PROBE_DIFF: (
        "it would re-run gate 2's canary over real and probe-fork combat "
        "pairs joined by mirror_of, for probed keywords only"
    ),
    CheckName.ROLE_POLARITY: (
        "it would compare the predicted mana-pool sign for {R} in cost "
        "position against effect position"
    ),
}


def not_implemented(name: CheckName) -> CheckResult:
    return CheckResult(
        name, CheckStatus.NOT_IMPLEMENTED,
        f"not implemented; {NOT_IMPLEMENTED_CHECKS[name]}",
    )


def breakdown_records(context: EvaluationContext) -> dict[str, list]:
    """Both validation strata, as the curated dataset's fixed samples: what
    best-checkpoint selection scored."""
    from effects.infrastructure.corpus_store import CorpusStore
    from effects.infrastructure.record_io import read_shard

    store = CorpusStore(context.corpus_path)
    out = {}
    for stratum in VALIDATION_STRATA:
        path = store.sample_path(stratum)
        out[stratum] = list(read_shard(path)) if path.exists() else []
    return out


def run_breakdowns(context: EvaluationContext) -> list[CheckResult]:
    """Score both validation strata once and report every breakdown."""
    from effects.application.breakdowns import breakdown_checks
    from effects.application.record_scoring import score_records

    strata = breakdown_records(context)
    everything = [record for rows in strata.values() for record in rows]
    if not everything:
        return [CheckResult(
            CheckName.BREAKDOWNS, CheckStatus.SKIPPED,
            "neither validation stratum holds a record to score",
        )]
    runnable = context.main_runnable(everything)
    manifest = context.manifest
    mode_counts = manifest.legality_mode_counts or None
    scored = {
        stratum: (
            score_records(rows, runnable, legality_mode_counts=mode_counts)
            if rows else []
        )
        for stratum, rows in strata.items()
    }
    games_by_text: dict[str, float] = dict(manifest.rarity)
    return breakdown_checks(
        scored, games_by_text=games_by_text,
        withheld_keyword=context.main.provenance.withheld_keyword,
    )


def run(config: EvaluateEffectModelConfig) -> EvaluationReport:
    """Load the checkpoints, run every check, and report."""
    from effects.application.gate_two import (
        evaluate_gate_two,
        gate_two_records,
        run_gate_two,
    )
    from effects.application.geometry_checks import run_gate_three

    context = load_context(config)
    variants = load_baselines(config, context.main)

    report = EvaluationReport()
    sidecars = _sidecar_index(context)
    cached = _load_cache(context, sidecars)
    report.add(run_gate_three(cached, config.abilities_root))
    report.add(_gate_one(context, variants))

    logger.info("gate 2: scoring the damage-step keywords on the game-disjoint stratum")
    report.keyword_verdicts = run_gate_two(
        gate_two_records(context.main.provenance, corpus_path=context.corpus_path),
        context.main_runnable,
    )
    report.add(evaluate_gate_two(report.keyword_verdicts))

    logger.info("breakdowns: scoring both validation strata")
    report.extend(run_breakdowns(context))
    report.extend(_geometry_checks(context, cached, sidecars))
    report.extend(_unmeasured_checks(context.main))
    return report


def _sidecar_index(context: EvaluationContext):
    """The card and token trees' sidecars (FR-123), keyed on the checkpoint's
    surface once for every variant's cache."""
    from effects.domain.ability_encoder import surface_of
    from effects.infrastructure.ability_cache_store import SidecarIndex

    return SidecarIndex(context.config.cards_folders, surface=surface_of(context.vocab_path))


def _load_cache(context: EvaluationContext, sidecars, variant: str = "full"):
    """One variant's cache: a vector per unique text on the checkpoint's surface."""
    from effects.application.geometry_checks import cached_vectors

    root = context.config.abilities_root
    logger.info("loading the %s ability cache under %s", variant, root)
    return cached_vectors(sidecars, root, variant=variant)


def _gate_one(context: EvaluationContext, variants: dict) -> CheckResult:
    """Gate 1, which is defined against the identity baseline and needs it."""
    from effects.application.gate_one import run_gate_one

    if Baseline.IDENTITY not in variants:
        return CheckResult(
            CheckName.GATE_1, CheckStatus.SKIPPED,
            "needs --variant-checkpoint identity=PATH: gate 1 is defined "
            "against that baseline",
        )
    return run_gate_one(context, variants[Baseline.IDENTITY])


def _geometry_checks(context: EvaluationContext, cached, sidecars) -> list[CheckResult]:
    """The e-geometry checks, all reported, none gating."""
    from effects.application.geometry_checks import (
        NEIGHBOUR_QUERIES,
        check_baseline_geometry,
        check_decodability,
        check_nearest_neighbours,
        check_umap,
        check_ward,
    )

    logger.info("reported checks: ward, neighbours, UMAP, decodability, baselines")
    checks = [
        check_ward(cached),
        check_nearest_neighbours(cached, NEIGHBOUR_QUERIES),
        check_umap(cached),
        check_decodability(
            cached, Path(context.config.win_rates),
            held_out_cards=context.main.provenance.held_out_cards,
        ),
    ]
    for variant in (Baseline.NO_STATE, Baseline.TAXONOMY):
        checks.append(check_baseline_geometry(
            cached, _load_cache(context, sidecars, variant), variant,
        ))
    return checks


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

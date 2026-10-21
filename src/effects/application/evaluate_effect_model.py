"""``evaluate-effect-model``: the three gates and the reported checks.

Two of the three gates block shipping, and they block for different reasons.
**Gate 1** asks whether the encoder is reading text at all, by requiring the
model to beat an ``identity`` baseline that can memorize every text it has seen
and knows nothing about one it has not. **Gate 3** asks whether the embeddings
have collapsed — a space where every ability sits on top of every other passes
every average-case metric and is useless.

**Gate 2 blocks nothing**, and that is deliberate: its per-keyword verdict is the
input to a build decision, not a ship decision. A keyword whose removal does not
move the prediction the way the rules say gets routed to a stage-three probe,
and a run where all eight pass means that machinery never has to be written.

Splits are never a flag. The held-out card list and the ``game_id`` sets come
from ``--checkpoint``, because the corpus is append-only: recomputing the split
against a grown corpus would score the gates partly on games the model trained
on. A ``--variant-checkpoint`` recording a different split fails fast, since a
baseline that saw different games is not a baseline.

This module is the orchestration: each gate lives in its own module
(``gate_one``, ``gate_two``, ``geometry_checks``), the breakdowns' report lines
in ``breakdowns``, and the report's values in ``domain.evaluation_report``.
"""

from __future__ import annotations

import logging
from pathlib import Path

from effects.application.evaluation_context import (
    EvaluateEffectModelConfig,
    EvaluationContext,
    load_context,
    load_variants,
)
from effects.domain.evaluation_report import (
    CheckName,
    CheckResult,
    CheckStatus,
    EvaluationReport,
)
from effects.domain.strata import VALIDATION_STRATA
from effects.domain.variants import Variant

logger = logging.getLogger(__name__)

CARD_DISJOINT, GAME_DISJOINT = VALIDATION_STRATA


# ── checks that wait for later stages (FR-111) ──────────────────────────

#: What each stage-gated check needs before it can run. Skipping rather than
#: failing is the contract: a check with no records has measured nothing, and
#: reporting that as a failure would make a stage-one run look broken.
STAGE_GATED_CHECKS: dict[CheckName, str] = {
    CheckName.MATCHED_REAL_VS_FORK: "fork records (stage three)",
    CheckName.PROBE_DIFF: "damage-step probe records (stage three)",
    CheckName.ROLE_POLARITY: "mana records (stage two)",
}


def skip_unavailable(name: CheckName) -> CheckResult:
    return CheckResult(
        name, CheckStatus.SKIPPED,
        f"needs {STAGE_GATED_CHECKS[name]}, which the corpus does not have yet",
    )


# ── breakdowns (FR-064 – FR-068) ────────────────────────────────────────

#: Records per stratum read from the raw corpus when the checkpoint names no
#: curated dataset, whose fixed validation samples are otherwise the source.
#: The curated samples are a few thousand records; reading every record of a
#: thousand raw games would hold gigabytes for the same breakdown.
BREAKDOWN_RAW_RECORDS = 4096


def breakdown_records(context: EvaluationContext) -> dict[str, list]:
    """Both validation strata, as the breakdowns score them.

    From the curated dataset's fixed samples when the checkpoint trained on
    one: they are what best-checkpoint selection scored, and the card-disjoint
    sample is filled round-robin over held-out texts (FR-051), so every text
    shows up in the per-text mean. Otherwise from the raw corpus: the games
    the checkpoint recorded for each stratum, up to a bound.
    """
    from effects.infrastructure.record_io import read_records_matching, read_shard

    main = context.main
    if context.corpus_path is not None:
        from effects.infrastructure.corpus_store import CorpusStore

        store = CorpusStore(context.corpus_path)
        out = {}
        for stratum in (CARD_DISJOINT, GAME_DISJOINT):
            path = store.sample_path(stratum)
            out[stratum] = list(read_shard(path)) if path.exists() else []
        return out

    games = {
        CARD_DISJOINT: frozenset(main.provenance.card_disjoint_games),
        GAME_DISJOINT: frozenset(main.provenance.game_disjoint_games),
    }
    out = {stratum: [] for stratum in games}
    if not any(games.values()):
        return out
    wanted = games[CARD_DISJOINT] | games[GAME_DISJOINT]
    records_dir = Path(context.config.records_dir)
    for record in read_records_matching(records_dir, game_ids=wanted):
        for stratum, members in games.items():
            if record.game_id in members and len(out[stratum]) < BREAKDOWN_RAW_RECORDS:
                out[stratum].append(record)
        if all(len(rows) >= BREAKDOWN_RAW_RECORDS for rows in out.values()):
            break
    return out


def run_breakdowns(context: EvaluationContext) -> list[CheckResult]:
    """Score both validation strata once and report every breakdown."""
    from effects.application.breakdowns import breakdown_checks
    from effects.application.gate_one import score_records

    strata = breakdown_records(context)
    everything = [record for rows in strata.values() for record in rows]
    if not everything:
        return [CheckResult(
            CheckName.BREAKDOWNS, CheckStatus.SKIPPED,
            "neither validation stratum holds a record to score",
        )]
    runnable = context.main_runnable(everything)
    manifest = context.manifest
    mode_counts = (manifest.legality_mode_counts or None) if manifest else None
    scored = {
        stratum: (
            score_records(
                rows, runnable.encoder, runnable.model, runnable.batcher,
                fields=runnable.fields, legality_mode_counts=mode_counts,
            )
            if rows else []
        )
        for stratum, rows in strata.items()
    }
    games_by_text: dict[str, float] = dict(manifest.rarity) if manifest else {}
    return breakdown_checks(
        scored, games_by_text=games_by_text,
        withheld_keyword=context.main.provenance.withheld_keyword,
    )


# ── the command ─────────────────────────────────────────────────────────


def run(config: EvaluateEffectModelConfig) -> EvaluationReport:
    """Load the checkpoints, run every check, and report.

    Splits come from ``--checkpoint`` and are never recomputed; a
    ``--variant-checkpoint`` recording a different split fails fast here, before
    any number is produced that would look comparable and not be.
    """
    from effects.application.gate_two import (
        evaluate_gate_two,
        gate_two_records,
        run_gate_two,
    )
    from effects.application.geometry_checks import run_gate_three

    context = load_context(config)
    variants = load_variants(config, context.main)

    report = EvaluationReport()
    cached = _load_cache(context)
    report.add(run_gate_three(cached, config.abilities_root))
    report.add(_gate_one(context, variants))

    logger.info("gate 2: scoring the damage-step keywords on the game-disjoint stratum")
    report.keyword_verdicts = run_gate_two(
        gate_two_records(context.config, context.main.provenance),
        context.main_runnable,
    )
    report.add(evaluate_gate_two(report.keyword_verdicts))

    logger.info("breakdowns: scoring both validation strata")
    report.extend(run_breakdowns(context))
    report.extend(_geometry_checks(context, cached))
    report.extend(_skipped_checks(context.main))
    return report


def _load_cache(context: EvaluationContext, variant: str = "full"):
    """One variant's cache, keyed on the surface the checkpoint encoded from.

    One vector per unique ability text on that surface — the population gate 3
    is defined over (FR-123) and the one every other geometry check reads —
    over the card and token trees: the variant-script tree's perturbed scripts
    are synthetic collection input, and the shipping cache serves the real card
    and token lines.
    """
    from effects.application.geometry_checks import load_cache
    from effects.domain.ability_encoder import surface_of

    config = context.config
    logger.info("loading the %s ability cache under %s", variant, config.abilities_root)
    return load_cache(
        config.abilities_root, config.cards_folders,
        surface=surface_of(context.vocab_path), variant=variant,
    )


def _gate_one(context: EvaluationContext, variants: dict) -> CheckResult:
    """Gate 1, which is defined against the identity baseline and needs it."""
    from effects.application.gate_one import run_gate_one

    if Variant.IDENTITY not in variants:
        return CheckResult(
            CheckName.GATE_1, CheckStatus.SKIPPED,
            "needs --variant-checkpoint identity=PATH: gate 1 is defined "
            "against that baseline",
        )
    return run_gate_one(context, variants[Variant.IDENTITY])


def _geometry_checks(context: EvaluationContext, cached) -> list[CheckResult]:
    """The e-geometry checks, all reported, none gating."""
    from effects.application.geometry_checks import (
        NEIGHBOUR_QUERIES,
        check_decodability,
        check_nearest_neighbours,
        check_umap,
        check_variant_geometry,
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
    for variant in (Variant.NO_STATE, Variant.TAXONOMY):
        checks.append(check_variant_geometry(
            cached, _load_cache(context, variant), variant,
        ))
    return checks


def _skipped_checks(main) -> list[CheckResult]:
    """The checks whose records do not exist yet, each named and skipped."""
    checks = [skip_unavailable(name) for name in STAGE_GATED_CHECKS]
    if not main.provenance.withheld_keyword:
        checks.append(CheckResult(
            CheckName.ZERO_SHOT_KEYWORD, CheckStatus.SKIPPED,
            "the checkpoint withheld no keyword; retrain with "
            "--withhold-keyword to give this check something to measure",
        ))
    return checks

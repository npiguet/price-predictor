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
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from itertools import chain, islice
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from effects.application.gate_one import GateOneMetrics
from effects.application.gate_two import FieldAgreement, KeywordScore
from effects.domain.damage_step_keywords import (
    DAMAGE_STEP_KEYWORDS,
    MIN_DIRECTION_AGREEMENT,
    MIN_QUALIFYING_RECORDS,
    DamageStepKeyword,
)
from effects.domain.gate_metrics import (
    GATE1_MIN_DEVIANCE_REDUCTION,
    GATE1_MIN_GATE_F1_GAIN,
    GATE1_MIN_ZONE_ACCURACY_GAIN,
    GATE3_MAX_MEAN_COSINE,
    GATE3_MAX_TOP_COMPONENT,
    cosine_distance,
    mean_pairwise_cosine,
    top_component_share,
)
from effects.domain.line_query import LineQuery
from effects.domain.records import LegalityDecision
from effects.domain.strata import VALIDATION_STRATA

if TYPE_CHECKING:
    from effects.domain.corpus_manifest import CorpusManifest
    from effects.infrastructure.effect_model_store import (
        EffectCheckpoint,
        SplitProvenance,
    )
    from effects.infrastructure.model_runner import LoadedModel, Runnable
    from effects.infrastructure.sidecar_io import SidecarCache

logger = logging.getLogger(__name__)

#: Records read eagerly before gate 2 loads its checkpoint. The loader measures
#: each slot kind's width from real records, and every combat record on a board
#: with a creature and a player answers that identically.
SLOT_WIDTH_SAMPLE = 8


class CheckStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    #: The records this check needs do not exist yet (FR-111).
    SKIPPED = "skipped"
    #: Reported, gates nothing.
    REPORTED = "reported"


@dataclass(frozen=True, slots=True)
class CheckResult:
    name: str
    status: CheckStatus
    detail: str = ""
    values: dict[str, float] = field(default_factory=dict)

    @property
    def blocks(self) -> bool:
        return self.status is CheckStatus.FAIL


# ── what every stage reads ──────────────────────────────────────────────


@dataclass
class EvaluationContext:
    """What every stage of one evaluation reads, gathered once in :func:`run`.

    The main checkpoint's weights are loaded on first use and kept: gate 1,
    the breakdowns and gate 2 all run it, and each used to load it afresh with
    a sidecar cache of its own. Each record set still gets a batcher of its own
    (``LoadedModel.runnable``), since the slot widths and the active fields are
    measured from the records it scores.
    """

    config: EvaluateEffectModelConfig
    main: EffectCheckpoint
    vocab_path: Path
    keyword_path: Path
    corpus_path: Path | None = None
    #: The curated dataset's manifest, read once; None for a raw-corpus run.
    manifest: CorpusManifest | None = None
    _main_model: LoadedModel | None = field(default=None, repr=False)
    _sidecars: SidecarCache | None = field(default=None, repr=False)

    def main_runnable(self, records) -> Runnable:
        """The main checkpoint, ready to run over ``records``."""
        if self._main_model is None:
            self._main_model = self._load(self.main)
        return self._main_model.runnable(records)

    def runnable_of(self, checkpoint, records) -> Runnable:
        """Another checkpoint over ``records``, loaded now and not kept."""
        return self._load(checkpoint).runnable(records)

    def _load(self, checkpoint) -> LoadedModel:
        from effects.infrastructure.model_runner import build_sidecars, load_model

        if self._sidecars is None:
            self._sidecars = build_sidecars(self.config)
        return load_model(
            self.config, checkpoint,
            vocab_path=self.vocab_path, keyword_path=self.keyword_path,
            sidecars=self._sidecars,
        )


# ── gate 1: the identity baseline ───────────────────────────────────────


def run_gate_one(context: EvaluationContext, identity) -> CheckResult:
    """Score both models on the split's unique-text stratum and compare.

    Loading a second model and running the corpus twice is the expensive part
    of an evaluation, so it happens only when the baseline is actually
    supplied. The baseline is loaded for this gate alone and released after it.
    """
    from effects.application.gate_one import (
        measure,
        training_texts,
        unique_text_records,
    )
    from effects.infrastructure.model_runner import stratum_records

    records, training_acting, message = stratum_records(
        context.config, context.main,
    )
    if not records:
        return CheckResult("gate-1", CheckStatus.SKIPPED, message)

    # Both checkpoints read one vocabulary (require_same_split), so one
    # surface's texts decide the stratum for both.
    stratum = None
    scores = {}
    for name, checkpoint in (("model", None), ("identity", identity)):
        runnable = (
            context.main_runnable(records) if checkpoint is None
            else context.runnable_of(checkpoint, records)
        )
        batcher = runnable.batcher
        if stratum is None:
            stratum = unique_text_records(
                records, batcher, seen=training_texts(training_acting, batcher),
            )
        if not stratum:
            return CheckResult(
                "gate-1", CheckStatus.SKIPPED,
                "no resolution record in the card-disjoint split has an "
                "ability text absent from training; the identity baseline "
                "could recall every text in it, so the comparison would "
                "measure memorization on both sides rather than reading",
            )
        scores[name] = measure(
            stratum, runnable.encoder, runnable.model, batcher,
            fields=runnable.fields,
        )

    return evaluate_gate_one(scores["model"], scores["identity"])


def _margin(
    label: str, value: float, breached: bool, comparison: str,
) -> str | None:
    """One margin's failure, or None where it holds.

    A margin that could not be measured fails rather than passes: NaN compares
    False against every threshold, so without this an empty stratum or an
    all-zero embedding read as a gate that held.
    """
    if math.isnan(value):
        return f"{label} not measured"
    return f"{label} {comparison}" if breached else None


def _breaches(*margins: str | None) -> list[str]:
    return [margin for margin in margins if margin is not None]


def _relative_reduction(baseline: float, value: float) -> float:
    """How far ``value`` falls below ``baseline``, as a share of it.

    An unmeasured baseline leaves the reduction unmeasured; a zero one leaves
    nothing to reduce.
    """
    if math.isnan(baseline):
        return baseline
    return (baseline - value) / baseline if baseline > 0 else 0.0


def evaluate_gate_one(
    model: GateOneMetrics, identity: GateOneMetrics,
) -> CheckResult:
    """All three margins must hold (FR-118). Blocks shipping.

    Three margins rather than one because they fail differently: the gate F1
    says whether the model knows *that* something happens, the zone accuracy
    whether it knows *what*, and the deviance whether it knows *how much*. A
    model can beat the baseline on one and tie on the others by learning the
    corpus's priors.
    """
    gate_gain = model.affected_gate_f1 - identity.affected_gate_f1
    zone_gain = model.zone_outcome_accuracy - identity.zone_outcome_accuracy
    deviance_reduction = _relative_reduction(
        identity.mean_poisson_deviance, model.mean_poisson_deviance,
    )
    values = {
        "affected_gate_f1_gain": gate_gain,
        "zone_outcome_accuracy_gain": zone_gain,
        "poisson_deviance_reduction": deviance_reduction,
    }
    failures = _breaches(
        _margin(
            "affected-gate F1 gain", gate_gain,
            gate_gain < GATE1_MIN_GATE_F1_GAIN,
            f"{gate_gain:+.3f} < {GATE1_MIN_GATE_F1_GAIN}",
        ),
        _margin(
            "zone-outcome accuracy gain", zone_gain,
            zone_gain < GATE1_MIN_ZONE_ACCURACY_GAIN,
            f"{zone_gain:+.3f} < {GATE1_MIN_ZONE_ACCURACY_GAIN}",
        ),
        _margin(
            "Poisson deviance reduction", deviance_reduction,
            deviance_reduction < GATE1_MIN_DEVIANCE_REDUCTION,
            f"{deviance_reduction:+.1%} < {GATE1_MIN_DEVIANCE_REDUCTION:.0%}",
        ),
    )
    if failures:
        return CheckResult(
            "gate-1", CheckStatus.FAIL,
            "the encoder is not reading text: " + "; ".join(failures),
            values,
        )
    return CheckResult(
        "gate-1", CheckStatus.PASS,
        "beats the identity baseline on all three margins", values,
    )


# ── gate 2: the damage-step keyword canary ──────────────────────────────


@dataclass(frozen=True, slots=True)
class KeywordVerdict:
    """One keyword's routing verdict. Blocks nothing."""

    keyword: str
    qualifying_records: int
    direction_agreement: float
    routed_to_probe: bool
    reason: str

    @property
    def passed(self) -> bool:
        return not self.routed_to_probe


def evaluate_keyword(
    row: DamageStepKeyword,
    *,
    qualifying_records: int,
    agreeing_records: int,
    per_field: dict[str, FieldAgreement] | None = None,
) -> KeywordVerdict:
    """Route one keyword: pass, or send it to a stage-three probe (FR-119).

    Under-sampled routes the same way as wrong. A keyword with 40 qualifying
    records has not been tested, and treating "untested" as "passed" is how a
    canary stops being one.

    ``per_field`` is reported but never routes: a row's verdict is "did every
    field move", and the breakdown says *which* one did not — a keyword whose
    magnitude moves and whose consequence does not is a different problem from
    one the model ignores entirely, and the combined percentage cannot tell them
    apart.
    """
    agreement = (
        agreeing_records / qualifying_records if qualifying_records else 0.0
    )
    breakdown = _render_per_field(per_field)
    if qualifying_records < MIN_QUALIFYING_RECORDS:
        return KeywordVerdict(
            row.keyword, qualifying_records, agreement, routed_to_probe=True,
            reason=(
                f"under-sampled: {qualifying_records} qualifying records < "
                f"{MIN_QUALIFYING_RECORDS}" + breakdown
            ),
        )
    if agreement < MIN_DIRECTION_AGREEMENT:
        return KeywordVerdict(
            row.keyword, qualifying_records, agreement, routed_to_probe=True,
            reason=(
                f"direction agreement {agreement:.1%} < "
                f"{MIN_DIRECTION_AGREEMENT:.0%} on {', '.join(row.fields)}"
                + breakdown
            ),
        )
    return KeywordVerdict(
        row.keyword, qualifying_records, agreement, routed_to_probe=False,
        reason=f"direction agreement {agreement:.1%}" + breakdown,
    )


def _render_per_field(per_field: dict[str, FieldAgreement] | None) -> str:
    """The per-effect tail of a keyword's report line, or nothing."""
    if not per_field:
        return ""
    parts = [
        f"{label} {agreeing / scored:.0%}"
        for label, (agreeing, scored) in per_field.items() if scored
    ]
    return f" ({', '.join(parts)})" if parts else ""


def evaluate_gate_two(verdicts: list[KeywordVerdict]) -> CheckResult:
    """The canary's routing report. **Blocks nothing** (FR-119).

    Its output is a build decision, not a ship decision: the keywords routed
    here are exactly the ones stage three needs probe machinery for, and a run
    where none are routed means that machinery is never written.
    """
    routed = [v.keyword for v in verdicts if v.routed_to_probe]
    detail = (
        "all eight keywords pass; no stage-three probe machinery is needed"
        if not routed
        else f"routed to a stage-three probe: {', '.join(routed)}"
    )
    return CheckResult(
        "gate-2", CheckStatus.REPORTED, detail,
        {v.keyword: v.direction_agreement for v in verdicts},
    )


# ── gate 3: collapse canaries ───────────────────────────────────────────


def evaluate_gate_three(vectors: np.ndarray, *, seed: int = 42) -> CheckResult:
    """Two collapse canaries, both of which block shipping (FR-123).

    One vector per unique ability text. A collapsed space passes every
    average-case metric — every prediction is the corpus mean and every mean is
    close — so this is checked on the geometry rather than on the loss.
    """
    cosine = mean_pairwise_cosine(vectors, seed=seed)
    top = top_component_share(vectors)
    values = {"mean_pairwise_cosine": cosine, "top_component_share": top}
    failures = _breaches(
        _margin(
            "mean pairwise cosine", cosine, cosine > GATE3_MAX_MEAN_COSINE,
            f"{cosine:.3f} > {GATE3_MAX_MEAN_COSINE}",
        ),
        _margin(
            "top principal component", top, top > GATE3_MAX_TOP_COMPONENT,
            f"explains {top:.1%} > {GATE3_MAX_TOP_COMPONENT:.0%}",
        ),
    )
    if failures:
        return CheckResult(
            "gate-3", CheckStatus.FAIL,
            "the embedding space has collapsed: " + "; ".join(failures), values,
        )
    return CheckResult("gate-3", CheckStatus.PASS, "no collapse", values)


# ── the ward canary (FR-112) ────────────────────────────────────────────


def evaluate_ward_canary(
    ward: np.ndarray,
    twins: list[np.ndarray],
    bare_keywords: list[np.ndarray],
    *,
    unresolved: Sequence[str] = (),
) -> CheckResult:
    """Is ward nearer its meaning than it is to keyword-shaped text in general?

    Passes when ward is closer to **each** functional twin than to the median of
    its distances to all bare single-keyword vectors. A median rather than a
    fixed threshold, so the criterion stays meaningful whatever scale the
    embedding settles at.

    ``unresolved`` names the checked-in lines the cache had no vector for. They
    are carried into the detail, and the counts compared into the values, so a
    verdict over fewer twins than the table lists says so.
    """
    tail = f"; unresolved: {'; '.join(unresolved)}" if unresolved else ""
    if not twins or not bare_keywords:
        return CheckResult(
            "ward-canary", CheckStatus.SKIPPED,
            "no ward, twin, or bare-keyword vectors in the cache" + tail,
        )
    reference = float(np.median([cosine_distance(ward, k) for k in bare_keywords]))
    distances = [cosine_distance(ward, twin) for twin in twins]
    farther = [d for d in distances if not (d < reference)]
    values = {
        "median_bare_keyword_distance": reference,
        "min_twin_distance": min(distances),
        "max_twin_distance": max(distances),
        "twins_compared": float(len(twins)),
        "bare_keywords_compared": float(len(bare_keywords)),
    }
    if farther:
        return CheckResult(
            "ward-canary", CheckStatus.REPORTED,
            f"{len(farther)} of {len(twins)} functional twins sit farther from "
            f"ward than the median bare keyword ({reference:.3f})" + tail,
            values,
        )
    return CheckResult(
        "ward-canary", CheckStatus.REPORTED,
        f"every functional twin is nearer than the median bare keyword "
        f"({reference:.3f})" + tail,
        values,
    )


# ── checks that wait for later stages (FR-111) ──────────────────────────

#: What each stage-gated check needs before it can run. Skipping rather than
#: failing is the contract: a check with no records has measured nothing, and
#: reporting that as a failure would make a stage-one run look broken.
STAGE_GATED_CHECKS: dict[str, str] = {
    "matched-real-vs-fork": "fork records (stage three)",
    "probe-diff": "damage-step probe records (stage three)",
    "role-polarity": "mana records (stage two)",
}


def skip_unavailable(name: str) -> CheckResult:
    return CheckResult(
        name, CheckStatus.SKIPPED,
        f"needs {STAGE_GATED_CHECKS[name]}, which the corpus does not have yet",
    )


# ── the report ──────────────────────────────────────────────────────────


@dataclass
class EvaluationReport:
    """Everything one evaluation run produced."""

    checks: list[CheckResult] = field(default_factory=list)
    keyword_verdicts: list[KeywordVerdict] = field(default_factory=list)

    def add(self, result: CheckResult) -> CheckResult:
        self.checks.append(result)
        return result

    @property
    def blocking_failures(self) -> list[CheckResult]:
        return [check for check in self.checks if check.blocks]

    @property
    def ships(self) -> bool:
        """Whether gates 1 and 3 both hold.

        Only gates 1 and 3 can answer no. Gate 2 is routing information and the
        remaining checks are diagnostics. The verdict is reported; it never
        stops the command.
        """
        return not self.blocking_failures

    def render(self) -> str:
        lines = []
        for check in self.checks:
            marker = {
                CheckStatus.PASS: "PASS",
                CheckStatus.FAIL: "FAIL",
                CheckStatus.SKIPPED: "skip",
                CheckStatus.REPORTED: "----",
            }[check.status]
            lines.append(f"[{marker}] {check.name}: {check.detail}")
            # Every figure a check measured, whatever its verdict: the design
            # record's Outcome section needs a passing margin as much as a
            # failing one.
            if check.values:
                lines.append("       " + "  ".join(
                    f"{name}={value:+.4f}"
                    for name, value in check.values.items()
                ))
        for verdict in self.keyword_verdicts:
            state = "pass" if verdict.passed else "→ probe"
            lines.append(
                f"    {verdict.keyword:<16} {state:<8} "
                f"n={verdict.qualifying_records:<6} {verdict.reason}"
            )
        lines.append(
            "SHIPS" if self.ships
            else "GATES FAILED: " + ", ".join(c.name for c in self.blocking_failures)
        )
        return "\n".join(lines)


#: Where the sealed pipeline's per-card winnability labels live by default;
#: ``--win-rates`` names another table. The decodability battery skips when it
#: is absent rather than failing: the labels are an operator's training data,
#: not a repository artifact.
DEFAULT_WIN_RATES = Path("output/sealed/cards-win-rates.txt")


@dataclass
class EvaluateEffectModelConfig:
    checkpoint: Path = field(
        default_factory=lambda: Path("models/effects/effect-model/latest.pt"),
    )
    variant_checkpoints: dict[str, Path] = field(default_factory=dict)
    records_dir: Path = field(default_factory=lambda: Path("output/effects/records/"))
    cards_folders: tuple[Path, ...] = (
        Path("output/cardsfolder/"), Path("output/tokenscripts/"),
    )
    variant_scripts: Path | None = None
    vocab_path: Path | None = None
    keyword_definitions: Path | None = None
    #: A curated dataset directory (`build-corpus`); defaults to the one
    #: `--checkpoint` recorded, if any (FR-147).
    corpus: Path | None = None
    abilities_root: Path = field(
        default_factory=lambda: Path("output/effects/abilities"),
    )
    #: The per-card win-rate table the decodability battery reads (FR-070).
    win_rates: Path = field(default_factory=lambda: DEFAULT_WIN_RATES)


def parse_variant_checkpoint(value: str) -> tuple[str, Path]:
    """``NAME=PATH`` for ``--variant-checkpoint``."""
    name, separator, path = value.partition("=")
    if not separator or not name.strip() or not path.strip():
        raise ValueError(
            f"--variant-checkpoint takes NAME=PATH, got {value!r}"
        )
    return name.strip(), Path(path.strip())


# ── corpus provenance (FR-147) ──────────────────────────────────────────


class CorpusMismatchError(RuntimeError):
    """The curated corpus was rebuilt since the checkpoint trained on it."""


def check_corpus(provenance: SplitProvenance, *, actual_digest: str) -> None:
    """Refuse a dataset that is not the one the checkpoint read (FR-147).

    A rebuild is a different split, so scoring the gates against it would score
    them partly on games the model trained on — the failure the recorded
    ``game_id`` sets exist to prevent, arriving through the corpus instead.
    """
    if not provenance.corpus_digest:
        return
    if provenance.corpus_digest != actual_digest:
        raise CorpusMismatchError(
            f"{provenance.corpus_path} has been rebuilt since this checkpoint "
            f"trained on it (recorded {provenance.corpus_digest}, found "
            f"{actual_digest}). Evaluate against the dataset it read, or "
            "retrain against this one."
        )


def resolve_corpus_path(
    provenance: SplitProvenance, *, corpus: Path | None, checkpoint: Path,
) -> Path | None:
    """Where to read the curated corpus ``check_corpus`` verifies, if anywhere.

    Defaults to what the checkpoint recorded; an explicit ``--corpus``
    overrides it. That much matches ``resolve_inference_paths``, but the two
    diverge on a checkpoint with no recorded corpus: ``resolve_inference_paths``
    always resolves a path and hash-checks it afterwards, override or not,
    while here there is no digest to check an override against, so the
    override is discarded rather than resolved. Refusing outright would be too
    strong — it would break a batch-eval script that passes ``--corpus``
    uniformly across a mix of curated and legacy checkpoints — and honouring
    it would be meaningless, so a warning names the checkpoint and says why,
    rather than the flag silently doing nothing.
    """
    if not provenance.corpus_digest:
        if corpus is not None:
            logger.warning(
                "%s records no curated corpus, so --corpus %s has nothing "
                "to check its digest against and is ignored.",
                checkpoint, corpus,
            )
        return None
    if corpus is not None:
        return Path(corpus)
    return Path(provenance.corpus_path)


def keyword_rows() -> tuple[DamageStepKeyword, ...]:
    """Gate 2's table, so the evaluator and the report read one source."""
    return DAMAGE_STEP_KEYWORDS


def load_shipping_cache(
    config: EvaluateEffectModelConfig, *, surface: str, variant: str = "full",
):
    """The cache the evaluator reads: the card and token trees, on ``surface``.

    One vector per unique ability text on the checkpoint's encoding surface —
    the population gate 3 is defined over (FR-123) and the one every other
    geometry check reads. Keyed on the text rather than on the vector's bytes,
    because one text encoded in two batches differs in float noise, and a
    byte key counts those copies as distinct points. The variant-script tree
    is left out: its perturbed scripts are synthetic collection input, and the
    shipping cache serves the real card and token lines.
    """
    from effects.application.geometry_checks import load_cache

    return load_cache(
        config.abilities_root, config.cards_folders,
        surface=surface, variant=variant,
    )


def run(config: EvaluateEffectModelConfig) -> EvaluationReport:
    """Load the checkpoints, run every check, and report.

    Splits come from ``--checkpoint`` and are never recomputed; a
    ``--variant-checkpoint`` recording a different split fails fast here, before
    any number is produced that would look comparable and not be.
    """
    from effects.infrastructure.effect_model_store import (
        load_effect_checkpoint,
        load_for_inference,
        require_same_split,
    )

    main_path = Path(config.checkpoint)
    main, vocab_path, keyword_path = load_for_inference(
        main_path,
        vocab_path=config.vocab_path, keyword_path=config.keyword_definitions,
    )
    corpus_path = resolve_corpus_path(
        main.provenance, corpus=config.corpus, checkpoint=main_path,
    )
    manifest = None
    if corpus_path is not None:
        from effects.infrastructure.corpus_store import CorpusStore

        manifest = CorpusStore(corpus_path).load()
    check_corpus(
        main.provenance, actual_digest=manifest.digest() if manifest else "",
    )
    if corpus_path is not None:
        from effects.infrastructure.record_io import iter_shards, refuse_mixed_generations

        # Before any record is read (FR-033): the two generations read
        # actor_player on playability records differently.
        refuse_mixed_generations(iter_shards(Path(corpus_path)))

    variants = {}
    for name, path in config.variant_checkpoints.items():
        loaded = load_effect_checkpoint(path)
        require_same_split(
            main, loaded, main_path=main_path, variant_path=Path(path),
        )
        variants[name] = loaded

    from effects.application.geometry_checks import (
        check_decodability,
        check_nearest_neighbours,
        check_umap,
        check_variant_geometry,
        check_ward,
    )
    from effects.domain.ability_encoder import surface_of

    context = EvaluationContext(
        config, main, vocab_path, keyword_path,
        corpus_path=corpus_path, manifest=manifest,
    )
    report = EvaluationReport()
    logger.info("loading the ability cache under %s", config.abilities_root)
    # The surface the checkpoint encoded from, so every geometry check keys
    # the cache on the texts its vectors actually encode.
    surface = surface_of(vocab_path)
    cached = load_shipping_cache(config, surface=surface)

    # ── gate 3: geometry, read off the cache ──
    logger.info("gate 3: geometry over %d unique cached texts", len(cached.by_text))
    vectors = cached.matrix()
    if vectors.shape[0] < 2:
        report.add(CheckResult(
            "gate-3", CheckStatus.SKIPPED,
            f"no ability cache under {config.abilities_root}; run "
            "encode-abilities first",
        ))
    else:
        report.add(evaluate_gate_three(vectors))

    # ── gate 1: needs the identity baseline ──
    if "identity" not in variants:
        report.add(CheckResult(
            "gate-1", CheckStatus.SKIPPED,
            "needs --variant-checkpoint identity=PATH: gate 1 is defined "
            "against that baseline",
        ))
    else:
        report.add(run_gate_one(context, variants["identity"]))

    # ── gate 2: per keyword, routing only ──
    logger.info("gate 2: scoring the damage-step keywords on the game-disjoint stratum")
    verdicts = run_gate_two(context)
    report.keyword_verdicts = verdicts
    report.add(evaluate_gate_two(verdicts))

    # ── breakdowns: per text, rarity, family, policy, decision (FR-064–068) ──
    logger.info("breakdowns: scoring both validation strata")
    for check in run_breakdowns(context):
        report.add(check)

    # ── e-geometry checks, all reported ──
    logger.info("reported checks: ward, neighbours, UMAP, decodability, baselines")
    report.add(check_ward(cached))
    report.add(check_nearest_neighbours(cached, NEIGHBOUR_QUERIES))
    report.add(check_umap(cached))
    report.add(check_decodability(
        cached, Path(config.win_rates),
        held_out_cards=main.provenance.held_out_cards,
    ))
    for name in ("no-state", "taxonomy"):
        report.add(check_variant_geometry(
            cached,
            load_shipping_cache(config, surface=surface, variant=name),
            name,
        ))

    # ── checks whose records do not exist yet ──
    for name in STAGE_GATED_CHECKS:
        report.add(skip_unavailable(name))

    if not main.provenance.withheld_keyword:
        report.add(CheckResult(
            "zero-shot-keyword", CheckStatus.SKIPPED,
            "the checkpoint withheld no keyword; retrain with "
            "--withhold-keyword to give this check something to measure",
        ))
    return report


#: Lines whose neighbours a reader can judge at a glance. Reported rather than
#: scored: the point is for a person to see whether the neighbours of a removal
#: spell are removal spells, which no metric asks. Each prose line is printed
#: with several scripts across the corpus, so each pins the card whose script
#: the query stands for (see :mod:`effects.domain.line_query`).
NEIGHBOUR_QUERIES: tuple[LineQuery, ...] = (
    LineQuery("destroy target creature.", card="murder"),
    LineQuery("draw a card.", card="think twice"),
    LineQuery("CARDNAME deals 3 damage to any target.", card="lightning bolt"),
    LineQuery(
        "target creature gets +2/+2 until end of turn.", card="artful maneuver",
    ),
)

# ── breakdowns (FR-064 – FR-068) ────────────────────────────────────────
#
# The grouping lives in `effects.application.breakdowns` and the per-record
# scoring in `gate_one.measure`; what is left here is which records are scored
# and how each group becomes a report line.

#: Records per stratum read from the raw corpus when the checkpoint names no
#: curated dataset, whose fixed validation samples are otherwise the source.
#: The curated samples are a few thousand records; reading every record of a
#: thousand raw games would hold gigabytes for the same breakdown.
BREAKDOWN_RAW_RECORDS = 4096

CARD_DISJOINT, GAME_DISJOINT = VALIDATION_STRATA


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
    from effects.application.gate_one import measure

    strata = breakdown_records(context)
    everything = [record for rows in strata.values() for record in rows]
    if not everything:
        return [CheckResult(
            "breakdowns", CheckStatus.SKIPPED,
            "neither validation stratum holds a record to score",
        )]
    runnable = context.main_runnable(everything)
    manifest = context.manifest
    mode_counts = (manifest.legality_mode_counts or None) if manifest else None
    scored = {
        stratum: (
            list(measure(
                rows, runnable.encoder, runnable.model, runnable.batcher,
                fields=runnable.fields, legality_mode_counts=mode_counts,
            ).records)
            if rows else []
        )
        for stratum, rows in strata.items()
    }
    games_by_text: dict[str, float] = dict(manifest.rarity) if manifest else {}
    return breakdown_checks(
        scored, games_by_text=games_by_text,
        withheld_keyword=context.main.provenance.withheld_keyword,
    )


def breakdown_checks(
    scored: dict[str, list],
    *,
    games_by_text: dict[str, float],
    withheld_keyword: str | None,
) -> list[CheckResult]:
    """Every breakdown of the scored strata, one report line per group.

    Pure: ``scored`` maps a stratum to its ``RecordResult`` rows, so the
    report's shape is testable without a model.
    """
    from effects.application import breakdowns as bd

    card = scored.get(CARD_DISJOINT, [])
    game = scored.get(GAME_DISJOINT, [])
    checks: list[CheckResult] = []
    if card:
        checks.append(_group_check(CARD_DISJOINT, bd.overall(card)))
        for bucket, group in bd.by_rarity_bucket(card, games_by_text).items():
            checks.append(_group_check(f"{CARD_DISJOINT} rarity {bucket}", group))
        families = bd.by_family(card)
        for family, group in families.items():
            checks.append(_group_check(f"{CARD_DISJOINT} family {family}", group))
        if families:
            checks.append(_group_check(
                f"{CARD_DISJOINT} mean over families",
                bd.mean_over_families(families),
                detail=f"{len(families)} families, each weighted once",
            ))
    else:
        checks.append(CheckResult(
            CARD_DISJOINT, CheckStatus.SKIPPED,
            "the card-disjoint stratum holds no record to break down",
        ))

    gaps = bd.memorization_gap(bd.by_family(game), bd.by_family(card))
    if gaps:
        for family, values in gaps.items():
            checks.append(CheckResult(
                f"memorization-gap {family}", CheckStatus.REPORTED,
                "game-disjoint minus card-disjoint loss, per field", values,
            ))
    else:
        checks.append(CheckResult(
            "memorization-gap", CheckStatus.SKIPPED,
            "no family has records in both validation strata",
        ))

    for stratum, rows in ((CARD_DISJOINT, card), (GAME_DISJOINT, game)):
        policy = bd.policy_slice(rows)
        if bd.OFF_POLICY in policy:
            for name, group in policy.items():
                checks.append(_group_check(f"{stratum} policy {name}", group))
        else:
            checks.append(CheckResult(
                f"{stratum} off-policy", CheckStatus.SKIPPED,
                "no record in this stratum was acted by the random seat",
            ))
        decisions = bd.decision_slice(rows)
        if LegalityDecision.REAL in decisions:
            for name, group in decisions.items():
                checks.append(_group_check(f"{stratum} legality {name}", group))
        else:
            checks.append(CheckResult(
                f"{stratum} real-decision legality", CheckStatus.SKIPPED,
                "no legality record in this stratum is classed a real decision",
            ))

    if withheld_keyword:
        groups = bd.keyword_slice([*card, *game], withheld_keyword)
        if bd.WITHHELD in groups:
            for name, group in groups.items():
                checks.append(_group_check(
                    f"zero-shot-keyword {withheld_keyword} {name}", group,
                ))
        else:
            checks.append(CheckResult(
                "zero-shot-keyword", CheckStatus.SKIPPED,
                f"no validation record carries the withheld keyword "
                f"{withheld_keyword!r}",
            ))
    return checks


def _group_check(name: str, group, *, detail: str | None = None) -> CheckResult:
    """One group's report line: per-record field means, then per-text ones.

    Per-text means are prefixed ``text:``; the detail carries both counts, so a
    mean over three records reads as one. A mean of group means passes its own
    ``detail``, since its counts are groups rather than records.
    """
    from effects.application.breakdowns import TOTAL

    values = {
        **group.per_record.means,
        **{f"text:{key}": value for key, value in group.per_text.means.items()},
    }
    texts = group.per_text.counts.get(TOTAL, 0)
    return CheckResult(
        name, CheckStatus.REPORTED,
        detail or f"{group.per_record.records} records, {texts} texts", values,
    )

# ── gate 2: the damage-step keyword canary ──────────────────────────────
#
# The scoring itself lives in `effects.application.gate_two`, the way gate 1's
# does in `gate_one`; what is left here is the selection, the routing and the
# report line.


def _gate_two_records(config, provenance):
    """The combat records of the checkpoint's **game-disjoint** games (FR-120).

    Game-disjoint rather than card-disjoint: the gate asks whether the model
    uses a keyword it has seen, not whether it generalizes to text it has not,
    and holding out the carriers as well would empty six of the eight
    populations to answer a question gate 1 already answers.

    An iterator, not a list. The split is a thousand games and a parsed record
    costs about 45 KB, so the scorer reads it a chunk at a time.
    """
    from effects.domain.records import RecordKind
    from effects.infrastructure.record_io import read_records_matching

    games = frozenset(provenance.game_disjoint_games)
    if not games:
        return iter(())
    return read_records_matching(
        Path(config.records_dir), game_ids=games, kinds={RecordKind.COMBAT.value},
    )


def _gate_two_runnable(context: EvaluationContext, records):
    """The main checkpoint with a batcher measured from ``records``.

    A seam of its own so a test can hand gate 2 a stand-in model; the weights
    come from the context, loaded once for the whole evaluation.
    """
    return context.main_runnable(records)


def run_gate_two(context: EvaluationContext) -> list[KeywordVerdict]:
    """Score the eight keywords and route each one (FR-119 – FR-122).

    Nothing is loaded when the split holds no combat record to score: every
    keyword is then under-sampled at n=0, which is the same verdict a corpus
    with forty records gets and for the same reason.
    """
    from effects.application.gate_two import score_keywords

    rows = keyword_rows()
    records = iter(_gate_two_records(context.config, context.main.provenance))
    # A handful of records up front, because the loader measures the slot
    # widths from real ones; the rest stays a stream the scorer reads in chunks.
    head = list(islice(records, SLOT_WIDTH_SAMPLE))
    if not head:
        scores = {row.keyword: KeywordScore() for row in rows}
    else:
        runnable = _gate_two_runnable(context, head)
        scores = score_keywords(
            chain(head, records), rows, runnable.encoder, runnable.model,
            runnable.batcher, fields=runnable.fields,
        )
    return [
        evaluate_keyword(
            row,
            qualifying_records=scores[row.keyword].qualifying,
            agreeing_records=scores[row.keyword].agreeing,
            per_field=scores[row.keyword].per_field,
        )
        for row in rows
    ]

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
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from itertools import chain, islice
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from effects.application.gate_one import GateOneMetrics
from effects.application.gate_two import KeywordScore
from effects.domain.damage_step_keywords import (
    DAMAGE_STEP_KEYWORDS,
    MIN_DIRECTION_AGREEMENT,
    MIN_QUALIFYING_RECORDS,
    DamageStepKeyword,
)
from effects.domain.line_query import LineQuery

if TYPE_CHECKING:
    from effects.infrastructure.effect_model_store import SplitProvenance

logger = logging.getLogger(__name__)

# ── gate thresholds (FR-118, FR-119, FR-123). Contract, not tuning. ──────
GATE1_MIN_GATE_F1_GAIN = 0.05
GATE1_MIN_ZONE_ACCURACY_GAIN = 0.05
GATE1_MIN_DEVIANCE_REDUCTION = 0.05
GATE3_MAX_MEAN_COSINE = 0.5
GATE3_COSINE_PAIRS = 10_000
GATE3_MAX_TOP_COMPONENT = 0.30

#: Records read eagerly before gate 2 loads its checkpoint. The loader measures
#: each slot kind's width from real records, and every combat record on a board
#: with a creature and a player answers that identically.
SLOT_WIDTH_SAMPLE = 8


class Stratum(StrEnum):
    """The four held-out strata (FR-106).

    They separate "has the model seen this exact text" from "has it seen these
    parts": a model that only memorized texts scores well on ``shared_text``
    and badly on the other three.
    """

    #: The line's text appears on no training card.
    UNIQUE_TEXT = "unique-text"
    #: The line's text also appears on a training card.
    SHARED_TEXT = "shared-text"
    #: Every sub-ability API type appears in training; their combination does not.
    NOVEL_COMBINATION = "novel-combination"
    #: A numeric parameter outside the range seen in training.
    NUMERIC_EXTRAPOLATION = "numeric-extrapolation"


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


# ── gate 1: the identity baseline ───────────────────────────────────────


def run_gate_one(
    config, main, identity, *, vocab_path: Path, keyword_path: Path,
) -> CheckResult:
    """Score both models on the split's unique-text stratum and compare.

    Loading two models and running the corpus twice is the expensive part of an
    evaluation, so it happens only when the baseline is actually supplied.
    """
    from effects.application.gate_one import (
        measure,
        training_texts,
        unique_text_records,
    )
    from effects.infrastructure.model_runner import (
        load_runnable,
        stratum_records,
    )

    records, training, message = stratum_records(config, main)
    if not records:
        return CheckResult("gate-1", CheckStatus.SKIPPED, message)

    scores = {}
    for name, checkpoint in (("model", main), ("identity", identity)):
        encoder, model, batcher, fields = load_runnable(
            config, checkpoint,
            vocab_path=vocab_path, keyword_path=keyword_path,
            records=records,
        )
        stratum = unique_text_records(
            records, batcher, seen=training_texts(training, batcher),
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
            stratum, encoder, model, batcher, fields=fields,
        )

    return evaluate_gate_one(scores["model"], scores["identity"])


def poisson_deviance(predicted: np.ndarray, observed: np.ndarray) -> float:
    """Mean Poisson deviance, the count analogue of squared error.

    Gate 1 reads it over the magnitude side of the count-valued fields, which is
    where "how much damage" and "how many cards" live — the numbers a model that
    is not reading the text can only predict at the corpus average.
    """
    predicted = np.asarray(predicted, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)
    if predicted.size == 0:
        return float("nan")
    safe_predicted = np.clip(predicted, 1e-9, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(observed > 0, observed * np.log(observed / safe_predicted), 0.0)
    return float(2.0 * np.mean(ratio - (observed - safe_predicted)))


def f1_score(predicted: np.ndarray, observed: np.ndarray) -> float:
    """Binary F1 over the affected/unaffected gate."""
    predicted = np.asarray(predicted).astype(bool)
    observed = np.asarray(observed).astype(bool)
    true_positive = float(np.sum(predicted & observed))
    if true_positive == 0:
        return 0.0
    precision = true_positive / float(np.sum(predicted))
    recall = true_positive / float(np.sum(observed))
    return 2 * precision * recall / (precision + recall)


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
    deviance_reduction = (
        (identity.mean_poisson_deviance - model.mean_poisson_deviance)
        / identity.mean_poisson_deviance
        if identity.mean_poisson_deviance > 0
        else 0.0
    )
    values = {
        "affected_gate_f1_gain": gate_gain,
        "zone_outcome_accuracy_gain": zone_gain,
        "poisson_deviance_reduction": deviance_reduction,
    }
    failures = []
    if gate_gain < GATE1_MIN_GATE_F1_GAIN:
        failures.append(
            f"affected-gate F1 gain {gate_gain:+.3f} < {GATE1_MIN_GATE_F1_GAIN}"
        )
    if zone_gain < GATE1_MIN_ZONE_ACCURACY_GAIN:
        failures.append(
            f"zone-outcome accuracy gain {zone_gain:+.3f} < "
            f"{GATE1_MIN_ZONE_ACCURACY_GAIN}"
        )
    if deviance_reduction < GATE1_MIN_DEVIANCE_REDUCTION:
        failures.append(
            f"Poisson deviance reduction {deviance_reduction:+.1%} < "
            f"{GATE1_MIN_DEVIANCE_REDUCTION:.0%}"
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
    per_field: dict[str, tuple[int, int]] | None = None,
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


def _render_per_field(per_field: dict[str, tuple[int, int]] | None) -> str:
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


def mean_pairwise_cosine(
    vectors: np.ndarray, *, pairs: int = GATE3_COSINE_PAIRS, seed: int = 42,
) -> float:
    """Mean cosine similarity over ``pairs`` random distinct pairs.

    Sampled rather than exhaustive because the corpus has ~38k unique texts and
    the full pairwise matrix is 700 million entries for a number a sample
    estimates to three decimals.
    """
    count = vectors.shape[0]
    if count < 2:
        return float("nan")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    unit = vectors / np.clip(norms, 1e-12, None)
    rng = random.Random(seed)
    total = 0.0
    for _ in range(pairs):
        i = rng.randrange(count)
        j = rng.randrange(count - 1)
        if j >= i:
            j += 1
        total += float(np.dot(unit[i], unit[j]))
    return total / pairs


def top_component_share(vectors: np.ndarray) -> float:
    """Fraction of total variance the largest principal component explains."""
    if vectors.shape[0] < 2:
        return float("nan")
    centred = vectors - vectors.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centred, compute_uv=False)
    variance = singular ** 2
    total = float(variance.sum())
    return float(variance[0] / total) if total > 0 else float("nan")


def evaluate_gate_three(vectors: np.ndarray, *, seed: int = 42) -> CheckResult:
    """Two collapse canaries, both of which block shipping (FR-123).

    One vector per unique ability text. A collapsed space passes every
    average-case metric — every prediction is the corpus mean and every mean is
    close — so this is checked on the geometry rather than on the loss.
    """
    cosine = mean_pairwise_cosine(vectors, seed=seed)
    top = top_component_share(vectors)
    values = {"mean_pairwise_cosine": cosine, "top_component_share": top}
    failures = []
    if not math.isnan(cosine) and cosine > GATE3_MAX_MEAN_COSINE:
        failures.append(
            f"mean pairwise cosine {cosine:.3f} > {GATE3_MAX_MEAN_COSINE}"
        )
    if not math.isnan(top) and top > GATE3_MAX_TOP_COMPONENT:
        failures.append(
            f"top principal component explains {top:.1%} > "
            f"{GATE3_MAX_TOP_COMPONENT:.0%}"
        )
    if failures:
        return CheckResult(
            "gate-3", CheckStatus.FAIL,
            "the embedding space has collapsed: " + "; ".join(failures), values,
        )
    return CheckResult("gate-3", CheckStatus.PASS, "no collapse", values)


# ── the ward canary (FR-112) ────────────────────────────────────────────


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0.0:
        return float("nan")
    return 1.0 - float(np.dot(a, b)) / denominator


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


# ── the role-polarity probe (FR-116) ────────────────────────────────────


def evaluate_role_polarity(
    cost_position_sign: float | None, effect_position_sign: float | None,
) -> CheckResult:
    """Does ``{R}`` in a cost mean the opposite of ``{R}`` in an effect?

    The narrowest possible test of the role embedding. The same mana symbol
    appears on both sides of an activated ability, and the two mean opposite
    things: paying ``{R}`` takes red mana out of the pool, producing ``{R}``
    puts it in. A model that read the symbol without its role would predict the
    same sign for both.

    Needs mana records, so it waits for stage two: the cost half is observable
    from stage one, but the effect half only exists once mana abilities are
    collected.
    """
    if cost_position_sign is None or effect_position_sign is None:
        return skip_unavailable("role-polarity")
    values = {
        "cost_position_mana_sign": cost_position_sign,
        "effect_position_mana_sign": effect_position_sign,
    }
    if cost_position_sign < 0 < effect_position_sign:
        return CheckResult(
            "role-polarity", CheckStatus.REPORTED,
            f"{{R}} in cost position predicts a mana decrease "
            f"({cost_position_sign:+.3f}) and in effect position an increase "
            f"({effect_position_sign:+.3f})",
            values,
        )
    return CheckResult(
        "role-polarity", CheckStatus.REPORTED,
        f"{{R}} predicts the same direction in both positions "
        f"(cost {cost_position_sign:+.3f}, effect {effect_position_sign:+.3f}) "
        "— the role embedding is not separating them",
        values,
    )


# ── the fork checks (FR-042, FR-116) ────────────────────────────────────


def pair_forks(records: list) -> list[tuple]:
    """Join each fork record to the real record it mirrors.

    The pairing lives in ``mirror_of`` rather than being reconstructed from
    board state, because two combats on one turn can look identical and the
    pairing has to be exact for the difference to mean anything.
    """
    by_id = {record.record_id: record for record in records}
    pairs = []
    for record in records:
        if record.fork and record.mirror_of:
            real = by_id.get(record.mirror_of)
            if real is not None:
                pairs.append((real, record))
    return pairs


def evaluate_matched_forks(
    pairs: list[tuple], agreements: list[bool],
) -> CheckResult:
    """How often the model's predictions agree across a matched pair.

    The real-versus-fork difference is computed **here**, at evaluation time,
    and never becomes a training target (FR-042). A model trained on its own
    disagreements would learn to reproduce its errors rather than correct them.
    """
    if not pairs:
        return skip_unavailable("matched-real-vs-fork")
    if not agreements:
        return CheckResult(
            "matched-real-vs-fork", CheckStatus.REPORTED,
            f"{len(pairs)} matched pairs, none scored",
        )
    rate = sum(agreements) / len(agreements)
    return CheckResult(
        "matched-real-vs-fork", CheckStatus.REPORTED,
        f"predictions agree on {rate:.1%} of {len(pairs)} matched pairs",
        {"agreement": rate, "pairs": float(len(pairs))},
    )


def evaluate_probe_diff(
    verdicts_by_keyword: dict[str, KeywordVerdict],
    probed_keywords: tuple[str, ...],
) -> CheckResult:
    """Gate 2's canary, re-run over the real-and-fork combat pairs.

    Only for keywords a probe was actually taken for: the point is to check
    whether an isolated counterfactual agrees with the model-side perturbation
    gate 2 used, and a keyword with no probe has nothing to compare against.
    """
    if not probed_keywords:
        return skip_unavailable("probe-diff")
    rendered = []
    for keyword in probed_keywords:
        verdict = verdicts_by_keyword.get(keyword)
        if verdict is None:
            continue
        rendered.append(
            f"{keyword} {verdict.direction_agreement:.1%} over "
            f"{verdict.qualifying_records} pairs"
        )
    return CheckResult(
        "probe-diff", CheckStatus.REPORTED,
        "; ".join(rendered) if rendered else "no probed keyword had pairs",
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
        """Whether the model and cache may ship.

        Only gates 1 and 3 can answer no. Gate 2 is routing information and the
        remaining checks are diagnostics.
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
            else "BLOCKED: " + ", ".join(c.name for c in self.blocking_failures)
        )
        return "\n".join(lines)


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
    sealed_encoder_checkpoint: Path = field(
        default_factory=lambda: Path("models/sealed/encoder/latest.pt"),
    )


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
        EffectModelStore,
        require_same_split,
        resolve_inference_paths,
    )

    main_path = Path(config.checkpoint)
    main = EffectModelStore(main_path.parent).load(main_path)
    vocab_path, keyword_path = resolve_inference_paths(
        main.provenance,
        vocab_path=config.vocab_path,
        keyword_path=config.keyword_definitions,
    )
    main.provenance.verify_hashes(
        vocab_path=vocab_path, keyword_path=keyword_path,
    )
    corpus_path = resolve_corpus_path(
        main.provenance, corpus=config.corpus, checkpoint=main_path,
    )
    actual_corpus_digest = ""
    if corpus_path is not None:
        from effects.infrastructure.corpus_store import CorpusStore

        actual_corpus_digest = CorpusStore(corpus_path).load().digest()
    check_corpus(main.provenance, actual_digest=actual_corpus_digest)

    variants = {}
    for name, path in config.variant_checkpoints.items():
        loaded = EffectModelStore(Path(path).parent).load(Path(path))
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

    report = EvaluationReport()
    # The surface the checkpoint encoded from, so every geometry check keys
    # the cache on the texts its vectors actually encode.
    surface = surface_of(vocab_path)
    cached = load_shipping_cache(config, surface=surface)

    # ── gate 3: geometry, read off the cache ──
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
        report.add(run_gate_one(
            config, main, variants["identity"],
            vocab_path=vocab_path, keyword_path=keyword_path,
        ))

    # ── gate 2: per keyword, routing only ──
    verdicts = run_gate_two(
        config, main, vocab_path=vocab_path, keyword_path=keyword_path,
    )
    report.keyword_verdicts = verdicts
    report.add(evaluate_gate_two(verdicts))

    # ── e-geometry checks, all reported ──
    report.add(check_ward(cached))
    report.add(check_nearest_neighbours(cached, NEIGHBOUR_QUERIES))
    report.add(check_umap(cached))
    report.add(check_decodability(
        cached, {}, DEFAULT_WIN_RATES,
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

    if main.provenance.withheld_keyword:
        report.add(CheckResult(
            "zero-shot-keyword", CheckStatus.REPORTED,
            f"the checkpoint withheld {main.provenance.withheld_keyword!r}",
        ))
    else:
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

#: Where the sealed pipeline's per-card winnability labels live. The
#: decodability battery skips when it is absent rather than failing: the labels
#: are an operator's training data, not a repository artifact.
DEFAULT_WIN_RATES = Path("output/sealed/cards-win-rates.txt")

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
    from effects.infrastructure.record_io import read_records

    games = frozenset(provenance.game_disjoint_games)
    if not games:
        return iter(())
    return (
        record for record in read_records(Path(config.records_dir))
        if record.kind is RecordKind.COMBAT and record.game_id in games
    )


def _gate_two_runnable(config, checkpoint, *, vocab_path, keyword_path, records):
    """The checkpoint, ready to run over ``records``. Loaded exactly once."""
    from effects.infrastructure.model_runner import load_runnable

    return load_runnable(
        config, checkpoint,
        vocab_path=vocab_path, keyword_path=keyword_path, records=records,
    )


def run_gate_two(
    config, main, *, vocab_path: Path, keyword_path: Path,
) -> list[KeywordVerdict]:
    """Score the eight keywords and route each one (FR-119 – FR-122).

    Nothing is loaded when the split holds no combat record to score: every
    keyword is then under-sampled at n=0, which is the same verdict a corpus
    with forty records gets and for the same reason.
    """
    from effects.application.gate_two import score_keywords

    rows = keyword_rows()
    records = iter(_gate_two_records(config, main.provenance))
    # A handful of records up front, because the loader measures the slot
    # widths from real ones; the rest stays a stream the scorer reads in chunks.
    head = list(islice(records, SLOT_WIDTH_SAMPLE))
    if not head:
        scores = {row.keyword: KeywordScore() for row in rows}
    else:
        encoder, model, batcher, fields = _gate_two_runnable(
            config, main,
            vocab_path=vocab_path, keyword_path=keyword_path, records=head,
        )
        # The batcher's own cache, not a second one built from the config: the
        # resolver decides which entity carries a keyword and the surface
        # builder decides which slot to drop for it, and the two reading
        # different joins is exactly how a perturbation strips nothing.
        scores = score_keywords(
            chain(head, records), rows, encoder, model, batcher,
            batcher.sidecars, fields=fields,
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

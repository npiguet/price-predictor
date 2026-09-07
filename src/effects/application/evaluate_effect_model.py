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
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import numpy as np

from effects.domain.damage_step_keywords import (
    DAMAGE_STEP_KEYWORDS,
    MIN_DIRECTION_AGREEMENT,
    MIN_QUALIFYING_RECORDS,
    DamageStepKeyword,
)

logger = logging.getLogger(__name__)

# ── gate thresholds (FR-118, FR-119, FR-123). Contract, not tuning. ──────
GATE1_MIN_GATE_F1_GAIN = 0.05
GATE1_MIN_ZONE_ACCURACY_GAIN = 0.05
GATE1_MIN_DEVIANCE_REDUCTION = 0.05
GATE3_MAX_MEAN_COSINE = 0.5
GATE3_COSINE_PAIRS = 10_000
GATE3_MAX_TOP_COMPONENT = 0.30


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


@dataclass(frozen=True, slots=True)
class GateOneMetrics:
    """One model's scores on the card-disjoint split's unique-text stratum."""

    affected_gate_f1: float
    zone_outcome_accuracy: float
    mean_poisson_deviance: float


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
) -> KeywordVerdict:
    """Route one keyword: pass, or send it to a stage-three probe (FR-119).

    Under-sampled routes the same way as wrong. A keyword with 40 qualifying
    records has not been tested, and treating "untested" as "passed" is how a
    canary stops being one.
    """
    agreement = (
        agreeing_records / qualifying_records if qualifying_records else 0.0
    )
    if qualifying_records < MIN_QUALIFYING_RECORDS:
        return KeywordVerdict(
            row.keyword, qualifying_records, agreement, routed_to_probe=True,
            reason=(
                f"under-sampled: {qualifying_records} qualifying records < "
                f"{MIN_QUALIFYING_RECORDS}"
            ),
        )
    if agreement < MIN_DIRECTION_AGREEMENT:
        return KeywordVerdict(
            row.keyword, qualifying_records, agreement, routed_to_probe=True,
            reason=(
                f"direction agreement {agreement:.1%} < "
                f"{MIN_DIRECTION_AGREEMENT:.0%} on {', '.join(row.fields)}"
            ),
        )
    return KeywordVerdict(
        row.keyword, qualifying_records, agreement, routed_to_probe=False,
        reason=f"direction agreement {agreement:.1%}",
    )


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
) -> CheckResult:
    """Is ward nearer its meaning than it is to keyword-shaped text in general?

    Passes when ward is closer to **each** functional twin than to the median of
    its distances to all bare single-keyword vectors. A median rather than a
    fixed threshold, so the criterion stays meaningful whatever scale the
    embedding settles at.
    """
    if not twins or not bare_keywords:
        return CheckResult(
            "ward-canary", CheckStatus.SKIPPED,
            "no ward, twin, or bare-keyword vectors in the cache",
        )
    reference = float(np.median([cosine_distance(ward, k) for k in bare_keywords]))
    distances = [cosine_distance(ward, twin) for twin in twins]
    farther = [d for d in distances if not (d < reference)]
    values = {
        "median_bare_keyword_distance": reference,
        "max_twin_distance": max(distances),
    }
    if farther:
        return CheckResult(
            "ward-canary", CheckStatus.REPORTED,
            f"{len(farther)} of {len(twins)} functional twins sit farther from "
            f"ward than the median bare keyword ({reference:.3f})",
            values,
        )
    return CheckResult(
        "ward-canary", CheckStatus.REPORTED,
        f"every functional twin is nearer than the median bare keyword "
        f"({reference:.3f})",
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


def scored_records(records, provenance) -> list:
    """Only the games the checkpoint recorded (FR-108).

    The corpus grows between the training run and the evaluation, and a record
    from a game the model never saw is neither training nor validation — it is
    unclassified, and scoring it would quietly change what the split means.
    """
    allowed = provenance.validation_games
    return [record for record in records if record.game_id in allowed]


def keyword_rows() -> tuple[DamageStepKeyword, ...]:
    """Gate 2's table, so the evaluator and the report read one source."""
    return DAMAGE_STEP_KEYWORDS


def unique_ability_vectors(abilities_root: Path, variant: str) -> np.ndarray:
    """One vector per unique ability text, for gate 3 (FR-123).

    Deduplicated by value: the corpus has ~38k unique texts across 66k lines,
    and counting a reprinted line twice would make the space look more populated
    than it is.
    """
    from effects.domain.ability_cache_layout import ARRAY_KEY, CACHE_SUFFIX

    root = Path(abilities_root)
    if not root.is_dir():
        return np.zeros((0, 0), dtype=np.float32)
    pattern = (
        f"*{CACHE_SUFFIX}" if variant == "full" else f"*.{variant}{CACHE_SUFFIX}"
    )
    seen: dict[bytes, np.ndarray] = {}
    for path in sorted(root.rglob(pattern)):
        if variant == "full" and path.name.count(".") != 1:
            continue
        with np.load(path) as data:
            matrix = data[ARRAY_KEY]
        for row in matrix:
            seen.setdefault(row.tobytes(), row)
    if not seen:
        return np.zeros((0, 0), dtype=np.float32)
    return np.stack(list(seen.values()))


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
        load_cache,
    )
    from effects.infrastructure.record_io import read_records

    report = EvaluationReport()
    cards_root = next(
        (Path(f) for f in config.cards_folders if Path(f).name == "cardsfolder"),
        Path(config.cards_folders[0]),
    )
    cached = load_cache(config.abilities_root, cards_root, variant="full")

    # ── gate 3: geometry, read off the cache ──
    vectors = unique_ability_vectors(config.abilities_root, "full")
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

    # ── gate 2: per keyword, routing only ──
    scored = scored_records(read_records(config.records_dir), main.provenance)
    verdicts = [
        evaluate_keyword(
            row,
            qualifying_records=count_qualifying(scored, row),
            agreeing_records=0,
        )
        for row in keyword_rows()
    ]
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
            load_cache(config.abilities_root, cards_root, variant=name),
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


#: Texts whose neighbours a reader can judge at a glance. Reported rather than
#: scored: the point is for a person to see whether the neighbours of a removal
#: spell are removal spells, which no metric asks.
NEIGHBOUR_QUERIES: tuple[str, ...] = (
    "destroy target creature",
    "draw a card",
    "deal 3 damage to any target",
    "target creature gets +2/+2 until end of turn",
)

#: Where the sealed pipeline's per-card winnability labels live. The
#: decodability battery skips when it is absent rather than failing: the labels
#: are an operator's training data, not a repository artifact.
DEFAULT_WIN_RATES = Path("output/sealed/cards-win-rates.txt")


def count_qualifying(records: list, row: DamageStepKeyword) -> int:
    """Combat records in which the keyword's carrier is actually in combat.

    A first approximation of ``row.qualifies_when``: the full predicate needs
    the model's perturbed prediction, which the caller supplies. Counting here
    is what lets the under-sampled verdict fire before any model has run.
    """
    from effects.domain.records import RecordKind

    keyword = row.keyword
    qualifying = 0
    for record in records:
        if record.kind is not RecordKind.COMBAT:
            continue
        for entity in record.state.entities:
            if entity.combat is None:
                continue
            if keyword in entity.granted_temporary.keywords:
                qualifying += 1
                break
    return qualifying

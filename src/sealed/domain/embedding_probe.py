"""embedding_probe: linear-probe validation of card embedding quality."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sealed.domain.mana_scorer import (
    compute_mana_value,
    count_actual_sources,
    count_pips,
)

# ─── Value objects ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CardData:
    """A single card's embedding and text, loaded from disk."""
    name: str
    embedding: np.ndarray
    text: str


@dataclass(frozen=True)
class ProbeSpec:
    """Specification for a single linear probe."""
    feature_name: str
    probe_type: str   # "classification", "regression", or "ordinal"
    threshold: float
    extract_labels: Callable[[list[CardData]], np.ndarray]
    round_to: float | None = None  # regression only: snap predictions to this step size
    class_values: list[float] | None = None  # ordinal only: sorted class boundary values


@dataclass(frozen=True)
class ProbeResult:
    """Result of running a single probe."""
    feature_name: str
    score: float
    threshold: float
    passed: bool
    n_samples: int
    rounded_score: float | None = None  # R² after rounding predictions to nearest valid value
    bucket_stats: list[tuple[float, int, float]] | None = None  # (value, count, exact_match)


@dataclass(frozen=True)
class ValidationResult:
    """Aggregate result of all probes."""
    probe_results: list[ProbeResult]
    n_cards: int
    n_lands: int
    all_passed: bool


# ─── Ground truth extraction ──────────────────────────────────────────────────

_LAND_LINE_RE = re.compile(r"^type", re.IGNORECASE)


def _is_land_text(text: str) -> bool:
    for line in text.splitlines():
        if _LAND_LINE_RE.match(line.strip()) and "land" in line.lower():
            return True
    return False


def extract_is_land(cards: list[CardData]) -> np.ndarray:
    """Binary label: 1 if card is a land, 0 otherwise."""
    return np.array([1.0 if _is_land_text(c.text) else 0.0 for c in cards])


_DEVOID_RE = re.compile(r"^static:\s*devoid\s*$", re.IGNORECASE | re.MULTILINE)


def _has_devoid(text: str) -> bool:
    """Return True if card text contains a 'static: devoid' line."""
    return bool(_DEVOID_RE.search(text))


def extract_card_color(cards: list[CardData], color: str) -> np.ndarray:
    """Binary label: 1 if card has the given color, 0 otherwise.

    Corrected definition (FR-003):
    - W/U/B/R/G = 1 if pip count > 0 AND card is not devoid
    - C = 1 if card is colorless: all W/U/B/R/G labels are 0
      (i.e., no colored pips, or devoid, or no mana cost)
    """
    labels = []
    for c in cards:
        pips = count_pips([c.text])
        devoid = _has_devoid(c.text)
        if color == "C":
            # Colorless = no colored pips or devoid
            is_colorless = all(
                pips.counts.get(col, 0.0) <= 0.0 or devoid
                for col in ("W", "U", "B", "R", "G")
            )
            labels.append(1.0 if is_colorless else 0.0)
        else:
            pip_val = pips.counts.get(color, 0.0)
            labels.append(0.0 if devoid or pip_val <= 0.0 else 1.0)
    return np.array(labels)


def extract_pip_counts(cards: list[CardData], color: str) -> np.ndarray:
    """Regression label: number of pips of the given color (fractional for hybrids)."""
    labels = []
    for c in cards:
        pips = count_pips([c.text])
        labels.append(pips.counts.get(color, 0.0))
    return np.array(labels)


def extract_mana_value(cards: list[CardData]) -> np.ndarray:
    """Regression label: total mana value from mana cost line."""
    labels = []
    for c in cards:
        mv = 0.0
        for line in c.text.splitlines():
            stripped = line.strip()
            if stripped.lower().startswith("mana cost:"):
                cost_part = stripped[len("mana cost:"):].strip()
                mv = compute_mana_value(cost_part)
                break
        labels.append(mv)
    return np.array(labels)


def extract_mana_produced(cards: list[CardData], color: str) -> np.ndarray:
    """Binary label: 1 if card produces mana of the given color, 0 otherwise.

    Works for all card types — lands, artifacts (Sol Ring), creatures (Llanowar
    Elves). Any card without a {T}: add ability gets 0.
    """
    labels = []
    for c in cards:
        sources = count_actual_sources([c.text])
        labels.append(1.0 if sources.sources.get(color, 0.0) > 0 else 0.0)
    return np.array(labels)


# ─── Auxiliary label computation ─────────────────────────────────────────────

_COLORS = ("W", "U", "B", "R", "G", "C")


def compute_aux_labels(card_text: str) -> np.ndarray:
    """Return the 20-element auxiliary label vector for a single card text.

    Label order (data-model.md):
      0:     is_land
      1–6:   card_color W/U/B/R/G/C (corrected: devoid → colorless)
      7–12:  pip_count W/U/B/R/G/C
      13:    mana_value
      14–19: mana_produced W/U/B/R/G/C
    """
    labels = np.zeros(20, dtype=np.float32)

    # 0: is_land
    labels[0] = 1.0 if _is_land_text(card_text) else 0.0

    # 1–6: card_color (corrected definition: devoid → colorless)
    pips = count_pips([card_text])
    devoid = _has_devoid(card_text)
    for i, color in enumerate(_COLORS):
        if color == "C":
            is_colorless = all(
                pips.counts.get(col, 0.0) <= 0.0 or devoid
                for col in ("W", "U", "B", "R", "G")
            )
            labels[1 + i] = 1.0 if is_colorless else 0.0
        else:
            pip_val = pips.counts.get(color, 0.0)
            labels[1 + i] = 0.0 if devoid or pip_val <= 0.0 else 1.0

    # 7–12: pip_count (raw pip counts, not affected by devoid)
    for i, color in enumerate(_COLORS):
        labels[7 + i] = float(pips.counts.get(color, 0.0))

    # 13: mana_value
    mv = 0.0
    for line in card_text.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("mana cost:"):
            cost_part = stripped[len("mana cost:"):].strip()
            mv = compute_mana_value(cost_part)
            break
    labels[13] = float(mv)

    # 14–19: mana_produced
    sources = count_actual_sources([card_text])
    for i, color in enumerate(_COLORS):
        labels[14 + i] = 1.0 if sources.sources.get(color, 0.0) > 0 else 0.0

    return labels


# ─── Probe runner ─────────────────────────────────────────────────────────────


def build_default_probes(
    threshold_accuracy: float = 0.95,
    threshold_r2: float = 0.85,
) -> list[ProbeSpec]:
    """Return the 21 standard embedding validation probes.

    Thresholds per FR-007:
      - is-land: max(threshold_accuracy, 0.99)
      - mana value: max(threshold_r2, 0.90)
      - all other classification probes: threshold_accuracy
      - all other regression probes: threshold_r2
    """
    probes: list[ProbeSpec] = []

    # Is land (1 probe)
    probes.append(ProbeSpec(
        feature_name="Is land",
        probe_type="classification",
        threshold=max(threshold_accuracy, 0.99),
        extract_labels=extract_is_land,
    ))

    # Card color — one per color (6 probes)
    for color in _COLORS:
        probes.append(ProbeSpec(
            feature_name=f"Card color ({color})",
            probe_type="classification",
            threshold=threshold_accuracy,
            extract_labels=lambda cards, c=color: extract_card_color(cards, c),
        ))

    # Pip counts — one per color (6 ordinal probes)
    _wubrg_classes = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0]
    _c_classes = [0.0, 1.0, 2.0, 2.5, 3.0]
    for color in _COLORS:
        probes.append(ProbeSpec(
            feature_name=f"Pip counts ({color})",
            probe_type="ordinal",
            threshold=threshold_accuracy,
            extract_labels=lambda cards, c=color: extract_pip_counts(cards, c),
            class_values=_c_classes if color == "C" else _wubrg_classes,
        ))

    # Mana value (1 ordinal probe)
    probes.append(ProbeSpec(
        feature_name="Mana value",
        probe_type="ordinal",
        threshold=threshold_accuracy,
        extract_labels=extract_mana_value,
        class_values=[float(v) for v in range(17)],
    ))

    # Mana produced — one per color (6 probes)
    for color in _COLORS:
        probes.append(ProbeSpec(
            feature_name=f"Mana produced ({color})",
            probe_type="classification",
            threshold=threshold_accuracy,
            extract_labels=lambda cards, c=color: extract_mana_produced(cards, c),
        ))

    return probes


def run_probes(
    cards: list[CardData],
    probes: list[ProbeSpec],
    on_result: Callable[[ProbeResult], None] | None = None,
    embed_dim: int | None = None,
) -> list[ProbeResult]:
    """Run each probe with 5-fold cross-validation and return results.

    If *on_result* is provided it is called immediately after each probe
    finishes, allowing the caller to display progress in real time.

    If *embed_dim* is provided, only the first *embed_dim* dimensions of each
    embedding are used. This allows probing only the transformer-learned portion
    when embeddings include appended explicit features.
    """
    embeddings = np.stack([c.embedding for c in cards])
    if embed_dim is not None:
        embeddings = embeddings[:, :embed_dim]

    results: list[ProbeResult] = []
    for spec in probes:
        labels = spec.extract_labels(cards)

        rounded_score: float | None = None
        bucket_stats: list[tuple[float, int, float]] | None = None
        if spec.probe_type == "classification":
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=1000, solver="lbfgs"),
            )
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            fold_scores = cross_val_score(
                model, embeddings, labels, cv=cv, scoring="accuracy", error_score=0.0
            )
            preds = cross_val_predict(model, embeddings, labels, cv=cv)
            rounded_score = float(np.mean(preds == labels))
        elif spec.probe_type == "ordinal":
            # Multinomial logistic regression on ordinal class indices — matches
            # the K-logit linear training head, so training and probing use the
            # same model class (K linear functions of the embedding).
            class_arr = np.array(spec.class_values)
            class_indices = np.array([
                int(np.argmin(np.abs(class_arr - v))) for v in labels
            ])
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=5000, solver="lbfgs"),
            )
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            fold_scores = cross_val_score(
                model, embeddings, class_indices, cv=cv, scoring="accuracy",
                error_score=0.0,
            )
            preds = cross_val_predict(model, embeddings, class_indices, cv=cv)
            rounded_score = float(np.mean(preds == class_indices))
            bucket_stats = []
            for k, val in enumerate(spec.class_values):
                mask = class_indices == k
                count = int(np.sum(mask))
                exact = float(np.mean(preds[mask] == k)) if count > 0 else 0.0
                bucket_stats.append((val, count, exact))
        else:  # regression
            model = LinearRegression()
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            fold_scores = cross_val_score(
                model, embeddings, labels, cv=cv, scoring="r2", error_score=0.0
            )
            if spec.round_to is not None:
                preds = cross_val_predict(model, embeddings, labels, cv=cv)
                pred_steps = np.round(preds / spec.round_to)
                true_steps = np.round(labels / spec.round_to)
                rounded_score = float(np.mean(pred_steps == true_steps))
                bucket_stats = []
                for step in sorted(np.unique(true_steps)):
                    mask = true_steps == step
                    count = int(np.sum(mask))
                    exact = float(np.mean(pred_steps[mask] == step))
                    bucket_stats.append((float(step * spec.round_to), count, exact))

        mean_score = float(np.mean(fold_scores))
        # For regression probes with rounding, also accept if exact match meets threshold.
        # R² is undefined (≈0) for near-constant distributions, so exact match is the
        # reliable signal in those cases (e.g., pip_count_C where ~99% of cards have 0).
        if rounded_score is not None:
            passed = mean_score >= spec.threshold or rounded_score >= spec.threshold
        else:
            passed = mean_score >= spec.threshold
        result = ProbeResult(
            feature_name=spec.feature_name,
            score=mean_score,
            threshold=spec.threshold,
            passed=passed,
            n_samples=len(cards),
            rounded_score=rounded_score,
            bucket_stats=bucket_stats,
        )
        results.append(result)
        if on_result is not None:
            on_result(result)

    return results

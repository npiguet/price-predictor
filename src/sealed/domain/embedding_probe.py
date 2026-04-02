"""embedding_probe: linear-probe validation of card embedding quality."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

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
    probe_type: str   # "classification" or "regression"
    threshold: float
    extract_labels: Callable[[list[CardData]], np.ndarray]


@dataclass(frozen=True)
class ProbeResult:
    """Result of running a single probe."""
    feature_name: str
    score: float
    threshold: float
    passed: bool
    n_samples: int


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


def extract_card_color(cards: list[CardData], color: str) -> np.ndarray:
    """Binary label: 1 if card has ≥1 pip of the given color, 0 otherwise."""
    labels = []
    for c in cards:
        pips = count_pips([c.text])
        labels.append(1.0 if pips.counts.get(color, 0.0) > 0 else 0.0)
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


# ─── Probe runner ─────────────────────────────────────────────────────────────

_COLORS = ("W", "U", "B", "R", "G", "C")


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

    # Pip counts — one per color (6 probes)
    for color in _COLORS:
        probes.append(ProbeSpec(
            feature_name=f"Pip counts ({color})",
            probe_type="regression",
            threshold=threshold_r2,
            extract_labels=lambda cards, c=color: extract_pip_counts(cards, c),
        ))

    # Mana value (1 probe)
    probes.append(ProbeSpec(
        feature_name="Mana value",
        probe_type="regression",
        threshold=max(threshold_r2, 0.90),
        extract_labels=extract_mana_value,
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


def run_probes(cards: list[CardData], probes: list[ProbeSpec]) -> list[ProbeResult]:
    """Run each probe with 5-fold cross-validation and return results."""
    embeddings = np.stack([c.embedding for c in cards])

    results: list[ProbeResult] = []
    for spec in probes:
        labels = spec.extract_labels(cards)

        if spec.probe_type == "classification":
            model = LogisticRegression(max_iter=1000, solver="lbfgs")
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            fold_scores = cross_val_score(
                model, embeddings, labels, cv=cv, scoring="accuracy", error_score=0.0
            )
        else:
            model = LinearRegression()
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            fold_scores = cross_val_score(
                model, embeddings, labels, cv=cv, scoring="r2", error_score=0.0
            )

        mean_score = float(np.mean(fold_scores))
        results.append(ProbeResult(
            feature_name=spec.feature_name,
            score=mean_score,
            threshold=spec.threshold,
            passed=mean_score >= spec.threshold,
            n_samples=len(cards),
        ))

    return results

"""The checks that read the ability cache rather than the model.

They ask about the geometry of ``e`` rather than a prediction, which catches
what the loss cannot: a collapsed space passes every average-case metric.
Only gate 3 is a verdict; the rest are reported.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from effects.domain.ability_cache_layout import pooled_card_vector
from effects.domain.baselines import Baseline
from effects.domain.evaluation_report import (
    Bound,
    CheckName,
    CheckResult,
    CheckStatus,
    Margin,
    gate_verdict,
)
from effects.domain.gate_metrics import (
    GATE3_MAX_MEAN_COSINE,
    GATE3_MAX_TOP_COMPONENT,
    cosine_distance,
    mean_pairwise_cosine,
    top_component_share,
)
from effects.domain.line_query import LineQuery, normalize_prose
from effects.domain.provenance import SourceTree

if TYPE_CHECKING:
    from effects.infrastructure.ability_cache_store import SidecarIndex

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Resolution:
    """Where a :class:`LineQuery` landed: a cache key, or why it did not."""

    key: str | None
    problem: str = ""


@dataclass(frozen=True, slots=True)
class CacheGeometry:
    """The two collapse measures of one set of vectors (FR-123)."""

    mean_pairwise_cosine: float
    top_component_share: float

    @classmethod
    def of(cls, vectors: np.ndarray, *, seed: int = 42) -> CacheGeometry:
        return cls(
            mean_pairwise_cosine(vectors, seed=seed),
            top_component_share(vectors),
        )


@dataclass
class CachedVectors:
    """One variant's cache, loaded once and indexed by ability text.

    By the text on the checkpoint's encoding surface, so a reprint is one
    point (not one per card or per float-noisy batch) and two scripts sharing
    prose are two. ``by_prose`` maps each normalized prose line to the keys it
    encodes to, with the cards printing each, for queries a person reads.
    The cached properties are kept, so nothing may add to ``by_text`` once
    any has been read.
    """

    by_text: dict[str, np.ndarray] = field(default_factory=dict)
    by_card: dict[str, np.ndarray] = field(default_factory=dict)
    display: dict[str, str] = field(default_factory=dict)
    by_prose: dict[str, dict[str, set[str]]] = field(default_factory=dict)

    @cached_property
    def matrix(self) -> np.ndarray:
        """One row per text, in ``texts`` order."""
        if not self.by_text:
            return np.zeros((0, 0), dtype=np.float32)
        return np.stack(list(self.by_text.values()))

    @cached_property
    def unit(self) -> np.ndarray:
        norms = np.linalg.norm(self.matrix, axis=1, keepdims=True)
        return self.matrix / np.clip(norms, 1e-12, None)

    @cached_property
    def texts(self) -> list[str]:
        return list(self.by_text)

    @cached_property
    def geometry(self) -> CacheGeometry:
        return CacheGeometry.of(self.matrix)

    def label(self, key: str) -> str:
        """``key`` as a person reads it: its prose, where it has any."""
        return self.display.get(key, key)

    def resolve(self, query: LineQuery) -> Resolution:
        """The cache key ``query`` names, or the reason it names none. Several
        keys under one prose is ambiguity: the query has to pin its card."""
        keys = self.by_prose.get(normalize_prose(query.prose), {})
        if query.card is not None:
            card = normalize_prose(query.card)
            keys = {key: cards for key, cards in keys.items() if card in cards}
            if not keys:
                return Resolution(
                    None, f"no line {query.prose!r} on {query.card!r}",
                )
        if not keys:
            return Resolution(None, f"no line {query.prose!r} in the cache")
        if len(keys) > 1:
            return Resolution(
                None,
                f"{query.prose!r} is printed with {len(keys)} scripts; "
                "pin its card",
            )
        return Resolution(next(iter(keys)))

    def __len__(self) -> int:
        return len(self.by_text)


def cached_vectors(
    index: SidecarIndex, abilities_root: Path, *, variant: str = Baseline.FULL,
) -> CachedVectors:
    """One variant's cache over sidecars already read. Only a ``cardsfolder``
    source contributes a pooled per-card vector."""
    cached = CachedVectors()
    for source, matrix in index.matrices(abilities_root, variant=variant):
        card = normalize_prose(source.card)
        for row, line in zip(matrix, source.lines):
            if line is None:
                continue
            cached.by_text.setdefault(line.text, row)
            cached.display.setdefault(line.text, line.prose or line.text)
            if line.prose:
                cached.by_prose.setdefault(
                    normalize_prose(line.prose), {},
                ).setdefault(line.text, set()).add(card)
        if matrix.size and Path(source.script_file).parts[:1] == (SourceTree.CARDSFOLDER,):
            cached.by_card[source.card] = pooled_card_vector(matrix)
    return cached


# ── gate 3: collapse canaries ───────────────────────────────────────────


def gate_three_verdict(geometry: CacheGeometry) -> CheckResult:
    """Gate 3 on measured geometry, one vector per unique ability text."""
    return gate_verdict(
        CheckName.GATE_3,
        (
            Margin(
                "mean_pairwise_cosine", "mean pairwise cosine",
                geometry.mean_pairwise_cosine, GATE3_MAX_MEAN_COSINE,
                Bound.CEILING, value_format=".3f",
            ),
            Margin(
                "top_component_share", "top principal component",
                geometry.top_component_share, GATE3_MAX_TOP_COMPONENT,
                Bound.CEILING, value_format=".1%", limit_format=".0%",
                verb="explains",
            ),
        ),
        fail_prefix="the embedding space has collapsed",
        pass_detail="no collapse",
    )


def run_gate_three(cached: CachedVectors, abilities_root: Path) -> CheckResult:
    """Gate 3, read off the cache, or a skip where there is none to read."""
    logger.info("gate 3: geometry over %d unique cached texts", len(cached.by_text))
    if len(cached) < 2:
        return CheckResult(
            CheckName.GATE_3, CheckStatus.SKIPPED,
            f"no ability cache under {abilities_root}; run "
            "encode-abilities first",
        )
    return gate_three_verdict(cached.geometry)


# ── nearest neighbours ──────────────────────────────────────────────────

#: Lines whose neighbours a reader can judge at a glance, each pinned to a card
#: because its prose compiles to several scripts.
NEIGHBOUR_QUERIES: tuple[LineQuery, ...] = (
    LineQuery("destroy target creature.", card="murder"),
    LineQuery("draw a card.", card="think twice"),
    LineQuery("CARDNAME deals 3 damage to any target.", card="lightning bolt"),
    LineQuery(
        "target creature gets +2/+2 until end of turn.", card="artful maneuver",
    ),
)


def nearest_neighbours(
    cached: CachedVectors, query: str, *, k: int = 5,
) -> list[tuple[str, float]]:
    """The ``k`` texts nearest ``query`` by cosine similarity, for a person to
    judge, since no metric asks whether a removal spell's neighbours are."""
    target = cached.by_text.get(query)
    if target is None or len(cached) < 2:
        return []
    texts = cached.texts
    target_unit = target / max(float(np.linalg.norm(target)), 1e-12)
    scores = cached.unit @ target_unit
    order = np.argsort(-scores)
    out: list[tuple[str, float]] = []
    for index in order:
        if texts[index] == query:
            continue
        out.append((texts[index], float(scores[index])))
        if len(out) >= k:
            break
    return out


def umap_projection(matrix: np.ndarray, *, seed: int = 42) -> np.ndarray | None:
    """A 2-D UMAP projection, or None when the library or the data is absent:
    a run that cannot draw the picture has not failed."""
    if matrix.shape[0] < 10:
        return None
    try:
        import umap
    except ImportError:
        logger.warning("umap-learn is not installed; skipping the projection")
        return None
    # A seed forces UMAP single-threaded; saying so keeps it from warning.
    reducer = umap.UMAP(n_components=2, random_state=seed, n_jobs=1)
    return reducer.fit_transform(matrix)


def check_nearest_neighbours(
    cached: CachedVectors, queries: tuple[LineQuery, ...],
) -> CheckResult:
    """Each query's three nearest lines, in prose; an unresolved query is
    listed as such rather than left out."""
    if len(cached) < 2:
        return CheckResult(
            CheckName.NEAREST_NEIGHBOUR, CheckStatus.SKIPPED,
            "the ability cache is empty; run encode-abilities first",
        )
    lines = []
    unresolved = 0
    for query in queries:
        resolution = cached.resolve(query)
        if resolution.key is None:
            unresolved += 1
            lines.append(f"{query.label()!r} → unresolved: {resolution.problem}")
            continue
        neighbours = nearest_neighbours(cached, resolution.key, k=3)
        rendered = ", ".join(
            f"{cached.label(text)[:70]!r} ({score:.2f})"
            for text, score in neighbours
        )
        lines.append(f"{query.label()!r} → {rendered}")
    return CheckResult(
        CheckName.NEAREST_NEIGHBOUR, CheckStatus.REPORTED, "; ".join(lines),
        {"unresolved_queries": float(unresolved)},
    )


def check_umap(cached: CachedVectors) -> CheckResult:
    projection = umap_projection(cached.matrix)
    if projection is None:
        return CheckResult(
            CheckName.UMAP, CheckStatus.SKIPPED,
            "too few vectors, or umap-learn is not installed",
        )
    return CheckResult(
        CheckName.UMAP, CheckStatus.REPORTED,
        f"projected {projection.shape[0]} unique ability texts to 2-D",
    )


# ── the ward canary (FR-112) ────────────────────────────────────────────


def evaluate_ward_canary(
    ward: np.ndarray,
    twins: list[np.ndarray],
    bare_keywords: list[np.ndarray],
    *,
    unresolved: Sequence[str] = (),
) -> CheckResult:
    """Is ward nearer its meaning than it is to keyword-shaped text in general?

    Holds when ward is closer to each functional twin than the median of its
    distances to the bare keywords, a median so the criterion survives any
    embedding scale. ``unresolved`` lines are named in the detail, so a
    verdict over fewer twins than the table lists says so.
    """
    tail = f"; unresolved: {'; '.join(unresolved)}" if unresolved else ""
    if not twins or not bare_keywords:
        return CheckResult(
            CheckName.WARD_CANARY, CheckStatus.SKIPPED,
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
            CheckName.WARD_CANARY, CheckStatus.REPORTED,
            f"{len(farther)} of {len(twins)} functional twins sit farther from "
            f"ward than the median bare keyword ({reference:.3f})" + tail,
            values,
        )
    return CheckResult(
        CheckName.WARD_CANARY, CheckStatus.REPORTED,
        f"every functional twin is nearer than the median bare keyword "
        f"({reference:.3f})" + tail,
        values,
    )



def check_ward(cached: CachedVectors) -> CheckResult:
    """:func:`evaluate_ward_canary` over the cache, each line resolved through
    its prose to the key the checkpoint encoded."""
    from effects.domain.ward_twins import BARE_KEYWORDS, WARD, WARD_TWINS

    ward = cached.resolve(WARD)
    if ward.key is None:
        return CheckResult(
            CheckName.WARD_CANARY, CheckStatus.SKIPPED,
            f"ward is not in the cache: {ward.problem}",
        )
    unresolved: list[str] = []

    def vectors(queries: tuple[LineQuery, ...]) -> list[np.ndarray]:
        found = []
        for query in queries:
            resolution = cached.resolve(query)
            if resolution.key is None:
                unresolved.append(resolution.problem)
            else:
                found.append(cached.by_text[resolution.key])
        return found

    return evaluate_ward_canary(
        cached.by_text[ward.key], vectors(WARD_TWINS), vectors(BARE_KEYWORDS),
        unresolved=unresolved,
    )


# ── the decodability battery (FR-114) ───────────────────────────────────


def check_decodability(
    cached: CachedVectors,
    win_rates_path: Path,
    *,
    held_out_cards: tuple[str, ...] = (),
) -> CheckResult:
    """How much of each winnability label is linearly readable from pooled
    ``e``, validated on the checkpoint's held-out cards."""
    if not cached.by_card:
        return CheckResult(
            CheckName.DECODABILITY, CheckStatus.SKIPPED,
            "the ability cache is empty; run encode-abilities first",
        )
    if not Path(win_rates_path).exists():
        return CheckResult(
            CheckName.DECODABILITY, CheckStatus.SKIPPED,
            f"no win-rate table at {win_rates_path}",
        )
    from price_predictor.application.ridge_probes import (
        build_label_table,
        fit_probes,
        load_labels,
    )

    shared = sorted(cached.by_card)
    # Sidecars name cards in lowercase, the win-rate table in Forge's case:
    # joined case-insensitively, in the cache's order.
    canonical = {name.casefold(): name for name in load_labels(Path(win_rates_path))}
    labelled = [canonical.get(name.casefold(), name) for name in shared]
    with_rate = sum(name.casefold() in canonical for name in shared)
    if with_rate < 50:
        return CheckResult(
            CheckName.DECODABILITY, CheckStatus.SKIPPED,
            f"only {with_rate} of {len(shared)} cached cards have a win rate, "
            "too few to fit probes",
        )
    table = build_label_table(
        labelled,
        win_rates_path=Path(win_rates_path),
        val_names={canonical.get(name.casefold(), name) for name in held_out_cards},
    )
    effects_matrix = np.stack([cached.by_card[name] for name in shared])
    effects_probes = fit_probes(table, effects_matrix, mode="honest")
    rendered = ", ".join(
        f"{name} {probe.metrics.get('val_r2', float('nan')):.3f}"
        for name, probe in effects_probes.probes.items()
    )
    return CheckResult(
        CheckName.DECODABILITY, CheckStatus.REPORTED, f"effects: {rendered}",
    )


# ── the average-effect control and the taxonomy comparison ──────────────


def check_baseline_geometry(
    full: CachedVectors, baseline_vectors: CachedVectors, name: str,
) -> CheckResult:
    """Compare a baseline's cache geometry against the shipping one."""
    if len(baseline_vectors) < 2 or len(full) < 2:
        return CheckResult(
            CheckName.baseline_comparison(name), CheckStatus.SKIPPED,
            f"no {name} cache; run encode-abilities --variant {name}",
        )
    shipping, baseline = full.geometry, baseline_vectors.geometry
    return CheckResult(
        CheckName.baseline_comparison(name), CheckStatus.REPORTED,
        f"mean pairwise cosine {shipping.mean_pairwise_cosine:.3f} vs "
        f"{baseline.mean_pairwise_cosine:.3f}; top component "
        f"{shipping.top_component_share:.1%} vs "
        f"{baseline.top_component_share:.1%}",
        {
            "full_cosine": shipping.mean_pairwise_cosine,
            f"{name}_cosine": baseline.mean_pairwise_cosine,
        },
    )

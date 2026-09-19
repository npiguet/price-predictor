"""The checks that read the ability cache rather than the model.

Gate 3, the ward canary, nearest-neighbour inspection, UMAP, the decodability
battery and the scorer smoke test all ask about the *geometry* of ``e`` rather
than about a prediction. They are grouped here because they share one input —
the cache — and because a check on the geometry catches a failure the loss
cannot: a collapsed space, where every ability sits on top of every other,
passes every average-case metric and is useless.

None of them blocks except gate 3. The rest are reported, and they are reported
because a number that gates nothing still tells an operator which way to go
next.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from effects.application.evaluate_effect_model import CheckResult, CheckStatus
from effects.domain.ability_cache_layout import ARRAY_KEY, CACHE_SUFFIX

logger = logging.getLogger(__name__)


@dataclass
class CachedVectors:
    """One variant's cache, loaded once and indexed by ability text.

    Indexed by *text* rather than by provenance key because every geometry
    check asks about unique texts: a line reprinted on forty cards is one point
    in the space, and counting it forty times would make the space look more
    populated than it is.
    """

    by_text: dict[str, np.ndarray] = field(default_factory=dict)
    by_card: dict[str, np.ndarray] = field(default_factory=dict)

    def matrix(self) -> np.ndarray:
        if not self.by_text:
            return np.zeros((0, 0), dtype=np.float32)
        return np.stack(list(self.by_text.values()))

    def texts(self) -> list[str]:
        return list(self.by_text)

    def __len__(self) -> int:
        return len(self.by_text)


def load_cache(
    abilities_root: Path,
    sidecars_root: Path,
    *,
    variant: str = "full",
    surface: str = "prose",
) -> CachedVectors:
    """Load a variant's cache, keyed by the ability text each row encodes.

    ``sidecars_root`` supplies the texts: the cache is row-aligned with each
    source's sidecar, so the pairing is positional and needs no index. The key
    is the text on the surface the checkpoint encoded from — prose through
    stage three, script from stage four — because a check that keyed on the
    other surface would be comparing the vectors against texts they do not
    encode.
    """
    from effects.domain.ability_encoder import encoding_text
    from effects.infrastructure.sidecar_io import (
        SIDECAR_SUFFIX,
        converted_text_path,
        prose_for,
        prose_lines,
        read_sidecar,
    )

    suffix = CACHE_SUFFIX if variant == "full" else f".{variant}{CACHE_SUFFIX}"
    cached = CachedVectors()
    root = Path(abilities_root)
    if not root.is_dir():
        return cached

    for sidecar_path in sorted(Path(sidecars_root).rglob(f"*{SIDECAR_SUFFIX}")):
        sidecar = read_sidecar(sidecar_path)
        relative = Path(sidecar.script_file)
        cache_path = root / relative.parent / f"{relative.stem}{suffix}"
        if not cache_path.exists():
            continue
        with np.load(cache_path) as data:
            matrix = data[ARRAY_KEY]
        if matrix.shape[0] != len(sidecar.lines):
            logger.warning(
                "skipping %s: %d cache rows for %d sidecar lines",
                cache_path, matrix.shape[0], len(sidecar.lines),
            )
            continue
        rendered = prose_lines(converted_text_path(sidecar_path))
        for row, line in zip(matrix, sidecar.lines):
            text = encoding_text(line, prose_for(line, rendered), surface)
            if text:
                cached.by_text.setdefault(text, row)
        if matrix.size:
            cached.by_card[sidecar.card] = pooled(matrix)
    return cached


def pooled(matrix: np.ndarray) -> np.ndarray:
    """Mean and max over a card's ability rows, concatenated (FR-114)."""
    from effects.domain.ability_cache_layout import pooled_card_vector

    return pooled_card_vector(matrix)


# ── nearest neighbours ──────────────────────────────────────────────────


def nearest_neighbours(
    cached: CachedVectors, query: str, *, k: int = 5,
) -> list[tuple[str, float]]:
    """The ``k`` texts nearest ``query`` by cosine similarity.

    Reported rather than scored: the point is for a person to read the list and
    see whether the neighbours of "destroy target creature" are removal spells.
    A metric cannot ask that question.
    """
    target = cached.by_text.get(query)
    if target is None or len(cached) < 2:
        return []
    texts = cached.texts()
    matrix = cached.matrix()
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    unit = matrix / np.clip(norms, 1e-12, None)
    target_unit = target / max(float(np.linalg.norm(target)), 1e-12)
    scores = unit @ target_unit
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
    """A 2-D UMAP projection, or None when the library or the data is absent.

    Returns None rather than raising: the projection is a picture for a person
    to look at, and a run that cannot draw it has not failed.
    """
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
    cached: CachedVectors, queries: tuple[str, ...],
) -> CheckResult:
    if len(cached) < 2:
        return CheckResult(
            "nearest-neighbour", CheckStatus.SKIPPED,
            "the ability cache is empty; run encode-abilities first",
        )
    lines = []
    for query in queries:
        neighbours = nearest_neighbours(cached, query, k=3)
        if neighbours:
            rendered = ", ".join(f"{t[:40]!r} ({s:.2f})" for t, s in neighbours)
            lines.append(f"{query[:40]!r} → {rendered}")
    return CheckResult(
        "nearest-neighbour", CheckStatus.REPORTED,
        "; ".join(lines) if lines else "none of the query texts are in the cache",
    )


def check_umap(cached: CachedVectors) -> CheckResult:
    projection = umap_projection(cached.matrix())
    if projection is None:
        return CheckResult(
            "umap", CheckStatus.SKIPPED,
            "too few vectors, or umap-learn is not installed",
        )
    return CheckResult(
        "umap", CheckStatus.REPORTED,
        f"projected {projection.shape[0]} unique ability texts to 2-D",
    )


# ── the ward canary ─────────────────────────────────────────────────────


def check_ward(cached: CachedVectors) -> CheckResult:
    """Is ward's ``e`` nearer its longhand twins than a bare keyword generally?"""
    from effects.application.evaluate_effect_model import evaluate_ward_canary
    from effects.domain.ward_twins import BARE_KEYWORDS, WARD_TEXT, WARD_TWINS

    ward = _nearest_by_prefix(cached, WARD_TEXT)
    if ward is None:
        return CheckResult(
            "ward-canary", CheckStatus.SKIPPED,
            "no ward line in the cache",
        )
    twins = [v for v in (_nearest_by_prefix(cached, t) for t in WARD_TWINS) if v is not None]
    bare = [
        v for v in (_nearest_by_prefix(cached, k) for k in BARE_KEYWORDS)
        if v is not None
    ]
    return evaluate_ward_canary(ward, twins, bare)


def _nearest_by_prefix(cached: CachedVectors, text: str) -> np.ndarray | None:
    """The cached vector for ``text``, matched loosely.

    Converted text differs from the checked-in twin wording in whitespace and
    punctuation, so an exact lookup would find nothing on a real corpus. The
    match is on a normalized prefix, which is specific enough for these texts.
    """
    exact = cached.by_text.get(text)
    if exact is not None:
        return exact
    needle = _normalize(text)[:60]
    if not needle:
        return None
    for candidate, vector in cached.by_text.items():
        if _normalize(candidate).startswith(needle):
            return vector
    return None


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


# ── the decodability battery (FR-114) ───────────────────────────────────


def check_decodability(
    cached: CachedVectors,
    sealed_vectors: dict[str, np.ndarray],
    win_rates_path: Path,
    *,
    held_out_cards: tuple[str, ...] = (),
) -> CheckResult:
    """How much of each winnability label is linearly readable from pooled ``e``.

    Run side by side with the sealed encoder on the **same** feature table, so
    the comparison is between two representations of the same cards rather than
    between two studies.
    """
    if not cached.by_card:
        return CheckResult(
            "decodability", CheckStatus.SKIPPED,
            "the ability cache is empty; run encode-abilities first",
        )
    if not Path(win_rates_path).exists():
        return CheckResult(
            "decodability", CheckStatus.SKIPPED,
            f"no win-rate table at {win_rates_path}",
        )
    from price_predictor.application.ridge_probes import build_label_table, fit_probes

    shared = sorted(set(cached.by_card) & set(sealed_vectors)) if sealed_vectors \
        else sorted(cached.by_card)
    if len(shared) < 50:
        return CheckResult(
            "decodability", CheckStatus.SKIPPED,
            f"only {len(shared)} cards in both caches; too few to fit probes",
        )
    table = build_label_table(
        shared, win_rates_path=Path(win_rates_path), val_names=set(held_out_cards),
    )
    effects_matrix = np.stack([cached.by_card[name] for name in shared])
    effects_probes = fit_probes(table, effects_matrix, mode="honest")
    detail = _probe_summary("effects", effects_probes)
    if sealed_vectors:
        sealed_matrix = np.stack([sealed_vectors[name] for name in shared])
        sealed_probes = fit_probes(table, sealed_matrix, mode="honest")
        detail += " | " + _probe_summary("sealed", sealed_probes)
    return CheckResult("decodability", CheckStatus.REPORTED, detail)


def _probe_summary(label: str, probes) -> str:
    scores = {
        name: probe.metrics.get("val_r2", float("nan"))
        for name, probe in probes.probes.items()
    }
    rendered = ", ".join(f"{name} {value:.3f}" for name, value in scores.items())
    return f"{label}: {rendered}"


# ── the average-effect control and the taxonomy comparison ──────────────


def check_variant_geometry(
    full: CachedVectors, variant: CachedVectors, name: str,
) -> CheckResult:
    """Compare a baseline's cache geometry against the shipping one.

    ``no-state`` is the average-effect control and ``taxonomy`` the
    script-structure floor; both are read here rather than through a prediction,
    because what they test is whether the encoder's space carries more than
    theirs does.
    """
    if len(variant) < 2 or len(full) < 2:
        return CheckResult(
            f"{name}-comparison", CheckStatus.SKIPPED,
            f"no {name} cache; run encode-abilities --variant {name}",
        )
    from effects.application.evaluate_effect_model import (
        mean_pairwise_cosine,
        top_component_share,
    )

    full_matrix, variant_matrix = full.matrix(), variant.matrix()
    return CheckResult(
        f"{name}-comparison", CheckStatus.REPORTED,
        f"mean pairwise cosine {mean_pairwise_cosine(full_matrix):.3f} vs "
        f"{mean_pairwise_cosine(variant_matrix):.3f}; top component "
        f"{top_component_share(full_matrix):.1%} vs "
        f"{top_component_share(variant_matrix):.1%}",
        {
            "full_cosine": mean_pairwise_cosine(full_matrix),
            f"{name}_cosine": mean_pairwise_cosine(variant_matrix),
        },
    )


# ── the scorer smoke test (FR-115) ──────────────────────────────────────


def write_scorer_smoke_cache(
    cached: CachedVectors,
    sealed_vectors: dict[str, np.ndarray],
    scratch_folder: Path,
    locator,
) -> int:
    """Write pooled ``e`` **concatenated** with the sealed vector, into scratch.

    Concatenated rather than replacing, so the test asks "does this add
    anything" rather than "is this better than nothing". Written into a scratch
    copy of the cards folder and **never** into ``output/cardsfolder/``: the
    sealed pipeline reads that tree, and a run of this check must not change
    what the scorer trains on.
    """
    scratch_folder = Path(scratch_folder)
    written = 0
    for name, pooled_vector in cached.by_card.items():
        base = sealed_vectors.get(name)
        if base is None:
            continue
        combined = np.concatenate([base, pooled_vector]).astype(np.float32)
        path = locator.expected_path(name, ".npz")
        target = scratch_folder / path.relative_to(path.parents[1])
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, embedding=combined)
        written += 1
    return written

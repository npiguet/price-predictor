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
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from effects.application.evaluate_effect_model import CheckResult, CheckStatus
from effects.domain.ability_cache_layout import ARRAY_KEY, CACHE_SUFFIX
from effects.domain.line_query import LineQuery, normalize_prose

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Resolution:
    """Where a :class:`LineQuery` landed: a cache key, or why it did not."""

    key: str | None
    problem: str = ""


@dataclass
class CachedVectors:
    """One variant's cache, loaded once and indexed by ability text.

    Indexed by *text* rather than by provenance key because every geometry
    check asks about unique texts: a line reprinted on forty cards is one point
    in the space, and counting it forty times would make the space look more
    populated than it is. The text is the one on the checkpoint's encoding
    surface — what the model read — so two lines sharing prose but compiling to
    different scripts are two points on a script checkpoint, and one reprint
    encoded in two batches, whose vectors differ in float noise, is one.

    ``display`` and ``by_prose`` exist for the checks a person reads. A script
    key is unreadable in a report, and a query written as script would break on
    the first reconversion, so queries name lines in prose and results print in
    prose; ``by_prose`` maps each normalized prose line to the keys it encodes
    to, with the cards printing each.
    """

    by_text: dict[str, np.ndarray] = field(default_factory=dict)
    by_card: dict[str, np.ndarray] = field(default_factory=dict)
    display: dict[str, str] = field(default_factory=dict)
    by_prose: dict[str, dict[str, set[str]]] = field(default_factory=dict)

    def matrix(self) -> np.ndarray:
        if not self.by_text:
            return np.zeros((0, 0), dtype=np.float32)
        return np.stack(list(self.by_text.values()))

    def texts(self) -> list[str]:
        return list(self.by_text)

    def label(self, key: str) -> str:
        """``key`` as a person reads it: its prose, where it has any."""
        return self.display.get(key, key)

    def resolve(self, query: LineQuery) -> Resolution:
        """The cache key ``query`` names, or the reason it names none.

        Several keys under one prose is ambiguity rather than a choice to make
        here: picking the first would compare against whichever card happened
        to load first, so the query has to pin its card instead.
        """
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


def load_cache(
    abilities_root: Path,
    sidecar_roots: Path | Iterable[Path],
    *,
    surface: str,
    variant: str = "full",
) -> CachedVectors:
    """Load a variant's cache, keyed by the ability text each row encodes.

    ``sidecar_roots`` supplies the texts: the cache is row-aligned with each
    source's sidecar, so the pairing is positional and needs no index. The key
    is the text on ``surface``, which the caller takes from the checkpoint the
    cache was encoded with, and which has no default: a check keyed on the
    other surface compares the vectors against texts they do not encode, and
    on a script checkpoint silently merges every pair of scripts sharing a
    prose line.

    Only a ``cardsfolder`` source contributes a pooled per-card vector: a token
    is not a card the sealed pipeline scores.
    """
    from effects.domain.ability_encoder import encoding_text
    from effects.infrastructure.sidecar_io import (
        SIDECAR_SUFFIX,
        converted_text_path,
        prose_for,
        prose_lines,
        read_sidecar,
    )

    roots = (
        (Path(sidecar_roots),) if isinstance(sidecar_roots, (str, Path))
        else tuple(Path(r) for r in sidecar_roots)
    )
    suffix = CACHE_SUFFIX if variant == "full" else f".{variant}{CACHE_SUFFIX}"
    cached = CachedVectors()
    root = Path(abilities_root)
    if not root.is_dir():
        return cached

    for sidecar_root in roots:
        for sidecar_path in sorted(sidecar_root.rglob(f"*{SIDECAR_SUFFIX}")):
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
            card = normalize_prose(sidecar.card)
            for row, line in zip(matrix, sidecar.lines):
                prose = prose_for(line, rendered)
                text = encoding_text(line, prose, surface)
                if not text:
                    continue
                cached.by_text.setdefault(text, row)
                cached.display.setdefault(text, prose or text)
                if prose:
                    cached.by_prose.setdefault(
                        normalize_prose(prose), {},
                    ).setdefault(text, set()).add(card)
            if matrix.size and relative.parts[:1] == ("cardsfolder",):
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
    cached: CachedVectors, queries: tuple[LineQuery, ...],
) -> CheckResult:
    """Each query's three nearest lines, printed in prose for a person to judge.

    A query that resolves to no line is listed as unresolved rather than left
    out, so a list shorter than the query table says why.
    """
    if len(cached) < 2:
        return CheckResult(
            "nearest-neighbour", CheckStatus.SKIPPED,
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
        "nearest-neighbour", CheckStatus.REPORTED, "; ".join(lines),
        {"unresolved_queries": float(unresolved)},
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
    """Is ward's ``e`` nearer its longhand twins than a bare keyword generally?

    Every line resolves through its prose to the key the checkpoint encoded. A
    twin or keyword that resolves to nothing is named in the result rather
    than dropped: a canary quietly comparing against fewer twins than it lists
    is weaker than it claims to be.
    """
    from effects.application.evaluate_effect_model import evaluate_ward_canary
    from effects.domain.ward_twins import BARE_KEYWORDS, WARD, WARD_TWINS

    ward = cached.resolve(WARD)
    if ward.key is None:
        return CheckResult(
            "ward-canary", CheckStatus.SKIPPED,
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

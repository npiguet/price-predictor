"""Method B: how much the frozen trunk relies on ``e`` (FR-085).

Each output field's loss is measured over validation records as the model
reads them, then again with ``e`` replaced by one of three substitutes:

===========  ================================================================
noise        Gaussian noise matched to the cache's mean and covariance — does
             the trunk rely on ``e`` at all?
api-mean     the mean ``e`` of the line's API type — does it rely on ``e``
             beyond the API type?
nearest      the ``e`` of the other text whose vector lies closest — how
             sensitive is it to fine differences between texts?
===========  ================================================================

Each replacement is applied to every ability slot, to ``[ACT]`` alone, and to
the card slots alone. The model takes ``e`` as an input of its own, so a
replacement is a different tensor in that argument and the model is never
modified.

Nothing here does work at import time.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace

import numpy as np

REPLACEMENTS: tuple[str, ...] = ("noise", "api-mean", "nearest")
SCOPES: tuple[str, ...] = ("all", "act", "cards")
JITTER = 1e-6


@dataclass
class CacheStatistics:
    """What the three replacements draw on, computed once from the cache."""

    texts: list[str]
    matrix: np.ndarray
    mean: np.ndarray
    cholesky: np.ndarray
    api_means: dict[str | None, np.ndarray]


def cache_statistics(items) -> CacheStatistics:
    """From ``common.line_items`` with ``e``: one row per unique text."""
    items = [item for item in items if item.e is not None]
    matrix = np.stack([item.e for item in items]).astype(np.float64)
    mean = matrix.mean(axis=0)
    cov = np.cov(matrix, rowvar=False) + JITTER * np.eye(matrix.shape[1])
    by_api: dict[str | None, list[np.ndarray]] = defaultdict(list)
    for item, row in zip(items, matrix):
        by_api[item.api].append(row)
    return CacheStatistics(
        texts=[item.text for item in items],
        matrix=matrix.astype(np.float32),
        mean=mean.astype(np.float32),
        cholesky=np.linalg.cholesky(cov).astype(np.float32),
        api_means={api: np.mean(rows, axis=0).astype(np.float32)
                   for api, rows in by_api.items()},
    )


def scope_mask(slot_kinds, scope: str):
    """Which slots a scope replaces: ``[ACT]`` is slot 1, cards are ABILITY slots."""
    from effects.domain.input_layout import SlotKind

    act = slot_kinds == int(SlotKind.ACT)
    cards = slot_kinds == int(SlotKind.ABILITY)
    return {"all": act | cards, "act": act, "cards": cards}[scope]


@dataclass(frozen=True)
class SubstituteRequest:
    """The batch's distinct texts, their lines and the live ``e`` rows to replace."""

    texts: list[str]
    lines: dict
    live: object


def _noise_vectors(stats: CacheStatistics, request: SubstituteRequest,
                   rng: np.random.Generator) -> np.ndarray:
    z = rng.standard_normal((len(request.texts), stats.mean.shape[0])).astype(np.float32)
    return stats.mean + z @ stats.cholesky.T


def _api_mean_vectors(stats: CacheStatistics, request: SubstituteRequest,
                      rng: np.random.Generator) -> np.ndarray:
    out = []
    for text in request.texts:
        line = request.lines.get(text)
        api = getattr(line, "script_api_type", None)
        out.append(stats.api_means.get(api, stats.mean))
    return np.stack(out)


def _nearest_vectors(stats: CacheStatistics, request: SubstituteRequest,
                     rng: np.random.Generator) -> np.ndarray:
    import torch

    live = request.live
    cache = torch.tensor(stats.matrix, device=live.device)
    cache = torch.nn.functional.normalize(cache, dim=-1)
    query = torch.nn.functional.normalize(live.float(), dim=-1)
    similarity = query @ cache.T
    own = {text: index for index, text in enumerate(stats.texts)}
    for row, text in enumerate(request.texts):
        if text in own:
            similarity[row, own[text]] = -2.0
    nearest = similarity.argmax(dim=-1).cpu().numpy()
    return stats.matrix[nearest]


_SUBSTITUTES = {
    "noise": _noise_vectors, "api-mean": _api_mean_vectors, "nearest": _nearest_vectors,
}


def replacement_vectors(stats: CacheStatistics, request: SubstituteRequest,
                        replacement: str, rng: np.random.Generator) -> np.ndarray:
    """One substitute ``e`` per batch text (rows aligned with ``request.texts``)."""
    return _SUBSTITUTES[replacement](stats, request, rng)


@dataclass(frozen=True)
class AblationSettings:
    """How :func:`run_ablation` batches the records and seeds its noise."""

    batch_size: int = 32
    seed: int = 42


@dataclass
class _Chunk:
    """One batch of records as the model reads it, with its targets."""

    records: list
    batch: object
    surfaces: list
    encoded: object
    targets: object
    rows: object = None


class _Ablation:
    """The running loss totals of one ablation over successive record chunks."""

    def __init__(self, probe_model, stats: CacheStatistics, settings: AblationSettings) -> None:
        self.probe_model = probe_model
        self.stats = stats
        self.settings = settings
        self.rng = np.random.default_rng(settings.seed)
        self.totals: dict[tuple, float] = defaultdict(float)
        self.count = 0

    def run(self, records: list) -> dict:
        import torch

        size = self.settings.batch_size
        with torch.no_grad():
            for start in range(0, len(records), size):
                self._add_chunk(records[start : start + size])
        return self._report()

    def _losses(self, chunk: _Chunk, e_vectors) -> dict[str, float]:
        batch = replace(chunk.batch, e_vectors=e_vectors)
        return entity_loss_parts(self.probe_model.model, batch, chunk.targets,
                                 self.probe_model.fields)

    def _add_chunk(self, records: list) -> None:
        from effects.domain.surface_collation import per_entity_targets

        probe_model = self.probe_model
        batch, surfaces, encoded = probe_model.batcher.build(records, probe_model.encoder)
        targets = per_entity_targets(records, surfaces, probe_model.fields)
        chunk = _Chunk(records, batch, surfaces, encoded, targets)
        for name, value in self._losses(chunk, batch.e_vectors).items():
            self.totals[(name, "base", "")] += value * len(records)
        if encoded.matrix is not None:
            chunk.rows = _e_rows(chunk)
            for replacement in REPLACEMENTS:
                self._score_replacement(chunk, replacement)
        self.count += len(records)

    def _score_replacement(self, chunk: _Chunk, replacement: str) -> None:
        import torch

        batch, encoded = chunk.batch, chunk.encoded
        substitute = torch.tensor(replacement_vectors(
            self.stats, SubstituteRequest(encoded.texts, encoded.lines, encoded.matrix),
            replacement, self.rng,
        ), device=batch.e_vectors.device, dtype=batch.e_vectors.dtype)
        for scope in SCOPES:
            self._score_scope(chunk, replacement, scope, substitute)

    def _score_scope(self, chunk: _Chunk, replacement: str, scope: str, substitute) -> None:
        batch, rows = chunk.batch, chunk.rows
        chosen = scope_mask(batch.slot_kinds, scope) & (rows >= 0)
        e_vectors = batch.e_vectors.clone()
        e_vectors[chosen] = substitute[rows[chosen]]
        for name, value in self._losses(chunk, e_vectors).items():
            self.totals[(name, replacement, scope)] += value * len(chunk.records)

    def _report(self) -> dict:
        totals, count = self.totals, max(self.count, 1)
        out: dict[str, dict] = {}
        for name in sorted({name for name, kind, _ in totals if kind == "base"}):
            base = totals[(name, "base", "")] / count
            out[name] = {
                replacement: {
                    scope: totals[(name, replacement, scope)] / count - base
                    for scope in SCOPES
                }
                for replacement in REPLACEMENTS
            }
            out[name]["base_loss"] = base
        return out


def _e_rows(chunk: _Chunk):
    """Each slot's row of the batch's ``e`` matrix, ``-1`` where it has none."""
    import torch

    batch = chunk.batch
    rows = torch.full(batch.e_vectors.shape[:2], -1, dtype=torch.long)
    for b, surface in enumerate(chunk.surfaces):
        for s, slot in enumerate(surface.slots):
            if slot.e_row is not None:
                rows[b, s] = slot.e_row
    return rows.to(batch.e_vectors.device)


def run_ablation(probe_model, records: list, stats: CacheStatistics, *,
                 settings: AblationSettings = AblationSettings()) -> dict:
    """``{field: {replacement: {scope: loss increase}}}`` over ``records``.

    Losses are the per-entity head's own terms, the affected gate among them,
    averaged per record exactly as training reports them.
    """
    return _Ablation(probe_model, stats, settings).run(records)


def entity_loss_parts(model, batch, targets, fields) -> dict[str, float]:
    """The per-entity loss's terms when ``model`` reads one built ``batch``."""
    from effects.domain.effect_losses import EntityLossScoring, per_entity_loss

    hidden = model(batch)
    gathered = model.per_entity_at(hidden, targets.index.to(hidden.device)).float()
    _total, parts = per_entity_loss(
        gathered, targets.fields_on(gathered.device),
        EntityLossScoring(
            fields=fields,
            log_dispersion=model.per_entity_dispersion(),
            report_parts=True,
        ),
    )
    return parts.as_dict()


def field_losses(model, probe_model, records: list, *, batch_size: int = 32) -> dict:
    """Each per-entity field's mean loss when ``model`` reads ``records``.

    ``model`` may be another trunk than ``probe_model``'s own — method C scores
    its shallow trunk this way, over the same batches and targets.
    """
    import torch

    from effects.domain.surface_collation import per_entity_targets

    totals: dict[str, float] = defaultdict(float)
    count = 0
    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            chunk = records[start : start + batch_size]
            batch, surfaces, _encoded = probe_model.batcher.build(chunk, probe_model.encoder)
            targets = per_entity_targets(chunk, surfaces, probe_model.fields)
            parts = entity_loss_parts(model, batch, targets, probe_model.fields)
            for name, value in parts.items():
                totals[name] += value * len(chunk)
            count += len(chunk)
    return {name: value / max(count, 1) for name, value in totals.items()}

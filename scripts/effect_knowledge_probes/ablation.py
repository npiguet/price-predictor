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
from dataclasses import dataclass

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
    from effects.domain.effect_head_input import SlotKind

    act = slot_kinds == int(SlotKind.ACT)
    cards = slot_kinds == int(SlotKind.ABILITY)
    return {"all": act | cards, "act": act, "cards": cards}[scope]


def replacement_vectors(stats: CacheStatistics, texts: list[str], lines: dict,
                        live, replacement: str, rng: np.random.Generator) -> np.ndarray:
    """One substitute ``e`` per batch text (rows aligned with ``texts``)."""
    import torch

    if replacement == "noise":
        z = rng.standard_normal((len(texts), stats.mean.shape[0])).astype(np.float32)
        return stats.mean + z @ stats.cholesky.T
    if replacement == "api-mean":
        out = []
        for text in texts:
            line = lines.get(text)
            api = getattr(line, "script_api_type", None)
            out.append(stats.api_means.get(api, stats.mean))
        return np.stack(out)
    if replacement == "nearest":
        cache = torch.tensor(stats.matrix, device=live.device)
        cache = torch.nn.functional.normalize(cache, dim=-1)
        query = torch.nn.functional.normalize(live.float(), dim=-1)
        similarity = query @ cache.T
        own = {text: index for index, text in enumerate(stats.texts)}
        for row, text in enumerate(texts):
            if text in own:
                similarity[row, own[text]] = -2.0
        nearest = similarity.argmax(dim=-1).cpu().numpy()
        return stats.matrix[nearest]
    raise KeyError(replacement)


def run_ablation(probe_model, records: list, stats: CacheStatistics, *,
                 batch_size: int = 32, seed: int = 42) -> dict:
    """``{field: {replacement: {scope: loss increase}}}`` over ``records``.

    Losses are the per-entity head's own terms, the affected gate among them,
    averaged per record exactly as training reports them.
    """
    import torch

    from effects.domain.effect_model import entity_target_tensors, per_entity_loss
    from effects.domain.effect_targets import derive_targets

    model, encoder, batcher = probe_model.model, probe_model.encoder, probe_model.batcher
    fields = probe_model.fields
    rng = np.random.default_rng(seed)
    totals: dict[tuple, float] = defaultdict(float)
    count = 0

    def losses(batch, surfaces, targets, e_vectors) -> dict[str, float]:
        hidden = model(**{**batch, "e_vectors": e_vectors})
        outputs = model.per_entity(hidden)
        gate, field_targets, mask, index = entity_target_tensors(surfaces, targets, fields)
        index = index.to(outputs.device)
        gathered = outputs.gather(1, index.unsqueeze(-1).expand(-1, -1, outputs.shape[-1]))
        device = gathered.device
        _total, parts = per_entity_loss(
            gathered.float(), gate.to(device),
            {k: v.to(device) for k, v in field_targets.items()},
            mask.to(device), fields=fields, report_parts=True,
        )
        return parts

    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            chunk = records[start : start + batch_size]
            batch, surfaces = batcher.build(chunk, encoder)
            encoded = batcher.encoded
            targets = [derive_targets(record) for record in chunk]
            base = losses(batch, surfaces, targets, batch["e_vectors"])
            for name, value in base.items():
                totals[(name, "base", "")] += value * len(chunk)
            if encoded.matrix is not None:
                rows = torch.full(batch["e_vectors"].shape[:2], -1, dtype=torch.long)
                for b, surface in enumerate(surfaces):
                    for s, slot in enumerate(surface.slots):
                        if isinstance(slot.e, int):
                            rows[b, s] = slot.e
                rows = rows.to(batch["e_vectors"].device)
                has_row = rows >= 0
                for replacement in REPLACEMENTS:
                    substitute = torch.tensor(replacement_vectors(
                        stats, encoded.texts, encoded.lines, encoded.matrix,
                        replacement, rng,
                    ), device=batch["e_vectors"].device, dtype=batch["e_vectors"].dtype)
                    for scope in SCOPES:
                        chosen = scope_mask(batch["slot_kinds"], scope) & has_row
                        e_vectors = batch["e_vectors"].clone()
                        e_vectors[chosen] = substitute[rows[chosen]]
                        for name, value in losses(batch, surfaces, targets, e_vectors).items():
                            totals[(name, replacement, scope)] += value * len(chunk)
            count += len(chunk)
    out: dict[str, dict] = {}
    names = sorted({name for name, kind, _ in totals if kind == "base"})
    for name in names:
        base = totals[(name, "base", "")] / max(count, 1)
        out[name] = {
            replacement: {
                scope: totals[(name, replacement, scope)] / max(count, 1) - base
                for scope in SCOPES
            }
            for replacement in REPLACEMENTS
        }
        out[name]["base_loss"] = base
    return out


def field_losses(model, probe_model, records: list, *, batch_size: int = 32) -> dict:
    """Each per-entity field's mean loss when ``model`` reads ``records``.

    ``model`` may be another trunk than ``probe_model``'s own — method C scores
    its shallow trunk this way, over the same batches and targets.
    """
    import torch

    from effects.domain.effect_model import entity_target_tensors, per_entity_loss
    from effects.domain.effect_targets import derive_targets

    totals: dict[str, float] = defaultdict(float)
    count = 0
    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            chunk = records[start : start + batch_size]
            batch, surfaces = probe_model.batcher.build(chunk, probe_model.encoder)
            outputs = model.per_entity(model(**batch))
            targets = [derive_targets(record) for record in chunk]
            gate, field_targets, mask, index = entity_target_tensors(
                surfaces, targets, probe_model.fields)
            index = index.to(outputs.device)
            gathered = outputs.gather(
                1, index.unsqueeze(-1).expand(-1, -1, outputs.shape[-1])).float()
            device = gathered.device
            _total, parts = per_entity_loss(
                gathered, gate.to(device),
                {k: v.to(device) for k, v in field_targets.items()},
                mask.to(device), fields=probe_model.fields, report_parts=True,
            )
            for name, value in parts.items():
                totals[name] += value * len(chunk)
            count += len(chunk)
    return {name: value / max(count, 1) for name, value in totals.items()}

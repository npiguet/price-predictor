"""Every record's own losses and grouping keys, for the evaluation's breakdowns.

Gate 1's forward pass (:func:`effects.application.record_forward.entity_chunks`),
split by row into the trainer's per-entity loss over each record alone.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from effects.application.breakdowns import TOTAL, RecordResult
from effects.application.record_forward import EntityScoring, entity_chunks
from effects.domain.damage_step_keywords import KeywordResolver
from effects.domain.effect_families import effect_family
from effects.domain.effect_losses import GATE_TERM, EntityLossScoring, per_entity_loss
from effects.domain.entity_field_targets import EntityTargetBatch
from effects.domain.keyword_names import normalize_keyword
from effects.domain.provenance import first_text
from effects.domain.records import acting_keys

if TYPE_CHECKING:
    from effects.application.model_loading import Runnable
    from effects.application.surface_batching import SurfaceBatcher


def score_records(
    records, runnable: Runnable, *, legality_mode_counts: Mapping[str, int],
) -> list[RecordResult]:
    """Every record's own losses and grouping keys, for the breakdowns.

    ``legality_mode_counts`` is the dataset manifest's, so each record's family
    is the one the build placed it in.
    """
    # One resolver for the whole pass: it memoizes each key's keyword.
    resolver = runnable.keyword_resolver()
    grouping = _Grouping(runnable.batcher, resolver, legality_mode_counts)
    results: list[RecordResult] = []
    dispersion = runnable.model.per_entity_dispersion().detach().cpu()
    for chunk, gathered, targets in entity_chunks(
            records, EntityScoring(runnable.forward(), runnable.batcher, runnable.fields)):
        scored = _ChunkOutputs(gathered, targets, runnable.fields, dispersion)
        for row, record in enumerate(chunk):
            results.append(grouping.result(record, _record_losses(row, scored)))
    return results


@dataclass(frozen=True)
class _ChunkOutputs:
    """One forward chunk: the head's outputs gathered per entity, the targets
    they are scored against, the fields scored and the dispersions."""

    gathered: torch.Tensor
    targets: EntityTargetBatch
    fields: Sequence
    log_dispersion: torch.Tensor


def _record_losses(row: int, chunk: _ChunkOutputs) -> dict[str, float]:
    """One record's per-entity loss by field: ``per_entity_loss`` on it alone,
    so the breakdowns cannot drift from what training minimized. A field with
    no supervised entity is left out rather than reported as zero, the gate
    included: a record that scores no gate entity (a playability record) has no
    ``gate`` key. A record that supervises no term at all has no ``total``
    either, so a breakdown's total mean and count read only records that train
    something."""
    rows = slice(row, row + 1)
    alone = chunk.targets.select(rows)
    _total, parts = per_entity_loss(
        chunk.gathered[rows], alone,
        EntityLossScoring(
            fields=chunk.fields, log_dispersion=chunk.log_dispersion, report_parts=True,
        ),
    )
    losses = parts.as_dict()
    if not bool(alone.gate_scored.any()):
        del losses[GATE_TERM]
    if losses:
        losses[TOTAL] = sum(losses.values())
    return losses


@dataclass(frozen=True)
class _Grouping:
    """What names a record's grouping keys: the sidecars' batcher, the pass's
    keyword resolver and the manifest's legality mode counts."""

    batcher: SurfaceBatcher
    resolver: KeywordResolver
    legality_mode_counts: Mapping[str, int]

    def result(self, record, losses: dict[str, float]) -> RecordResult:
        """The grouping keys of one scored record, beside its losses."""
        return RecordResult(
            game_id=record.game_id,
            subkind=record.subkind.value if record.subkind else None,
            text=first_text(acting_keys(record), self.batcher.text_of),
            family=self._family(record),
            random_seat=bool(record.random_seat),
            what_if=record.what_if,
            keywords=record_keywords(record, self.batcher, self.resolver),
            losses=losses,
        )

    def _family(self, record) -> str | None:
        return effect_family(
            record, self.batcher.sidecars, resolver=self.resolver,
            legality_mode_counts=self.legality_mode_counts,
        )


def record_keywords(
    record, batcher: SurfaceBatcher, resolver: KeywordResolver,
) -> frozenset[str]:
    """Every keyword on the acting line (a ward trigger, say) or carried by a
    board entity."""
    found: set[str] = set()
    for key in acting_keys(record):
        line = batcher.sidecars.line_or_none(key)
        keyword = line.bare_keyword if line is not None else None
        if keyword is not None:
            found.add(normalize_keyword(keyword))
    for entity in record.state.entities:
        found.update(normalize_keyword(k) for k in resolver.keywords_of(entity))
    return frozenset(found)

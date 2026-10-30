"""Every record's own losses and grouping keys, for the evaluation's breakdowns.

Gate 1's forward pass (:func:`effects.application.model_loading.forward_chunks`),
split by row into the trainer's per-entity loss over each record alone.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from effects.application.breakdowns import TOTAL, RecordResult
from effects.application.model_loading import forward_chunks
from effects.domain.damage_step_keywords import KeywordResolver
from effects.domain.effect_losses import per_entity_loss
from effects.domain.entity_field_targets import EntityTargetBatch
from effects.domain.keyword_names import normalize_keyword
from effects.domain.provenance import first_text
from effects.domain.records import acting_keys
from effects.domain.rule_families import rule_family

if TYPE_CHECKING:
    from effects.application.model_loading import Runnable
    from effects.application.surface_batching import SurfaceBatcher


def score_records(
    records, runnable: Runnable, *, legality_mode_counts=None,
) -> list[RecordResult]:
    """Every record's own losses and grouping keys, for the breakdowns.

    ``legality_mode_counts`` is the dataset manifest's, so each record's family
    is the one the build placed it in.
    """
    # One resolver for the whole pass: it memoizes each key's keyword.
    resolver = runnable.keyword_resolver()
    results: list[RecordResult] = []
    for chunk, gathered, targets in forward_chunks(records, runnable):
        for row, record in enumerate(chunk):
            results.append(_record_result(
                record, runnable.batcher, resolver,
                _record_losses(row, gathered, targets, runnable.fields),
                legality_mode_counts,
            ))
    return results


def _record_losses(
    row: int, gathered: torch.Tensor, targets: EntityTargetBatch, fields,
) -> dict[str, float]:
    """One record's per-entity loss by field: ``per_entity_loss`` on it alone,
    so the breakdowns cannot drift from what training minimized. A field with
    no supervised entity is left out rather than reported as zero."""
    rows = slice(row, row + 1)
    _total, parts = per_entity_loss(
        gathered[rows], targets.select(rows), fields=fields, report_parts=True,
    )
    losses = parts.as_dict()
    losses[TOTAL] = sum(losses.values())
    return losses


def _record_result(
    record, batcher: SurfaceBatcher, resolver: KeywordResolver,
    losses: dict[str, float], legality_mode_counts=None,
) -> RecordResult:
    """The grouping keys of one scored record, beside its losses."""
    try:
        family = rule_family(
            record, batcher.sidecars, resolver=resolver,
            legality_mode_counts=legality_mode_counts,
        )
    except KeyError:
        # The sidecar does not describe the card the record names.
        family = None
    return RecordResult(
        game_id=record.game_id,
        subkind=record.subkind.value if record.subkind else None,
        text=first_text(acting_keys(record), batcher.text_of),
        family=family,
        random_seat=bool(record.random_seat),
        what_if=record.what_if,
        keywords=record_keywords(record, batcher, resolver),
        losses=losses,
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

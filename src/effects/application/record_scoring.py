"""Every record's own losses and grouping keys, for the evaluation's breakdowns.

The breakdowns compare how a field's loss moves across slices of the
validation strata, so each record is scored on its own: the same forward pass
gate 1 pools into its margins (:func:`effects.application.gate_one.forward_chunks`),
split by row here into the trainer's own per-entity loss over that one record.
The grouping itself is :mod:`effects.application.breakdowns`' job.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from effects.application.breakdowns import TOTAL, RecordResult
from effects.application.gate_one import acting_text, forward_chunks
from effects.domain.damage_step_keywords import KeywordResolver, keyword_of_line
from effects.domain.effect_head_layout import EntityFieldTargets
from effects.domain.effect_losses import per_entity_loss
from effects.domain.keyword_names import normalize_keyword
from effects.domain.records import acting_keys
from effects.domain.rule_families import rule_family

if TYPE_CHECKING:
    from effects.application.surface_batching import SurfaceBatcher
    from effects.infrastructure.model_runner import Runnable


def score_records(
    records, runnable: Runnable, *, legality_mode_counts=None,
) -> list[RecordResult]:
    """Every record's own losses and grouping keys, for the breakdowns.

    ``legality_mode_counts`` is the dataset manifest's, so each record's family
    is the one the build placed it in.
    """
    # One resolver for the whole pass: it memoizes each provenance key's
    # keyword, and the strata repeat the same few hundred cards.
    resolver = runnable.keyword_resolver()
    results: list[RecordResult] = []
    for chunk, (gathered, gate, field_targets, mask) in forward_chunks(
        records, runnable,
    ):
        for row, record in enumerate(chunk):
            results.append(_record_result(
                record, runnable.batcher, resolver,
                _record_losses(
                    row, gathered, gate, field_targets, mask, runnable.fields,
                ),
                legality_mode_counts,
            ))
    return results


def _record_losses(
    row: int, gathered: torch.Tensor, gate: torch.Tensor,
    field_targets: EntityFieldTargets, mask: torch.Tensor, fields,
) -> dict[str, float]:
    """One record's per-entity loss by field: ``per_entity_loss`` on it alone.

    The trainer's own loss over a batch of one, so the breakdowns cannot drift
    from what training minimized: the gate over every real entity, then each
    field over the affected entities its record supervises. A field with no
    such entity is left out rather than reported as zero. One record means the
    per-record normalization divides by one, so each part is that record's raw
    sum. Read on the host, where the batch already is, so scoring a record
    costs no device synchronization.
    """
    rows = slice(row, row + 1)
    _total, parts = per_entity_loss(
        gathered[rows], gate[rows],
        field_targets.select(rows),
        mask[rows], fields=fields, report_parts=True,
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
        # The sidecar does not describe the card the record names; the record
        # still scored, it just cannot be placed in a family.
        family = None
    return RecordResult(
        record_id=record.record_id,
        game_id=record.game_id,
        kind=record.kind.value,
        subkind=record.subkind.value if record.subkind else None,
        text=acting_text(record, batcher),
        family=family,
        random_seat=bool(record.random_seat),
        what_if=record.what_if,
        keywords=record_keywords(record, batcher, resolver),
        losses=losses,
    )


def record_keywords(
    record, batcher: SurfaceBatcher, resolver: KeywordResolver,
) -> frozenset[str]:
    """Every keyword on the acting line or carried by a board entity.

    The acting line is a keyword when the record is that keyword's own
    resolution (a ward trigger, say); the board's entities carry theirs through
    both channels the resolver reads.
    """
    found: set[str] = set()
    for key in acting_keys(record):
        line = batcher.sidecars.line_or_none(key)
        keyword = keyword_of_line(line) if line is not None else None
        if keyword is not None:
            found.add(normalize_keyword(keyword))
    for entity in record.state.entities:
        found.update(normalize_keyword(k) for k in resolver.keywords_of(entity))
    return frozenset(found)

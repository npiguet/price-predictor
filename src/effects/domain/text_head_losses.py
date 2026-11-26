"""The text heads' losses: the value head's, the masked-token head's and the
script-API head's, one row per encoded text (FR-058, FR-058b).

A row's loss is averaged over the rows of the pass
(:func:`~effects.domain.loss_rows.per_row`).
"""

from __future__ import annotations

import torch
import torch.nn.functional as functional

from effects.domain.effect_head_layout import (
    AT_LEAST_SLICES,
    AT_LEAST_STEPS,
    VALUE_DISPERSED,
    VALUE_SLICES,
)
from effects.domain.field_kinds import FIELD_KINDS, FieldScoring
from effects.domain.field_types import FieldType
from effects.domain.loss_rows import Rows, per_row
from effects.domain.value_targets import VALUE_TARGETS


def at_least_targets(amounts: torch.Tensor) -> torch.Tensor:
    """``(n,)`` amounts as ``(n, AT_LEAST_STEPS)`` bits: column k-1 is ``|amount| >= k``."""
    steps = torch.arange(
        1, AT_LEAST_STEPS + 1, device=amounts.device, dtype=amounts.dtype,
    )
    return (amounts.abs().unsqueeze(-1) >= steps).to(amounts.dtype)


_VALUE_DISPERSION_INDEX: dict[str, int] = {
    name: i for i, name in enumerate(VALUE_DISPERSED)
}


def value_loss(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
    *, log_dispersion: torch.Tensor | None = None,
) -> torch.Tensor:
    """The value head's loss, one row per unique text (FR-058, FR-058b): the
    per-entity fields' losses by type, plus each amount's at-least bits;
    ``log_dispersion`` in :data:`VALUE_DISPERSED` order."""
    rows = int(prediction.shape[0])
    mask = mask.detach().bool().cpu()
    target = target.to(prediction.device)
    total = prediction.new_zeros((), dtype=torch.float32)
    for index, entry in enumerate(VALUE_TARGETS):
        selected = Rows.of(mask[:, index], prediction.device)
        if selected.empty:
            continue
        predicted = selected.select(prediction).float()
        wanted = selected.select(target)[:, index]
        for term in _value_target_terms(predicted, wanted, entry, log_dispersion):
            total = total + term
    return per_row(total, rows)


def _value_target_terms(
    predicted: torch.Tensor, wanted: torch.Tensor, entry: tuple[str, FieldType],
    log_dispersion: torch.Tensor | None,
) -> list[torch.Tensor]:
    """One value target's terms, in the order they are summed: its field's
    loss, then its at-least bits where it has them."""
    name, field_type = entry
    dispersion = None
    if field_type.has_magnitude:
        if log_dispersion is None:
            raise ValueError(
                f"the value head's {name} is scored by a negative binomial "
                "and needs the head's log-dispersions"
            )
        dispersion = log_dispersion[_VALUE_DISPERSION_INDEX[name]]
    start, end = VALUE_SLICES[name]
    # Uncapped: a stated amount is the text's own, which no game compounds.
    terms = [FIELD_KINDS[field_type].loss(
        predicted[:, start:end], wanted, FieldScoring(1, None, dispersion),
    )]
    if name in AT_LEAST_SLICES:
        # Averaged over the steps, so forty bits weigh what one binary
        # target does rather than dominating the encoder's gradient.
        start, end = AT_LEAST_SLICES[name]
        terms.append(functional.binary_cross_entropy_with_logits(
            predicted[:, start:end], at_least_targets(wanted.float()), reduction="sum",
        ) / AT_LEAST_STEPS)
    return terms


def mlm_loss(
    logits: torch.Tensor, targets: torch.Tensor, *, rows: int,
) -> torch.Tensor:
    """Cross-entropy at the masked positions, per text of the pass."""
    return per_row(functional.cross_entropy(logits, targets, reduction="sum"), rows)


def api_loss(
    type_logits: torch.Tensor,
    type_targets: torch.Tensor,
    key_logits: torch.Tensor | None,
    key_targets: torch.Tensor | None,
) -> torch.Tensor:
    """Script-API classification from ``e``, one row per text: the API type's
    cross-entropy plus every parameter key's binary cross-entropy. The keys
    are None for a run whose sidecars name no parameter key, which builds no
    key head."""
    rows = int(type_logits.shape[0])
    total = None
    # A line with no API type carries -1 and is skipped.
    known = Rows.of(type_targets.detach().cpu() >= 0, type_logits.device)
    if not known.empty:
        total = functional.cross_entropy(
            known.select(type_logits),
            known.select(type_targets).to(type_logits.device),
            reduction="sum",
        )
    if key_logits is not None and key_targets is not None:
        keys = functional.binary_cross_entropy_with_logits(
            key_logits, key_targets.float(), reduction="sum",
        )
        total = keys if total is None else total + keys
    if total is None:
        return type_logits.new_zeros(())
    return per_row(total, rows)

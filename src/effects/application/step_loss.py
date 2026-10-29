"""A step's device half: one prepared step's losses, training's and validation's.

Both read the same computer, so the number validation prints is the objective
training descends.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

import torch

from effects.application.step_preparation import DeviceStep, HeadTargets
from effects.application.surface_batching import EncodedBatch, Materializer
from effects.application.text_staging import MaskedTokens, MlmMasker
from effects.application.train_effect_model import TrainEffectModelConfig
from effects.domain.ability_encoder import AbilityEncoder
from effects.domain.aux_heads import (
    AUX_HEAD,
    AUX_HEADS,
    SHIPPED_HEADS,
    TRAINING_HEADS,
    Head,
    HeadInput,
)
from effects.domain.effect_losses import LossParts, mlm_loss, per_entity_loss
from effects.domain.effect_model import EffectModel
from effects.domain.surface_collation import gather_entities


def autocast_enabled(device: torch.device) -> bool:
    """Whether the forward pass and loss run under bf16 autocast: CUDA only,
    and only when the card supports bf16.

    Guarded on ``device_count()`` because under ``CUDA_VISIBLE_DEVICES=""``
    ``is_available()`` can still be true and ``is_bf16_supported()`` would
    raise rather than read as unsupported.
    """
    return (
        device.type == "cuda"
        and torch.cuda.device_count() > 0
        and torch.cuda.is_bf16_supported()
    )


def read_back(scalars: Mapping[str, torch.Tensor]) -> dict[str, float]:
    """Device scalars as floats by name, in one synchronization.

    Stacked in float64, so each value is the one ``float()`` would have read;
    moved to the first one's device, because a group with no gradient reports
    its norm as a host zero.
    """
    if not scalars:
        return {}
    values = list(scalars.values())
    device = values[0].device
    stacked = torch.stack([
        value.detach().to(device, torch.float64) for value in values
    ])
    return dict(zip(scalars, stacked.tolist()))


@dataclass
class StepLoss:
    """One batch's losses, on the device."""

    #: What backward runs on: ``shipped`` plus the weighted training-only heads.
    total: torch.Tensor
    #: The objective validation scores and the train line prints: the
    #: per-entity loss and the shipped heads.
    shipped: torch.Tensor
    parts: LossParts
    terms: dict[str, torch.Tensor]
    #: The MLM pass's masked tokens, left for after the main backward.
    pending_mlm: MaskedTokens | None = None
    #: The per-entity head's output at each target entity, detached.
    gathered: torch.Tensor | None = None


def combine_terms(
    loss: torch.Tensor, terms: Mapping[str, torch.Tensor],
    weights: Mapping[Head, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(total, shipped)`` from the per-entity loss and the head terms (FR-060a).

    Shipped heads join at unit weight, training-only heads at their own
    weights for backward alone. Summed in :data:`AUX_HEADS` order, which is
    part of what makes a run repeat bit for bit.
    """
    shipped = loss
    for spec in SHIPPED_HEADS:
        if spec.head in terms:
            shipped = shipped + terms[spec.head]
    total = shipped
    for spec in TRAINING_HEADS:
        if spec.head in terms:
            total = total + weights[spec.head] * terms[spec.head]
    return total, shipped


class TrainedModules(Protocol):
    """What a step's losses read off the run: the two modules it trains."""

    encoder: AbilityEncoder
    model: EffectModel


class StepLossComputer:
    """One prepared step's losses on the device, for training and validation
    alike."""

    def __init__(self, config: TrainEffectModelConfig, device: torch.device) -> None:
        self.config = config
        self.device = device
        #: bf16 autocast on CUDA (FR-095).
        self.autocast: bool = autocast_enabled(device)
        #: The training-only heads' weights in the backward total (FR-060a).
        self.weights: dict[Head, float] = {
            spec.head: spec.weight(config) for spec in TRAINING_HEADS
        }

    def _autocast(self):
        """bf16 autocast on CUDA (FR-095); a no-op context elsewhere."""
        return torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=self.autocast,
        )

    def device_loss(
        self, prepared: DeviceStep, modules: TrainedModules,
        materializer: Materializer, *, step: int | None = None,
        report_parts: bool = False,
    ) -> StepLoss:
        """One prepared step's losses: the device half.

        ``step`` is a training step's place in the run, for the noise on ``e``;
        ``report_parts`` keeps each field's term, stacked on the device.
        Autocast encloses ``materialize`` too, because the encoder's forward
        runs there; backward and the optimizer step stay outside it (bf16
        needs no grad scaler).
        """
        with self._autocast():
            batch, _surfaces, encoded = materializer.materialize(
                prepared.batch, modules.encoder, step=step,
            )
            hidden = modules.model(**batch)
            targets = prepared.targets
            gathered = modules.model.per_entity(gather_entities(
                hidden, targets.index.to(self.device, non_blocking=True),
            ))
            loss, parts = per_entity_loss(
                gathered, targets.fields_on(self.device, non_blocking=True),
                fields=prepared.fields, report_parts=report_parts,
            )
            terms = self.head_terms(prepared.heads, encoded, modules.model, hidden)
        total, shipped = combine_terms(loss, terms, self.weights)
        return StepLoss(
            total=total, shipped=shipped, parts=parts, terms=terms,
            pending_mlm=prepared.heads.get(Head.MLM), gathered=gathered.detach(),
        )

    def head_terms(
        self, targets: HeadTargets, encoded: EncodedBatch, model, hidden,
    ) -> dict[str, torch.Tensor]:
        """Every head's loss term but the MLM pass's, as device tensors; a head
        the host half prepared no targets for is not computed."""
        inputs = {HeadInput.TRUNK: hidden, HeadInput.ENCODED: encoded.matrix}
        terms: dict[str, torch.Tensor] = {}
        for spec in AUX_HEADS:
            given = targets.get(spec.head)
            source = inputs.get(spec.reads)
            if spec.loss is None or given is None or source is None:
                continue
            terms[spec.head] = spec.loss(
                model, source, *spec.to_device(given, self.device),
            )
        return terms

    def mlm_backward(self, loss: StepLoss, encoder, model) -> None:
        """The MLM term's forward and backward, after the main loss's backward
        has freed its activations: with both graphs alive at once, long texts
        overflowed 8 GB. The accumulated gradient is the same sum either way.
        """
        masked: MaskedTokens | None = loss.pending_mlm
        if masked is None:
            return
        loss.pending_mlm = None
        with self._autocast():
            token_hidden, token_targets = MlmMasker.forward(encoder, masked, self.device)
            term = mlm_loss(
                model.mlm_head(token_hidden).float(), token_targets, rows=masked.rows,
            )
        weight = AUX_HEAD[Head.MLM].weight(self.config)
        (weight * term / self.config.grad_accum).backward()
        loss.terms[Head.MLM] = term.detach()

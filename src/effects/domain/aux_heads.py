"""The heads beside the per-entity one, one row each.

A head is **shipped** or **training-only** (FR-060a, FR-063). A shipped head
joins the per-entity loss at unit weight, is what validation scores, and is
saved with the model. A training-only head shapes the encoder: it is added for
the backward pass alone, at its own weight, and filtered out of the saved
artifact (FR-076), so a loader must accept a checkpoint without it.

Every place that treats the heads differently — which a step prepares targets
for, how the terms combine, how each loss reads its input, which modules a
checkpoint drops — reads :data:`AUX_HEADS`, so a new head is a row here and a
module on :class:`~effects.domain.effect_model.EffectModel`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

import torch

from effects.domain.effect_losses import (
    api_loss,
    created_objects_loss,
    value_loss,
    verdict_loss,
)


class Head(StrEnum):
    """The heads beside the per-entity one, by the loss term each adds."""

    VERDICT = "verdict"
    CREATED_OBJECTS = "created_objects"
    VALUE = "value"
    MLM = "mlm"
    API = "api"


class HeadInput(StrEnum):
    """What a head's loss reads."""

    #: The trunk's output, one row per record.
    TRUNK = "trunk"
    #: The batch's encoded ``e`` matrix, one row per unique text.
    ENCODED = "encoded"
    #: A masked second encoder pass over the batch's texts, run after the main
    #: backward has freed the first pass's activations.
    MASKED_TOKENS = "masked_tokens"


def _verdict(model, hidden, target, mask) -> torch.Tensor:
    return verdict_loss(model.verdict(hidden).float(), target, mask)


def _created_objects(model, hidden, target, mask) -> torch.Tensor:
    return created_objects_loss(model.created_objects(hidden).float(), target, mask)


def _value(model, matrix, values, mask) -> torch.Tensor:
    return value_loss(model.value_head(matrix).float(), values, mask)


def _api(model, matrix, types, keys) -> torch.Tensor:
    keyed = model.param_key_head is not None
    return api_loss(
        model.api_type_head(matrix).float(), types,
        model.param_key_head(matrix).float() if keyed else None,
        keys if keyed else None,
    )


@dataclass(frozen=True, slots=True)
class HeadSpec:
    """One auxiliary head: what it reads, what it weighs, what it is made of."""

    head: Head
    reads: HeadInput
    #: The ``EffectModel`` modules that make up the head.
    modules: tuple[str, ...]
    #: The ``TrainEffectModelConfig`` field holding a training-only head's
    #: weight; None for a shipped head, which joins at unit weight.
    weight_field: str | None = None
    #: ``(model, input, *targets) -> loss``, the targets as the step's host half
    #: built them with :attr:`on_device` moved to the device. None for the MLM
    #: head, whose second encoder pass the training loop runs itself.
    loss: Callable[..., torch.Tensor] | None = None
    #: Which of the head's target tensors the loss reads on the device. The rest
    #: are masks and class indices it selects its rows by on the host, where
    #: they were built, so no step waits for the device to say which rows.
    on_device: tuple[int, ...] = (0,)
    #: Whether the head predicts the script-API vocabularies, and so exists only
    #: in a run that has them.
    needs_api_vocabulary: bool = False

    @property
    def shipped(self) -> bool:
        return self.weight_field is None

    def weight(self, config) -> float:
        """The head's weight in the backward pass."""
        return 1.0 if self.weight_field is None else getattr(config, self.weight_field)

    def to_device(self, targets: tuple, device) -> tuple:
        """``targets`` with the ones the loss reads on the device moved there."""
        return tuple(
            target.to(device, non_blocking=True) if index in self.on_device else target
            for index, target in enumerate(targets)
        )


#: Every auxiliary head, shipped ones first. The order is the order the terms
#: are computed and summed in, and float addition is not associative: reordered,
#: a run would no longer repeat bit for bit.
AUX_HEADS: tuple[HeadSpec, ...] = (
    HeadSpec(Head.VERDICT, HeadInput.TRUNK, ("verdict_head",), loss=_verdict),
    HeadSpec(
        Head.CREATED_OBJECTS, HeadInput.TRUNK, ("created_objects_head",),
        loss=_created_objects,
    ),
    HeadSpec(
        Head.VALUE, HeadInput.ENCODED, ("value_head",), weight_field="value_weight",
        loss=_value,
    ),
    HeadSpec(
        Head.MLM, HeadInput.MASKED_TOKENS, ("mlm_head",), weight_field="mlm_weight",
    ),
    HeadSpec(
        Head.API, HeadInput.ENCODED, ("api_type_head", "param_key_head"),
        weight_field="api_weight", loss=_api, on_device=(1,),
        needs_api_vocabulary=True,
    ),
)

AUX_HEAD: dict[Head, HeadSpec] = {spec.head: spec for spec in AUX_HEADS}
SHIPPED_HEADS: tuple[HeadSpec, ...] = tuple(spec for spec in AUX_HEADS if spec.shipped)
TRAINING_HEADS: tuple[HeadSpec, ...] = tuple(
    spec for spec in AUX_HEADS if not spec.shipped
)
#: The modules filtered out of the saved artifact (FR-076, FR-059).
TRAINING_ONLY_MODULES: tuple[str, ...] = tuple(
    module for spec in TRAINING_HEADS for module in spec.modules
)

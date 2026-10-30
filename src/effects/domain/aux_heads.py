"""The heads beside the per-entity one, one row each.

A shipped head joins the per-entity loss at unit weight and is saved; a
training-only head is added for backward alone, at its own weight, and is
filtered out of the saved artifact (FR-060a, FR-063, FR-076). A new head is a
row of :data:`AUX_HEADS` and a module on ``EffectModel``.
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
    #: A masked second encoder pass over the batch's texts.
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
    #: weight; None for a shipped head.
    weight_field: str | None = None
    #: ``(model, input, *targets) -> loss``; None for the MLM head, whose
    #: second encoder pass the training loop runs itself.
    loss: Callable[..., torch.Tensor] | None = None
    #: Which target tensors the loss reads on the device; the rest are masks
    #: and indices it selects rows by on the host.
    on_device: tuple[int, ...] = (0,)
    needs_api_vocabulary: bool = False

    @property
    def shipped(self) -> bool:
        return self.weight_field is None

    def weight(self, config) -> float:
        return 1.0 if self.weight_field is None else getattr(config, self.weight_field)

    def to_device(self, targets: tuple, device) -> tuple:
        return tuple(
            target.to(device, non_blocking=True) if index in self.on_device else target
            for index, target in enumerate(targets)
        )


#: Every auxiliary head, shipped ones first, in the order the terms are summed:
#: reordered, a run would no longer repeat bit for bit.
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
#: Their state-dict key prefixes: what saving drops and loading forgives.
TRAINING_ONLY_PREFIXES: tuple[str, ...] = tuple(
    f"{module}." for module in TRAINING_ONLY_MODULES
)

"""Each output term's loss denominator (FR-062, research T3).

A term's loss is divided by the loss its best constant prediction reaches on
a sample of the training stratum, clamped below at ``--deviance-floor``, so
every field trains on one scale: a field the constant already predicts well
does not vanish from the objective, and a few large fields do not dominate
it. Computed once at a run's startup and kept with the run, since a resumed
run must divide by the same numbers.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch

from effects.domain.aux_heads import Head
from effects.domain.effect_head_layout import (
    CREATED_TERMS,
    GLOBAL_LAYOUT,
    PER_ENTITY_TERMS,
    VERDICT_LAYOUT,
)
from effects.domain.loss_floor import PER_ENTITY

#: Each head's terms in the order its loss reads a per-term vector.
HEAD_TERMS: dict[str, tuple[str, ...]] = {
    PER_ENTITY: PER_ENTITY_TERMS,
    Head.VERDICT.value: VERDICT_LAYOUT.terms,
    Head.CREATED_OBJECTS.value: CREATED_TERMS,
    Head.GLOBAL.value: GLOBAL_LAYOUT.terms,
}


@dataclass(frozen=True)
class LossDenominators:
    """Every term's denominator, by head: its constant-prediction loss on the
    startup sample, never below ``floor``. A term the sample never supervised
    divides by ``floor``."""

    #: ``head -> term -> denominator``, each already at least ``floor``.
    values: Mapping[str, Mapping[str, float]]
    floor: float

    @classmethod
    def of(cls, losses: Mapping[str, Mapping[str, float]], *, floor: float) -> LossDenominators:
        """Clamp ``losses`` (:func:`~effects.domain.loss_floor.constant_predictor_losses`)
        below at ``floor``."""
        if not floor > 0:
            raise ValueError(f"--deviance-floor must be above zero, not {floor}")
        return cls(
            {
                str(head): {str(term): max(float(loss), floor) for term, loss in terms.items()}
                for head, terms in losses.items()
            },
            float(floor),
        )

    def of_term(self, head: str, term: str) -> float:
        return self.values.get(head, {}).get(term, self.floor)

    def reciprocals(self, head: str) -> torch.Tensor:
        """``1 / denominator`` for each of ``head``'s terms, in
        :data:`HEAD_TERMS` order: the ``scale`` its loss takes."""
        return torch.tensor([1.0 / self.of_term(head, term) for term in HEAD_TERMS[head]])

    def as_dict(self) -> dict:
        """A plain value for ``resume.pt`` and the checkpoint."""
        return {
            "floor": self.floor,
            "values": {head: dict(terms) for head, terms in self.values.items()},
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> LossDenominators:
        return cls(
            {head: dict(terms) for head, terms in data["values"].items()},
            float(data["floor"]),
        )


@dataclass(frozen=True)
class HeadScales:
    """The reciprocals of a run's denominators per head, on the training
    device, built once: the ``scale`` each shipped head's loss takes."""

    per_entity: torch.Tensor
    heads: Mapping[Head, torch.Tensor]

    @classmethod
    def of(cls, denominators: LossDenominators, device: torch.device) -> HeadScales:
        return cls(
            denominators.reciprocals(PER_ENTITY).to(device),
            {
                head: denominators.reciprocals(head.value).to(device)
                for head in (Head.VERDICT, Head.CREATED_OBJECTS, Head.GLOBAL)
            },
        )

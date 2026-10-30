"""The MLM pass (FR-060b): which of a batch's encoder inputs it masks, drawn
on the host, and the encoder forward over them on the device.

The mask draws only from the generator it is handed, after the step's own
draws, so a step's mask follows its place in the run wherever it is prepared.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch

from effects.application.text_staging import encode_by_length
from effects.domain.ability_encoder import AbilityEncoder
from effects.infrastructure.torch_device import pinned


@dataclass
class MaskedTokens:
    """The MLM pass's inputs, drawn on the host: ``positions`` are the
    flattened indices of the masked tokens, ``targets`` the ids that stood
    there."""

    inputs: dict[str, torch.Tensor]
    positions: torch.Tensor
    targets: torch.Tensor

    @property
    def rows(self) -> int:
        """The texts the pass masked tokens in: the MLM term's rows."""
        return int(self.inputs["token_ids"].shape[0])

    def pin(self) -> None:
        self.inputs = pinned(self.inputs)
        self.positions = pinned(self.positions)
        self.targets = pinned(self.targets)


@dataclass(frozen=True)
class MlmMasker:
    """The MLM pass: a second encoder forward over a batch's encoder inputs,
    so the effect head keeps reading the unmasked ``e``. The mask is drawn on
    the host, so the device never has to report which positions it picked."""

    pad_id: int
    mask_id: int

    def draw(
        self, inputs: dict[str, torch.Tensor] | None, mask_prob: float,
        rng: random.Random,
    ) -> MaskedTokens | None:
        """Which of ``inputs``' tokens the pass masks, or None when it masks
        none or there are no inputs. Draws from ``rng`` after the build's own
        draws.

        ``inputs`` are the batch's collated encoder inputs, the main pass's
        own: every tensor this replaces is a new one in a copy of the dict,
        so the main pass still reads them unmasked.
        """
        if inputs is None or mask_prob <= 0.0:
            return None
        batch = dict(inputs)
        ids = batch["token_ids"]
        generator = torch.Generator().manual_seed(rng.getrandbits(63))
        picks = (torch.rand(ids.shape, generator=generator) < mask_prob)
        picks &= batch["attention_mask"].bool()
        picks[:, 0] = False  # never [CLS]
        positions = picks.flatten().nonzero().squeeze(1)
        if positions.numel() == 0:
            return None
        targets = ids.flatten().index_select(0, positions)
        batch["token_ids"] = ids.masked_fill(picks, self.mask_id)
        # A masked number's value goes too, or its embedding hands it back.
        batch["numbers"] = batch["numbers"].masked_fill(picks, 0.0)
        batch["number_mask"] = batch["number_mask"].masked_fill(picks, 0)
        return MaskedTokens(inputs=batch, positions=positions, targets=targets)

    @staticmethod
    def forward(
        encoder: AbilityEncoder, masked: MaskedTokens, device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The device half: the encoder's outputs at the masked positions, and
        the ids that stood there."""
        _e, hidden = encode_by_length(encoder, masked.inputs, device, hidden=True)
        selected = hidden.reshape(-1, hidden.shape[-1]).index_select(
            0, masked.positions.to(device, non_blocking=True),
        )
        return selected, masked.targets.to(device, non_blocking=True)

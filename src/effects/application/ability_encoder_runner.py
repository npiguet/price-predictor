"""Runs a trained ability encoder over the corpus's ability lines, on the GPU.

Each distinct encoder input is encoded once, through the trainer's own path
(``TextStager``, then ``encode_by_length``). The surface follows the
checkpoint's vocabulary rather than a flag, so a cache can never be built on a
different surface from the model that reads it.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Callable, Sequence

import numpy as np
import torch

from effects.application.text_staging import TextStager, encode_by_length
from effects.domain.ability_encoder import (
    AbilityEncoder,
    AbilityEncoderConfig,
    TruncationLog,
    collate_lines,
    encoding_role_spans,
    encoding_text,
)
from effects.domain.ability_tokenizer import AbilityTokenizer, Surface
from effects.domain.provenance import SidecarLine
from effects.infrastructure.torch_device import default_device

logger = logging.getLogger(__name__)

#: Texts collated per ``encode_by_length`` call, so the host never holds the
#: whole tree's inputs padded at once.
BLOCK_TEXTS = 1024


class AbilityEncoderRunner:
    """A loaded encoder plus its text stager, loaded once for the whole run."""

    def __init__(
        self,
        config: AbilityEncoderConfig,
        state: dict,
        tokenizer: AbilityTokenizer,
        *,
        surface: Surface,
        device: torch.device | None = None,
        withheld_keyword: str | None = None,
        truncations: TruncationLog | None = None,
    ) -> None:
        self.device = device or default_device()
        self.encoder = AbilityEncoder(config).to(self.device)
        self.encoder.load_state_dict(state)
        # Dropout off, without which a cache would not be reproducible.
        self.encoder.eval()
        # Inference probability zero, so two runs write the same cache; the
        # withheld keyword expands as it did in training.
        self.stager = TextStager.for_inference(
            tokenizer, surface, withhold_keyword=withheld_keyword,
            truncations=truncations,
        )
        self.e_dim = config.e_dim

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint,
        tokenizer: AbilityTokenizer,
        *,
        surface: Surface,
        device: torch.device | None = None,
        truncations: TruncationLog | None = None,
    ) -> AbilityEncoderRunner:
        """``surface`` is the checkpoint's vocabulary's, which the tokenizer's
        own grammar may not be (``tokenizer_surface``)."""
        return cls(
            checkpoint.encoder_config,
            checkpoint.encoder_state,
            tokenizer,
            surface=surface,
            device=device,
            withheld_keyword=checkpoint.provenance.withheld_keyword,
            truncations=truncations,
        )

    def encode(
        self, units: Sequence[tuple[str, tuple]],
        keys_of: Callable[[str], Sequence] = lambda text: (),
    ) -> np.ndarray:
        """``(len(units), e_dim)``, one row per ``(text, role spans)``, in order.

        A text the window cuts short is reported under ``keys_of(text)``.
        """
        # At an expansion probability of zero the draws decide nothing.
        rng = random.Random(0)
        prepared = [
            self.stager.prepare(text, spans, rng=rng, keys_of=keys_of)
            for text, spans in units
        ]
        rows = np.zeros((len(units), self.e_dim), dtype=np.float32)
        order = sorted(range(len(prepared)), key=lambda row: len(prepared[row]))
        with torch.no_grad():
            for start in range(0, len(order), BLOCK_TEXTS):
                block = order[start:start + BLOCK_TEXTS]
                inputs = collate_lines(
                    [prepared[row] for row in block], self.stager.tokenizer.pad_id,
                )
                vectors, _hidden = encode_by_length(self.encoder, inputs, self.device)
                rows[block] = vectors.cpu().numpy()
        return rows

    def unit_of(self, line: SidecarLine, prose: str | None) -> tuple[str, tuple]:
        """``(text, role spans)`` one sidecar line is encoded from, given its
        converted prose."""
        surface = self.stager.surface
        text = encoding_text(line, prose, surface) or ""
        return text, encoding_role_spans(line, text, surface)

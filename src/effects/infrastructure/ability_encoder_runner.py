"""Runs a trained ability encoder over the corpus's ability lines, on the GPU.

``encode-abilities`` computes one vector per ability line for the whole corpus —
roughly a hundred thousand lines. The corpus repeats itself heavily (every
"Flying", every "deals 3 damage to any target"), so the command hands the
runner each distinct encoder input once for the whole tree, and the runner
encodes them through the trainer's own path: tokenized by ``TextStager``, sorted
by length and run by ``encode_by_length``, so a short line never pays for a long
one's padding.

The surface a line is encoded on follows the checkpoint's vocabulary rather than
a flag, so a cache can never be built on a different surface from the model that
reads it.
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
    surface_of,
)
from effects.domain.ability_tokenizer import (
    INFERENCE_KEYWORD_EXPAND_P,
    AbilityTokenizer,
)
from effects.domain.provenance import SidecarLine
from effects.infrastructure.torch_device import default_device

logger = logging.getLogger(__name__)

#: Texts collated per ``encode_by_length`` call. They are length-sorted first,
#: so a block pads to about its own texts' length; the bound keeps the host from
#: holding the whole tree's inputs padded at once, which at the corpus's longest
#: line would be most of a gigabyte.
BLOCK_TEXTS = 1024


class AbilityEncoderRunner:
    """A loaded encoder plus its text stager, reused across every source file.

    Load-once: the checkpoint is read and the weights moved to the device a
    single time for the whole run, rather than per card.
    """

    def __init__(
        self,
        config: AbilityEncoderConfig,
        state: dict,
        tokenizer: AbilityTokenizer,
        *,
        surface: str,
        device: torch.device | None = None,
        withheld_keyword: str | None = None,
        truncations: TruncationLog | None = None,
    ) -> None:
        self.device = device or default_device()
        self.encoder = AbilityEncoder(config).to(self.device)
        self.encoder.load_state_dict(state)
        # eval() turns dropout off, without which a cache would not be
        # reproducible. The encoder carries no noise of its own: noise on e is
        # a training-batch concern (FR-056) and never reaches this path.
        self.encoder.eval()
        # A known keyword stays the token it trained as and only an unknown one
        # expands (FR-012): the shared inference probability is zero, and the
        # cache is an artifact, so a sampled expansion would make two runs of
        # the same command disagree. The keyword the checkpoint trained without
        # expands as it did in training, its token dropped.
        self.stager = TextStager(
            tokenizer, surface, withhold_keyword=withheld_keyword,
            keyword_expand_p=INFERENCE_KEYWORD_EXPAND_P, truncations=truncations,
        )
        self.e_dim = config.e_dim

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint,
        tokenizer: AbilityTokenizer,
        *,
        vocab_path,
        device: torch.device | None = None,
        truncations: TruncationLog | None = None,
    ) -> AbilityEncoderRunner:
        return cls(
            checkpoint.encoder_config,
            checkpoint.encoder_state,
            tokenizer,
            surface=surface_of(vocab_path),
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
                # One transfer per block rather than per line.
                rows[block] = vectors.cpu().numpy()
        return rows

    def unit_of(self, line: SidecarLine, prose: list[str]) -> tuple[str, tuple]:
        """``(text, role spans)`` one sidecar line is encoded from."""
        from effects.infrastructure.sidecar_io import prose_for

        surface = self.stager.surface
        text = encoding_text(line, prose_for(line, prose), surface) or ""
        return text, encoding_role_spans(line, text, surface)

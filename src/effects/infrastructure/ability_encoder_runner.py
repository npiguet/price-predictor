"""Runs a trained ability encoder over sidecar lines, in batches, on the GPU.

``encode-abilities`` computes one vector per ability line for the whole corpus —
roughly a hundred thousand lines. A forward pass per line would spend nearly all
of its time on kernel launches for tensors of one row, so lines are batched, and
they are batched by *unique text*: the corpus repeats itself heavily (every
"Flying", every "deals 3 damage to any target"), and an encode of one text
serves every line that shares it.

The surface a line is encoded on follows the checkpoint's vocabulary rather than
a flag, so a cache can never be built on a different surface from the model that
reads it.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

from effects.domain.ability_encoder import (
    AbilityEncoder,
    AbilityEncoderConfig,
    collate_lines,
    encoding_text,
    prepare_line,
    surface_of,
)
from effects.domain.ability_tokenizer import AbilityTokenizer
from effects.domain.provenance import ProvenanceSidecar, SidecarLine

logger = logging.getLogger(__name__)

#: Lines per forward pass. The encoder is four layers at d_model 256 over lines
#: of a few dozen tokens, so this is far below the 8 GB budget even at the
#: longest lines the position table allows.
DEFAULT_BATCH_LINES = 256


class AbilityEncoderRunner:
    """A loaded encoder plus its tokenizer, reused across every source file.

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
        batch_lines: int = DEFAULT_BATCH_LINES,
    ) -> None:
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.encoder = AbilityEncoder(config).to(self.device)
        self.encoder.load_state_dict(state)
        # eval() matters beyond dropout here: the encoder adds noise to e while
        # training, and a cache built with it would not be reproducible.
        self.encoder.eval()
        self.tokenizer = tokenizer
        self.surface = surface
        self.batch_lines = batch_lines
        self.e_dim = config.e_dim

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint,
        tokenizer: AbilityTokenizer,
        *,
        vocab_path,
        device: torch.device | None = None,
    ) -> AbilityEncoderRunner:
        return cls(
            checkpoint.encoder_config,
            checkpoint.encoder_state,
            tokenizer,
            surface=surface_of(vocab_path),
            device=device,
        )

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """``(len(texts), e_dim)``, one row per text, in order.

        Keywords are expanded deterministically here (probability 1.0, no RNG):
        the cache is an artifact, and a sampled expansion would make two runs of
        the same command disagree.
        """
        if not texts:
            return np.zeros((0, self.e_dim), dtype=np.float32)
        rows: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_lines):
                chunk = texts[start : start + self.batch_lines]
                lines = [
                    prepare_line(
                        self.tokenizer,
                        self.tokenizer.expand_keywords(
                            self.tokenizer.tokenize(text), probability=1.0,
                        ),
                    )
                    for text in chunk
                ]
                batch = collate_lines(lines, self.tokenizer.pad_id)
                batch = {k: v.to(self.device) for k, v in batch.items()}
                vectors, _hidden = self.encoder(**batch)
                # One transfer per batch rather than per line.
                rows.append(vectors.cpu().numpy())
        return np.concatenate(rows).astype(np.float32)

    def encode_sidecar_lines(
        self, sidecar: ProvenanceSidecar, prose: list[str],
    ) -> np.ndarray:
        """``(n_lines, e_dim)`` for one source, row-aligned to its sidecar.

        Deduplicated by text before the forward pass: a card printing the same
        keyword on two faces encodes it once, and the two rows share the result.
        """
        texts = [
            self._text_for(line, prose) or "" for line in sidecar.lines
        ]
        unique = sorted(set(texts))
        index = {text: row for row, text in enumerate(unique)}
        encoded = self.encode_texts(unique)
        if not texts:
            return np.zeros((0, self.e_dim), dtype=np.float32)
        return encoded[[index[text] for text in texts]]

    def _text_for(self, line: SidecarLine, prose: list[str]) -> str | None:
        from effects.infrastructure.sidecar_io import prose_for

        return encoding_text(line, prose_for(line, prose), self.surface)

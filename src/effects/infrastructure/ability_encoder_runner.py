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
    encoding_role_spans,
    encoding_text,
    prepare_line,
    surface_of,
)
from effects.domain.ability_tokenizer import (
    INFERENCE_KEYWORD_EXPAND_P,
    AbilityTokenizer,
    withheld_keyword_tokens,
)
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
        withheld_keyword: str | None = None,
    ) -> None:
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.encoder = AbilityEncoder(config).to(self.device)
        self.encoder.load_state_dict(state)
        # eval() turns dropout off, without which a cache would not be
        # reproducible. The encoder carries no noise of its own: noise on e is
        # a training-batch concern (FR-056) and never reaches this path.
        self.encoder.eval()
        self.tokenizer = tokenizer
        self.surface = surface
        self.batch_lines = batch_lines
        self.e_dim = config.e_dim
        #: The keyword the checkpoint trained without: the cache reads it as
        #: its definition, as training did, never as a token nothing updated.
        self.withheld = withheld_keyword_tokens(withheld_keyword)

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
            withheld_keyword=checkpoint.provenance.withheld_keyword,
        )

    def encode_texts(
        self, texts: list[str], role_spans: list[tuple] | None = None,
    ) -> np.ndarray:
        """``(len(texts), e_dim)``, one row per text, in order.

        ``role_spans`` holds each text's spans where it is prose (FR-069).

        A known keyword stays the token it trained as and only an unknown one
        expands (FR-012): the shared inference probability is zero, and the
        cache is an artifact, so a sampled expansion would make two runs of the
        same command disagree. The withheld keyword expands as it did in
        training, its token dropped.
        """
        if not texts:
            return np.zeros((0, self.e_dim), dtype=np.float32)
        spans = role_spans or [()] * len(texts)
        rows: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_lines):
                end = start + self.batch_lines
                lines = [
                    self.prepare(text, span)
                    for text, span in zip(texts[start:end], spans[start:end])
                ]
                batch = collate_lines(lines, self.tokenizer.pad_id)
                batch = {k: v.to(self.device) for k, v in batch.items()}
                vectors, _hidden = self.encoder(**batch)
                # One transfer per batch rather than per line.
                rows.append(vectors.cpu().numpy())
        return np.concatenate(rows).astype(np.float32)

    def prepare(self, text: str, role_spans: tuple = ()):
        """One text as the encoder reads it, through the trainer's own path."""
        return prepare_line(self.tokenizer, self.tokenizer.encoding_tokens(
            text, role_spans=role_spans, withheld=self.withheld,
            probability=INFERENCE_KEYWORD_EXPAND_P,
        ))

    def encode_sidecar_lines(
        self, sidecar: ProvenanceSidecar, prose: list[str],
    ) -> np.ndarray:
        """``(n_lines, e_dim)`` for one source, row-aligned to its sidecar.

        Deduplicated by text and spans before the forward pass: a card
        printing the same keyword on two faces encodes it once, and the two
        rows share the result.
        """
        units = [self.unit_of(line, prose) for line in sidecar.lines]
        if not units:
            return np.zeros((0, self.e_dim), dtype=np.float32)
        unique = sorted(set(units))
        index = {unit: row for row, unit in enumerate(unique)}
        encoded = self.encode_texts(
            [text for text, _ in unique], [spans for _, spans in unique],
        )
        return encoded[[index[unit] for unit in units]]

    def unit_of(self, line: SidecarLine, prose: list[str]) -> tuple[str, tuple]:
        """``(text, role spans)`` one sidecar line is encoded from."""
        from effects.infrastructure.sidecar_io import prose_for

        text = encoding_text(line, prose_for(line, prose), self.surface) or ""
        return text, encoding_role_spans(line, text, self.surface)

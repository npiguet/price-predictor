"""Ability texts to the encoder's inputs, and the encoder pass over them.

The stager draws only from the generator it is handed, so a step's draws
follow its place in the run wherever it is prepared. The MLM pass over the
same inputs is ``mlm_masking``'s.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
import torch

from effects.domain.ability_encoder import (
    AbilityEncoder,
    EncodedLine,
    TruncationLog,
    collate_lines,
    encoding_role_spans,
    prepare_line,
)
from effects.domain.ability_tokenizer import (
    INFERENCE_KEYWORD_EXPAND_P,
    AbilityTokenizer,
    Surface,
    withheld_keyword_tokens,
)
from effects.domain.baselines import BaselineMasks
from effects.domain.provenance import SidecarLine
from effects.infrastructure.torch_device import pinned

#: Texts per length-sorted encoder pass. Padded to the batch's longest instead,
#: one 512-token chain makes every line cost 512 positions and the saved
#: activations overrun the 8 GB card.
ENCODE_CHUNK = 32


def _length_chunks(attention_mask: torch.Tensor) -> list[tuple[torch.Tensor, int]]:
    """``(rows, width)`` per encoder pass, rows grouped by length, on the host."""
    lengths = attention_mask.sum(dim=1)
    order = torch.argsort(lengths, stable=True)
    return [
        (rows, int(lengths[rows].max()))
        for rows in order.split(ENCODE_CHUNK)
    ]


def encode_by_length(
    encoder: AbilityEncoder, inputs: dict[str, torch.Tensor], device,
    *, hidden: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """``encoder(**inputs)`` in length-sorted groups, in the inputs' row order.

    Padding is masked out of attention, so grouping changes no output. With
    ``hidden`` the token outputs come back padded to the batch's width; what
    sits at a padding position is unspecified.
    """
    chunks = _length_chunks(inputs["attention_mask"])
    if len(chunks) == 1:
        batch = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
        vectors, tokens = encoder(**batch)
        return vectors, tokens if hidden else None
    parts, outputs = [], []
    for rows, width in chunks:
        batch = {
            k: v.index_select(0, rows)[:, :width].to(device, non_blocking=True)
            for k, v in inputs.items()
        }
        vectors, tokens = encoder(**batch)
        parts.append(vectors)
        if hidden:
            outputs.append(tokens)
    order = torch.cat([rows for rows, _ in chunks])
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(order.numel())
    vectors = torch.cat(parts).index_select(0, inverse.to(device, non_blocking=True))
    if not hidden:
        return vectors, None
    full = outputs[0].new_zeros(
        (order.numel(), inputs["attention_mask"].shape[1], outputs[0].shape[-1]),
    )
    for (rows, width), tokens in zip(chunks, outputs):
        full[rows.to(device, non_blocking=True), :width] = tokens
    return vectors, full


# ── the encoder's inputs ────────────────────────────────────────────────


@dataclass
class StagedTexts:
    """A batch's unique texts, staged for whatever produces their ``e``.

    At most one of ``encoder_inputs`` and ``taxonomy_vectors`` is set;
    neither, for the ``identity`` baseline.
    """

    #: Each text's row of the batch's ``e`` matrix.
    rows: dict[str, int]
    #: Empty for the two baselines that never reach the encoder.
    prepared: list[EncodedLine]
    #: None when the encoder is not called.
    encoder_inputs: dict[str, torch.Tensor] | None
    #: The ``taxonomy`` baseline's hashed ``e``, which no encoder computes.
    taxonomy_vectors: torch.Tensor | None

    @property
    def ordered(self) -> list[str]:
        """The texts in matrix-row order."""
        return sorted(self.rows, key=self.rows.__getitem__)

    def pin(self) -> None:
        self.encoder_inputs = pinned(self.encoder_inputs)
        self.taxonomy_vectors = pinned(self.taxonomy_vectors)


@dataclass(frozen=True)
class TextStager:
    """Ability texts to encoder inputs, the one way every caller agrees on."""

    tokenizer: AbilityTokenizer
    surface: Surface
    withhold_keyword: str | None = None
    keyword_expand_p: float = 0.0
    truncations: TruncationLog | None = None

    @classmethod
    def for_inference(
        cls, tokenizer: AbilityTokenizer, surface: Surface, *,
        withhold_keyword: str | None, truncations: TruncationLog | None = None,
    ) -> TextStager:
        """The stager every pass outside a training step encodes through:
        validation, evaluation and the cache. Expansion at the inference
        probability, so two passes read one text alike; the withheld keyword
        stays withheld, being a split rather than an augmentation."""
        return cls(
            tokenizer, surface, withhold_keyword=withhold_keyword,
            keyword_expand_p=INFERENCE_KEYWORD_EXPAND_P, truncations=truncations,
        )

    def stage(
        self, texts: Mapping[str, SidecarLine], *, masks: BaselineMasks, e_dim: int,
        rng: random.Random, keys_of: Callable[[str], list],
    ) -> StagedTexts:
        """``texts`` (each to its sidecar line) staged for this variant;
        ``keys_of`` names a truncated text by the keys that resolved to it."""
        if not texts:
            return StagedTexts({}, [], None, None)
        ordered = sorted(texts)
        rows = {text: row for row, text in enumerate(ordered)}
        if masks.identity_embedding:
            return StagedTexts(rows, [], None, None)
        if masks.taxonomy_embedding:
            return StagedTexts(rows, [], None, _taxonomy_vectors(ordered, texts, e_dim))
        prepared = self.prepare_texts(ordered, texts, rng=rng, keys_of=keys_of)
        return StagedTexts(
            rows, prepared, collate_lines(prepared, self.tokenizer.pad_id), None,
        )

    def prepare_texts(
        self, ordered: list[str], lines_of: Mapping[str, SidecarLine], *,
        rng: random.Random, keys_of: Callable[[str], list],
    ) -> list[EncodedLine]:
        """Each text through :meth:`prepare`, with the role spans of its
        sidecar line in ``lines_of``."""
        return [
            self.prepare(
                text, encoding_role_spans(lines_of.get(text), text, self.surface),
                rng=rng, keys_of=keys_of,
            )
            for text in ordered
        ]

    def prepare(
        self, text: str, role_spans: tuple, *, rng: random.Random,
        keys_of: Callable[[str], list],
    ) -> EncodedLine:
        """One text tokenized with ``role_spans`` and its keywords expanded; a
        truncation is reported under ``keys_of(text)``."""
        tokens = self.tokenizer.encoding_tokens(
            text, withheld=withheld_keyword_tokens(self.withhold_keyword),
            role_spans=role_spans, probability=self.keyword_expand_p, rng=rng,
        )
        line = prepare_line(self.tokenizer, tokens)
        if line.truncated and self.truncations is not None:
            self.truncations.note(text, keys_of(text))
        return line


def _taxonomy_vectors(
    ordered: list[str], lines: Mapping[str, SidecarLine], e_dim: int,
) -> torch.Tensor:
    """The taxonomy baseline's stand-in for ``e``: the hash embedding its cache
    writes, so the trained baseline and its cache agree."""
    from effects.domain.baselines import taxonomy_vector

    rows = np.stack([taxonomy_vector(lines[text], e_dim) for text in ordered])
    return torch.from_numpy(rows)

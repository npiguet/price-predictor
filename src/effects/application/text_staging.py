"""Ability texts to the encoder's inputs, and the MLM pass's mask over them.

Both collaborators draw only from the generator they are handed, so a step's
draws follow its place in the run wherever it is prepared.
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
from effects.domain.ability_tokenizer import AbilityTokenizer, Surface, withheld_keyword_tokens
from effects.domain.baselines import BaselineMasks
from effects.domain.provenance import SidecarLine

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


def pinned(value):
    """``value`` with every tensor in it copied to page-locked host memory, so
    a ``non_blocking`` copy does not hold the host."""
    if isinstance(value, torch.Tensor):
        return value.pin_memory()
    if isinstance(value, dict):
        return {key: pinned(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(pinned(item) for item in value)
    return value


# ── the encoder's inputs ────────────────────────────────────────────────


@dataclass
class StagedTexts:
    """A batch's unique texts, staged for whatever produces their ``e``.

    At most one of ``encoder_inputs`` and ``stand_in`` (the ``taxonomy``
    baseline's vectors) is set; neither, for the ``identity`` baseline.
    """

    rows: dict[str, int]
    prepared: list[EncodedLine]
    encoder_inputs: dict[str, torch.Tensor] | None
    stand_in: torch.Tensor | None

    @property
    def ordered(self) -> list[str]:
        return sorted(self.rows, key=self.rows.__getitem__)


@dataclass(frozen=True)
class TextStager:
    """Ability texts to encoder inputs, the one way every caller agrees on."""

    tokenizer: AbilityTokenizer
    surface: Surface
    withhold_keyword: str | None = None
    keyword_expand_p: float = 0.0
    truncations: TruncationLog | None = None

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
        self, ordered: list[str], lines_of: Mapping[str, SidecarLine] | None = None,
        *, rng: random.Random, keys_of: Callable[[str], list] = lambda text: [],
    ) -> list[EncodedLine]:
        """Each text through :meth:`prepare`, with the role spans of its
        sidecar line in ``lines_of``."""
        lines_of = lines_of or {}
        return [
            self.prepare(
                text, encoding_role_spans(lines_of.get(text), text, self.surface),
                rng=rng, keys_of=keys_of,
            )
            for text in ordered
        ]

    def prepare(
        self, text: str, role_spans: tuple = (), *, rng: random.Random,
        keys_of: Callable[[str], list] = lambda text: [],
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


# ── the MLM pass (FR-060b) ──────────────────────────────────────────────


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
    """The MLM pass: a second encoder forward over a batch's prepared lines,
    so the effect head keeps reading the unmasked ``e``. The mask is drawn on
    the host, so the device never has to report which positions it picked."""

    pad_id: int
    mask_id: int

    def draw(
        self, prepared: list[EncodedLine], mask_prob: float, rng: random.Random,
    ) -> MaskedTokens | None:
        """Which tokens the pass masks, or None when it masks none. Draws from
        ``rng`` after the build's own draws."""
        if not prepared or mask_prob <= 0.0:
            return None
        batch = collate_lines(prepared, self.pad_id)
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

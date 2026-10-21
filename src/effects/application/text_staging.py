"""Ability texts to the encoder's inputs, and the MLM pass's mask over them.

The batcher's two collaborators on the host side. :class:`TextStager` turns a
batch's unique ability texts into what the encoder reads — or, for the two
baselines that never reach the encoder, into what stands in for it — and
:class:`MlmMasker` draws which of those tokens the MLM pass hides. Both draw
from the generator they are handed, so a step's draws follow its place in the
run wherever it is prepared.
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
from effects.domain.ability_tokenizer import AbilityTokenizer, withheld_keyword_tokens
from effects.domain.variants import VariantMasks

#: Texts per encoder pass. A batch's texts are sorted by length and encoded in
#: groups this size, each padded to its own longest text: padded to the
#: batch's longest, one 512-token chain makes every short line in the batch cost
#: 512 positions, most of the encoder's input is padding, and the activations
#: a batch saves for backward overrun the 8 GB card.
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

    ``inputs`` is a host batch from ``collate_lines``. Padding is masked out of
    attention, so each text encodes as it would padded to any width. With
    ``hidden`` the token outputs come back padded to the batch's width as one
    tensor, matching the one-batch outputs at every real token; what sits at a
    padding position is unspecified, and the MLM pass never reads one.
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
    """``value`` with every tensor in it copied to page-locked host memory.

    A copy from pageable memory holds the host until the device has the bytes;
    from pinned memory a ``non_blocking`` copy lets it go on queueing work.
    Walks the dicts, lists and tuples a prepared batch is built from.
    """
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

    ``rows`` numbers the texts in matrix-row order. Exactly one of the rest is
    used: ``encoder_inputs`` for the encoder, over the ``prepared`` lines;
    ``stand_in``, the ``taxonomy`` baseline's vectors; or neither, for the
    ``identity`` baseline, whose table looks its rows up on the device.
    """

    rows: dict[str, int]
    prepared: list[EncodedLine]
    encoder_inputs: dict[str, torch.Tensor] | None
    stand_in: torch.Tensor | None

    @property
    def ordered(self) -> list[str]:
        """The texts in matrix-row order."""
        return sorted(self.rows, key=self.rows.__getitem__)


@dataclass(frozen=True)
class TextStager:
    """Ability texts to encoder inputs, the way every caller must agree on.

    The withheld keyword always expands and its token is dropped everywhere
    (``AbilityTokenizer.encoding_tokens``); a prose text reads its line's role
    spans (FR-069); a text the encoder's window cuts short is reported once per
    run, to ``truncations``.
    """

    tokenizer: AbilityTokenizer
    surface: str
    withhold_keyword: str | None = None
    keyword_expand_p: float = 0.0
    truncations: TruncationLog | None = None

    def stage(
        self, texts: Mapping[str, object], *, masks: VariantMasks, e_dim: int,
        rng: random.Random, keys_of: Callable[[str], list],
    ) -> StagedTexts:
        """``texts`` (each to its sidecar line) staged for this variant.

        Tokenizing draws from ``rng`` (keyword expansion), so it runs wherever
        the batch is prepared, and in the order batches are. ``keys_of`` names
        a truncated text by the keys that resolved to it.
        """
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
        self, ordered: list[str], lines_of: Mapping[str, object] | None = None,
        *, rng: random.Random, keys_of: Callable[[str], list] = lambda text: [],
    ) -> list[EncodedLine]:
        """Each text tokenized and its keywords expanded, the withheld one always.

        The withheld keyword's token is dropped wherever it is left, inside an
        expansion too, so the model never reads it. ``lines_of`` maps a text to
        its sidecar line, whose role spans a prose text is tokenized with.
        """
        withheld = withheld_keyword_tokens(self.withhold_keyword)
        lines_of = lines_of or {}
        lines = []
        for text in ordered:
            tokens = self.tokenizer.encoding_tokens(
                text, withheld=withheld,
                role_spans=encoding_role_spans(lines_of.get(text), text, self.surface),
                probability=self.keyword_expand_p, rng=rng,
            )
            line = prepare_line(self.tokenizer, tokens)
            if line.truncated and self.truncations is not None:
                self.truncations.note(text, keys_of(text))
            lines.append(line)
        return lines


def _taxonomy_vectors(
    ordered: list[str], lines: Mapping[str, object], e_dim: int,
) -> torch.Tensor:
    """The taxonomy baseline's stand-in for an encoded ``e``.

    Deterministic and unlearned — the same hash embedding the cache writes for
    this variant, so the trained baseline and its cache agree.
    """
    from effects.application.encode_abilities import taxonomy_vector

    rows = np.stack([taxonomy_vector(lines[text], e_dim) for text in ordered])
    return torch.from_numpy(rows)


# ── the MLM pass (FR-060b) ──────────────────────────────────────────────


@dataclass
class MaskedTokens:
    """The MLM pass's inputs, drawn on the host.

    ``inputs`` are the encoder's keyword arguments with the picked tokens
    replaced by ``[MASK]``, ``positions`` the flattened indices of those
    tokens, and ``targets`` the ids that stood there.
    """

    inputs: dict[str, torch.Tensor]
    positions: torch.Tensor
    targets: torch.Tensor

    @property
    def rows(self) -> int:
        """The texts the pass masked tokens in: the MLM term's rows."""
        return int(self.inputs["token_ids"].shape[0])

    def pin(self) -> None:
        """Every tensor in page-locked memory; see ``pinned``."""
        self.inputs = pinned(self.inputs)
        self.positions = pinned(self.positions)
        self.targets = pinned(self.targets)


@dataclass(frozen=True)
class MlmMasker:
    """The MLM pass: a second encoder forward over a batch's prepared lines.

    A share of their tokens is replaced by ``[MASK]`` — never ``[CLS]``, never
    padding — so the effect head keeps reading the unmasked ``e`` and the
    corruption shapes only what the encoder's token outputs must recover. The
    mask is drawn and the positions picked on the host, so the device never has
    to report which they were; and the vocabulary projection then runs over the
    masked positions alone rather than over every position of the batch.
    """

    pad_id: int
    mask_id: int

    def draw(
        self, prepared: list[EncodedLine], mask_prob: float, rng: random.Random,
    ) -> MaskedTokens | None:
        """Which tokens the pass masks, or None when it masks none.

        Draws from ``rng`` after the build's own draws, so a step prepared
        ahead of its turn takes the same draws as one prepared in place.
        """
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
        # A masked number's value goes too: the encoder adds the number
        # embedding beside the token's, and it would hand the answer back.
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

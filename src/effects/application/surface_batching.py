"""Records to model inputs: the one place that assembly is defined, shared by
the trainer and every evaluator so they cannot disagree.

:meth:`SurfaceBatcher.prepare` is the host half and needs no model;
:meth:`Materializer.materialize` is the device half.
"""

from __future__ import annotations

import dataclasses
import random
from collections.abc import Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import NamedTuple

import torch

from effects.application.ability_text import KeyResolver
from effects.application.identity_table import (
    IDENTITY_TABLE_KEY as IDENTITY_TABLE_KEY,  # re-exported: where importers look for it
)
from effects.application.identity_table import IdentityTable
from effects.application.input_masking import apply_masks
from effects.application.mlm_masking import MaskedTokens, MlmMasker
from effects.application.noise_state import NoiseState
from effects.application.text_resolution import StagedLines, TextResolution
from effects.application.text_staging import (
    StagedTexts,
    StageInputs,
    TextStager,
    encode_by_length,
)
from effects.domain.ability_encoder import AbilityEncoder
from effects.domain.ability_tokenizer import Surface
from effects.domain.baselines import BaselineMasks, baseline_masks
from effects.domain.grid_edits import TextOverride
from effects.domain.input_layout import SlotKind
from effects.domain.provenance import SidecarLine, SidecarLookup
from effects.domain.records import acting_keys
from effects.domain.slot_lines import slot_lines
from effects.domain.slot_widths import layout_widths
from effects.domain.surface_builder import NO_STRIP, SurfaceBuilder, board_entities
from effects.domain.surface_collation import TrunkInputs, collate_surfaces
from effects.domain.value_targets import ValueLine


@dataclass
class EncodedBatch:
    """What one batch encoded, for the heads that read ``e`` once per text.

    ``texts`` are in matrix-row order; ``matrix`` is after noise, in training.
    """

    texts: list[str]
    lines: Mapping[str, SidecarLine | None]
    matrix: torch.Tensor | None


@dataclass
class BatchLines:
    """The lines of one batch: its unique texts, each with the first line met,
    and every distinct line the value head reads a row of ``e`` for."""

    texts: dict[str, SidecarLine | None] = dataclasses.field(default_factory=dict)
    _value: dict[ValueLine, None] = dataclasses.field(default_factory=dict)

    def note(self, text: str, line: SidecarLine | None) -> None:
        self.texts.setdefault(text, line)
        self._value[ValueLine.of(text, line)] = None

    def value_lines(self) -> tuple[ValueLine, ...]:
        """The distinct lines in a fixed order, whichever was met first."""
        return tuple(sorted(self._value, key=ValueLine.order))


@dataclass
class PreparedBatch:
    """The host half of one ``build``: CPU tensors and plain structures only.

    ``e`` is never computed here: the head's loss reaches the encoder only
    through the rows ``materialize`` scatters, so a vector produced ahead of
    the step would carry no gradient.
    """

    #: Each unique ability text to its sidecar line; None for a text no line
    #: carries.
    texts: dict[str, SidecarLine | None]
    #: The same texts staged for whatever produces their ``e``.
    staged: StagedTexts
    surfaces: list
    #: The variant's masks already applied.
    inputs: TrunkInputs
    #: Each distinct line the batch's texts read, in a fixed order: the value
    #: head's rows, each reading the ``e`` of its text's row.
    value_lines: tuple[ValueLine, ...] = ()

    def pin(self) -> None:
        self.staged.pin()
        self.inputs = self.inputs.pin()


class Materialized(NamedTuple):
    """One batch on the device: the trunk's inputs, the surfaces they came
    from, and the batch's encoded texts."""

    inputs: TrunkInputs
    surfaces: list
    encoded: EncodedBatch


@dataclass
class Materializer:
    """The device half of a batch: whatever turns staged texts into ``e``.

    The encoder runs here so ``e`` is part of the step's graph, and the noise
    so its draws read the device generator in step order. ``noise`` is
    training's alone.
    """

    device: torch.device
    masks: BaselineMasks
    identity_table: IdentityTable | None = None
    noise: NoiseState | None = None

    def encode(self, staged: StagedTexts, encoder: AbilityEncoder) -> torch.Tensor | None:
        """The batch's ``e`` matrix, one row per staged text in row order, or
        None; ``identity`` and ``taxonomy`` never reach the encoder."""
        if self.masks.identity_embedding and staged.rows:
            if self.identity_table is None:
                raise ValueError(
                    "the identity baseline needs its embedding table; construct "
                    "the materializer with identity_table="
                )
            return self.identity_table(
                self.identity_table.rows(staged.ordered).to(self.device, non_blocking=True),
            )
        if staged.taxonomy_vectors is not None:
            return staged.taxonomy_vectors.to(self.device, non_blocking=True)
        if staged.encoder_inputs is None:
            return None
        vectors, _hidden = encode_by_length(encoder, staged.encoder_inputs, self.device)
        return vectors

    def materialize(
        self, prepared: PreparedBatch, encoder: AbilityEncoder, *,
        step: int | None = None,
    ) -> Materialized:
        """``prepared`` on the device, its texts encoded and scattered;
        ``step`` is the training step, which places the noise on its ramp."""
        matrix = self.encode(prepared.staged, encoder)
        if self.noise is not None:
            # Every e of the batch, whichever variant produced it (FR-056).
            matrix = self.noise.apply(matrix, step=step)
        # The one place the encoder's output enters the head.
        inputs = prepared.inputs.to(self.device, non_blocking=True).with_e(matrix)
        return Materialized(
            inputs, prepared.surfaces,
            EncodedBatch(texts=prepared.staged.ordered, lines=prepared.texts, matrix=matrix),
        )


@dataclass(frozen=True, slots=True)
class SurfaceSpec:
    """What one model's surfaces are laid out by: the width of ``e``, each
    slot kind's feature width, the variant's masks and the input layout.

    Built from the run's config in training and from the checkpoint wherever
    a saved model runs, so the two are assembled one way; the widths are the
    layout's (:func:`~effects.domain.slot_widths.layout_widths`) in both, a
    loaded checkpoint's layout being this code's (``require_input_layout``).
    """

    e_dim: int
    widths: dict[SlotKind, int]
    masks: BaselineMasks

    @classmethod
    def for_training(cls, config) -> SurfaceSpec:
        """A training run's, under the layout every run records."""
        return cls(config.e_dim, layout_widths(), baseline_masks(config.variant))

    @classmethod
    def for_saved_model(cls, checkpoint) -> SurfaceSpec:
        """A saved model's; the store has already refused a checkpoint
        trained on another input layout."""
        return cls(
            checkpoint.encoder_config.e_dim, layout_widths(),
            baseline_masks(checkpoint.variant),
        )


@dataclass(frozen=True, slots=True)
class StepDraws:
    """What a batcher draws with: its generator and the context-dropout
    probability, the two that move with the step."""

    context_dropout: float = 0.0
    #: The step's generator: keyword expansion, context dropout and the MLM
    #: mask draw from it, in that order. None is a generator seeded 0.
    rng: random.Random | None = None


class SurfaceBatcher:
    """Turns a list of records into the trunk's inputs.

    Owns the host half; a batcher built only to prepare steps, as a worker's
    is, has no ``materializer``.
    """

    def __init__(
        self,
        *,
        text: TextResolution,
        spec: SurfaceSpec,
        materializer: Materializer | None = None,
        draws: StepDraws = StepDraws(),
    ) -> None:
        #: How a key becomes its text; a step's batcher borrows the run's join.
        self.text = text
        self.spec = spec
        self.materializer = materializer
        self.context_dropout = draws.context_dropout
        self.rng = draws.rng or random.Random(0)

    @property
    def stager(self) -> TextStager:
        return self.text.stager

    @property
    def resolver(self) -> KeyResolver:
        return self.text.resolver

    @property
    def sidecars(self) -> SidecarLookup:
        return self.resolver.sidecars

    @property
    def surface(self) -> Surface:
        return self.stager.surface

    def text_of(self, key) -> str | None:
        """The text one ability key is encoded from, on this run's surface:
        its single definition. A key the sidecar does not describe raises."""
        return self.text.text_of(key)

    def line_of(self, key):
        return self.text.line_of(key)

    def override_text(self, edit: TextOverride) -> None:
        """Make ``edit.key`` read ``edit.text`` for this batcher's life
        (``TextResolution.override``)."""
        self.text.override(edit)

    def reading_as(self, key, text: str) -> AbstractContextManager[None]:
        """Inside the block ``key`` reads ``text``, keeping its own line."""
        return self.text.reading_as(key, text)

    def batch_texts(self, records) -> dict[str, SidecarLine | None]:
        """Every ability text this batch's surfaces reference, to its line."""
        return self.batch_lines(records).texts

    def batch_lines(self, records) -> BatchLines:
        """Every ability text this batch's surfaces reference: the acting line
        of each record and every ability on every card the surface lays out,
        command-zone objects included, since a text left out reaches the model
        as a zero vector."""
        found = BatchLines()
        for record in records:
            for key in acting_keys(record):
                if self._note(found, key) is not None:
                    break
            self._note_board(found, record)
        return found

    def _note_board(self, found: BatchLines, record) -> None:
        """The lines of ``record``'s board, and the modes that follow each."""
        for entity in board_entities(record):
            for line in slot_lines(entity, self.board_keys):
                self._note_with_modes(found, line.key)

    def _note_with_modes(self, found: BatchLines, key) -> None:
        """``key``'s text into ``found``, then its charm modes' where it has one."""
        if self._note(found, key) is None:
            return
        for mode in self.options_for(key):
            self._note(found, mode)

    def _note(self, found: BatchLines, key) -> str | None:
        """``key``'s text into ``found``, with its line; None for a key that
        reads no text."""
        text, line = self.text.resolve(key)
        if text is not None:
            found.note(text, line)
        return text

    def options_for(self, key) -> tuple:
        return self.text.options_for(key)

    def board_keys(self, key) -> tuple:
        return self.text.board_keys(key)

    def room_of(self, key) -> str | None:
        return self.text.room_of(key)

    def keyword_of_key(self, key) -> str | None:
        return self.text.keyword_of_key(key)

    def stage(self, texts: Mapping[str, SidecarLine | None]) -> StagedTexts:
        return self.stager.stage(texts, draw=StageInputs(
            self.spec.masks, self.spec.e_dim, self.rng, self.text.keys_of,
        ))

    def encode_texts(
        self, texts: Mapping[str, SidecarLine | None], encoder: AbilityEncoder,
    ) -> tuple[dict[str, int], torch.Tensor | None]:
        """Encode each unique ability text once: ``(row_of_text, matrix)``."""
        staged = self.stage(texts)
        return staged.rows, self.materializer.encode(staged, encoder)

    def draw_mlm_mask(
        self, inputs: dict[str, torch.Tensor] | None, mask_prob: float,
    ) -> MaskedTokens | None:
        """Which of a batch's collated encoder ``inputs`` the MLM pass masks,
        from this step's generator after the build's own draws
        (``MlmMasker.draw``)."""
        tokenizer = self.stager.tokenizer
        return MlmMasker(tokenizer.pad_id, tokenizer.mask_id).draw(
            inputs, mask_prob, self.rng,
        )

    def surface_builder(self, rows: dict[str, int]) -> SurfaceBuilder:
        """The builder of one batch's surfaces, each ``e`` a row of ``rows``;
        it draws context dropout from the batcher's generator."""
        return SurfaceBuilder(
            StagedLines(self.text, rows), e_dim=self.spec.e_dim, rng=self.rng,
            context_dropout=self.context_dropout,
        )

    def build(self, records, encoder: AbilityEncoder, strip_per_record=None) -> Materialized:
        """One batch of records on the device: ``prepare`` then ``materialize``.

        ``strip_per_record`` holds one ``stripped`` map per record
        (``NO_STRIP`` for one built as collected), because gate 2 batches a
        record with its perturbed twin.
        """
        return self.materializer.materialize(
            self.prepare(records, strip_per_record), encoder,
        )

    def prepare(self, records, strip_per_record=None) -> PreparedBatch:
        """``build``'s host half. Draws keyword expansion, then context
        dropout, from the batcher's generator, in that order."""
        found = self.batch_lines(records)
        texts = found.texts
        staged = self.stage(texts)
        strips = strip_per_record or [NO_STRIP] * len(records)
        builder = self.surface_builder(staged.rows)
        surfaces = [builder.build(record, strip) for record, strip in zip(records, strips)]
        inputs = apply_masks(collate_surfaces(
            surfaces, e_dim=self.spec.e_dim, widths=self.spec.widths,
        ), self.spec.masks)
        return PreparedBatch(
            texts=texts, staged=staged, surfaces=surfaces, inputs=inputs,
            value_lines=found.value_lines(),
        )

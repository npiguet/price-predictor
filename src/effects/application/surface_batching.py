"""Records to model inputs: the one place that assembly is defined, shared by
the trainer and every evaluator so they cannot disagree.

:meth:`SurfaceBatcher.prepare` is the host half and needs no model;
:meth:`Materializer.materialize` is the device half.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import NamedTuple

import torch

from effects.application.ability_text import KeyResolver
from effects.application.text_staging import (
    MaskedTokens,
    MlmMasker,
    StagedTexts,
    TextStager,
    encode_by_length,
    pinned,
)
from effects.domain.ability_encoder import AbilityEncoder, EncodedLine
from effects.domain.ability_tokenizer import Surface
from effects.domain.baselines import BaselineMasks
from effects.domain.damage_step_keywords import keyword_of_line
from effects.domain.effect_head_input import (
    ACT_SLOT,
    INPUT_LAYOUT,
    KIND_FLAG_WIDTH,
    InputLayout,
    SlotKind,
    build_effect_head_input,
    continuous_masked_keywords,
)
from effects.domain.provenance import SidecarLookup
from effects.domain.records import PlayabilityDecisionPayload, acting_keys
from effects.domain.surface_collation import collate_surfaces, scatter_e_rows

#: Running-covariance decay for the noise on ``e`` (FR-056).
NOISE_DECAY = 0.99

#: Steps between two factorizations of the running covariance: a GPU
#: eigendecomposition stalls the step on the host. ``Σ`` still updates every
#: step.
NOISE_REFACTOR_EVERY = 10

class NoiseState:
    """Gaussian noise on ``e`` scaled to the spread of ``e`` itself (FR-056).

    The noise has covariance ``r² Σ``: ``Σ`` a running covariance of the
    batches' ``e`` with the gradient stopped, ``r`` rising linearly from 0 to
    the ratio over the first ``ramp_steps`` optimizer steps.

    Update and draw run outside autocast and in float64: a bf16 covariance is
    neither symmetric nor positive semi-definite, and its factorization fails
    within the first steps.
    """

    def __init__(
        self, ratio: float, ramp_steps: int, *,
        decay: float = NOISE_DECAY,
        refactor_every: int = NOISE_REFACTOR_EVERY,
    ) -> None:
        self.ratio = ratio
        self.ramp_steps = max(int(ramp_steps), 1)
        self.decay = decay
        self.sigma: torch.Tensor | None = None
        #: The training step of the last batch, kept for the resume file.
        self.step = 0
        self.refactor_every = max(int(refactor_every), 1)
        #: ``V √Λ`` of the last factorized ``Σ`` and the draw it was taken at.
        self._factor: torch.Tensor | None = None
        self._draws_since_factor = 0

    def state_dict(self) -> dict:
        return {
            "sigma": self.sigma, "step": self.step,
            "factor": self._factor, "draws_since_factor": self._draws_since_factor,
        }

    def load_state_dict(self, state: dict, device: torch.device) -> None:
        def on(tensor):
            return tensor.to(device) if tensor is not None else None

        self.sigma = on(state["sigma"])
        self.step = state["step"]
        self._factor = on(state["factor"])
        self._draws_since_factor = state["draws_since_factor"]

    def scale(self, step: int) -> float:
        """``r`` at training step ``step``."""
        return self.ratio * min(1.0, step / self.ramp_steps)

    def update(self, matrix: torch.Tensor) -> None:
        """Fold one batch's covariance into ``Σ``."""
        if matrix.shape[0] < 2:
            return
        with torch.no_grad(), _exact(matrix):
            vectors = matrix.detach().double()
            centred = vectors - vectors.mean(dim=0, keepdim=True)
            covariance = centred.T @ centred / (vectors.shape[0] - 1)
            covariance = (covariance + covariance.T) / 2
            if self.sigma is None:
                self.sigma = covariance
            else:
                self.sigma = (
                    self.decay * self.sigma + (1.0 - self.decay) * covariance
                )

    def apply(self, matrix: torch.Tensor | None, *, step: int) -> torch.Tensor | None:
        """``matrix`` plus ``L z``, where ``L Lᵀ = r²Σ``, at training step
        ``step``; a batch with no ``e`` still records its step.

        ``L = V √Λ`` with eigenvalues clamped at zero, so a singular ``Σ``
        (a batch narrower than ``e_dim``) never fails to factorize.
        """
        self.step = step
        if matrix is None:
            return None
        self.update(matrix)
        r = self.scale(step)
        if r <= 0.0 or self.sigma is None:
            return matrix
        with torch.no_grad(), _exact(matrix):
            if self._factor is None or self._draws_since_factor >= self.refactor_every:
                values, vectors = torch.linalg.eigh(self.sigma)
                self._factor = vectors * values.clamp(min=0.0).sqrt()
                self._draws_since_factor = 0
            self._draws_since_factor += 1
            factor = r * self._factor
            draw = torch.randn(
                matrix.shape[0], factor.shape[0],
                device=matrix.device, dtype=factor.dtype,
            ) @ factor.T
        return matrix + draw.to(matrix.dtype)


def _exact(matrix: torch.Tensor):
    """A region where autocast does not lower the noise arithmetic's precision."""
    return torch.autocast(device_type=matrix.device.type, enabled=False)


@dataclass
class EncodedBatch:
    """What one batch encoded, for the heads that read ``e`` once per text.

    ``texts`` are in matrix-row order; ``matrix`` is after noise, in training.
    """

    texts: list[str]
    lines: dict[str, object]
    matrix: torch.Tensor | None


@dataclass
class PreparedBatch:
    """The host half of one ``build``: CPU tensors and plain structures only.

    ``e`` is never computed here: the head's loss reaches the encoder only
    through the rows ``materialize`` scatters, so a vector produced ahead of
    the step would carry no gradient.
    """

    #: Each unique ability text to its sidecar line.
    texts: dict[str, object]
    #: Each text's row of the batch's ``e`` matrix.
    rows: dict[str, int]
    #: Empty for the two baselines that never reach the encoder.
    prepared: list[EncodedLine]
    #: None when the encoder is not called.
    encoder_inputs: dict[str, torch.Tensor] | None
    #: The ``taxonomy`` baseline's vectors, which stand in for the encoder.
    stand_in: torch.Tensor | None
    surfaces: list
    #: The variant's masks already applied.
    inputs: dict

    def pin(self) -> None:
        self.encoder_inputs = pinned(self.encoder_inputs)
        self.stand_in = pinned(self.stand_in)
        self.inputs = pinned(self.inputs)


class Materialized(NamedTuple):
    """One batch on the device: the trunk's keyword arguments, the surfaces
    they came from, and the batch's encoded texts."""

    inputs: dict
    surfaces: list
    encoded: EncodedBatch


#: Where a checkpoint's ``extra`` holds the identity baseline's table.
IDENTITY_TABLE_KEY = "identity_table"


class IdentityTable(torch.nn.Module):
    """The ``identity`` baseline's free ``e`` per ability text, indexed exactly.

    One row per text ``ability_text.known_texts`` yields, plus one row every
    other text shares (a counterfactual copy's, say). Never a hash: two texts
    sharing a row is the memorization the baseline exists to measure. The
    texts ride in the checkpoint beside the weights (:meth:`state`).
    """

    UNKNOWN_ROW = 0

    def __init__(self, texts: Iterable[str], e_dim: int) -> None:
        super().__init__()
        self.texts: tuple[str, ...] = tuple(sorted(set(texts)))
        self._rows = {text: row for row, text in enumerate(self.texts, start=1)}
        self.embedding = torch.nn.Embedding(len(self.texts) + 1, e_dim)

    def rows(self, texts: Iterable[str]) -> torch.Tensor:
        """Each text's row, on the host; the unknown row for a text not indexed."""
        return torch.tensor(
            [self._rows.get(text, self.UNKNOWN_ROW) for text in texts],
            dtype=torch.long,
        )

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        return self.embedding(rows)

    def state(self) -> dict:
        return {"texts": list(self.texts), "weight": self.embedding.weight.detach().cpu()}

    @classmethod
    def from_state(cls, state: dict) -> IdentityTable:
        table = cls(state["texts"], int(state["weight"].shape[1]))
        table.embedding.weight.data.copy_(state["weight"])
        return table


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

    def encode(
        self, ordered: list[str], inputs: dict[str, torch.Tensor] | None,
        stand_in: torch.Tensor | None, encoder: AbilityEncoder,
    ) -> torch.Tensor | None:
        """The batch's ``e`` matrix, one row per text of ``ordered``, or None;
        ``identity`` and ``taxonomy`` never reach the encoder."""
        if self.masks.identity_embedding and ordered:
            if self.identity_table is None:
                raise ValueError(
                    "the identity baseline needs its embedding table; construct "
                    "the materializer with identity_table="
                )
            return self.identity_table(
                self.identity_table.rows(ordered).to(self.device, non_blocking=True),
            )
        if stand_in is not None:
            return stand_in.to(self.device, non_blocking=True)
        if inputs is None:
            return None
        vectors, _hidden = encode_by_length(encoder, inputs, self.device)
        return vectors

    def materialize(
        self, prepared: PreparedBatch, encoder: AbilityEncoder, *,
        step: int | None = None,
    ) -> Materialized:
        """``prepared`` on the device, its texts encoded and scattered;
        ``step`` is the training step, which places the noise on its ramp."""
        rows = prepared.rows
        ordered = sorted(rows, key=rows.__getitem__)
        matrix = self.encode(ordered, prepared.encoder_inputs, prepared.stand_in, encoder)
        if self.noise is not None:
            # Every e of the batch, whichever variant produced it (FR-056).
            matrix = self.noise.apply(matrix, step=step)
        inputs = prepared.inputs
        batch = {
            "slot_features": {
                k: v.to(self.device, non_blocking=True)
                for k, v in inputs["slot_features"].items()
            },
            **{
                k: v.to(self.device, non_blocking=True)
                for k, v in inputs.items() if k != "slot_features"
            },
        }
        e_rows = batch.pop("e_rows")
        if matrix is not None:
            # The one place the encoder's output enters the head.
            batch["e_vectors"] = scatter_e_rows(batch["e_vectors"], e_rows, matrix)
        return Materialized(
            batch, prepared.surfaces,
            EncodedBatch(texts=ordered, lines=prepared.texts, matrix=matrix),
        )


class SurfaceBatcher:
    """Turns a list of records into the trunk's keyword arguments.

    Owns the host half; a batcher built only to prepare steps, as a worker's
    is, has no ``materializer``.
    """

    def __init__(
        self,
        *,
        stager: TextStager,
        resolver: KeyResolver,
        masks: BaselineMasks,
        e_dim: int,
        widths: dict[SlotKind, int],
        materializer: Materializer | None = None,
        context_dropout: float = 0.0,
        rng: random.Random | None = None,
        input_layout: InputLayout = INPUT_LAYOUT,
    ) -> None:
        self.stager = stager
        #: The memoised key-to-text join; a step's batcher borrows the run's.
        self.resolver = resolver
        self.masks = masks
        self.e_dim = e_dim
        self.widths = widths
        self.materializer = materializer
        self.context_dropout = context_dropout
        #: The step's generator: keyword expansion, context dropout and the MLM
        #: mask draw from it, in that order.
        self.rng = rng or random.Random(0)
        #: Training's own, or the one a loaded checkpoint recorded (FR-063c).
        self.input_layout = input_layout
        #: ``key -> (text, line)`` for :meth:`stand_in` keys, apart from the
        #: resolver's memo because a copy's text belongs to its step alone.
        self._staged: dict[object, tuple[str, object]] = {}

    @property
    def sidecars(self) -> SidecarLookup:
        return self.resolver.sidecars

    @property
    def surface(self) -> Surface:
        return self.stager.surface

    def _resolve(self, key) -> tuple[str | None, object]:
        """``(text, line)`` for one key: a staged stand-in, else the join."""
        staged = self._staged.get(key)
        return staged if staged is not None else self.resolver.resolve(key)

    def text_of(self, key) -> str | None:
        """The text one ability key is encoded from, on this run's surface:
        its single definition. A key the sidecar does not describe raises."""
        return self._resolve(key)[0]

    def line_of(self, key):
        return self._resolve(key)[1]

    def stand_in(
        self, key, text: str, like, *, script: str | None = None,
        role_spans: tuple | None = None,
    ) -> None:
        """Make ``key`` read ``text`` for this batcher's life, with ``like``'s
        line, its ``script`` and ``role_spans`` replaced where given (FR-063e)."""
        line = self.line_of(like)
        if line is not None and (script is not None or role_spans is not None):
            line = replace(
                line,
                script_text=line.script_text if script is None else script,
                role_spans=line.role_spans if role_spans is None else role_spans,
            )
        self._staged[key] = (text, line)

    @contextmanager
    def reading_as(self, key, text: str) -> Iterator[None]:
        """Inside the block ``key`` reads ``text``, keeping its own line: how a
        probe has the encoder read one edited script."""
        previous = self._staged.get(key)
        self.stand_in(key, text, like=key)
        try:
            yield
        finally:
            if previous is None:
                del self._staged[key]
            else:
                self._staged[key] = previous

    def batch_texts(self, records) -> dict[str, object]:
        """Every ability text this batch's surfaces reference, to its line:
        the acting line of each record and every ability on every entity, since
        a text left out reaches the model as a zero vector."""
        texts: dict[str, object] = {}

        def note(key) -> str | None:
            text, line = self._resolve(key)
            if text is not None:
                texts.setdefault(text, line)
            return text

        for record in records:
            for key in acting_keys(record):
                if note(key) is not None:
                    break
            for entity in record.state.entities:
                for key in (*entity.printed, *entity.granted_attached):
                    if note(key) is not None:
                        for mode in self.options_for(key):
                            note(mode)
        return texts

    def options_for(self, key) -> tuple:
        """The keys of the charm-mode lines that follow ``key``'s line
        (FR-063a); none for a stood-in key."""
        return () if key in self._staged else self.resolver.options_for(key)

    def _keys_of(self, text: str) -> list:
        """The keys resolved to ``text`` so far: how a truncation is named."""
        staged = [key for key, (known, _) in self._staged.items() if known == text]
        return self.resolver.keys_of(text) + staged

    def stage(self, texts: Mapping[str, object]) -> StagedTexts:
        return self.stager.stage(
            texts, masks=self.masks, e_dim=self.e_dim, rng=self.rng,
            keys_of=self._keys_of,
        )

    def encode_texts(
        self, texts: dict[str, object], encoder: AbilityEncoder,
    ) -> tuple[dict[str, int], torch.Tensor | None]:
        """Encode each unique ability text once: ``(row_of_text, matrix)``."""
        staged = self.stage(texts)
        return staged.rows, self.materializer.encode(
            staged.ordered, staged.encoder_inputs, staged.stand_in, encoder,
        )

    def draw_mlm_mask(
        self, prepared: list[EncodedLine], mask_prob: float,
    ) -> MaskedTokens | None:
        """Which of ``prepared``'s tokens the MLM pass masks, from this step's
        generator after the build's own draws (``MlmMasker.draw``)."""
        tokenizer = self.stager.tokenizer
        return MlmMasker(tokenizer.pad_id, tokenizer.mask_id).draw(
            prepared, mask_prob, self.rng,
        )

    def keyword_of_key(self, key) -> str | None:
        """Which damage-step keyword one ability key's line *is*, if any."""
        return keyword_of_line(self._resolve(key)[1])

    def surface_for(
        self,
        record,
        rows: dict[str, int],
        strip_keywords: dict[str, frozenset[str]] | None = None,
    ):
        """One record's token surface.

        ``strip_keywords`` is gate 2's perturbation: per entity, the keywords
        to remove from that entity's input through both channels.
        """

        def e_for(key):
            text = self.text_of(key)
            return None if text is None else rows.get(text)

        def has_line(key) -> bool:
            # A dropped key gets no token; a line a variant masks keeps its
            # token, zeroed in ``prepare``, so the geometry is the model's.
            return self.text_of(key) is not None

        return build_effect_head_input(
            record,
            e_for=e_for,
            has_line=has_line,
            e_dim=self.e_dim,
            # A decision record's [ACT] reads its candidate's e (FR-060a).
            candidate_index=(
                0 if isinstance(record.payload, PlayabilityDecisionPayload)
                else None
            ),
            context_dropout=self.context_dropout,
            rng=self.rng,
            masked_keywords=continuous_masked_keywords(record),
            stripped_keywords=strip_keywords,
            keyword_of=self.keyword_of_key,
            options_for=self.options_for,
            layout=self.input_layout,
        )

    def build(self, records, encoder: AbilityEncoder, strip_per_record=None) -> Materialized:
        """One batch of records on the device: ``prepare`` then ``materialize``.

        ``strip_per_record`` holds one ``strip_keywords`` map (or None) per
        record, because gate 2 batches a record with its perturbed twin.
        """
        return self.materializer.materialize(
            self.prepare(records, strip_per_record), encoder,
        )

    def prepare(self, records, strip_per_record=None) -> PreparedBatch:
        """``build``'s host half. Draws keyword expansion, then context
        dropout, from the batcher's generator, in that order."""
        texts = self.batch_texts(records)
        staged = self.stage(texts)
        strips = strip_per_record or [None] * len(records)
        surfaces = [
            self.surface_for(record, staged.rows, strip)
            for record, strip in zip(records, strips)
        ]
        inputs = self._masked(collate_surfaces(
            surfaces, e_dim=self.e_dim, widths=self.widths,
        ))
        return PreparedBatch(
            texts=texts, rows=staged.rows, prepared=staged.prepared,
            encoder_inputs=staged.encoder_inputs, stand_in=staged.stand_in,
            surfaces=surfaces, inputs=inputs,
        )

    def _masked(self, inputs: dict) -> dict:
        """The collated inputs with the variant's masks applied (FR-094),
        zeroed rather than dropped so the per-entity targets keep their shape."""
        masks = self.masks
        features = inputs["slot_features"]
        if masks.zero_state:
            for kind in (SlotKind.PLAYER, SlotKind.CARD):
                features[kind] = torch.zeros_like(features[kind])
            features[SlotKind.GLOBAL][..., :-KIND_FLAG_WIDTH] = 0.0
        if masks.zero_act_e or masks.zero_context_e:
            zeroed = torch.full(
                (inputs["e_rows"].shape[1],), masks.zero_context_e, dtype=torch.bool,
            )
            if zeroed.numel() > ACT_SLOT:
                zeroed[ACT_SLOT] = masks.zero_act_e
            inputs["e_rows"][:, zeroed] = -1
            inputs["e_vectors"][:, zeroed] = 0.0
        return inputs

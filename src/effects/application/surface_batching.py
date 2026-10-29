"""Records to model inputs: the one place that assembly is defined.

The trainer and the gate-1 evaluator both need it, and they must agree exactly.
When they did not — one keying the batch's encodings by prose, the other looking
them up by script — the effect head trained on an all-zero ``e`` and every loss
still fell, so nothing about the run looked wrong. Sharing the code is what
stops that recurring, and it is why the encoding text has a single definition
here rather than one in each caller.

The variant masks live here too, for the same reason: a baseline that differs
from the shipping model in any way other than its masked inputs is not a
baseline, and gate 1 is decided by comparing the two.

A batch is built in two halves. :meth:`SurfaceBatcher.prepare` is the host
half — the texts staged (:class:`~effects.application.text_staging.TextStager`),
the surfaces built and collated, the masks applied — and needs no model.
:meth:`Materializer.materialize` is the device half: the copies, the encoder,
the noise on ``e``, the rows scattered into the head's input. It returns what
it computed rather than leaving it on an object for a later read.
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
from effects.domain.records import PlayabilityDecisionPayload, acting_keys
from effects.domain.surface_collation import collate_surfaces, scatter_e_rows

#: Running-covariance decay for the noise on ``e`` (FR-056).
NOISE_DECAY = 0.99

#: Steps between two factorizations of the running covariance. A GPU
#: eigendecomposition stalls the step on the host, which cost about a tenth of
#: a training step when taken every step. ``Σ`` itself still updates every
#: step; at decay 0.99 ten steps move it by about a tenth of the way towards
#: the recent batches, so the factor drawn from lags it by at most that much.
NOISE_REFACTOR_EVERY = 10

class NoiseState:
    """Gaussian noise on ``e`` scaled to the spread of ``e`` itself (FR-056).

    A fixed σ meant different things at different points of a run: the scale
    of ``e`` is learned, so a σ that is a nudge at step zero is nothing once the
    vectors have grown. Here the noise has covariance ``r² Σ``, where ``Σ`` is a
    running average of the covariance of the batch's ``e`` vectors about their
    mean, held with the gradient stopped, and ``r`` rises linearly from 0 to the
    ratio over the first ``ramp_steps`` optimizer steps.

    Owned by the training loop and handed to the training steps' materializer;
    validation, evaluation and encoding build theirs without one, so nothing
    outside training is ever perturbed. Everything stays on the device:
    the update is one covariance, the draw one eigendecomposition and one
    matmul, and nothing is read back to the host.

    Both run outside autocast and in float64. Training runs under bf16
    autocast, which would otherwise compute the covariance in bf16: the result
    is neither symmetric nor positive semi-definite, and its factorization
    fails within the first steps. ``e_dim`` is small (64 by default), so the
    double-precision work is negligible next to a forward pass.
    """

    def __init__(
        self, ratio: float, ramp_steps: int, *,
        decay: float = NOISE_DECAY,
        refactor_every: int = NOISE_REFACTOR_EVERY,
    ) -> None:
        self.ratio = ratio
        self.ramp_steps = max(int(ramp_steps), 1)
        self.decay = decay
        #: The running covariance, seeded from the first batch.
        self.sigma: torch.Tensor | None = None
        #: Optimizer steps taken; set by the training loop before each batch.
        self.step = 0
        self.refactor_every = max(int(refactor_every), 1)
        #: ``V √Λ`` of the last factorized ``Σ`` and the draw it was taken at.
        self._factor: torch.Tensor | None = None
        self._draws_since_factor = 0

    def state_dict(self) -> dict:
        """Everything a resumed run needs to draw the noise the next step would."""
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

    def scale(self) -> float:
        """``r`` at the current step."""
        return self.ratio * min(1.0, self.step / self.ramp_steps)

    def update(self, matrix: torch.Tensor) -> None:
        """Fold one batch's covariance into ``Σ``. No gradient flows here."""
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

    def apply(self, matrix: torch.Tensor) -> torch.Tensor:
        """``matrix`` plus ``L z``, where ``L Lᵀ = r²Σ``.

        ``L = V √Λ`` from the eigendecomposition of ``Σ``, eigenvalues clamped
        at zero: exact on a singular ``Σ``, which a batch narrower than
        ``e_dim`` produces, and never a factorization failure.
        """
        self.update(matrix)
        r = self.scale()
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
    """What one batch encoded, for the heads that read ``e`` directly.

    ``texts`` are the batch's unique ability texts in matrix-row order,
    ``lines`` each text's sidecar line, and ``matrix`` the ``e`` rows the
    surfaces scattered (after noise, in training). The value and script-API
    heads read these rows, one per text rather than one per slot.
    """

    texts: list[str]
    lines: dict[str, object]
    matrix: torch.Tensor | None


@dataclass
class PreparedBatch:
    """The host half of one ``build``: everything that does not need the model.

    CPU tensors and plain structures only, so it can be assembled on a thread
    other than the one driving the device. What is left for ``materialize`` is
    the part the model has to be present for: moving these to the device,
    encoding the texts and scattering the rows. ``e`` is never computed here —
    the head's loss reaches the encoder only through the rows ``materialize``
    scatters, so a vector produced ahead of the step would carry no gradient.
    """

    #: Each unique ability text to its sidecar line.
    texts: dict[str, object]
    #: Each text's row of the batch's ``e`` matrix.
    rows: dict[str, int]
    #: ``EncodedLine`` per text in row order; empty for the two baselines
    #: that never reach the encoder.
    prepared: list[EncodedLine]
    #: ``collate_lines`` over ``prepared``, or None when the encoder is not
    #: called.
    encoder_inputs: dict[str, torch.Tensor] | None
    #: The ``taxonomy`` baseline's vectors, which stand in for the encoder.
    stand_in: torch.Tensor | None
    surfaces: list
    #: ``collate_surfaces``' output, the variant's masks applied.
    inputs: dict

    def pin(self) -> None:
        """Every tensor ``materialize`` copies, in page-locked memory."""
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

    One row per text the run can see — every sidecar line, spelled as the
    key-to-text join spells it (``ability_text.known_texts``) — plus one row
    every other text shares, such as a counterfactual copy's edited text. An
    exact index rather than a hash: a hash into a fixed table put about a
    quarter of the corpus's texts on a row another text also trained, which is
    exactly the memorization the baseline exists to measure.

    The texts ride in the checkpoint beside the weights (:meth:`state`), so a
    loaded baseline reads each text's own vector.
    """

    #: The row every text outside the index reads.
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
        """The texts and their vectors: what a checkpoint keeps."""
        return {"texts": list(self.texts), "weight": self.embedding.weight.detach().cpu()}

    @classmethod
    def from_state(cls, state: dict) -> IdentityTable:
        table = cls(state["texts"], int(state["weight"].shape[1]))
        table.embedding.weight.data.copy_(state["weight"])
        return table


@dataclass
class Materializer:
    """The device half of a batch: whatever turns staged texts into ``e``.

    The encoder runs here, on the thread that runs the model, so ``e`` is part
    of the step's graph; and so does the noise, whose draw reads the device's
    generator and has to take it in step order. ``noise`` is training's alone:
    validation, evaluation and encoding build theirs without it, so nothing
    outside training is ever perturbed.
    """

    device: torch.device
    masks: BaselineMasks
    identity_table: IdentityTable | None = None
    noise: NoiseState | None = None

    def encode(
        self, ordered: list[str], inputs: dict[str, torch.Tensor] | None,
        stand_in: torch.Tensor | None, encoder: AbilityEncoder,
    ) -> torch.Tensor | None:
        """The batch's ``e`` matrix, one row per text of ``ordered``, or None.

        Two baselines never reach the encoder at all. ``identity`` reads a free
        vector per text — the control for "is the encoder reading the words, or
        just memorizing which ability this is" — and ``taxonomy`` reads the
        sidecar's API type and parameter keys, the control for "is it reading
        more than the script's shape".
        """
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

    def materialize(self, prepared: PreparedBatch, encoder: AbilityEncoder) -> Materialized:
        """``prepared`` on the device, its texts encoded and scattered."""
        rows = prepared.rows
        ordered = sorted(rows, key=rows.__getitem__)
        matrix = self.encode(ordered, prepared.encoder_inputs, prepared.stand_in, encoder)
        if self.noise is not None and matrix is not None:
            # Every e of the batch, whichever variant produced it (FR-056).
            matrix = self.noise.apply(matrix)
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
            # The one place the encoder's output enters the head. Done here
            # rather than in collate because the vectors must stay on the
            # device and attached to the graph.
            batch["e_vectors"] = scatter_e_rows(batch["e_vectors"], e_rows, matrix)
        return Materialized(
            batch, prepared.surfaces,
            EncodedBatch(texts=ordered, lines=prepared.texts, matrix=matrix),
        )


class SurfaceBatcher:
    """Turns a list of records into the trunk's keyword arguments.

    The host half's owner: the key-to-text join (``resolver``), the text
    staging (``stager``), the surfaces and the variant's masks. Its device half
    is ``materializer``'s; a batcher built only to prepare steps, as a worker's
    is, has none.
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
        #: The key-to-text join, memoised. A training step's batcher borrows
        #: the run's, so the answers outlive the step.
        self.resolver = resolver
        self.masks = masks
        self.e_dim = e_dim
        self.widths = widths
        self.materializer = materializer
        self.context_dropout = context_dropout
        #: The step's generator: keyword expansion, context dropout and the MLM
        #: mask draw from it, in that order.
        self.rng = rng or random.Random(0)
        #: The input layout every surface is built in (FR-063c): training's
        #: own, or the one a loaded checkpoint recorded.
        self.input_layout = input_layout
        #: ``key -> (text, line)`` for the keys this batcher stood in for
        #: (:meth:`stand_in`). Kept apart from the resolver's memo, because a
        #: counterfactual copy's text belongs to its step alone.
        self._staged: dict[object, tuple[str, object]] = {}

    @property
    def sidecars(self):
        return self.resolver.sidecars

    @property
    def surface(self) -> str:
        return self.stager.surface

    # ── the encoding text ───────────────────────────────────────────────

    def _resolve(self, key) -> tuple[str | None, object]:
        """``(text, line)`` for one key: a staged stand-in, else the join."""
        staged = self._staged.get(key)
        return staged if staged is not None else self.resolver.resolve(key)

    def text_of(self, key) -> str | None:
        """The text one ability key is encoded from, on this run's surface.

        The single definition of it. Two of these that disagree miss every
        lookup, and the model then trains on a zero vector while every number
        it reports still moves.

        A key the sidecar does not describe raises: that is the contract's
        fail-loudly case, and reading it as no text is what hid it.
        """
        return self._resolve(key)[0]

    def line_of(self, key):
        """The sidecar line one ability key resolves to, or None."""
        return self._resolve(key)[1]

    def stand_in(
        self, key, text: str, like, *, script: str | None = None,
        role_spans: tuple | None = None,
    ) -> None:
        """Make ``key`` read ``text`` for this batcher's life, with ``like``'s line.

        A counterfactual copy's acting key names no sidecar (FR-063e): its text
        is the real line's with the amount edited, and everything read off the
        line itself, such as the API type, is the real line's — except its
        ``script`` and ``role_spans`` where the copy edited those too, which
        the value head's targets and the prose's roles read.
        """
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
        """Inside the block ``key`` reads ``text``, keeping its own line.

        How a probe edits one ability's script — a damage amount, a card count —
        and has the checkpoint's own encoder read the edit with nothing else
        moved. The key's previous reading is back when the block ends.
        """
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
        """Every ability text this batch's surfaces reference, to its line.

        The acting line of each record **and** every ability on every entity in
        its state: the context abilities are the board the head reads, and a
        text left out of the encode reaches the model as a zero vector.

        The line comes along because the ``taxonomy`` baseline reads the
        sidecar's script facts rather than the text; it rides in the memo
        beside the text so noting a key is one lookup rather than a join
        followed by a second join inside ``text_of``.
        """
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
        """The keys of the charm-mode lines that follow ``key``'s line (FR-063a).

        A stood-in key has none: it is one mode's or one spell's edited copy.
        """
        return () if key in self._staged else self.resolver.options_for(key)

    def _keys_of(self, text: str) -> list:
        """The keys resolved to ``text`` so far: how a truncation is named."""
        staged = [key for key, (known, _) in self._staged.items() if known == text]
        return self.resolver.keys_of(text) + staged

    # ── the ability vectors ─────────────────────────────────────────────

    def stage(self, texts: Mapping[str, object]) -> StagedTexts:
        """``texts`` staged for this variant's encoder or its stand-in."""
        return self.stager.stage(
            texts, masks=self.masks, e_dim=self.e_dim, rng=self.rng,
            keys_of=self._keys_of,
        )

    def encode_texts(
        self, texts: dict[str, object], encoder: AbilityEncoder,
    ) -> tuple[dict[str, int], torch.Tensor | None]:
        """Encode each unique ability text once: ``(row_of_text, matrix)``.

        This is what grouping a batch by game buys: the abilities on one board
        recur across that game's records, so one forward pass serves many. The
        surface carries row indices and the vectors stay on the device in one
        tensor, so the gradient reaches the encoder and no ability vector makes
        a host round trip.
        """
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

    # ── the surface ─────────────────────────────────────────────────────

    def keyword_of_key(self, key) -> str | None:
        """Which damage-step keyword one ability key's line *is*, if any.

        Memoised through ``_resolve`` like every other answer about a key, so
        gate 2 asking it once per entity per build costs a dict hit.
        """
        return keyword_of_line(self._resolve(key)[1])

    def surface_for(
        self,
        record,
        rows: dict[str, int],
        strip_keywords: dict[str, frozenset[str]] | None = None,
    ):
        """One record's token surface.

        ``strip_keywords`` is gate 2's perturbation: per entity, the keywords to
        remove from that entity's input through **both** channels. It is a
        per-build argument rather than batcher state because the gate compares
        the same record with and without the keyword, and the two builds have to
        be able to share everything else.
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
            # A decision record's [ACT] reads its candidate's e (FR-060a);
            # without it [ACT] read a zero vector and the verdict head would
            # have trained on nothing.
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

        ``strip_per_record`` is a list as long as ``records``, each entry the
        ``strip_keywords`` map for that record or None. A list rather than one
        map for the batch because gate 2 puts a record and its perturbed twin in
        the same batch: they share every encoded ability text, and one map would
        strip both.

        The trainer calls the two halves apart, so the next batch's host half
        overlaps this one's device half; every other caller wants both at once.
        """
        return self.materializer.materialize(
            self.prepare(records, strip_per_record), encoder,
        )

    def prepare(self, records, strip_per_record=None) -> PreparedBatch:
        """``build``'s host half: no model, no device, only CPU tensors.

        Draws from the batcher's generator twice — keyword expansion while
        tokenizing, then context dropout while building each surface — in that
        order.
        """
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
        """The collated inputs with the variant's masks applied (FR-094).

        Zeroed here rather than left out of the surface, so the slots, and
        therefore the per-entity targets, keep their shape. ``no-state`` is
        the average-effect control: no board, no global state but the record's
        kind flags, and no ability on the board — only ``[ACT]`` and its ``e``.
        """
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

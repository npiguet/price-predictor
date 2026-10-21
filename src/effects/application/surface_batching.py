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
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field, replace

import numpy as np
import torch

from effects.application.ability_text import KeyResolver
from effects.application.train_effect_model import withheld_keyword_tokens
from effects.domain.ability_encoder import (
    AbilityEncoder,
    TruncationLog,
    collate_lines,
    encoding_role_spans,
    prepare_line,
)
from effects.domain.ability_tokenizer import AbilityTokenizer
from effects.domain.damage_step_keywords import keyword_of_line
from effects.domain.effect_head_input import (
    ACT_SLOT,
    INPUT_LAYOUT,
    KIND_FLAG_WIDTH,
    SlotKind,
    build_effect_head_input,
    continuous_masked_keywords,
)
from effects.domain.records import PlayabilityDecisionPayload, acting_keys
from effects.domain.surface_collation import collate_surfaces, scatter_e_rows
from effects.domain.variants import VariantMasks

#: Running-covariance decay for the noise on ``e`` (FR-056).
NOISE_DECAY = 0.99

#: Steps between two factorizations of the running covariance. A GPU
#: eigendecomposition stalls the step on the host, which cost about a tenth of
#: a training step when taken every step. ``Σ`` itself still updates every
#: step; at decay 0.99 ten steps move it by about a tenth of the way towards
#: the recent batches, so the factor drawn from lags it by at most that much.
NOISE_REFACTOR_EVERY = 10

#: Texts per encoder pass. A batch's texts are sorted by length and encoded in
#: groups this size, each padded to its own longest text: padded to the
#: batch's longest, one 512-token chain made every short line in the batch cost
#: 512 positions, four fifths of the encoder's input was padding, and the
#: activations a batch saved for backward overran the 8 GB card.
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


class NoiseState:
    """Gaussian noise on ``e`` scaled to the spread of ``e`` itself (FR-056).

    A fixed σ meant different things at different points of a run: the scale
    of ``e`` is learned, so a σ that is a nudge at step zero is nothing once the
    vectors have grown. Here the noise has covariance ``r² Σ``, where ``Σ`` is a
    running average of the covariance of the batch's ``e`` vectors about their
    mean, held with the gradient stopped, and ``r`` rises linearly from 0 to the
    ratio over the first ``ramp_steps`` optimizer steps.

    Owned by the training loop and handed to each training batch's batcher;
    validation, evaluation and encoding build their batchers without one, so
    nothing outside training is ever perturbed. Everything stays on the device:
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
    """What one ``build`` encoded, for the heads that read ``e`` directly.

    ``texts`` are the batch's unique ability texts in matrix-row order,
    ``lines`` each text's sidecar line, and ``matrix`` the ``e`` rows the
    surfaces scattered (after noise, in training). The value and script-API
    heads read these rows, one per text rather than one per slot.
    """

    texts: list[str]
    lines: dict[str, object]
    matrix: torch.Tensor | None
    #: ``EncodedLine`` per text, as fed to the encoder; the MLM pass masks them.
    prepared: list = field(default_factory=list)


@dataclass
class MaskedTokens:
    """The MLM pass's inputs, drawn on the host (FR-060b).

    ``inputs`` are the encoder's keyword arguments with the picked tokens
    replaced by ``[MASK]``, ``positions`` the flattened indices of those
    tokens, and ``targets`` the ids that stood there.
    """

    inputs: dict[str, torch.Tensor]
    positions: torch.Tensor
    targets: torch.Tensor

    def pin(self) -> None:
        """Every tensor in page-locked memory; see ``pinned``."""
        self.inputs = pinned(self.inputs)
        self.positions = pinned(self.positions)
        self.targets = pinned(self.targets)


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
    prepared: list
    #: ``collate_lines`` over ``prepared``, or None when the encoder is not
    #: called.
    encoder_inputs: dict[str, torch.Tensor] | None
    #: The ``taxonomy`` baseline's vectors, which stand in for the encoder.
    stand_in: torch.Tensor | None
    surfaces: list
    #: ``collate_surfaces``' output, the variant's state mask applied.
    inputs: dict

    def pin(self) -> None:
        """Every tensor ``materialize`` copies, in page-locked memory."""
        self.encoder_inputs = pinned(self.encoder_inputs)
        self.stand_in = pinned(self.stand_in)
        self.inputs = pinned(self.inputs)


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


class SurfaceBatcher:
    """Turns a list of records into the trunk's keyword arguments."""

    def __init__(
        self,
        *,
        tokenizer: AbilityTokenizer,
        sidecars,
        masks: VariantMasks,
        surface: str,
        e_dim: int,
        widths: dict[SlotKind, int],
        device: torch.device,
        withhold_keyword: str | None = None,
        keyword_expand_p: float = 0.0,
        context_dropout: float = 0.0,
        rng: random.Random | None = None,
        identity_table: IdentityTable | None = None,
        noise: NoiseState | None = None,
        truncations: TruncationLog | None = None,
        input_layout: str | None = INPUT_LAYOUT,
        resolver: KeyResolver | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        #: The input layout every surface is built in (FR-063c): training's
        #: own, or the one a loaded checkpoint recorded.
        self.input_layout = input_layout
        #: Where a text the encoder's window cuts short is reported, once per
        #: run (FR-006). The caller owns it, so it outlives this batch.
        self.truncations = truncations
        self.sidecars = sidecars
        self.masks = masks
        self.surface = surface
        self.e_dim = e_dim
        self.widths = widths
        self.device = device
        self.withhold_keyword = withhold_keyword
        self.keyword_expand_p = keyword_expand_p
        self.context_dropout = context_dropout
        self.rng = rng or random.Random(0)
        self.identity_table = identity_table
        #: Training only: the noise on every ``e`` this batch carries.
        self.noise = noise
        #: The last ``build``'s unique texts and their ``e`` rows.
        self.encoded: EncodedBatch | None = None
        self._prepared: list = []
        #: The key-to-text join, memoised. A training step's batcher borrows
        #: the run's, so the answers outlive the step.
        self.resolver = resolver or KeyResolver(sidecars, surface)
        #: ``key -> (text, line)`` for the keys this batcher stood in for
        #: (:meth:`stand_in`). Kept apart from the resolver's memo, because a
        #: counterfactual copy's text belongs to its step alone.
        self._staged: dict[object, tuple[str, object]] = {}

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

    # ── the ability vectors ─────────────────────────────────────────────

    def encode_texts(
        self, texts: dict[str, object], encoder: AbilityEncoder,
    ) -> tuple[dict[str, int], torch.Tensor | None]:
        """Encode each unique ability text once for the whole batch.

        This is what grouping a batch by game buys: the abilities on one board
        recur across that game's records, so one forward pass serves many.

        Returns ``(row_of_text, matrix)`` rather than per-text vectors: the
        surface carries row indices and the vectors stay on the device in one
        tensor, so the gradient reaches the encoder and no ability vector makes
        a host round trip.

        Two baselines never reach the encoder at all. ``identity`` reads a free
        vector per text — the control for "is the encoder reading the words, or
        just memorizing which ability this is" — and ``taxonomy`` reads the
        sidecar's API type and parameter keys, the control for "is it reading
        more than the script's shape".
        """
        rows, prepared, inputs, stand_in = self._stage_texts(texts)
        self._prepared = prepared
        return rows, self._encode_staged(
            sorted(rows, key=rows.__getitem__), inputs, stand_in, encoder,
        )

    def _stage_texts(self, texts: dict[str, object]):
        """``encode_texts``' host half: ``(rows, prepared, inputs, stand_in)``.

        Tokenizing draws from the batcher's generator (keyword expansion), so
        it runs wherever the batch is prepared, and in the order batches are.
        """
        if not texts:
            return {}, [], None, None
        ordered = sorted(texts)
        rows = {text: row for row, text in enumerate(ordered)}
        if self.masks.identity_embedding:
            # The table looks its rows up in the device half, where it lives.
            return rows, [], None, None
        if self.masks.taxonomy_embedding:
            return rows, [], None, self._taxonomy_vectors(ordered, texts)
        prepared = self.prepare_texts(ordered, texts)
        return rows, prepared, collate_lines(prepared, self.tokenizer.pad_id), None

    def _encode_staged(
        self, ordered: list[str], inputs: dict[str, torch.Tensor] | None,
        stand_in: torch.Tensor | None, encoder: AbilityEncoder,
    ) -> torch.Tensor | None:
        """``encode_texts``' device half: the batch's ``e`` matrix, or None.

        ``ordered`` is the batch's texts in matrix-row order.
        """
        if self.masks.identity_embedding and ordered:
            if self.identity_table is None:
                raise ValueError(
                    "the identity baseline needs its embedding table; construct "
                    "the batcher with identity_table="
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

    def prepare_texts(
        self, ordered: list[str], lines_of: Mapping[str, object] | None = None,
    ) -> list:
        """Each text tokenized and its keywords expanded, the withheld one always.

        The withheld keyword's token is dropped wherever it is left, inside an
        expansion too (``AbilityTokenizer.encoding_tokens``), so the model
        never reads it. ``lines_of`` maps a text to its sidecar line, whose
        role spans a prose text is tokenized with (FR-069).
        """
        withheld = withheld_keyword_tokens(self.withhold_keyword)
        lines_of = lines_of or {}
        lines = []
        for text in ordered:
            tokens = self.tokenizer.encoding_tokens(
                text, withheld=withheld,
                role_spans=encoding_role_spans(lines_of.get(text), text, self.surface),
                probability=self.keyword_expand_p, rng=self.rng,
            )
            line = prepare_line(self.tokenizer, tokens)
            if line.truncated and self.truncations is not None:
                self.truncations.note(text, self._keys_of(text))
            lines.append(line)
        return lines

    def _keys_of(self, text: str) -> list:
        """The keys resolved to ``text`` so far: how a truncation is named."""
        staged = [key for key, (known, _) in self._staged.items() if known == text]
        return self.resolver.keys_of(text) + staged

    def mlm_inputs(self, encoder: AbilityEncoder, mask_prob: float):
        """The MLM pass over the last build's texts (FR-060b).

        A second forward pass over the same prepared lines with a share
        ``mask_prob`` of their tokens replaced by ``[MASK]`` — never ``[CLS]``,
        never padding — so the effect head keeps reading the unmasked ``e`` and
        the corruption shapes only what the encoder's token outputs must
        recover.

        Returns ``(hidden, targets)`` on the device, already reduced to the
        masked positions, or None when nothing was masked. The mask is drawn and
        the positions picked on the host, so the device never has to report
        which they were; and the vocabulary projection then runs over the
        masked positions alone rather than over every position of the batch,
        which at a few hundred texts was half a gigabyte of logits to score a
        seventh of them.
        """
        masked = self.draw_mlm_mask(self._prepared, mask_prob)
        return None if masked is None else self.mlm_forward(encoder, masked)

    def draw_mlm_mask(
        self, prepared: list, mask_prob: float,
    ) -> MaskedTokens | None:
        """``mlm_inputs``' host half: which tokens the pass masks.

        Draws from the batcher's generator after the build's own draws, so a
        step prepared ahead of its turn takes the same draws as one prepared
        in place.
        """
        if not prepared or mask_prob <= 0.0:
            return None
        batch = collate_lines(prepared, self.tokenizer.pad_id)
        ids = batch["token_ids"]
        generator = torch.Generator().manual_seed(self.rng.getrandbits(63))
        picks = (torch.rand(ids.shape, generator=generator) < mask_prob)
        picks &= batch["attention_mask"].bool()
        picks[:, 0] = False  # never [CLS]
        positions = picks.flatten().nonzero().squeeze(1)
        if positions.numel() == 0:
            return None
        targets = ids.flatten().index_select(0, positions)
        batch["token_ids"] = ids.masked_fill(picks, self.tokenizer.mask_id)
        # A masked number's value goes too: the encoder adds the number
        # embedding beside the token's, and it would hand the answer back.
        batch["numbers"] = batch["numbers"].masked_fill(picks, 0.0)
        batch["number_mask"] = batch["number_mask"].masked_fill(picks, 0)
        return MaskedTokens(inputs=batch, positions=positions, targets=targets)

    def mlm_forward(self, encoder: AbilityEncoder, masked: MaskedTokens):
        """``mlm_inputs``' device half: the encoder's outputs at the masks."""
        _e, hidden = encode_by_length(
            encoder, masked.inputs, self.device, hidden=True,
        )
        selected = hidden.reshape(-1, hidden.shape[-1]).index_select(
            0, masked.positions.to(self.device, non_blocking=True),
        )
        return selected, masked.targets.to(self.device, non_blocking=True)

    def _taxonomy_vectors(
        self, ordered: list[str], lines: dict[str, object],
    ) -> torch.Tensor:
        """The taxonomy baseline's stand-in for an encoded ``e``.

        Deterministic and unlearned — the same hash embedding the cache writes
        for this variant, so the trained baseline and its cache agree.
        """
        from effects.application.encode_abilities import taxonomy_vector

        rows = np.stack([
            taxonomy_vector(lines[text], self.e_dim) for text in ordered
        ])
        return torch.from_numpy(rows)

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
        """One record's token surface, with the variant's masks applied.

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

    def build(self, records, encoder: AbilityEncoder, strip_per_record=None):
        """``(model_kwargs, surfaces)`` for one batch of records.

        ``strip_per_record`` is a list as long as ``records``, each entry the
        ``strip_keywords`` map for that record or None. A list rather than one
        map for the batch because gate 2 puts a record and its perturbed twin in
        the same batch: they share every encoded ability text, and one map would
        strip both.

        ``prepare`` then ``materialize``, back to back. The trainer calls the
        two apart, so the next batch's host half overlaps this one's device
        half; every other caller wants both at once.
        """
        return self.materialize(self.prepare(records, strip_per_record), encoder)

    def prepare(self, records, strip_per_record=None) -> PreparedBatch:
        """``build``'s host half: no model, no device, only CPU tensors.

        Draws from the batcher's generator twice — keyword expansion while
        tokenizing, then context dropout while building each surface — in the
        order ``build`` has always taken them.
        """
        texts = self.batch_texts(records)
        rows, prepared, encoder_inputs, stand_in = self._stage_texts(texts)
        strips = strip_per_record or [None] * len(records)
        surfaces = [
            self.surface_for(record, rows, strip)
            for record, strip in zip(records, strips)
        ]
        inputs = self._masked(collate_surfaces(
            surfaces, e_dim=self.e_dim, widths=self.widths,
        ))
        return PreparedBatch(
            texts=texts, rows=rows, prepared=prepared,
            encoder_inputs=encoder_inputs, stand_in=stand_in,
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

    def materialize(self, prepared: PreparedBatch, encoder: AbilityEncoder):
        """``build``'s device half: ``(model_kwargs, surfaces)``.

        The encoder runs here, on the thread that runs the model, so ``e`` is
        part of the step's graph; and so does the noise, whose draw reads the
        device's generator and has to take it in step order.
        """
        self._prepared = prepared.prepared
        rows = prepared.rows
        ordered = sorted(rows, key=rows.__getitem__)
        matrix = self._encode_staged(
            ordered, prepared.encoder_inputs, prepared.stand_in, encoder,
        )
        if self.noise is not None and matrix is not None:
            # Every e of the batch, whichever variant produced it (FR-056).
            matrix = self.noise.apply(matrix)
        self.encoded = EncodedBatch(
            texts=ordered, lines=prepared.texts,
            matrix=matrix, prepared=self._prepared,
        )
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
            batch["e_vectors"] = scatter_e_rows(
                batch["e_vectors"], e_rows, matrix,
            )
        return batch, prepared.surfaces

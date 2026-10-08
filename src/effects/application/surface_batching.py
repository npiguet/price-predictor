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

import hashlib
import random
from dataclasses import dataclass, field

import numpy as np
import torch

from effects.application.train_effect_model import (
    VariantMasks,
    withheld_keyword_rules,
)
from effects.domain.ability_encoder import (
    AbilityEncoder,
    TruncationLog,
    collate_lines,
    encoding_text,
    prepare_line,
)
from effects.domain.ability_tokenizer import AbilityTokenizer
from effects.domain.damage_step_keywords import keyword_of_line
from effects.domain.effect_head_input import (
    SlotKind,
    build_effect_head_input,
    continuous_masked_keywords,
)
from effects.domain.effect_model import collate_surfaces, scatter_e_rows
from effects.domain.records import PlayabilityDecisionPayload
from effects.infrastructure.sidecar_io import UnconfiguredTree

#: Rows in the ``identity`` baseline's free-embedding table. Ample for the
#: corpus's distinct ability texts, so collisions stay rare.
IDENTITY_TABLE_SIZE = 1 << 17


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


def text_slot(text: str, size: int) -> int:
    """A stable table row for an ability text.

    ``hash()`` is salted per process and would give the baseline a different
    table on every run, so this hashes the bytes explicitly.
    """
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % size


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
        identity_table: torch.nn.Embedding | None = None,
        noise: NoiseState | None = None,
        truncations: TruncationLog | None = None,
    ) -> None:
        self.tokenizer = tokenizer
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
        #: ``key -> option-line keys after its line``, memoised like the text.
        self._options_by_key: dict[object, tuple] = {}
        #: ``key -> (text, line)``, the join's answer for one key on this
        #: run's surface. ``SidecarCache`` already reads each card once, but
        #: the *key* is still resolved three or four times per batch — once by
        #: ``batch_texts`` and again by ``has_line`` and ``e_for`` for every
        #: record the key's entity appears in — and each of those walks the
        #: sidecar's lines and re-renders the text. The answer depends only on
        #: the key and on ``self.surface``, both fixed for the batcher's life,
        #: so it is computed once. (Swapping ``self.sidecars`` after the fact
        #: would serve stale answers; nothing but the tests does that.)
        self._text_by_key: dict[object, tuple[str | None, object]] = {}

    # ── the encoding text ───────────────────────────────────────────────

    def _resolve(self, key) -> tuple[str | None, object]:
        """``(text, line)`` for one key, memoised; see ``_text_by_key``.

        The mismatch ``KeyError`` is deliberately not cached — it is raised
        from inside the ``try`` and nothing is stored — because a failure
        remembered as "no text" is exactly the fail-loudly case going quiet.
        """
        cached = self._text_by_key.get(key)
        if cached is not None:
            return cached
        try:
            line = self.sidecars.line_for(key)
        except UnconfiguredTree:
            line = None
        if line is None:
            resolved: tuple[str | None, object] = (None, None)
        else:
            text = encoding_text(
                line, self.sidecars.prose_for(key), self.surface,
            )
            resolved = (
                text or (
                    f"{key.script_file}:{key.trait_kind}:"
                    f"{key.index_within_kind}"
                ),
                line,
            )
        self._text_by_key[key] = resolved
        return resolved

    def text_of(self, key) -> str | None:
        """The text one ability key is encoded from, on this run's surface.

        The single definition of it. Two of these that disagree miss every
        lookup, and the model then trains on a zero vector while every number
        it reports still moves.

        A key the sidecar does not describe raises: that is the contract's
        fail-loudly case, and reading it as no text is what hid it.
        """
        return self._resolve(key)[0]

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
            for key in self.acting_keys(record):
                if note(key) is not None:
                    break
            for entity in record.state.entities:
                for key in (*entity.printed, *entity.granted_attached):
                    if note(key) is not None:
                        for mode in self.options_for(key):
                            note(mode)
        return texts

    @staticmethod
    def acting_keys(record) -> tuple:
        """The keys ``[ACT]`` reads: the acting line, or a decision's candidate.

        A ``decision`` record names no acting line of its own; its ``[ACT]``
        carries the candidate's ``e`` (FR-060a), and Java writes one candidate
        per record, so it is the payload's first.
        """
        payload = record.payload
        if isinstance(payload, PlayabilityDecisionPayload):
            return payload.candidates[0].ability if payload.candidates else ()
        return record.ability or ()

    def options_for(self, key) -> tuple:
        """The keys of the charm-mode lines that follow ``key``'s line (FR-063a).

        Empty for every line that is not a charm's root, and for a sidecar
        source that cannot say (a stub, an unconverted script).
        """
        cached = self._options_by_key.get(key)
        if cached is not None:
            return cached
        found: tuple = ()
        get = getattr(self.sidecars, "get", None)
        # A mode is never a root: its siblings follow it in the sidecar too,
        # and must not be read as its own options.
        if get is not None and getattr(key, "option", None) is None:
            try:
                sidecar = get(key.script_file)
                row = sidecar.row_for(key)
            except KeyError:
                row = None
            if row is not None:
                found = tuple(
                    next(k for k in sidecar.lines[r].provenance if k.option is not None)
                    for r in sidecar.option_rows_after(row)
                )
        self._options_by_key[key] = found
        return found

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
        if not texts:
            return {}, None
        ordered = sorted(texts)
        rows = {text: row for row, text in enumerate(ordered)}
        if self.masks.identity_embedding:
            return rows, self._identity_vectors(ordered)
        if self.masks.taxonomy_embedding:
            return rows, self._taxonomy_vectors(ordered, texts)

        lines = self.prepare_texts(ordered)
        self._prepared = lines
        batch = collate_lines(lines, self.tokenizer.pad_id)
        batch = {k: v.to(self.device) for k, v in batch.items()}
        vectors, _hidden = encoder(**batch)
        return rows, vectors

    def prepare_texts(self, ordered: list[str]) -> list:
        """Each text tokenized, withheld keyword hidden, keywords expanded."""
        hidden_keywords, _force = withheld_keyword_rules(self.withhold_keyword)
        lines = []
        for text in ordered:
            tokens = self.tokenizer.tokenize(text)
            tokens = [t for t in tokens if t.text not in hidden_keywords]
            tokens = self.tokenizer.expand_keywords(
                tokens, probability=self.keyword_expand_p, rng=self.rng,
            )
            line = prepare_line(self.tokenizer, tokens)
            if line.truncated and self.truncations is not None:
                self.truncations.note(text, self._keys_of(text))
            lines.append(line)
        return lines

    def _keys_of(self, text: str) -> list:
        """The keys resolved to ``text`` so far: how a truncation is named."""
        return [key for key, (known, _) in self._text_by_key.items() if known == text]

    def mlm_inputs(self, encoder: AbilityEncoder, mask_prob: float):
        """The MLM pass over the last build's texts (FR-060b).

        A second forward pass over the same prepared lines with a share
        ``mask_prob`` of their tokens replaced by ``[MASK]`` — never ``[CLS]``,
        never padding — so the effect head keeps reading the unmasked ``e`` and
        the corruption shapes only what the encoder's token outputs must
        recover.

        Returns ``(hidden, targets, None)`` on the device, already reduced to the
        masked positions, or None when nothing was masked. The mask is drawn and
        the positions picked on the host, so the device never has to report
        which they were; and the vocabulary projection then runs over the
        masked positions alone rather than over every position of the batch,
        which at a few hundred texts was half a gigabyte of logits to score a
        seventh of them.
        """
        if not self._prepared or mask_prob <= 0.0:
            return None
        batch = collate_lines(self._prepared, self.tokenizer.pad_id)
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
        batch = {k: v.to(self.device) for k, v in batch.items()}
        _e, hidden = encoder(**batch)
        selected = hidden.reshape(-1, hidden.shape[-1]).index_select(
            0, positions.to(self.device, non_blocking=True),
        )
        return selected, targets.to(self.device, non_blocking=True), None

    def _identity_vectors(self, ordered: list[str]) -> torch.Tensor:
        """A free learned vector per ability text, keyed by hash.

        Hashed into a fixed table rather than indexed by a corpus-wide text
        list, so the baseline needs no pass over the corpus to number its texts
        and a text unseen at that pass cannot break it. Collisions cost the
        baseline a little, and the baseline is what the real model must beat.
        """
        if self.identity_table is None:
            raise ValueError(
                "the identity baseline needs its embedding table; construct "
                "the batcher with identity_table="
            )
        index = torch.tensor(
            [text_slot(text, IDENTITY_TABLE_SIZE) for text in ordered],
            dtype=torch.long, device=self.device,
        )
        return self.identity_table(index)

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
        return torch.from_numpy(rows).to(self.device)

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
            if self.masks.zero_e:
                return None
            text = self.text_of(key)
            return None if text is None else rows.get(text)

        def has_line(key) -> bool:
            # Asked before the mask, so a masked line keeps its zero token
            # and a dropped key gets none: the two are different facts.
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
        )

    def build(self, records, encoder: AbilityEncoder, strip_per_record=None):
        """``(model_kwargs, surfaces)`` for one batch of records.

        ``strip_per_record`` is a list as long as ``records``, each entry the
        ``strip_keywords`` map for that record or None. A list rather than one
        map for the batch because gate 2 puts a record and its perturbed twin in
        the same batch: they share every encoded ability text, and one map would
        strip both.
        """
        texts = self.batch_texts(records)
        self._prepared = []
        rows, matrix = self.encode_texts(texts, encoder)
        if self.noise is not None and matrix is not None:
            # Every e of the batch, whichever variant produced it (FR-056).
            matrix = self.noise.apply(matrix)
        self.encoded = EncodedBatch(
            texts=sorted(rows, key=rows.__getitem__), lines=texts,
            matrix=matrix, prepared=self._prepared,
        )
        strips = strip_per_record or [None] * len(records)
        surfaces = [
            self.surface_for(record, rows, strip)
            for record, strip in zip(records, strips)
        ]
        batch = collate_surfaces(
            surfaces, e_dim=self.e_dim, widths=self.widths,
        )
        if self.masks.zero_state:
            # The average-effect control: no board at all. Zeroed here rather
            # than left out of the surface so the slots, and therefore the
            # per-entity targets, keep their shape.
            for kind in (SlotKind.PLAYER, SlotKind.CARD):
                batch["slot_features"][kind] = torch.zeros_like(
                    batch["slot_features"][kind]
                )
        batch = {
            "slot_features": {
                k: v.to(self.device) for k, v in batch["slot_features"].items()
            },
            **{
                k: v.to(self.device)
                for k, v in batch.items() if k != "slot_features"
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
        return batch, surfaces

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

import numpy as np
import torch

from effects.application.train_effect_model import (
    VariantMasks,
    withheld_keyword_rules,
)
from effects.domain.ability_encoder import (
    AbilityEncoder,
    collate_lines,
    encoding_text,
    prepare_line,
)
from effects.domain.ability_tokenizer import AbilityTokenizer
from effects.domain.effect_head_input import (
    SlotKind,
    build_effect_head_input,
    continuous_masked_keywords,
)
from effects.domain.effect_model import collate_surfaces, scatter_e_rows

#: Rows in the ``identity`` baseline's free-embedding table. Ample for the
#: corpus's distinct ability texts, so collisions stay rare.
IDENTITY_TABLE_SIZE = 1 << 17


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
    ) -> None:
        self.tokenizer = tokenizer
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

    # ── the encoding text ───────────────────────────────────────────────

    def text_of(self, key) -> str | None:
        """The text one ability key is encoded from, on this run's surface.

        The single definition of it. Two of these that disagree miss every
        lookup, and the model then trains on a zero vector while every number
        it reports still moves.
        """
        try:
            line = self.sidecars.line_for(key)
        except KeyError:
            return None
        if line is None:
            return None
        text = encoding_text(line, self.sidecars.prose_for(key), self.surface)
        return text or (
            f"{key.script_file}:{key.trait_kind}:{key.index_within_kind}"
        )

    def batch_texts(self, records) -> dict[str, object]:
        """Every ability text this batch's surfaces reference, to its line.

        The acting line of each record **and** every ability on every entity in
        its state: the context abilities are the board the head reads, and a
        text left out of the encode reaches the model as a zero vector.

        The line comes along because the ``taxonomy`` baseline reads the
        sidecar's script facts rather than the text.
        """
        texts: dict[str, object] = {}

        def note(key) -> str | None:
            try:
                line = self.sidecars.line_for(key)
            except KeyError:
                return None
            if line is None:
                return None
            text = self.text_of(key)
            if text is not None:
                texts.setdefault(text, line)
            return text

        for record in records:
            for key in record.ability or ():
                if note(key) is not None:
                    break
            for entity in record.state.entities:
                for key in (*entity.printed, *entity.granted_attached):
                    note(key)
        return texts

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

        hidden_keywords, _force = withheld_keyword_rules(self.withhold_keyword)
        lines = []
        for text in ordered:
            tokens = self.tokenizer.tokenize(text)
            tokens = [t for t in tokens if t.text not in hidden_keywords]
            tokens = self.tokenizer.expand_keywords(
                tokens, probability=self.keyword_expand_p, rng=self.rng,
            )
            lines.append(prepare_line(self.tokenizer, tokens))
        batch = collate_lines(lines, self.tokenizer.pad_id)
        batch = {k: v.to(self.device) for k, v in batch.items()}
        vectors, _hidden = encoder(**batch)
        return rows, vectors

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

    def surface_for(self, record, rows: dict[str, int]):
        """One record's token surface, with the variant's masks applied."""

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
            context_dropout=self.context_dropout,
            rng=self.rng,
            masked_keywords=continuous_masked_keywords(record),
        )

    def build(self, records, encoder: AbilityEncoder):
        """``(model_kwargs, surfaces)`` for one batch of records."""
        rows, matrix = self.encode_texts(self.batch_texts(records), encoder)
        surfaces = [self.surface_for(record, rows) for record in records]
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

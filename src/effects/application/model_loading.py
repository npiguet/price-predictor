"""Turn a loaded checkpoint into something that can be run over records,
through the same ``surface_batching`` the trainer uses.

The weights themselves come off disk in the infrastructure
(:func:`effects.infrastructure.effect_model_store.load_weights`); what is
assembled here around them — the batcher, its text stager and key resolver,
the identity baseline's table, the evaluated fields — is the trainer's own
pipeline, so it lives beside it.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass

import torch

from effects.application.ability_text import KeyResolver
from effects.application.record_forward import BATCH_RECORDS as FORWARD_BATCH_RECORDS
from effects.application.record_forward import RecordForward
from effects.application.surface_batching import (
    IDENTITY_TABLE_KEY,
    IdentityTable,
    Materializer,
    SurfaceBatcher,
    SurfaceSpec,
)
from effects.application.text_staging import TextStager
from effects.domain.ability_encoder import (
    AbilityEncoder,
    TruncationLog,
    log_truncation,
)
from effects.domain.ability_tokenizer import AbilityTokenizer, Surface
from effects.domain.baselines import baseline_masks
from effects.domain.damage_step_keywords import KeywordResolver
from effects.domain.effect_head_layout import FieldSpec, active_fields
from effects.domain.effect_model import EffectModel
from effects.domain.sampling import sampling_class
from effects.domain.tokenizer_files import TokenizerFiles
from effects.infrastructure.effect_model_store import EffectCheckpoint, load_weights
from effects.infrastructure.sidecar_io import SidecarCache
from effects.infrastructure.tokenizer_loading import tokenizer_for
from effects.infrastructure.torch_device import default_device

logger = logging.getLogger(__name__)

#: Seeds every evaluation batcher's generator, so evaluations repeat.
EVALUATION_SEED = 42

#: Records per forward pass when gate 1 and the breakdowns score a checkpoint.
BATCH_RECORDS = FORWARD_BATCH_RECORDS


@dataclass(frozen=True, slots=True)
class Runnable:
    """One checkpoint, ready to run over one set of records."""

    encoder: AbilityEncoder
    model: EffectModel
    batcher: SurfaceBatcher
    fields: tuple[FieldSpec, ...]

    def forward(self) -> RecordForward:
        """The one chunked forward (:mod:`effects.application.record_forward`)
        over this checkpoint's modules and its batcher's device half."""
        return RecordForward(self.encoder, self.model, self.batcher.materializer)

    def keyword_resolver(self) -> KeywordResolver:
        """A resolver over the batcher's own sidecar cache, never a second
        one: reading different joins, a perturbation would strip nothing."""
        return KeywordResolver(self.batcher.sidecars)


@dataclass(frozen=True, slots=True)
class LoadedModel:
    """A checkpoint's weights on the device, before any record is seen; each
    record set gets a :class:`Runnable` with a batcher and generator of its
    own."""

    checkpoint: EffectCheckpoint
    encoder: AbilityEncoder
    model: EffectModel
    tokenizer: AbilityTokenizer
    identity_table: IdentityTable | None
    surface: Surface
    device: torch.device
    sidecars: SidecarCache

    def runnable(self, records) -> Runnable:
        return Runnable(
            self.encoder, self.model, self.batcher(), evaluation_fields(records),
        )

    def batcher(self) -> SurfaceBatcher:
        """A batcher of its own, laid out by the layout's widths, which a
        loaded checkpoint's trunk was built with."""
        checkpoint = self.checkpoint
        spec = SurfaceSpec.for_saved_model(checkpoint)
        return SurfaceBatcher(
            stager=TextStager.for_inference(
                self.tokenizer, self.surface,
                withhold_keyword=checkpoint.provenance.withheld_keyword,
                truncations=TruncationLog(log_truncation(logger, "evaluation")),
            ),
            resolver=KeyResolver(self.sidecars, self.surface),
            spec=spec,
            materializer=Materializer(self.device, spec.masks, self.identity_table),
            context_dropout=0.0,
            rng=random.Random(EVALUATION_SEED),
        )


def load_model(
    checkpoint, *, files: TokenizerFiles, sidecars: SidecarCache,
) -> LoadedModel:
    """``checkpoint``'s weights on the device, with its tokenizer over
    ``files`` (:func:`~effects.infrastructure.effect_model_store.inference_files`), joining
    keys through ``sidecars``, which several loads may share
    (:meth:`SourceTrees.sidecar_cache`)."""
    device = default_device()
    encoder, model = load_weights(checkpoint, device)
    tokenizer = tokenizer_for(checkpoint, files)
    return LoadedModel(
        checkpoint=checkpoint,
        encoder=encoder,
        model=model,
        tokenizer=tokenizer,
        identity_table=_identity_table(checkpoint, device),
        surface=tokenizer.surface,
        device=device,
        sidecars=sidecars,
    )


def load_runnable(
    checkpoint, *, files: TokenizerFiles, sidecars: SidecarCache, records,
) -> Runnable:
    """One checkpoint, loaded and ready to run over ``records``."""
    return load_model(checkpoint, files=files, sidecars=sidecars).runnable(records)


def _identity_table(checkpoint, device) -> IdentityTable | None:
    """The identity baseline's learned ``e`` table; None for any other variant."""
    if not baseline_masks(checkpoint.variant).identity_embedding:
        return None
    if IDENTITY_TABLE_KEY not in checkpoint.extra:
        raise ValueError(
            f"The identity baseline checkpoint carries no {IDENTITY_TABLE_KEY!r} "
            "(its embedding table): it would score on random e vectors and gate 1 "
            "would get a margin nobody earned. Retrain the baseline."
        )
    return IdentityTable.from_state(checkpoint.extra[IDENTITY_TABLE_KEY]).to(device)


def evaluation_fields(records) -> tuple[FieldSpec, ...]:
    """The fields active for ``records``' classes, the sparse group included:
    the gates score a finished model."""
    return active_fields(
        frozenset(sampling_class(r) for r in records), sparse=True,
    )

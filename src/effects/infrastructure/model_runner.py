"""Load a saved checkpoint back into something that can be run over records,
through the same ``surface_batching`` the trainer uses."""

from __future__ import annotations

import logging
import random
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from itertools import batched
from pathlib import Path

import torch

from effects.application.ability_text import KeyResolver
from effects.application.surface_batching import (
    IDENTITY_TABLE_KEY,
    IdentityTable,
    Materializer,
    SurfaceBatcher,
)
from effects.application.text_staging import TextStager
from effects.domain.ability_encoder import (
    AbilityEncoder,
    TruncationLog,
    log_truncation,
    surface_of,
)
from effects.domain.ability_tokenizer import (
    INFERENCE_KEYWORD_EXPAND_P,
    AbilityTokenizer,
)
from effects.domain.aux_heads import TRAINING_ONLY_MODULES
from effects.domain.baselines import baseline_masks
from effects.domain.damage_step_keywords import KeywordResolver
from effects.domain.effect_head_input import InputLayout, feature_widths
from effects.domain.effect_head_layout import FieldSpec, active_fields
from effects.domain.effect_model import EffectModel
from effects.domain.sampling import sampling_class
from effects.infrastructure.sidecar_io import SidecarCache, sidecar_roots
from effects.infrastructure.tokenizer_loading import load_tokenizer
from effects.infrastructure.torch_device import default_device

logger = logging.getLogger(__name__)

#: Seeds every evaluation batcher's generator, so evaluations repeat.
EVALUATION_SEED = 42


def build_sidecars(config) -> SidecarCache:
    """The sidecar roots ``--cards-folder`` names (FR-063b), each keyed by the
    tree its name ends in, so ``output/gen1-cardsfolder`` serves
    ``cardsfolder/`` keys."""
    return SidecarCache(sidecar_roots(
        config.cards_folders, getattr(config, "variant_scripts", None),
    ))


def tokenizer_for(checkpoint, vocab_path: Path, keyword_path: Path) -> AbilityTokenizer:
    """The tokenizer ``checkpoint`` trained with, under the rules it recorded
    (a gen-1 checkpoint reads its script vocabulary with the prose grammar)."""
    return load_tokenizer(
        vocab_path, keyword_path,
        training_settings=checkpoint.training_settings or {},
    )


#: Records per forward pass when gate 1 and the breakdowns score a checkpoint.
BATCH_RECORDS = 32

#: Observations per forward pass in gate 2: each is built twice, as collected
#: and perturbed, in one batch.
BATCH_OBSERVATIONS = BATCH_RECORDS // 2


@dataclass(frozen=True, slots=True)
class Runnable:
    """One checkpoint, ready to run over one set of records."""

    encoder: AbilityEncoder
    model: EffectModel
    batcher: SurfaceBatcher
    fields: tuple[FieldSpec, ...]

    def inference_chunks[T](self, items: Iterable[T], size: int) -> Iterator[list[T]]:
        """``items`` in chunks of ``size``, each yielded under inference mode
        and evaluation mode, since a probe's own ``Runnable`` may hand over
        modules still training."""
        self.encoder.eval()
        self.model.eval()
        with torch.inference_mode():
            for chunk in batched(items, size):
                yield list(chunk)

    def keyword_resolver(self) -> KeywordResolver:
        """A resolver over the batcher's own sidecar cache, never a second
        one: reading different joins, a perturbation would strip nothing."""
        return KeywordResolver(self.batcher.sidecars)


@dataclass(frozen=True, slots=True)
class LoadedModel:
    """A checkpoint's weights on the device, before any record is seen; each
    record set gets a :class:`Runnable` with a batcher and generator of its
    own."""

    checkpoint: object
    encoder: AbilityEncoder
    model: EffectModel
    tokenizer: AbilityTokenizer
    identity_table: IdentityTable | None
    surface: str
    device: torch.device
    sidecars: SidecarCache

    def runnable(self, records) -> Runnable:
        return Runnable(
            self.encoder, self.model, self._batcher(records),
            evaluation_fields(records),
        )

    def _batcher(self, records) -> SurfaceBatcher:
        checkpoint = self.checkpoint
        layout = InputLayout.of(checkpoint.training_settings)
        masks = baseline_masks(checkpoint.variant)
        return SurfaceBatcher(
            stager=TextStager(
                tokenizer=self.tokenizer,
                surface=self.surface,
                withhold_keyword=checkpoint.provenance.withheld_keyword,
                keyword_expand_p=INFERENCE_KEYWORD_EXPAND_P,
                truncations=TruncationLog(log_truncation(logger, "evaluation")),
            ),
            resolver=KeyResolver(self.sidecars, self.surface),
            masks=masks,
            e_dim=checkpoint.encoder_config.e_dim,
            widths=feature_widths(records, layout=layout),
            materializer=Materializer(self.device, masks, self.identity_table),
            context_dropout=0.0,
            rng=random.Random(EVALUATION_SEED),
            input_layout=layout,
        )


def load_model(
    config, checkpoint, *, vocab_path: Path, keyword_path: Path,
    sidecars: SidecarCache | None = None,
) -> LoadedModel:
    """``checkpoint``'s weights on the device, with its tokenizer;
    ``sidecars`` lets several loads share one cache."""
    device = default_device()
    encoder, model = _load_weights(checkpoint, device)
    return LoadedModel(
        checkpoint=checkpoint,
        encoder=encoder,
        model=model,
        tokenizer=tokenizer_for(checkpoint, vocab_path, keyword_path),
        identity_table=_identity_table(checkpoint, device),
        surface=surface_of(vocab_path),
        device=device,
        sidecars=sidecars if sidecars is not None else build_sidecars(config),
    )


def load_runnable(
    config, checkpoint, *, vocab_path: Path, keyword_path: Path, records,
) -> Runnable:
    """One checkpoint, loaded and ready to run over ``records``."""
    return load_model(
        config, checkpoint, vocab_path=vocab_path, keyword_path=keyword_path,
    ).runnable(records)


#: Model state a shipped checkpoint may lack: the training-only heads (FR-076)
#: and the zero-initialized option-kind embedding (FR-063a). Anything else
#: missing is refused, or it would run at random initial weights.
OPTIONAL_STATE_PREFIXES: tuple[str, ...] = (
    *(f"{module}." for module in TRAINING_ONLY_MODULES),
    "option_kind_embedding.",
)


def _load_weights(checkpoint, device) -> tuple[AbilityEncoder, EffectModel]:
    """The checkpoint's encoder and model on ``device``, in evaluation mode,
    at the size the checkpoint recorded (FR-061)."""
    encoder = AbilityEncoder(checkpoint.encoder_config).to(device)
    if checkpoint.encoder_state:
        encoder.load_state_dict(checkpoint.encoder_state)
    model = EffectModel(checkpoint.model_config).to(device)
    missing, unexpected = model.load_state_dict(checkpoint.model_state, strict=False)
    absent = [key for key in missing if not key.startswith(OPTIONAL_STATE_PREFIXES)]
    if absent or unexpected:
        raise RuntimeError(
            f"the checkpoint's model state does not fit the model it declares: "
            f"missing {absent}, unexpected {list(unexpected)}"
        )
    return encoder.eval(), model.eval()


def _identity_table(checkpoint, device) -> IdentityTable | None:
    """The identity baseline's learned ``e`` table; None for any other variant."""
    if not baseline_masks(checkpoint.variant).identity_embedding:
        return None
    saved = checkpoint.extra.get(IDENTITY_TABLE_KEY)
    if saved is not None:
        return IdentityTable.from_state(saved).to(device)
    logger.warning(
        "The identity baseline at %s carries no embedding table, so "
        "its e vectors are random rather than the ones it trained. "
        "Gate 1's margin against it is not meaningful — retrain the "
        "baseline with a build that saves the table.",
        checkpoint.variant,
    )
    return IdentityTable((), checkpoint.encoder_config.e_dim).to(device)


def evaluation_fields(records) -> tuple[FieldSpec, ...]:
    """The fields active for ``records``' classes, the sparse group included:
    the gates score a finished model."""
    return active_fields(
        frozenset(sampling_class(r) for r in records), sparse=True,
    )

"""Load a saved checkpoint back into something that can be run over records.

Training builds the encoder, the model and the batcher from a config; the
evaluator has to build the same three from a checkpoint, and they must match or
the numbers it reports describe a different model than the one that trained.
That is why the batcher comes from :mod:`effects.application.surface_batching`
rather than being assembled again here: the variant masks, the encoding surface
and the identity table are exactly what distinguish a baseline from the shipping
model, and gate 1 is decided by comparing the two.

The identity baseline is the case that would break quietly. Its ``e`` comes from
a free embedding table rather than the encoder, and that table is part of what it
learned, so a run that rebuilt it empty would score the baseline on random
vectors and hand gate 1 a margin it did not earn.
"""

from __future__ import annotations

import logging
import random
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from itertools import batched
from pathlib import Path

import torch

from effects.application.ability_text import KeyResolver
from effects.application.surface_batching import (
    IdentityTable,
    Materializer,
    SurfaceBatcher,
)
from effects.application.text_staging import TextStager
from effects.application.train_effect_model import RANDOM_SEED
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

IDENTITY_TABLE_KEY = "identity_table"


def build_sidecars(config) -> SidecarCache:
    """The sidecar roots ``--cards-folder`` names (FR-063b).

    A folder keyed by the tree its name ends in, so a kept-aside copy such as
    ``output/gen1-cardsfolder`` serves the keys that name ``cardsfolder/``.
    """
    return SidecarCache(sidecar_roots(
        config.cards_folders, getattr(config, "variant_scripts", None),
    ))


def tokenizer_for(checkpoint, vocab_path: Path, keyword_path: Path) -> AbilityTokenizer:
    """The tokenizer ``checkpoint`` trained with, over its verified files.

    Under the rules the checkpoint recorded rather than a training run's, so a
    gen-1 checkpoint reads its script vocabulary with the prose grammar.
    """
    return load_tokenizer(
        vocab_path, keyword_path,
        training_settings=checkpoint.training_settings or {},
    )


#: Records per forward pass when gate 1 and the breakdowns score a checkpoint.
BATCH_RECORDS = 32

#: Observations per forward pass in gate 2: half the record batch, because
#: every observation is built twice — as collected and perturbed — in one batch.
BATCH_OBSERVATIONS = BATCH_RECORDS // 2


@dataclass(frozen=True, slots=True)
class Runnable:
    """One checkpoint, ready to run over one set of records.

    ``fields`` is the active-field tuple for the classes those records carry,
    the sparse group included — the evaluator scores a finished model, not one
    partway through its schedule.
    """

    encoder: AbilityEncoder
    model: EffectModel
    batcher: SurfaceBatcher
    fields: tuple[FieldSpec, ...]

    def inference_chunks[T](self, items: Iterable[T], size: int) -> Iterator[list[T]]:
        """``items`` in chunks of ``size``, each yielded under inference mode.

        Sets evaluation mode first although a loaded checkpoint is already in
        it: a caller that builds its own ``Runnable``, such as a probe, may
        hand over modules that were training, and dropout would then make the
        numbers depend on a coin flip.
        """
        self.encoder.eval()
        self.model.eval()
        with torch.inference_mode():
            for chunk in batched(items, size):
                yield list(chunk)

    def keyword_resolver(self) -> KeywordResolver:
        """A resolver over the batcher's own sidecar cache, never a second one.

        The resolver decides which entity carries a keyword and the surface
        builder decides which slot to drop for it, and the two reading
        different joins is exactly how a perturbation strips nothing. One per
        pass: it memoizes each provenance key's keyword.
        """
        return KeywordResolver(self.batcher.sidecars)


@dataclass(frozen=True, slots=True)
class LoadedModel:
    """A checkpoint's weights on the device, before any record is seen.

    Split from :class:`Runnable` because the expensive part — the weights, the
    identity table, the tokenizer — does not depend on the records, while the
    batcher's slot widths and the active fields do. One load serves every
    record set an evaluation scores; each set gets a batcher of its own, with
    its own generator, so scoring one set never moves another's draws.
    """

    checkpoint: object
    encoder: AbilityEncoder
    model: EffectModel
    tokenizer: AbilityTokenizer
    identity_table: IdentityTable | None
    surface: str
    device: torch.device
    sidecars: SidecarCache

    def runnable(self, records) -> Runnable:
        """The model with a batcher measured from ``records``."""
        return Runnable(
            self.encoder, self.model, self._batcher(records),
            evaluation_fields(records),
        )

    def _batcher(self, records) -> SurfaceBatcher:
        checkpoint = self.checkpoint
        # The inputs the checkpoint learned on: a gen-1 checkpoint, or a gen-2
        # arm trained before the layout was recorded, reads the earlier layout
        # bit for bit (FR-063c).
        layout = InputLayout.of(checkpoint.training_settings)
        masks = baseline_masks(checkpoint.variant)
        return SurfaceBatcher(
            stager=TextStager(
                tokenizer=self.tokenizer,
                surface=self.surface,
                withhold_keyword=checkpoint.provenance.withheld_keyword,
                # Both are training-time augmentations. Sampling them at
                # evaluation would make the gate's numbers depend on a coin flip.
                keyword_expand_p=INFERENCE_KEYWORD_EXPAND_P,
                truncations=TruncationLog(log_truncation(logger, "evaluation")),
            ),
            resolver=KeyResolver(self.sidecars, self.surface),
            masks=masks,
            e_dim=checkpoint.encoder_config.e_dim,
            widths=feature_widths(records, layout=layout),
            materializer=Materializer(self.device, masks, self.identity_table),
            context_dropout=0.0,
            rng=random.Random(RANDOM_SEED),
            input_layout=layout,
        )


def load_model(
    config, checkpoint, *, vocab_path: Path, keyword_path: Path,
    sidecars: SidecarCache | None = None,
) -> LoadedModel:
    """``checkpoint``'s weights on the device, with its tokenizer.

    ``sidecars`` lets several loads share one cache; left out, the roots
    ``config`` names are read into a new one.
    """
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


#: Model state a shipped checkpoint may lack. The training-only heads are
#: filtered out at save time (FR-076), and the option-kind embedding, which
#: starts at zero, postdates checkpoints that read every row as it trained
#: without it (FR-063a). Anything else missing would run at its random
#: initial weights while every number it produced still looked like a result.
OPTIONAL_STATE_PREFIXES: tuple[str, ...] = (
    *(f"{module}." for module in TRAINING_ONLY_MODULES),
    "option_kind_embedding.",
)


def _load_weights(checkpoint, device) -> tuple[AbilityEncoder, EffectModel]:
    """The checkpoint's encoder and model on ``device``, in evaluation mode."""
    # Built at the size the checkpoint recorded (FR-061): a non-default arm
    # loads without its flags restated.
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
    """The fields active for ``records``' classes once training has finished.

    The sparse group included: the gates score the model as it finished
    training, not as it started.
    """
    return active_fields(
        frozenset(sampling_class(r) for r in records), sparse=True,
    )


#: Where a shard line names its record, read before the line is parsed.
_RECORD_ID = re.compile(r'"record_id":"([^"]*)"')
_RECORD_ID_SPAN = 256


def training_acting_keys(directory: Path) -> set[tuple]:
    """Every distinct ``acting_keys`` tuple among the records under ``directory``.

    ``directory`` is the curated corpus's ``training/``: what the checkpoint
    actually trained on, and so what the identity baseline could have
    memorized. A record curation wrote several times (``--reuse-cap``) is
    parsed once; the set is all gate 1 needs, and the records are never held.
    """
    from effects.domain.records import acting_keys
    from effects.infrastructure.record_io import (
        iter_shard_lines,
        iter_shards,
        parse_record_line,
        record_from_dict,
    )

    shards = iter_shards(directory)
    read: set[str] = set()
    acting: set[tuple] = set()
    for done, shard in enumerate(shards, start=1):
        for line in iter_shard_lines(shard):
            found = _RECORD_ID.search(line, 0, _RECORD_ID_SPAN)
            if found is not None:
                if found.group(1) in read:
                    continue
                read.add(found.group(1))
            stripped = line.strip()
            if not stripped:
                continue
            keys = acting_keys(record_from_dict(parse_record_line(stripped)))
            if keys:
                acting.add(keys)
        if done % 50 == 0:
            logger.info(
                "gate 1: %d of %d training shards read, %d acting lines",
                done, len(shards), len(acting),
            )
    return acting


def card_disjoint_resolutions(corpus_path: Path, games: frozenset[str]) -> Iterator:
    """The resolution records of the curated corpus's card-disjoint ``games``, streamed.

    Resolutions only, the one kind gate 1's stratum holds; the stratum's other
    kinds are never parsed.
    """
    from effects.domain.records import RecordKind
    from effects.infrastructure.corpus_store import CorpusStore
    from effects.infrastructure.record_io import read_records_matching

    return read_records_matching(
        CorpusStore(corpus_path).card_disjoint_dir,
        game_ids=games, kinds={RecordKind.RESOLUTION.value},
    )

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
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import torch

from effects.application.surface_batching import (
    IDENTITY_TABLE_SIZE,
    SurfaceBatcher,
)
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
    tokenizer_surface,
)
from effects.domain.effect_head_input import (
    INPUT_LAYOUT_KEY,
    feature_widths,
)
from effects.domain.effect_model import EffectModel, FieldSpec, active_fields
from effects.domain.sampling import sampling_class
from effects.domain.variants import variant_masks
from effects.infrastructure.sidecar_io import SidecarCache, sidecar_roots
from price_predictor.infrastructure.tokenizer_store import load_vocabulary

logger = logging.getLogger(__name__)

#: The step the evaluator reads the curriculum at. Past every curriculum
#: threshold, so the sparse fields are active and the gate scores the model as
#: it finished training rather than as it started.
EVALUATION_STEP = 1 << 30

IDENTITY_TABLE_KEY = "identity_table"


def build_sidecars(config) -> SidecarCache:
    """The sidecar roots ``--cards-folder`` names (FR-063b).

    A folder keyed by the tree its name ends in, so a kept-aside copy such as
    ``output/gen1-cardsfolder`` serves the keys that name ``cardsfolder/``.
    """
    return SidecarCache(sidecar_roots(
        config.cards_folders, getattr(config, "variant_scripts", None),
    ))


def input_layout_of(checkpoint) -> str | None:
    """The input layout a checkpoint trained on; None for one that recorded none (FR-063c)."""
    return (checkpoint.training_settings or {}).get(INPUT_LAYOUT_KEY)


def tokenizer_for(checkpoint, vocab_path: Path, keyword_path: Path) -> AbilityTokenizer:
    """The tokenizer ``checkpoint`` trained with, over its verified files.

    A missing keyword table tokenizes with no definitions rather than failing:
    a checkpoint trained before the table existed recorded none.
    """
    from effects.application.extract_keyword_definitions import (
        load_keyword_definitions,
    )

    definitions = (
        load_keyword_definitions(keyword_path) if Path(keyword_path).exists()
        else {}
    )
    # The grammar the checkpoint trained under, which for gen-1 is the prose
    # grammar even on its script vocabulary.
    return AbilityTokenizer(
        load_vocabulary(Path(vocab_path)), definitions,
        surface=tokenizer_surface(
            surface_of(vocab_path), checkpoint.training_settings,
        ),
    )


@dataclass(frozen=True, slots=True)
class Runnable:
    """One checkpoint, ready to run over one set of records.

    ``fields`` is the active-field tuple for the classes those records carry,
    read at a step past every curriculum threshold — the evaluator scores a
    finished model, not one partway through its schedule.
    """

    encoder: AbilityEncoder
    model: EffectModel
    batcher: SurfaceBatcher
    fields: tuple[FieldSpec, ...]


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
    identity_table: torch.nn.Embedding | None
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
        layout = input_layout_of(checkpoint)
        return SurfaceBatcher(
            tokenizer=self.tokenizer,
            sidecars=self.sidecars,
            masks=variant_masks(checkpoint.variant),
            surface=self.surface,
            e_dim=checkpoint.encoder_config.e_dim,
            widths=feature_widths(records, layout=layout),
            device=self.device,
            withhold_keyword=checkpoint.provenance.withheld_keyword,
            # Both are training-time augmentations. Sampling them at evaluation
            # would make the gate's numbers depend on a coin flip.
            keyword_expand_p=INFERENCE_KEYWORD_EXPAND_P,
            context_dropout=0.0,
            rng=random.Random(RANDOM_SEED),
            identity_table=self.identity_table,
            truncations=TruncationLog(log_truncation(logger, "evaluation")),
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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


def _load_weights(checkpoint, device) -> tuple[AbilityEncoder, EffectModel]:
    # Built at the size the checkpoint recorded (FR-061): a non-default arm
    # loads without its flags restated.
    encoder = AbilityEncoder(checkpoint.encoder_config).to(device)
    if checkpoint.encoder_state:
        encoder.load_state_dict(checkpoint.encoder_state)
    model = EffectModel(checkpoint.model_config).to(device)
    # The training-only heads are filtered out at save time, so the shipped
    # state is a strict subset of what the model declares.
    model.load_state_dict(checkpoint.model_state, strict=False)
    return encoder, model


def _identity_table(checkpoint, device) -> torch.nn.Embedding | None:
    """The identity baseline's learned ``e`` table; None for any other variant."""
    if not variant_masks(checkpoint.variant).identity_embedding:
        return None
    table = torch.nn.Embedding(
        IDENTITY_TABLE_SIZE, checkpoint.encoder_config.e_dim,
    ).to(device)
    saved = checkpoint.extra.get(IDENTITY_TABLE_KEY)
    if saved is not None:
        table.load_state_dict(saved)
    else:
        logger.warning(
            "The identity baseline at %s carries no embedding table, so "
            "its e vectors are random rather than the ones it trained. "
            "Gate 1's margin against it is not meaningful — retrain the "
            "baseline with a build that saves the table.",
            checkpoint.variant,
        )
    return table


def evaluation_fields(records) -> tuple[FieldSpec, ...]:
    """The fields active for ``records``' classes once training has finished."""
    return active_fields(
        present_classes=frozenset(sampling_class(r) for r in records),
        step=EVALUATION_STEP,
        curriculum_step=1,
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

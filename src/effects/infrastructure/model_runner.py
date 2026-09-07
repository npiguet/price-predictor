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
from pathlib import Path

import torch

from effects.application.surface_batching import (
    IDENTITY_TABLE_SIZE,
    SurfaceBatcher,
)
from effects.application.train_effect_model import (
    RANDOM_SEED,
    sampling_class,
    variant_masks,
)
from effects.domain.ability_encoder import AbilityEncoder, surface_of
from effects.domain.ability_tokenizer import AbilityTokenizer
from effects.domain.effect_head_input import (
    SlotKind,
    act_features,
    card_features,
    global_features,
    player_features,
)
from effects.domain.effect_model import EffectModel, active_fields
from effects.infrastructure.sidecar_io import SidecarCache
from price_predictor.infrastructure.tokenizer_store import load_vocabulary

logger = logging.getLogger(__name__)

#: The step the evaluator reads the curriculum at. Past every curriculum
#: threshold, so the sparse fields are active and the gate scores the model as
#: it finished training rather than as it started.
EVALUATION_STEP = 1 << 30

IDENTITY_TABLE_KEY = "identity_table"


def build_sidecars(config) -> SidecarCache:
    roots = {Path(f).name: Path(f) for f in config.cards_folders}
    variant_scripts = getattr(config, "variant_scripts", None)
    if variant_scripts:
        roots["variant-scripts"] = Path(variant_scripts)
    return SidecarCache(roots)


def feature_widths(records) -> dict[SlotKind, int]:
    """Measure each slot kind's width from real records, as training does."""
    probe = records[0]
    widths = {
        SlotKind.GLOBAL: len(global_features(probe)),
        SlotKind.ACT: len(act_features(probe)),
        SlotKind.PLAYER: 0,
        SlotKind.CARD: 0,
    }
    for record in records:
        if record.state.players:
            widths[SlotKind.PLAYER] = len(
                player_features(record.state.players[0], record)
            )
        if record.state.entities:
            widths[SlotKind.CARD] = len(
                card_features(record.state.entities[0], record)
            )
        if widths[SlotKind.PLAYER] and widths[SlotKind.CARD]:
            break
    return widths


def load_runnable(
    config, checkpoint, *, vocab_path: Path, keyword_path: Path, records,
):
    """``(encoder, model, batcher, fields)`` for one checkpoint.

    ``fields`` is the active-field tuple for the classes the records actually
    carry, read at a step past every curriculum threshold — the evaluator scores
    a finished model, not one partway through its schedule.
    """
    from effects.application.extract_keyword_definitions import (
        load_keyword_definitions,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    definitions = (
        load_keyword_definitions(keyword_path) if Path(keyword_path).exists()
        else {}
    )
    tokenizer = AbilityTokenizer(load_vocabulary(Path(vocab_path)), definitions)

    encoder = AbilityEncoder(checkpoint.encoder_config).to(device)
    if checkpoint.encoder_state:
        encoder.load_state_dict(checkpoint.encoder_state)
    model = EffectModel(checkpoint.model_config).to(device)
    # The training-only heads are filtered out at save time, so the shipped
    # state is a strict subset of what the model declares.
    model.load_state_dict(checkpoint.model_state, strict=False)

    masks = variant_masks(checkpoint.variant)
    identity_table = None
    if masks.identity_embedding:
        identity_table = torch.nn.Embedding(
            IDENTITY_TABLE_SIZE, checkpoint.encoder_config.e_dim,
        ).to(device)
        saved = checkpoint.extra.get(IDENTITY_TABLE_KEY)
        if saved is not None:
            identity_table.load_state_dict(saved)
        else:
            logger.warning(
                "The identity baseline at %s carries no embedding table, so "
                "its e vectors are random rather than the ones it trained. "
                "Gate 1's margin against it is not meaningful — retrain the "
                "baseline with a build that saves the table.",
                checkpoint.variant,
            )

    batcher = SurfaceBatcher(
        tokenizer=tokenizer,
        sidecars=build_sidecars(config),
        masks=masks,
        surface=surface_of(vocab_path),
        e_dim=checkpoint.encoder_config.e_dim,
        widths=feature_widths(records),
        device=device,
        withhold_keyword=checkpoint.provenance.withheld_keyword,
        # Both are training-time augmentations. Sampling them at evaluation
        # would make the gate's numbers depend on a coin flip.
        keyword_expand_p=0.0,
        context_dropout=0.0,
        rng=random.Random(RANDOM_SEED),
        identity_table=identity_table,
    )
    fields = active_fields(
        present_classes=frozenset(sampling_class(r) for r in records),
        step=EVALUATION_STEP,
        curriculum_step=1,
    )
    return encoder, model, batcher, fields


def stratum_records(config, checkpoint) -> tuple[list, list, str]:
    """``(card_disjoint, training, why_not)`` from the checkpoint's own split.

    The split comes from the checkpoint rather than being recomputed: the corpus
    is append-only, so a recomputed split would score the gate partly on games
    the model trained on.

    The training records come back too, because gate 1's stratum is defined by
    what the training games contain — a text the baseline never saw is the only
    one it cannot simply recall.
    """
    from effects.infrastructure.record_io import read_records

    held_out = frozenset(checkpoint.provenance.card_disjoint_games)
    if not held_out:
        return [], [], (
            "the checkpoint records no card-disjoint games, so there is no "
            "held-out stratum to score gate 1 on"
        )
    validation = held_out | frozenset(checkpoint.provenance.game_disjoint_games)
    card_disjoint, training = [], []
    for record in read_records(Path(config.records_dir)):
        if record.game_id in held_out:
            card_disjoint.append(record)
        elif record.game_id not in validation:
            training.append(record)
    if not card_disjoint:
        return [], [], (
            f"none of the checkpoint's {len(held_out)} card-disjoint games "
            f"appear in {config.records_dir}"
        )
    return card_disjoint, training, ""

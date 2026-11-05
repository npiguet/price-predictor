"""The loss denominators' startup pre-pass (FR-062, research T3).

Before the first step, a run reads a seeded sample of its training shards
once, prepares their records as validation does (no augmentation, no copies),
and keeps each shipped term's constant-prediction loss over them, clamped at
``--deviance-floor``. The run stores the result in ``resume.pt`` and its
checkpoints and a resume reuses it, so a run divides by one set of numbers
from its first step to its last.
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from effects.application.step_preparation import StepPreparer
from effects.domain.effect_head_layout import FieldSpec
from effects.domain.loss_denominators import LossDenominators
from effects.domain.loss_floor import TargetSample, constant_predictor_losses
from effects.infrastructure.shard_io import load_shard

logger = logging.getLogger(__name__)

#: Training shards the pre-pass reads: a few thousand records each, enough
#: for every term the corpus supervises often, read once at startup.
DENOMINATOR_SHARDS = 8


def denominator_shards(shards: Sequence[Path], *, seed: int) -> list[Path]:
    """The pre-pass's shards: :data:`DENOMINATOR_SHARDS` drawn from ``shards``
    under the run's seed, independent of every epoch's own draw."""
    pool = sorted(Path(shard) for shard in shards)
    draw = random.Random(f"loss-denominators:{seed}")
    return sorted(draw.sample(pool, min(DENOMINATOR_SHARDS, len(pool))))


def loss_denominators(
    preparer: StepPreparer, shards: Sequence[Path], *,
    fields: tuple[FieldSpec, ...], batch_size: int, floor: float,
    read: Callable[[Path], list] = load_shard,
) -> LossDenominators:
    """Each shipped term's constant-prediction loss over ``shards``' records,
    in the batches validation would score them in, clamped at ``floor``;
    ``read`` loads a shard's records."""
    started = time.monotonic()
    sample = TargetSample()
    for shard in shards:
        records = read(shard)
        for start in range(0, len(records), batch_size):
            step = preparer.prepare_scoring_step(records[start:start + batch_size], fields)
            sample.add(step.targets, step.heads)
    denominators = LossDenominators.of(
        constant_predictor_losses(sample, fields=fields), floor=floor,
    )
    terms = sum(len(values) for values in denominators.values.values())
    floored = sum(
        value <= floor for values in denominators.values.values() for value in values.values()
    )
    logger.info(
        "Loss denominators from %d training shard(s), %d records, in %.0fs: "
        "%d term(s) supervised, %d of them at the --deviance-floor %.3g; the "
        "rest divide by it too.",
        len(shards), sample.records, time.monotonic() - started, terms, floored, floor,
    )
    return denominators

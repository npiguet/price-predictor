"""The loss denominators' startup pre-pass (FR-062, research T3).

Before the first step, a run prepares a seeded sample of training steps as
training prepares them (the rarity-weighted draw, the lethality copies, the
rule mask; finding 15.4) and keeps each shipped term's constant-prediction
loss over their targets, clamped at ``--deviance-floor``. The run stores the
result in ``resume.pt`` and its checkpoints and a resume reuses it, so a run
divides by one set of numbers from its first step to its last.
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from effects.application.step_preparation import ShardTask, StepPreparer, StepReady
from effects.domain.effect_head_layout import PER_ENTITY_FIELDS
from effects.domain.loss_denominators import LossDenominators
from effects.domain.loss_floor import TargetSample, constant_predictor_losses
from effects.infrastructure.shard_io import load_shard

logger = logging.getLogger(__name__)

#: Training steps the pre-pass prepares: at the default 32 records a step,
#: about as many records as eight shards of a few thousand, enough for every
#: term the corpus supervises often, prepared once at startup.
DENOMINATOR_STEPS = 768

#: The epoch the pre-pass's tasks are keyed by: no training epoch (they count
#: from 1), so its draws are its own and the same under any worker count.
DENOMINATOR_EPOCH = 0


def denominator_shards(shards: Sequence[Path], *, seed: int, count: int) -> list[Path]:
    """The pre-pass's ``count`` shards, drawn from ``shards`` under the run's
    seed, independent of every epoch's own draw."""
    pool = sorted(Path(shard) for shard in shards)
    draw = random.Random(f"loss-denominators:{seed}")
    return sorted(draw.sample(pool, min(count, len(pool))))


def denominator_tasks(
    shards: Sequence[Path], *, seed: int, steps_per_epoch: int, shards_per_epoch: int,
) -> list[ShardTask]:
    """The pre-pass's shards, each read for the steps an epoch gives a shard
    (at most :data:`DENOMINATOR_STEPS`), so its records are drawn by rarity as
    an epoch draws them, until :data:`DENOMINATOR_STEPS` are planned. Every
    per-entity field is scored, the sparse group's included."""
    if not shards:
        return []
    drawn = max(1, min(shards_per_epoch, len(shards)))
    budget = min(-(-steps_per_epoch // drawn), DENOMINATOR_STEPS)
    count = -(-DENOMINATOR_STEPS // budget)
    return [
        ShardTask(
            shard=shard, epoch=DENOMINATOR_EPOCH, position=position, budget=budget,
            fields=PER_ENTITY_FIELDS,
        )
        for position, shard in enumerate(
            denominator_shards(shards, seed=seed, count=count), start=1,
        )
    ]


def loss_denominators(
    preparer: StepPreparer, tasks: Sequence[ShardTask], *, floor: float,
    read: Callable[[Path], list] = load_shard,
) -> LossDenominators:
    """Each shipped term's constant-prediction loss over the training steps
    ``preparer`` prepares for ``tasks``, over the per-entity fields the tasks
    train, clamped at ``floor``; ``read`` loads a shard's records. The
    training-only heads' targets are left out."""
    started = time.monotonic()
    sample = TargetSample()
    for task in tasks:
        for message in preparer.shard_steps(task, read(task.shard)):
            if isinstance(message, StepReady):
                step = message.prepared
                sample.add(step.targets, _shipped(step.heads, preparer.heads))
    fields = tasks[0].fields if tasks else ()
    denominators = LossDenominators.of(
        constant_predictor_losses(sample, fields=fields), floor=floor,
    )
    _report(denominators, shards=len(tasks), records=sample.records,
            seconds=time.monotonic() - started)
    return denominators


def _shipped(heads: dict, training_only) -> dict:
    """A step's head targets without the training-only heads', which no
    denominator divides."""
    return {head: targets for head, targets in heads.items() if head not in training_only}


def _report(
    denominators: LossDenominators, *, shards: int, records: int, seconds: float,
) -> None:
    floor = denominators.floor
    terms = sum(len(values) for values in denominators.values.values())
    floored = sum(
        value <= floor for values in denominators.values.values() for value in values.values()
    )
    logger.info(
        "Loss denominators from %d training shard(s), %d records prepared as "
        "training steps, in %.0fs: %d term(s) supervised, %d of them at the "
        "--deviance-floor %.3g; the rest divide by it too.",
        shards, records, seconds, terms, floored, floor,
    )

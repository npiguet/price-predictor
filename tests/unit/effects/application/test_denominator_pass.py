"""The loss denominators' startup pre-pass (FR-062, research T3, T144)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from effects.application.denominator_pass import (
    DENOMINATOR_SHARDS,
    denominator_shards,
    loss_denominators,
)
from effects.application.train_effect_model import TrainEffectModelConfig
from effects.domain.effect_head_layout import FIELDS_BY_NAME
from effects.domain.loss_denominators import LossDenominators
from effects.domain.loss_floor import PER_ENTITY, TargetSample, constant_predictor_losses
from effects.infrastructure.cli import build_parser
from tests.unit.effects.builders import fields_everywhere, make_entity_targets

FIELDS = (FIELDS_BY_NAME["damage_taken"], FIELDS_BY_NAME["tap_state"])


class _Preparer:
    """Prepares a record, here a damage amount, into one-entity targets, and
    remembers each batch it was handed."""

    def __init__(self) -> None:
        self.batches: list[list] = []

    def prepare_scoring_step(self, records, fields):
        self.batches.append(list(records))
        damage = torch.tensor([[float(amount)] for amount in records])
        tapped = torch.tensor([[float(amount > 2)] for amount in records])
        return SimpleNamespace(
            targets=make_entity_targets(
                torch.ones(len(records), 1),
                fields_everywhere({"damage_taken": damage, "tap_state": tapped}),
            ),
            heads={},
        )


SHARDS = {
    Path("a.jsonl.gz"): [0, 0, 1, 5, 0],
    Path("b.jsonl.gz"): [3, 0, 0],
}


def test_the_denominators_are_the_floored_constant_losses_of_the_shards():
    preparer = _Preparer()
    got = loss_denominators(
        preparer, list(SHARDS), fields=FIELDS, batch_size=2, floor=0.04,
        read=SHARDS.__getitem__,
    )
    # Each shard in validation-sized batches, never one batch across two shards.
    assert preparer.batches == [[0, 0], [1, 5], [0], [3, 0], [0]]

    sample = TargetSample()
    for records in SHARDS.values():
        sample.add(_Preparer().prepare_scoring_step(records, FIELDS).targets)
    expected = LossDenominators.of(constant_predictor_losses(sample, fields=FIELDS), floor=0.04)
    assert got == expected
    assert got.of_term(PER_ENTITY, "damage_taken") > 0.04


def test_a_term_below_the_floor_is_raised_to_it():
    """A field the sample's constant already predicts to 0.01 nats or less."""
    got = loss_denominators(
        _Preparer(), [Path("quiet")], fields=FIELDS, batch_size=4, floor=0.5,
        read=lambda path: [0] * 199 + [1],
    )
    assert got.of_term(PER_ENTITY, "damage_taken") == 0.5
    assert got.of_term(PER_ENTITY, "tap_state") == 0.5


def test_the_pre_pass_draws_eight_shards_by_the_run_s_seed():
    shards = [Path(f"shard-{index:03d}.jsonl.gz") for index in range(40)]
    drawn = denominator_shards(shards, seed=11)
    assert len(drawn) == DENOMINATOR_SHARDS == 8
    assert set(drawn) <= set(shards)
    assert denominator_shards(list(reversed(shards)), seed=11) == drawn
    assert denominator_shards(shards, seed=12) != drawn
    assert denominator_shards(shards[:3], seed=11) == sorted(shards[:3])


def test_the_deviance_floor_flag_defaults_to_four_hundredths_of_a_nat():
    parsed = build_parser().parse_args(["train-effect-model", "--corpus", "c"])
    assert parsed.deviance_floor == 0.04
    assert build_parser().parse_args(
        ["train-effect-model", "--corpus", "c", "--deviance-floor", "0.1"],
    ).deviance_floor == 0.1


def test_a_floor_of_zero_or_less_is_refused():
    with pytest.raises(ValueError, match="deviance-floor"):
        TrainEffectModelConfig(corpus="c", deviance_floor=0.0)

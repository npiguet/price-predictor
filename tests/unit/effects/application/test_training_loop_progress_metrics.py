"""Gate 1's progress margins come off the card-disjoint validation pass.

They used to come from a second pass over the same sample, ``gate_one.measure``,
on the same weights. One epoch trains end to end on the fixture records, at the
default batch size so the sample spans more than one batch; the margins the
validation pass fed are then set against ``measure`` run the old way on the
weights that pass scored.
"""

from __future__ import annotations

from dataclasses import astuple, replace

import numpy as np
import pytest
import torch

from effects.application import run_context
from effects.application.gate_one import measure
from effects.application.model_loading import Runnable
from effects.application.validation_pass import FieldParts, Validator
from effects.domain.effect_model import EffectModel
from effects.infrastructure.shard_io import write_shard
from tests.unit.effects.builders import gen1_training_config, make_training_loop

BATCH_SIZE = 32


@pytest.fixture(scope="module")
def scored(corpus, records, tmp_path_factory):
    """One epoch's margins both ways, and its card-disjoint loss with and
    without the margins being fed."""
    assert len(records) > BATCH_SIZE
    root = tmp_path_factory.mktemp("progress")
    sample = root / "card-disjoint.jsonl.gz"
    write_shard(sample, records)
    seen: dict = {}

    class _Measured(Validator):
        """The real validator, setting each epoch's margins beside the old
        way's on the weights it scored."""

        def epoch(self, run, samples, fields):
            card, game = super().epoch(run, samples, fields)
            seen["fed"] = card.progress()
            batcher = run.preparer.scoring_batcher()
            batcher.materializer = run.scoring
            seen["measured"] = measure(
                samples.card_disjoint, Runnable(run.encoder, run.model, batcher, fields),
            )
            seen["loss"] = card.loss
            seen["loss without margins"] = self.score(
                samples.card_disjoint, run, fields, (FieldParts(),),
            )
            run.encoder.train()
            run.model.train()
            return card, game

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(
            run_context, "EffectModel",
            lambda config: EffectModel(replace(
                config, d_model=32, n_layers=1, n_heads=2, ff_dim=64,
            )),
        )
        loop = make_training_loop(
            gen1_training_config(
                corpus, root / "out", prefetch_workers=0, epochs=1,
                batch_size=BATCH_SIZE,
            ),
            training_shards=corpus.shards,
            validation_samples={"card-disjoint": sample, "game-disjoint": corpus.sample},
        )
        loop.validator = _Measured(loop.config, loop.device, loop.losses)
        code = loop.execute()
    assert code == 0
    return seen


def test_the_fed_margins_are_the_ones_a_second_pass_measures(scored):
    """Exactly, on the CPU: the same weights, the same records per batch."""
    np.testing.assert_array_equal(astuple(scored["fed"]), astuple(scored["measured"]))


def test_the_margins_were_measured_at_all(scored):
    """Equal NaNs would pass the test above vacuously."""
    assert not np.isnan(astuple(scored["fed"])).any()


def test_feeding_them_leaves_the_validation_loss_as_it_was(scored):
    assert scored["loss"] == scored["loss without margins"]

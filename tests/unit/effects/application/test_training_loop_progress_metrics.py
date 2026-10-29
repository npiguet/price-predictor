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

from effects.application import training_loop as loop_module
from effects.application.gate_one import measure
from effects.application.training_loop import TrainingLoop
from effects.domain.effect_model import EffectModel
from effects.domain.split import CorpusSplit
from effects.infrastructure.record_io import write_shard
from tests.unit.effects.application.test_training_loop_workers import (  # noqa: F401
    _config,
    corpus,
    records,
)

BATCH_SIZE = 32


@pytest.fixture(scope="module")
def scored(corpus, records, tmp_path_factory):  # noqa: F811
    """One epoch's margins both ways, and its card-disjoint loss with and
    without the margins being fed."""
    assert len(records) > BATCH_SIZE
    root = tmp_path_factory.mktemp("progress")
    sample = root / "card-disjoint.jsonl.gz"
    write_shard(sample, records)
    seen: dict = {}
    real = TrainingLoop._validate_epoch

    def validate_epoch(self, epoch, run, tally, fields):
        result, parts, metrics = real(self, epoch, run, tally, fields)
        seen["fed"] = metrics
        seen["measured"] = measure(
            self.card_disjoint, run.encoder, run.model,
            self.preparer.batcher(training=False, materializer=self._scoring),
            fields=fields,
        )
        seen["loss"] = result.card_disjoint_loss
        seen["loss without margins"] = self._validate(
            self.card_disjoint, run, fields=fields, parts={},
        )
        run.encoder.train()
        run.model.train()
        return result, parts, metrics

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(
            loop_module, "EffectModel",
            lambda config: EffectModel(replace(
                config, d_model=32, n_layers=1, n_heads=2, ff_dim=64,
            )),
        )
        monkeypatch.setattr(TrainingLoop, "_validate_epoch", validate_epoch)
        code = TrainingLoop(
            _config(
                corpus, root / "out", prefetch_workers=0, epochs=1,
                batch_size=BATCH_SIZE,
            ),
            split=CorpusSplit(),
            training_shards=corpus.shards,
            validation_samples={"card-disjoint": sample, "game-disjoint": corpus.sample},
            holdout_permille=20, holdout_max_carriers=8, gate_one_records=0,
            rarity={}, corpus_digest="",
        ).execute()
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

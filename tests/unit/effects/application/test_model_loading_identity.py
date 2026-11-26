"""The identity baseline's embedding table rides in its checkpoint, and a
checkpoint without it is refused (task H04, finding 11.8)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from effects.application.model_loading import _identity_table
from effects.application.surface_batching import IDENTITY_TABLE_KEY, IdentityTable
from effects.domain.baselines import Baseline

_CPU = torch.device("cpu")


def _checkpoint(variant, extra):
    return SimpleNamespace(
        variant=variant, extra=extra, encoder_config=SimpleNamespace(e_dim=4),
    )


def test_an_identity_checkpoint_without_its_table_is_refused_by_the_tables_name():
    with pytest.raises(ValueError, match=IDENTITY_TABLE_KEY):
        _identity_table(_checkpoint(Baseline.IDENTITY, {}), _CPU)


def test_an_identity_checkpoint_rebuilds_the_table_it_trained():
    table = IdentityTable(["flying", "trample"], 4)
    rebuilt = _identity_table(
        _checkpoint(Baseline.IDENTITY, {IDENTITY_TABLE_KEY: table.state()}), _CPU,
    )
    assert rebuilt.texts == table.texts
    assert torch.equal(rebuilt.embedding.weight, table.embedding.weight)


def test_every_other_variant_has_no_table():
    assert _identity_table(_checkpoint(Baseline.FULL, {}), _CPU) is None

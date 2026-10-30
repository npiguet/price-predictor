"""What a pinned batch of per-entity targets page-locks, and what it leaves.

Only what a step copies to the device is pinned: the field values and the
gather index. The gate, mask and presence stay ordinary host tensors, since
the loss picks its rows from them on the host.
"""

from __future__ import annotations

import pytest
import torch

from tests.unit.effects.builders import make_entity_targets

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="page-locked memory needs a CUDA runtime",
)


def test_a_pinned_batch_pins_what_reaches_the_device_and_nothing_else():
    gate = torch.tensor([[1.0, 0.0]])
    batch = make_entity_targets(gate, {"damage_taken": torch.tensor([[2.0, 0.0]])})
    pinned = batch.pinned()
    assert pinned.fields.values["damage_taken"].is_pinned()
    assert pinned.index.is_pinned()
    assert not pinned.gate.is_pinned()
    assert not pinned.mask.is_pinned()
    assert not pinned.fields.present["damage_taken"].is_pinned()
    assert torch.equal(pinned.fields.values["damage_taken"], batch.fields.values["damage_taken"])
    assert torch.equal(pinned.index, batch.index)

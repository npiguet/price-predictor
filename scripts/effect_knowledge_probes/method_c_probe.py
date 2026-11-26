"""Method C: a shallow trunk trained on frozen ``e``, scored beside the full model.

Nothing here does work at import time.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from effect_knowledge_probes import ablation, record_walk


class _NoTrunk:
    """Method C's zero-layer trunk: the slots reach the heads as embedded."""

    def __new__(cls):
        from torch import nn

        class NoTrunk(nn.Module):
            def forward(self, src, src_key_padding_mask=None, **_):  # noqa: ARG002
                return src

        return NoTrunk()


def _shallow_model(probe_model, layers: int):
    """A fresh effect model of ``layers`` trunk layers (none for 0), on the probe's device."""
    from effects.domain.effect_model import EffectModel

    config = replace(probe_model.checkpoint.model_config, n_layers=max(layers, 1))
    shallow = EffectModel(config).to(probe_model.device)
    if layers == 0:
        shallow.trunk = _NoTrunk()
    return shallow


def _stream(shards: Sequence[Path]):
    """The listed training shards' rows as curated, copies included, forever."""
    while True:
        yield from record_walk.records(shards, distinct=False)


def _train_step(shallow, optimizer, probe_model, chunk: list) -> None:
    """One optimiser step of the shallow trunk over ``chunk``."""
    import torch

    from effects.domain.effect_losses import EntityLossScoring, per_entity_loss
    from effects.domain.surface_collation import per_entity_targets

    fields = probe_model.fields
    with torch.no_grad():
        batch, surfaces, _encoded = probe_model.batcher.build(chunk, probe_model.encoder)
    targets = per_entity_targets(chunk, surfaces, fields)
    hidden = shallow(batch)
    device = hidden.device
    gathered = shallow.per_entity_at(hidden, targets.index.to(device))
    loss, _ = per_entity_loss(
        gathered, targets.fields_on(device),
        EntityLossScoring(fields=fields, log_dispersion=shallow.per_entity_dispersion()),
    )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def method_c(args, probe_model, training_shards: Sequence[Path], eval_records: list) -> dict:
    """Train a shallow trunk on frozen ``e`` and score it beside the full model.

    It trains on the training stratum as curated, copies included: method C
    is a training run, not a probe item.
    """
    import torch

    layers = args.method_c_layers
    shallow = _shallow_model(probe_model, layers)
    optimizer = torch.optim.AdamW(shallow.parameters(), lr=3e-4, weight_decay=1e-4)
    records, steps, started = _stream(training_shards), 0, time.time()
    shallow.train()
    while steps < args.method_c_steps:
        _train_step(shallow, optimizer, probe_model, [next(records) for _ in range(16)])
        steps += 1
    shallow.eval()
    return {
        "layers": layers, "steps": steps, "seconds": time.time() - started,
        "shallow": ablation.field_losses(shallow, probe_model, eval_records),
        "full": ablation.field_losses(probe_model.model, probe_model, eval_records),
    }

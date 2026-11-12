"""Time one trunk training step with and without relations (025 T105, T141a; quickstart stage 2).

Reads a fixed curated shard, lays its records out as token surfaces through
``SurfaceBuilder.build``, and times a training step of the effect model's
trunk and shipped heads at ``--batch-size``: forward, a stand-in loss over
every head output, backward and an optimizer step, under bf16 autocast on
CUDA as training runs it. Reports the step time, the peak GPU memory, the
slots and the relation triples per record, and the triples per relation
family, once with relations and once without, so the difference is what the
relation terms cost.

The relations are the surface's own (``EffectHeadInput.relations``), the ones
training feeds the model: the builder reads a gen-3 record's ``state.links``
and adds the families it derives itself (``line_of``,
``line_of_the_same_card``, ``mode_of``, ``has_the_same_name``), after context
dropout and the strip, each link as a forward and a reverse triple. The run
without relations clears them and keeps every slot.

The model is built directly and no checkpoint is loaded: the trunk's cost does
not depend on its weights. No ability is encoded: each ability slot reads a
random ``e`` of ``--e-dim`` columns, since the step being measured is the
trunk's, not the encoder's. Every ability key the snapshot lists gets a slot
(no sidecar decides which keys have lines), so the slot and triple counts are
an upper bound on what training lays out. ``turn_action`` records, which no
curated stratum holds, are skipped, so a raw shard reads as training would.

    python -u scripts/measure_trunk_step.py --batch-size 32 \\
        --shard output/effects/corpus-gen3-pilot3/training/shard-00001.jsonl.gz
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

import torch

from effects.domain.effect_head_input import (
    EffectHeadInput,
    SlotKind,
    SurfaceBuilder,
    feature_widths,
    keyword_line_fragment,
)
from effects.domain.effect_model import EffectModel, EffectModelConfig
from effects.domain.keyword_names import normalize_keyword
from effects.domain.records import EffectRecord, RecordKind
from effects.domain.relation_types import RELATION_TYPES
from effects.domain.surface_collation import collate_surfaces, entity_index
from effects.infrastructure.shard_io import read_shard
from price_predictor.infrastructure.run_manifest import run_script


class _EveryKeyLookup:
    """Every ability key has a line, no key is a charm, a keyless keyword
    line is the keyword its fragment names (as the batcher reads it), and
    every ``e`` is the zero vector: the slot layout without a sidecar."""

    def e_for(self, key):
        return None

    def has_line(self, key) -> bool:
        return True

    def keyword_of(self, key):
        fragment = keyword_line_fragment(key)
        return normalize_keyword(fragment) if fragment is not None else None

    def options_for(self, key):
        return ()


def surfaces_of(records: list[EffectRecord], *, with_relations: bool) -> list[EffectHeadInput]:
    """Each record's surface as ``SurfaceBuilder.build`` lays it out, its
    relations cleared when ``with_relations`` is off."""
    builder = SurfaceBuilder(_EveryKeyLookup(), e_dim=1, rng=random.Random(0))
    built = [builder.build(record) for record in records]
    return built if with_relations else [replace(s, relations=()) for s in built]


def relation_family(kind: int) -> str:
    """The family a relation type row belongs to (``targets_2:reverse`` is
    ``targets``)."""
    name = RELATION_TYPES[kind].partition(":")[0]
    return "targets" if name.startswith("targets_") else name


def _percentile(values: list[int], share: float) -> int:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(share * len(ordered)))] if ordered else 0


def measure(records: list[EffectRecord], *, batch_size: int, steps: int, warmup: int,
            e_dim: int, with_relations: bool, device: torch.device, seed: int) -> dict:
    """Time ``steps`` training steps after ``warmup`` untimed ones."""
    needed = (steps + warmup) * batch_size
    if len(records) < needed:
        raise SystemExit(f"the shard holds {len(records)} records; {needed} are needed")
    records = records[:needed]
    widths = feature_widths(records)
    surfaces = surfaces_of(records, with_relations=with_relations)
    torch.manual_seed(seed)
    batches = []
    for start in range(0, needed, batch_size):
        chunk = surfaces[start:start + batch_size]
        inputs = collate_surfaces(chunk, e_dim=e_dim, widths=widths)
        abilities = inputs.slot_kinds != int(SlotKind.ABILITY)
        e = torch.randn(inputs.e_vectors.shape).masked_fill(abilities[..., None], 0.0)
        batches.append((replace(inputs, e_vectors=e, e_rows=None), entity_index(chunk)))

    config = EffectModelConfig(
        global_features=widths[SlotKind.GLOBAL], act_features=widths[SlotKind.ACT],
        player_features=widths[SlotKind.PLAYER], card_features=widths[SlotKind.CARD],
        ability_features=widths[SlotKind.ABILITY],
        e_dim=e_dim)
    model = EffectModel(config).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    cuda = device.type == "cuda"
    if cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    def step(inputs, index) -> None:
        inputs = inputs.to(device, non_blocking=True)
        index = index.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=cuda):
            hidden = model(inputs)
            loss = (model.per_entity_at(hidden, index).float().pow(2).mean()
                    + model.created_objects(hidden).float().pow(2).mean()
                    + model.verdict(hidden).float().pow(2).mean()
                    + model.global_outputs(hidden).float().pow(2).mean())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if cuda:
            torch.cuda.synchronize(device)

    times = []
    for number, (inputs, index) in enumerate(batches):
        started = time.perf_counter()
        step(inputs, index)
        if number >= warmup:
            times.append(time.perf_counter() - started)
    triples = [len(s.relations) for s in surfaces]
    slots = [len(s.slots) for s in surfaces]
    families = Counter(relation_family(kind) for s in surfaces for _, _, kind in s.relations)
    return {
        "relations": with_relations,
        "batch_size": batch_size,
        "steps": steps,
        "step_ms_median": round(1000 * statistics.median(times), 2),
        "step_ms_mean": round(1000 * statistics.fmean(times), 2),
        "peak_memory_mb": (round(torch.cuda.max_memory_allocated(device) / 2**20, 1)
                           if cuda else None),
        "peak_reserved_mb": (round(torch.cuda.max_memory_reserved(device) / 2**20, 1)
                             if cuda else None),
        "triples_per_record_mean": round(statistics.fmean(triples), 1),
        "triples_per_record_p50": _percentile(triples, 0.5),
        "triples_per_record_p99": _percentile(triples, 0.99),
        "triples_per_record_max": max(triples),
        "slots_per_record_p50": _percentile(slots, 0.5),
        "slots_per_record_p99": _percentile(slots, 0.99),
        "slots_per_record_max": max(slots),
        "triples_by_family": dict(families.most_common()),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--shard", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--e-dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=None,
                        help="also write the two measurements as JSON here")
    return parser


def read_records(shard: Path, needed: int) -> list[EffectRecord]:
    """The shard's first ``needed`` records a curated stratum can hold: every
    record but a ``turn_action`` one."""
    records = []
    for record in read_shard(shard):
        if record.kind is RecordKind.TURN_ACTION:
            continue
        records.append(record)
        if len(records) == needed:
            break
    return records


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_script(
        args.output.parent if args.output is not None else None, lambda: _run(args),
        script=__file__, argv=argv, inputs=[args.shard],
    )


def _run(args: argparse.Namespace) -> int:
    device = torch.device(args.device)
    records = read_records(args.shard, (args.steps + args.warmup) * args.batch_size)
    print(f"{len(records)} records read from {args.shard}", flush=True)
    results = []
    for with_relations in (True, False):
        result = measure(records, batch_size=args.batch_size, steps=args.steps,
                         warmup=args.warmup, e_dim=args.e_dim, with_relations=with_relations,
                         device=device, seed=args.seed)
        results.append(result)
        print(json.dumps(result), flush=True)
    if args.output is not None:
        args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

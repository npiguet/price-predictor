"""Time one trunk training step with and without relations (025 T105, quickstart stage 2).

Reads a fixed curated shard, lays its records out as token surfaces, builds
relations from what the snapshot already records, and times a training step
of the effect model's trunk and shipped heads at ``--batch-size``: forward,
a stand-in loss over every head output, backward and an optimizer step, under
bf16 autocast on CUDA as training runs it. Reports the step time, the peak GPU
memory and the relation triples per record, once with relations and once
without, so the difference is what the relation terms cost.

The relations come from the snapshot, not from a gen-3 record's ``links``:

- ``line_of``: each ability slot to its card's slot;
- ``line_of_the_same_card``: each ordered pair of one card's ability slots;
- ``attached_to``: an attached card to its host;
- ``attacks``: an attacker to the player or permanent it attacks;
- ``blocks``: a blocker to each attacker it blocks;
- ``targets``: ``[ACT]`` to the acting ability's targets, and a stack card to
  its own targets, by target index;
- ``source_of``: the acting ability's source card to ``[ACT]``;
- ``has_the_same_name``: each ordered pair of cards sharing a name.

Each link is two triples, the forward type from one end and the reverse type
from the other.

The model is built directly and no checkpoint is loaded: the trunk's cost does
not depend on its weights. No ability is encoded: each ability slot reads a
random ``e`` of ``--e-dim`` columns, since the step being measured is the
trunk's, not the encoder's. Every ability key the snapshot lists gets a slot
(no sidecar decides which keys have lines), so the slot counts are an upper
bound on what training lays out. The shard is read with ``shard_io``'s plain
reader, so a gen-2 shard measures here and a gen-3 one later (T141a).

    python -u scripts/measure_trunk_step.py --batch-size 32 \
        --shard output/effects/corpus-gen2/training/shard-00001.jsonl.gz
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import torch

from effects.domain.damage_step_keywords import defending_player
from effects.domain.effect_head_input import (
    ACT_SLOT,
    EffectHeadInput,
    SlotKind,
    SurfaceBuilder,
    feature_widths,
)
from effects.domain.effect_model import EffectModel, EffectModelConfig
from effects.domain.records import EffectRecord
from effects.domain.relation_types import Direction, relation_type
from effects.domain.surface_collation import collate_surfaces, entity_index
from effects.infrastructure.shard_io import read_shard


class _EveryKeyLookup:
    """Every ability key has a line, no key is a keyword or a charm, and every
    ``e`` is the zero vector: the slot layout without a sidecar."""

    def e_for(self, key):
        return None

    def has_line(self, key) -> bool:
        return True

    def keyword_of(self, key):
        return None

    def options_for(self, key):
        return ()


def _both(out: list, family: str, a: int, b: int, target_index: int | None = None) -> None:
    out.append((a, b, relation_type(family, Direction.FORWARD, target_index)))
    out.append((b, a, relation_type(family, Direction.REVERSE, target_index)))


#: Forge's rendering of an attacked permanent: its name, then its card id.
_CARD_DEFENDER = re.compile(r" \((\d+)\)$")


def _attacked(record: EffectRecord, attacker, subject) -> int | None:
    """The slot of what ``attacker`` attacks. A gen-2 snapshot records Forge's
    rendering of the defender rather than an id: a permanent as ``Name (id)``,
    whose entity is ``E<id>``, and a player by a spelling of its own, which
    ``defending_player`` recovers from the board."""
    attacking = attacker.combat.attacking
    if attacking is None:
        return None
    if subject(attacking) is not None:
        return subject(attacking)
    card = _CARD_DEFENDER.search(attacking)
    if card is not None:
        return subject(f"E{card.group(1)}")
    player = defending_player(record.state, attacker)
    return subject(player) if player is not None else None


def snapshot_relations(record: EffectRecord, surface: EffectHeadInput) -> tuple:
    """``(from slot, to slot, type)`` triples for one record's surface, built from
    what its snapshot records (the module docstring lists the families)."""
    card: dict[str, int] = {}
    player: dict[str, int] = {}
    lines: dict[str, list[int]] = defaultdict(list)
    for index, slot in enumerate(surface.slots):
        if slot.kind is SlotKind.CARD:
            card[slot.entity_id] = index
        elif slot.kind is SlotKind.PLAYER:
            player[slot.player_id] = index
        elif slot.kind is SlotKind.ABILITY:
            lines[slot.entity_id].append(index)

    def subject(identifier):
        return card.get(identifier, player.get(identifier))

    out: list[tuple[int, int, int]] = []
    for entity_id, rows in lines.items():
        for row in rows:
            _both(out, "line_of", row, card[entity_id])
        for i, a in enumerate(rows):
            for b in rows[i + 1:]:
                _both(out, "line_of_the_same_card", a, b)
    by_name: dict[str, list[int]] = defaultdict(list)
    for entity in record.state.entities:
        here = card[entity.id]
        if entity.name and not entity.face_down:
            by_name[entity.name].append(here)
        if entity.attached_to is not None and subject(entity.attached_to) is not None:
            _both(out, "attached_to", here, subject(entity.attached_to))
        combat = entity.combat
        if combat is not None:
            attacked = _attacked(record, entity, subject)
            if attacked is not None:
                _both(out, "attacks", here, attacked)
            for attacker in combat.blocking:
                if attacker in card:
                    _both(out, "blocks", here, card[attacker])
        if entity.stack_extras is not None:
            for index, target in enumerate(entity.stack_extras.targets):
                if subject(target) is not None:
                    _both(out, "targets", here, subject(target), index)
    for slots in by_name.values():
        for i, a in enumerate(slots):
            for b in slots[i + 1:]:
                _both(out, "has_the_same_name", a, b)
    refs = record.state.refs
    for index, target in enumerate(refs.targets):
        if subject(target) is not None:
            _both(out, "targets", ACT_SLOT, subject(target), index)
    if refs.source is not None and refs.source in card:
        _both(out, "source_of", card[refs.source], ACT_SLOT)
    return tuple(out)


def surfaces_of(records: list[EffectRecord], *, with_relations: bool) -> list[EffectHeadInput]:
    builder = SurfaceBuilder(_EveryKeyLookup(), e_dim=1, rng=random.Random(0))
    built = []
    for record in records:
        surface = builder.build(record)
        relations = snapshot_relations(record, surface) if with_relations else ()
        built.append(replace(surface, relations=relations))
    return built


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
                    + model.verdict(hidden).float().pow(2).mean())
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


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = torch.device(args.device)
    needed = (args.steps + args.warmup) * args.batch_size
    records = []
    for record in read_shard(args.shard):
        records.append(record)
        if len(records) == needed:
            break
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

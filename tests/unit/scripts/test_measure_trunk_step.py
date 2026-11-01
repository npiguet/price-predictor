"""``scripts/measure_trunk_step.py`` on the gen-2 fixture records, on CPU (025 T105)."""

from __future__ import annotations

import json
from pathlib import Path

import measure_trunk_step as mts
import pytest

from effects.domain.effect_head_input import ACT_SLOT, SlotKind
from effects.domain.relation_types import RELATION_TYPES, Direction, relation_type, reverse
from effects.infrastructure.shard_io import read_shard

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "effects" / "gen2-records.jsonl.gz"


@pytest.fixture(scope="module")
def records():
    return list(read_shard(FIXTURE))


@pytest.fixture(scope="module")
def surfaces(records):
    return mts.surfaces_of(records, with_relations=True)


def test_every_triple_names_two_slots_of_its_surface_and_a_known_type(surfaces):
    assert any(s.relations for s in surfaces)
    for surface in surfaces:
        for a, b, kind in surface.relations:
            assert 0 <= a < len(surface.slots) and 0 <= b < len(surface.slots)
            assert a != b
            assert 0 <= kind < len(RELATION_TYPES)


def test_every_link_is_read_from_both_ends(surfaces):
    for surface in surfaces:
        triples = set(surface.relations)
        for a, b, kind in triples:
            assert (b, a, reverse(kind)) in triples


def test_each_ability_slot_is_a_line_of_its_card(surfaces):
    line_of = relation_type("line_of", Direction.FORWARD)
    for surface in surfaces:
        cards = {s.entity_id: i for i, s in enumerate(surface.slots) if s.kind is SlotKind.CARD}
        for index, slot in enumerate(surface.slots):
            if slot.kind is SlotKind.ABILITY:
                assert (index, cards[slot.entity_id], line_of) in surface.relations


def test_act_targets_carry_their_target_index(records, surfaces):
    found = 0
    for record, surface in zip(records, surfaces):
        slots = {s.subject_id: i for i, s in enumerate(surface.slots)
                 if s.kind in (SlotKind.CARD, SlotKind.PLAYER)}
        for index, target in enumerate(record.state.refs.targets):
            if target in slots:
                kind = relation_type("targets", Direction.FORWARD, index)
                assert (ACT_SLOT, slots[target], kind) in surface.relations
                found += 1
    assert found


def test_attackers_and_blockers_are_linked(records, surfaces):
    attacks = relation_type("attacks", Direction.FORWARD)
    blocks = relation_type("blocks", Direction.FORWARD)
    kinds = {kind for surface in surfaces for _, _, kind in surface.relations}
    combat = [e for r in records for e in r.state.entities if e.combat is not None]
    if any(e.combat.attacking for e in combat):
        assert attacks in kinds
    if any(e.combat.blocking for e in combat):
        assert blocks in kinds


def test_without_relations_no_triple_is_built(records):
    assert all(not s.relations for s in mts.surfaces_of(records, with_relations=False))


def test_measure_reports_both_arms_on_cpu(records):
    import torch

    common = dict(batch_size=4, steps=2, warmup=1, e_dim=8, device=torch.device("cpu"), seed=0)
    on = mts.measure(records, with_relations=True, **common)
    off = mts.measure(records, with_relations=False, **common)
    assert on["relations"] and not off["relations"]
    assert on["triples_per_record_max"] > 0 and off["triples_per_record_max"] == 0
    assert on["slots_per_record_max"] == off["slots_per_record_max"]
    assert on["step_ms_median"] > 0 and on["peak_memory_mb"] is None


def test_too_short_a_shard_is_refused(records):
    import torch

    with pytest.raises(SystemExit):
        mts.measure(records, batch_size=64, steps=50, warmup=5, e_dim=8,
                    device=torch.device("cpu"), seed=0, with_relations=True)


def test_main_writes_the_two_measurements(tmp_path):
    out = tmp_path / "m.json"
    assert mts.main(["--shard", str(FIXTURE), "--batch-size", "4", "--steps", "2",
                     "--warmup", "1", "--e-dim", "8", "--device", "cpu",
                     "--output", str(out)]) == 0
    written = json.loads(out.read_text(encoding="utf-8"))
    assert [m["relations"] for m in written] == [True, False]

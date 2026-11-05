"""``scripts/measure_trunk_step.py`` on the gen-3 fixture records, on CPU (025 T105, T141a)."""

from __future__ import annotations

import json
import random

import measure_trunk_step as mts
import pytest

from effects.domain.effect_head_input import SurfaceBuilder
from effects.domain.relation_types import FAMILIES, RELATION_TYPES, reverse
from tests.unit.effects.builders import GEN3_RECORDS, curated_fixture_records


@pytest.fixture(scope="module")
def records():
    return curated_fixture_records()


@pytest.fixture(scope="module")
def surfaces(records):
    return mts.surfaces_of(records, with_relations=True)


def test_the_relations_are_the_ones_the_surface_builder_feeds_training(records, surfaces):
    """T141a: the measured triples are ``SurfaceBuilder.build``'s, read from
    ``links``, not a set the script builds for itself."""
    builder = SurfaceBuilder(mts._EveryKeyLookup(), e_dim=1, rng=random.Random(0))
    assert any(s.relations for s in surfaces)
    for record, surface in zip(records, surfaces):
        assert surface.relations == builder.build(record).relations


def test_link_families_reach_the_measurement(surfaces):
    """Families only a gen-3 record's ``links`` supply (``controls``,
    ``owns``) are among the measured triples."""
    families = {mts.relation_family(kind) for s in surfaces for _, _, kind in s.relations}
    assert {"controls", "owns", "line_of"} <= families
    assert families <= set(FAMILIES)


def test_every_triple_names_two_slots_of_its_surface_and_a_known_type(surfaces):
    for surface in surfaces:
        for a, b, kind in surface.relations:
            assert 0 <= a < len(surface.slots) and 0 <= b < len(surface.slots)
            assert 0 <= kind < len(RELATION_TYPES)


def test_every_link_is_read_from_both_ends(surfaces):
    for surface in surfaces:
        triples = set(surface.relations)
        for a, b, kind in triples:
            assert (b, a, reverse(kind)) in triples


def test_without_relations_no_triple_is_built_and_every_slot_stays(records, surfaces):
    bare = mts.surfaces_of(records, with_relations=False)
    assert all(not s.relations for s in bare)
    assert [len(s.slots) for s in bare] == [len(s.slots) for s in surfaces]


def test_measure_reports_both_arms_on_cpu(records):
    import torch

    common = dict(batch_size=4, steps=2, warmup=1, e_dim=8, device=torch.device("cpu"), seed=0)
    on = mts.measure(records, with_relations=True, **common)
    off = mts.measure(records, with_relations=False, **common)
    assert on["relations"] and not off["relations"]
    assert on["triples_per_record_max"] > 0 and off["triples_per_record_max"] == 0
    assert on["triples_by_family"] and off["triples_by_family"] == {}
    assert on["slots_per_record_max"] == off["slots_per_record_max"]
    assert on["step_ms_median"] > 0 and on["peak_memory_mb"] is None


def test_too_short_a_shard_is_refused(records):
    import torch

    with pytest.raises(SystemExit):
        mts.measure(records, batch_size=64, steps=50, warmup=5, e_dim=8,
                    device=torch.device("cpu"), seed=0, with_relations=True)


def test_a_raw_shard_is_read_without_its_turn_actions(records):
    from effects.domain.records import RecordKind
    from effects.infrastructure.shard_io import read_shard

    assert any(r.kind is RecordKind.TURN_ACTION for r in read_shard(GEN3_RECORDS))
    read = mts.read_records(GEN3_RECORDS, len(records) + 10)
    assert [r.record_id for r in read] == [r.record_id for r in records]
    assert len(mts.read_records(GEN3_RECORDS, 5)) == 5


def test_main_writes_the_two_measurements(tmp_path):
    out = tmp_path / "m.json"
    assert mts.main(["--shard", str(GEN3_RECORDS), "--batch-size", "4", "--steps", "2",
                     "--warmup", "1", "--e-dim", "8", "--device", "cpu",
                     "--output", str(out)]) == 0
    written = json.loads(out.read_text(encoding="utf-8"))
    assert [m["relations"] for m in written] == [True, False]

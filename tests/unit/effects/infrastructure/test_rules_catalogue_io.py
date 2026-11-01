"""``rules_catalogue_io`` over the real ``rules/`` tree (T011)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from effects.domain.rule_catalogue import MapVerdict, Polarity, is_model_output
from effects.infrastructure.rules_catalogue_io import (
    heads_digest,
    load_heads,
    load_scenarios,
    map_snapshot,
)

RULES = Path(__file__).resolve().parents[4] / "rules"


@pytest.fixture(scope="module")
def catalogue():
    return load_scenarios(RULES)


def test_every_scenario_is_read(catalogue):
    files = sorted((RULES / "scenarios").glob("*.json"))
    expected = sum(len(json.loads(p.read_text(encoding="utf-8"))["scenarios"]) for p in files)
    assert len(catalogue.scenarios) == expected
    assert all(s.variants for s in catalogue.scenarios)


def test_the_first_variant_is_the_base(catalogue):
    for scenario in catalogue.scenarios:
        assert [v.is_base for v in scenario.variants] == [
            i == 0 for i in range(len(scenario.variants))]


def test_a_known_scenario_reads_whole(catalogue):
    scenario = next(s for s in catalogue.scenarios if s.id == "704.3-1")
    assert scenario.unit == "704"
    assert scenario.rules == ("704.3",)
    assert scenario.record_kind == "resolution"
    assert [v.polarity for v in scenario.variants] == [Polarity.FIRES, Polarity.DOES_NOT_FIRE]
    assert "704.3" in catalogue.by_rule()


def test_heads_keep_only_untagged_model_outputs(catalogue):
    heads = load_heads(RULES, catalogue.rules())
    assert heads["704.5f"].verdict is MapVerdict.TRAINED
    assert heads["704.5f"].outputs == ("zone_outcome",)
    # 704.5a names only tagged outputs: a payload event and a proposed one.
    assert heads["704.5a"].outputs == ()
    assert all(is_model_output(o) for e in heads.values() for o in e.outputs)


def test_a_rule_with_no_heads_line_reads_missing(catalogue):
    heads = load_heads(RULES, frozenset({"999.9z"}))
    assert heads["999.9z"].verdict is MapVerdict.MISSING
    assert heads["999.9z"].outputs == ()


def test_the_snapshot_is_the_manifest_shape(catalogue):
    snapshot = map_snapshot(load_heads(RULES, catalogue.rules()))
    assert snapshot["704.5f"] == {"verdict": "trained", "outputs": ["zone_outcome"]}


def test_the_digest_follows_the_lines_not_their_files(tmp_path):
    copy = tmp_path / "rules"
    shutil.copytree(RULES / "heads", copy / "heads")
    assert heads_digest(copy) == heads_digest(RULES)
    first = sorted((copy / "heads").glob("*.jsonl"))[0]
    lines = first.read_text(encoding="utf-8").splitlines()
    edited = json.loads(lines[0]) | {"output": "missing"}
    first.write_text("\n".join([json.dumps(edited), *lines[1:]]) + "\n", encoding="utf-8")
    assert heads_digest(copy) != heads_digest(RULES)

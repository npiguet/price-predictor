"""``rule-counts`` over the real gen-2 fixture (T033).

The catalogue is a small ``rules/scenarios`` tree written per test, declaring
detectors the fixture's records show; the records are the real ones.
"""

from __future__ import annotations

import gzip
import json
import shutil
from pathlib import Path

import pytest

from effects.application.rule_counts import (
    RuleCountsConfig,
    RuleSource,
    count,
    report_lines,
    unit_of,
)

FIXTURES = Path(__file__).parents[3] / "fixtures" / "effects"
GEN2 = FIXTURES / "gen2-records.jsonl.gz"
SIDECARS = FIXTURES / "gen2-sidecars"


def _scenario(sid, rules, kind, detector):
    return {"id": sid, "rules": list(rules), "record_kind": kind, "detector": detector,
            "variants": [{"label": "base", "polarity": "fires", "delta": "", "expected": ""},
                         {"label": "near", "polarity": "does_not_fire", "delta": "one fact",
                          "expected": ""}]}


def _rules_dir(tmp_path, *scenarios) -> Path:
    root = tmp_path / "rules" / "scenarios"
    root.mkdir(parents=True)
    by_unit: dict[str, list] = {}
    for scenario in scenarios:
        by_unit.setdefault(scenario["id"].split(".")[0], []).append(scenario)
    for unit, found in by_unit.items():
        (root / f"{unit}.json").write_text(
            json.dumps({"unit": unit, "scenarios": found}), encoding="utf-8")
    return root.parent


_CATALOGUE = (
    _scenario("704.5f-1", ["704.5f"], "resolution", {
        "family": "sba_zone_move", "record_kind": "resolution", "outcome": "o",
        "near_miss": "n", "params": {"check": "toughness_zero"}}),
    _scenario("704.5g-1", ["704.5g"], "resolution", {
        "family": "sba_zone_move", "record_kind": "resolution", "outcome": "o",
        "near_miss": "n", "params": {"check": "lethal_damage"}}),
    _scenario("702.2b-1", ["702.2b"], "combat", {
        "family": "keyword_combat", "record_kind": "combat", "outcome": "o",
        "near_miss": "n", "params": {"keyword": "deathtouch"}}),
    _scenario("601.2-1", ["601.2"], "playability", {
        "family": "verdict_bit", "record_kind": "playability", "outcome": "o",
        "near_miss": "n", "params": {"bit": "timing"}}),
    _scenario("613.1-1", ["613.1"], "continuous", {"none": "layer order is in no field"}),
)


def _config(tmp_path, records_dir, workers=1):
    return RuleCountsConfig(
        records_dir=records_dir, output=tmp_path / "rule-counts.json",
        rules_dir=_rules_dir(tmp_path, *_CATALOGUE),
        sidecar_roots={"cardsfolder": SIDECARS / "cardsfolder",
                       "tokenscripts": SIDECARS / "tokenscripts"},
        workers=workers)


@pytest.fixture
def gen2_dir(tmp_path):
    directory = tmp_path / "records"
    directory.mkdir()
    shutil.copyfile(GEN2, directory / GEN2.name)
    return directory


def test_the_fixture_s_deaths_are_counted_per_side(tmp_path, gen2_dir):
    counts, _ = count(_config(tmp_path, gen2_dir))
    document = counts.to_json({})
    fires = sum(document["rules"]["704.5f"]["fires"].values())
    near = sum(document["rules"]["704.5f"]["does_not_fire"].values())
    assert fires >= 3 and near >= 3
    assert sum(document["rules"]["704.5g"]["fires"].values()) >= 3


def test_a_gen2_record_is_natural_and_a_forge_ai_game_counts_as_forge_ai(tmp_path, gen2_dir):
    counts, _ = count(_config(tmp_path, gen2_dir))
    sources = {source for (_, _, source) in counts.counts}
    assert str(RuleSource.AIMED) not in sources
    assert str(RuleSource.FORGE_AI) in sources


_DECREE = "50e3aa10-aa8d-4153-a614-e786a1dc6a6d.0-mv5fnkmsv6yb.1019"


def test_a_game_s_source_follows_any_of_its_records(tmp_path, gen2_dir):
    """The fixture's three 704.5f deaths are all in random-seat games, the
    random seat named by another record of each. Clearing the random seat from
    Decree of Pain's whole game moves its death, and no other, to ``forge_ai``.
    """
    original, _ = count(_config(tmp_path / "a", gen2_dir))
    assert original.to_json({})["rules"]["704.5f"]["fires"] == {
        str(RuleSource.RANDOM_SEAT): 3}
    path = gen2_dir / GEN2.name
    records = [json.loads(line) for line in gzip.open(path, "rt", encoding="utf-8")]
    game = next(r["game_id"] for r in records if r["record_id"] == _DECREE)
    for record in records:
        if record["game_id"] == game:
            record["random_seat"] = False
    with gzip.open(path, "wt", encoding="utf-8") as out:
        out.write("".join(json.dumps(r) + "\n" for r in records))
    counts, _ = count(_config(tmp_path / "b", gen2_dir))
    assert counts.to_json({})["rules"]["704.5f"]["fires"] == {
        str(RuleSource.RANDOM_SEAT): 2, str(RuleSource.FORGE_AI): 1}


def test_a_rule_needing_a_gen3_field_is_listed_not_countable(tmp_path, gen2_dir):
    counts, none = count(_config(tmp_path, gen2_dir))
    document = counts.to_json(none)
    assert document["not_countable"] == ["601.2"]
    assert "601.2" not in document["rules"]


def test_a_none_rule_is_listed_with_its_reason(tmp_path, gen2_dir):
    counts, none = count(_config(tmp_path, gen2_dir))
    assert none == {"613.1": "layer order is in no field"}


def test_the_review_list_names_record_and_scenario(tmp_path, gen2_dir):
    counts, _ = count(_config(tmp_path, gen2_dir))
    for record_id, scenario in counts.review:
        assert record_id.count(".") >= 2 and scenario.count("-") == 1


def test_one_and_several_workers_count_alike(tmp_path, gen2_dir):
    second = gen2_dir / "copy.jsonl.gz"
    shutil.copyfile(gen2_dir / GEN2.name, second)
    one, _ = count(_config(tmp_path, gen2_dir, workers=1))
    two, _ = count(_config(tmp_path / "b", gen2_dir, workers=2))
    assert one.counts == two.counts and one.records == two.records


def test_the_report_groups_rules_by_unit():
    document = {"rules": {"702.2b": {"fires": {"forge_ai": 2}},
                          "704.5f": {"fires": {"random_seat": 1, "forge_ai": 3}}},
                "none": {"613.1": "r"}, "not_countable": ["601.2"], "review": []}
    lines = report_lines(document)
    assert lines.index("-- 702.2") < lines.index("-- 704")
    assert any(line.startswith("704.5f") and "1/3/0" in line for line in lines)
    assert "  613.1: r" in lines and "  601.2" in lines


@pytest.mark.parametrize("rule, unit", [("704.5f", "704"), ("702.2b", "702.2"),
                                        ("701.10a", "701.10"), ("614.11a", "614")])
def test_unit_of(rule, unit):
    assert unit_of(rule) == unit


# --- carriers (T185, FR-058f) ---------------------------------------------------------------

GEN3 = FIXTURES / "gen3-records.jsonl.gz"
GEN3_SIDECARS = FIXTURES / "gen3-sidecars"

_CARRIER_CATALOGUE = (
    _scenario("702.2b-1", ["702.2b"], "combat", {
        "family": "keyword_combat", "record_kind": "combat", "outcome": "o",
        "near_miss": "n", "params": {"keyword": "deathtouch"}}),
    _scenario("106.3-1", ["106.3"], "resolution", {
        "family": "outcome_event", "record_kind": "resolution", "outcome": "o",
        "near_miss": "n", "params": {"event": "mana_produced"}}),
)


def _gen3_counts(tmp_path, workers=1, copies=1):
    records = tmp_path / "records"
    records.mkdir(parents=True)
    for copy in range(copies):
        shutil.copyfile(GEN3, records / f"{copy}-{GEN3.name}")
    config = RuleCountsConfig(
        records_dir=records, output=tmp_path / "rule-counts.json",
        rules_dir=_rules_dir(tmp_path, *_CARRIER_CATALOGUE),
        sidecar_roots={"cardsfolder": GEN3_SIDECARS / "cardsfolder",
                       "tokenscripts": GEN3_SIDECARS / "tokenscripts"},
        workers=workers)
    counts, none = count(config)
    return counts.to_json(none)["carriers"]


def _carrier(carriers, rule, script_file):
    return next(c for c in carriers[rule] if c["script_file"] == script_file)


def test_a_keyword_rule_s_carrier_acts_through_its_keyword_line(tmp_path):
    """A combat record names no acting line: the deathtouch carrier's own
    deathtouch line is what the rule's tags act through."""
    carriers = _gen3_counts(tmp_path)
    servant = _carrier(carriers, "702.2b", "cardsfolder/s/servant_of_the_stinger.txt")
    assert servant["lines"] == [{"key": {"face": 0, "trait_kind": "keyword",
                                         "index_within_kind": 0},
                                 "records": servant["records"]}]


def test_a_resolution_rule_s_carrier_acts_through_the_record_s_acting_line(tmp_path):
    carriers = _gen3_counts(tmp_path)
    acting = [line for carrier in carriers["106.3"] for line in carrier["lines"]]
    assert acting and all(line["key"]["trait_kind"] != "keyword" for line in acting)


def test_carriers_come_most_recorded_first_and_no_line_outcounts_its_card(tmp_path):
    carriers = _gen3_counts(tmp_path)
    for found in carriers.values():
        records = [carrier["records"] for carrier in found]
        assert records == sorted(records, reverse=True)
        for carrier in found:
            assert all(0 < line["records"] <= carrier["records"] for line in carrier["lines"])


def test_carriers_add_up_across_workers(tmp_path):
    one = _gen3_counts(tmp_path / "a", workers=1)
    two = _gen3_counts(tmp_path / "b", workers=2, copies=2)
    servant = _carrier(two, "702.2b", "cardsfolder/s/servant_of_the_stinger.txt")
    assert servant["records"] == 2 * _carrier(
        one, "702.2b", "cardsfolder/s/servant_of_the_stinger.txt")["records"]

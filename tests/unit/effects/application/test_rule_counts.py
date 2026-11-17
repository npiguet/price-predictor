"""``rule-counts`` over the real gen-3 fixture (T033).

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
GEN3 = FIXTURES / "gen3-records.jsonl.gz"
SIDECARS = FIXTURES / "gen3-sidecars"


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
    _scenario("704.5k-1", ["704.5k"], "resolution", {
        "family": "sba_zone_move", "record_kind": "resolution", "outcome": "o",
        "near_miss": "n", "params": {"check": "world_rule"}}),
    _scenario("121.4-1", ["121.4"], "resolution", {
        "family": "outcome_event", "record_kind": "resolution", "outcome": "o",
        "near_miss": "n", "params": {"event": "player_lost", "api": "Draw"}}),
    _scenario("704.5f-2", ["704.3"], "resolution", {
        "same_as": "704.5f-1", "reason": "the check 704.3 describes is 704.5f's"}),
)


def _config(tmp_path, records_dir, workers=1):
    return RuleCountsConfig(
        records_dir=records_dir, output=tmp_path / "rule-counts.json",
        rules_dir=_rules_dir(tmp_path, *_CATALOGUE),
        sidecar_roots={"cardsfolder": SIDECARS / "cardsfolder",
                       "tokenscripts": SIDECARS / "tokenscripts"},
        workers=workers)


@pytest.fixture
def gen3_dir(tmp_path):
    directory = tmp_path / "records"
    directory.mkdir()
    shutil.copyfile(GEN3, directory / GEN3.name)
    return directory


def test_the_fixture_s_deaths_are_counted_per_side(tmp_path, gen3_dir):
    counts, _ = count(_config(tmp_path, gen3_dir))
    document = counts.to_json({})
    assert document["records"] == 314
    assert document["generation"] == "gen-3"
    assert document["rules"]["704.5f"] == {
        "fires": {"random_seat": 2}, "does_not_fire": {"random_seat": 3}}
    assert document["rules"]["704.5g"] == {"fires": {"forge_ai": 6, "random_seat": 5}}
    assert document["rules"]["702.2b"] == {"does_not_fire": {"random_seat": 2}}
    assert document["rules"]["601.2"] == {
        "fires": {"aimed": 1, "forge_ai": 4, "random_seat": 18},
        "does_not_fire": {"aimed": 2, "forge_ai": 12, "random_seat": 10}}


def test_the_fixture_holds_every_source(tmp_path, gen3_dir):
    counts, _ = count(_config(tmp_path, gen3_dir))
    sources = {source for (_, _, source) in counts.counts}
    assert sources == {str(s) for s in RuleSource}


_FIRST_DEATH_GAME = "9442f17f-cb56-4893-8d4a-66b5538de180.0-mwhr12znmj0t.0"


def test_a_game_s_source_follows_any_of_its_records(tmp_path, gen3_dir):
    """The fixture's two 704.5f deaths and the first of its three non-deaths
    are in one game, a random-seat game by another record of it. Clearing the
    random seat from that whole game moves its three tags, and no other, to
    ``forge_ai``.
    """
    original, _ = count(_config(tmp_path / "a", gen3_dir))
    assert original.to_json({})["rules"]["704.5f"]["fires"] == {
        str(RuleSource.RANDOM_SEAT): 2}
    path = gen3_dir / GEN3.name
    records = [json.loads(line) for line in gzip.open(path, "rt", encoding="utf-8")]
    assert any(r["random_seat"] for r in records if r["game_id"] == _FIRST_DEATH_GAME)
    for record in records:
        if record["game_id"] == _FIRST_DEATH_GAME:
            record["random_seat"] = False
    with gzip.open(path, "wt", encoding="utf-8") as out:
        out.write("".join(json.dumps(r) + "\n" for r in records))
    counts, _ = count(_config(tmp_path / "b", gen3_dir))
    rule = counts.to_json({})["rules"]["704.5f"]
    assert rule["fires"] == {str(RuleSource.FORGE_AI): 2}
    assert rule["does_not_fire"] == {str(RuleSource.RANDOM_SEAT): 2, str(RuleSource.FORGE_AI): 1}


def test_every_declared_rule_is_countable_on_gen3_records(tmp_path, gen3_dir):
    counts, standing = count(_config(tmp_path, gen3_dir))
    document = counts.to_json(standing)
    assert document["not_countable"] == []
    assert "601.2" in document["rules"]


def test_a_none_rule_is_listed_with_its_reason(tmp_path, gen3_dir):
    counts, standing = count(_config(tmp_path, gen3_dir))
    assert standing.none == {"613.1": "layer order is in no field"}


def test_a_declared_rule_no_record_tags_is_listed_with_zero_records(tmp_path, gen3_dir):
    """D3(d), V5 #22: absent is no longer silent."""
    counts, standing = count(_config(tmp_path, gen3_dir))
    document = counts.to_json(standing)
    assert "704.5k" not in document["rules"]
    assert document["declared_zero"] == ["704.5k"]
    assert "rules declared with zero records: 1" in report_lines(document)
    assert "  704.5k" in report_lines(document)


def test_untaggable_and_same_as_scenarios_are_reported(tmp_path, gen3_dir):
    """V5 #14 and D4: a declaration asking a Draw resolution for the game's
    end tags nothing and is named with its reason; a same_as scenario is reported
    under the one it points at and counted nowhere else."""
    counts, standing = count(_config(tmp_path, gen3_dir))
    document = counts.to_json(standing)
    assert "121.4" not in document["rules"] and "121.4" not in document["declared_zero"]
    assert document["untaggable"]["121.4-1"]["rules"] == ["121.4"]
    assert "game-end record" in document["untaggable"]["121.4-1"]["reason"]
    assert document["same_as"] == {"704.5f-2": {
        "rules": ["704.3"], "same_as": ["704.5f-1"], "counted_under": ["704.5f"]}}
    assert "704.3" not in document["rules"] and "704.3" not in document["declared_zero"]
    lines = report_lines(document)
    assert "  704.5f-2 (704.3) -> 704.5f-1 (704.5f)" in lines


_NON_DEATH = "9442f17f-cb56-4893-8d4a-66b5538de180.0-mwhr12znmj0t.130"


def test_the_review_list_names_record_and_scenario(tmp_path, gen3_dir):
    """On a shard holding one record whose outcome disagrees with its side
    (a creature left at toughness 1 that dies anyway, 704.5f): the fixture
    alone holds none, so the list was never read."""
    from dataclasses import replace

    from effects.domain.event_schema import EventType
    from effects.infrastructure.shard_io import read_shard, write_shard

    counts, _ = count(_config(tmp_path / "plain", gen3_dir))
    assert counts.review == set()

    records = []
    for record in read_shard(gen3_dir / GEN3.name):
        if record.record_id == _NON_DEATH:
            shrink = next(e for e in record.payload.events if e.type is EventType.PT_CHANGE)
            death = replace(shrink, type=EventType.ZONE_CHANGE, attributed_to="unresolved",
                            params={"from_zone": "battlefield", "to_zone": "graveyard"})
            record = replace(record, payload=replace(
                record.payload, events=(*record.payload.events, death)))
        records.append(record)
    planted = tmp_path / "planted"
    write_shard(planted / GEN3.name, records)

    counts, _ = count(_config(tmp_path / "counted", planted))
    assert counts.review == {(_NON_DEATH, "704.5f-1")}


@pytest.mark.parametrize(("aimed_at", "source"), [
    (("704.5f",), "aimed"), ((), "aimed"), (None, "forge_ai"),
])
def test_a_record_carrying_aimed_at_counts_as_aimed_even_when_empty(
    tmp_path, gen3_dir, aimed_at, source,
):
    """The member's presence is the source (FR-009a): a coverage or variant
    run writes ``[]`` where it names no rule, and its records are aimed; a
    record without the member is natural."""
    from dataclasses import replace

    from effects.infrastructure.shard_io import read_shard, write_shard

    every = list(read_shard(gen3_dir / GEN3.name))
    seat_games = {record.game_id for record in every if record.random_seat}
    records = [replace(record, aimed_at=aimed_at) for record in every
               if record.game_id not in seat_games and record.directed is None]
    shard = tmp_path / "aimed" / GEN3.name
    write_shard(shard, records)

    counts, _ = count(_config(tmp_path, shard.parent))
    assert counts.counts and {found for (_rule, _side, found) in counts.counts} == {source}


def test_one_and_several_workers_count_alike(tmp_path, gen3_dir):
    second = gen3_dir / "copy.jsonl.gz"
    shutil.copyfile(gen3_dir / GEN3.name, second)
    one, _ = count(_config(tmp_path, gen3_dir, workers=1))
    two, _ = count(_config(tmp_path / "b", gen3_dir, workers=2))
    assert one.counts == two.counts and one.records == two.records


def test_a_gen2_set_is_refused_naming_its_generation(tmp_path):
    """A shard whose records carry no ``generation`` member and a
    ``random_seat`` one is gen-2: ``count`` refuses it, naming the generation."""
    from effects.domain.shard_generation import UnacceptedGenerationError

    records = tmp_path / "records"
    records.mkdir()
    with gzip.open(GEN3, "rt", encoding="utf-8") as source:
        data = json.loads(next(source))
    del data["generation"]
    (records / "gen2.jsonl").write_text(json.dumps(data) + "\n", encoding="utf-8")
    with pytest.raises(UnacceptedGenerationError, match=r"gen2\.jsonl holds gen-2 records"):
        count(_config(tmp_path, records))


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

_CARRIER_CATALOGUE = (
    _scenario("702.2b-1", ["702.2b"], "combat", {
        "family": "keyword_combat", "record_kind": "combat", "outcome": "o",
        "near_miss": "n", "params": {"keyword": "deathtouch"}}),
    _scenario("106.3-1", ["106.3"], "resolution", {
        "family": "outcome_event", "record_kind": "resolution", "outcome": "o",
        "near_miss": "n", "params": {"event": "mana_produced"}}),
)


def _gen3_document(tmp_path, workers=1, copies=1, *, ended=True):
    """The counts file over ``copies`` copies of the gen-3 fixture, each game
    ended by its game-end marker unless ``ended`` is False."""
    from tests.unit.effects.builders import write_cut_gen3_fixture

    records = tmp_path / "records"
    for copy in range(copies):
        (records / str(copy)).mkdir(parents=True)
        if ended:
            shutil.copyfile(GEN3, records / str(copy) / GEN3.name)
        else:
            write_cut_gen3_fixture(records / str(copy) / GEN3.name)
    config = RuleCountsConfig(
        records_dir=records, output=tmp_path / "rule-counts.json",
        rules_dir=_rules_dir(tmp_path, *_CARRIER_CATALOGUE),
        sidecar_roots={"cardsfolder": SIDECARS / "cardsfolder",
                       "tokenscripts": SIDECARS / "tokenscripts"},
        workers=workers)
    counts, standing = count(config)
    return counts.to_json(standing)


def _gen3_counts(tmp_path, workers=1, copies=1):
    return _gen3_document(tmp_path, workers, copies)["carriers"]


def test_a_cut_game_is_left_out_of_every_count_and_counted(tmp_path):
    """The fixture without its game-end markers, as a killed worker leaves a
    shard: every game in it reads as cut, build-corpus drops each, so no
    record of them may count toward a rule (WP7, P1 #1)."""
    from tests.unit.effects.builders import GEN3_FIXTURE_GAMES

    ended = _gen3_document(tmp_path / "ended")
    cut = _gen3_document(tmp_path / "cut", ended=False)

    assert ended["cut_games"] == 0 and ended["rules"]
    assert cut["cut_games"] == GEN3_FIXTURE_GAMES
    assert cut["rules"] == {} and cut["carriers"] == {} and cut["review"] == []
    # As the build's rule_counts: a cut game's records are read and count
    # nowhere, neither as records nor as rules they could not count.
    assert cut["records"] == 0 and ended["records"] > 0
    assert cut["not_countable"] == []


def _carrier(carriers, rule, script_file):
    return next(c for c in carriers[rule] if c["script_file"] == script_file)


def test_a_keyword_rule_s_carrier_acts_through_its_keyword_line(tmp_path):
    """A combat record names no acting line: the deathtouch carrier's own
    deathtouch line is what the rule's tags act through."""
    carriers = _gen3_counts(tmp_path)
    viper = _carrier(carriers, "702.2b", "cardsfolder/w/wasteland_viper.txt")
    assert viper["lines"] == [{"key": {"face": 0, "trait_kind": "keyword",
                                         "index_within_kind": 0},
                                 "records": viper["records"]}]


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
    viper = _carrier(two, "702.2b", "cardsfolder/w/wasteland_viper.txt")
    assert viper["records"] == 2 * _carrier(
        one, "702.2b", "cardsfolder/w/wasteland_viper.txt")["records"]

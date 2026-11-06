"""The selection set's freeze (contracts/selection-set.md; story 8 acceptance 7-9,
story 3 acceptance 5, story 1 acceptance 6-7, SC-010)."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import pytest

from effects.application.freeze_selection_set import (
    GridQualification,
    ProbeItem,
    RuleTestRules,
    SelectionSetRefused,
    blocked_rules,
    freeze_selection_set,
    left_out_lines,
    run,
)
from effects.domain.corpus_manifest import ClassCounts, CorpusManifest, SourceShard
from effects.domain.rule_catalogue import (
    Departure,
    DepartureResolution,
    DepartureStatus,
    Polarity,
    Scenario,
    ScenarioCatalogue,
)
from effects.domain.rule_detectors import RuleTag
from effects.domain.score_scales import component_standard_error
from effects.domain.selection_set import (
    NO_READABLE_OUTPUT,
    UNTESTED,
    error_of,
    item_key,
    load_selection_set,
    selection_set_path,
)

FIRES, NOT = Polarity.FIRES, Polarity.DOES_NOT_FIRE
GD = "game-disjoint"
#: Thresholds a handful of fake records can meet; the precision tests pass E5's.
LENIENT = dict(min_games_per_half=1, max_item_error=10.0, max_component_error=10.0)


@dataclass(frozen=True)
class _Record:
    record_id: str
    game_id: str
    synthetic: bool = False
    fork: bool = False
    interventional: bool = False
    aimed_at: tuple | None = None
    directed: dict | None = None
    curation: object = None


def _manifest(halves: dict) -> CorpusManifest:
    return CorpusManifest(
        seed=42, surface="script", vocab_path="models/effects/vocab-script.txt", variant_scripts="",
        holdout_permille=20, holdout_max_carriers=8, text_cap=200,
        card_disjoint_text_cap=50, training_records=0, class_mix={"rewrite": 1.0},
        delivered_mix={"rewrite": 1.0}, held_out_cards=(), held_out_texts=(),
        card_disjoint_games=tuple(g for h in halves.get("card-disjoint", {}).values() for g in h),
        game_disjoint_games=tuple(g for h in halves.get(GD, {}).values() for g in h),
        rarity={}, sources=(SourceShard("a.jsonl.gz", 1),),
        per_class={"rewrite": ClassCounts(1, 1, 0, 1)}, per_stratum={}, unique_texts={},
        shortfall={},
        halves={s: {h: tuple(sorted(g)) for h, g in by.items()} for s, by in halves.items()},
    )


def _reader(records):
    """A stratum of one shard per game, in record order."""
    by_game: dict[str, list] = {}
    for record in records:
        by_game.setdefault(record.game_id, []).append(record)
    return lambda: [list(rows) for rows in by_game.values()]


def _rules(tags: dict[str, list[RuleTag]], *, trained=None, declared=None, blocked=(),
           unreadable=()):
    every = {tag.rule for found in tags.values() for tag in found}
    return RuleTestRules(
        tag=lambda record: tags.get(record.record_id, []),
        declared=frozenset(declared if declared is not None else every),
        trained=frozenset(trained if trained is not None else every),
        blocked=frozenset(blocked),
        unreadable=frozenset(unreadable),
    )


def _tag(rule: str, polarity: Polarity, *entities: str) -> RuleTag:
    return RuleTag(rule, polarity, observed=polarity is FIRES, condition=(), outcome=entities,
                   scenario="s")


class _Probes:
    """One binary probe type, alternating labels, read at each record's E1."""

    def items(self, records):
        for i, record in enumerate(records):
            yield ProbeItem("dies", "binary", record.record_id, "E1", float(i % 2))


@dataclass
class _Grid:
    """An edit grid every record qualifies for, its outcome ``outcomes``."""

    name: str = "combat"
    cells: tuple = ("1x1", "1x2")
    outcomes: tuple = (1.0, 0.0)

    def qualify(self, record):
        return GridQualification(("E1", "E2"), self.outcomes)


class _Losses:
    """A loss of one on every copy under both masks, for one field."""

    def __init__(self) -> None:
        self.seen: list = []

    def losses(self, records):
        records = list(records)
        self.seen.append(records)
        return {"damage": {"hard": [1.0] * len(records), "soft": [1.0] * len(records)}}


def _two_halves(n: int, prefix: str = "g") -> dict:
    games = [f"{prefix}{i}" for i in range(2 * n)]
    return {GD: {"selection": games[0::2], "reporting": games[1::2]}}


def _balanced_records(halves: dict, per_game: int = 4) -> tuple[list, dict]:
    """Records of every game, each half carrying both polarities of rule ``R``."""
    records, tags = [], {}
    for by_half in halves.values():
        for games in by_half.values():
            for game in games:
                for k in range(per_game):
                    record = _Record(f"{game}.{k}", game)
                    records.append(record)
                    tags[record.record_id] = [_tag("R", FIRES if k % 2 else NOT, "E1")]
    return records, tags


def _freeze(halves, records, tags, **options):
    defaults = dict(rules=_rules(tags), probes=_Probes(), grids=[_Grid()],
                    deviance=_Losses(), **LENIENT)
    defaults.update(options)
    return freeze_selection_set(_manifest(halves), {GD: _reader(records)}, **defaults)


def _misses(refused) -> set[tuple[str, str, str]]:
    return {(m.stratum, m.half, m.component) for m in refused.value.misses}


# ── halves ──────────────────────────────────────────────────────────────


def test_a_half_under_the_game_minimum_refuses_before_reading() -> None:
    halves = _two_halves(3)

    def unread():
        raise AssertionError("the strata were read")

    with pytest.raises(SelectionSetRefused) as refused:
        freeze_selection_set(_manifest(halves), {GD: unread}, rules=_rules({}), probes=None,
                             grids=[], deviance=None, min_games_per_half=4)
    assert _misses(refused) == {(GD, "selection", "games"), (GD, "reporting", "games")}
    assert "3 game(s)" in refused.value.misses[0].detail


def test_halves_are_the_manifests_disjoint_and_deterministic() -> None:
    halves = _two_halves(5)
    records, tags = _balanced_records(halves)
    one = _freeze(halves, records, tags).sealed()
    two = _freeze(halves, records, tags).sealed()
    assert one.digest == two.digest
    assert one.halves == _manifest(halves).halves
    selection, reporting = (set(one.halves[GD][h]) for h in ("selection", "reporting"))
    assert not selection & reporting
    for half, games in one.halves[GD].items():
        read = {r.game_id for rows in one.rule_tests[GD][half]["R"].values() for r in rows}
        assert read <= set(games)


# ── rule tests ──────────────────────────────────────────────────────────


def test_no_two_polarity_rule_in_a_half_refuses_naming_it() -> None:
    """Story 8 acceptance 7."""
    halves = _two_halves(4)
    records, tags = _balanced_records(halves)
    for game in halves[GD]["selection"]:
        for k in range(4):
            tags[f"{game}.{k}"] = [_tag("R", FIRES)]
    with pytest.raises(SelectionSetRefused) as refused:
        _freeze(halves, records, tags)
    assert _misses(refused) == {(GD, "selection", "rule_tests")}
    left = refused.value.selection.left_out[GD]["selection"]["rule_tests"]
    assert [(item.item, item.reason) for item in left] == [
        ("R", "one polarity only: 16 fires, 0 does_not_fire")]
    assert any("R (one polarity only" in line for line in left_out_lines(refused.value.selection))


def test_three_rules_at_68_each_miss_the_component_target() -> None:
    """Story 8 acceptance 8: the smaller rules left out and named; three items
    at 0.0994 give the component 0.0574, above 0.05."""
    halves = {GD: {"selection": ["s"], "reporting": ["r"]}}
    records, tags = [], {}
    sizes = {"A": 68, "B": 68, "C": 68, "small1": 20, "small2": 5}
    for half, game in (("selection", "s"), ("reporting", "r")):
        for rule, n in sizes.items():
            for polarity in (FIRES, NOT):
                for i in range(n):
                    record = _Record(f"{game}.{rule}.{polarity}.{i}", game)
                    records.append(record)
                    tags[record.record_id] = [_tag(rule, polarity)]
    with pytest.raises(SelectionSetRefused) as refused:
        _freeze(halves, records, tags, min_games_per_half=1, max_item_error=0.1,
                max_component_error=0.05)
    misses = [m for m in refused.value.misses if m.component == "rule_tests"]
    assert {m.half for m in misses} == {"selection", "reporting"}
    assert "0.0574 over 3 item(s)" in misses[0].detail
    selection = refused.value.selection
    left = {item.item for item in selection.left_out[GD]["selection"]["rule_tests"]}
    assert left == {"small1", "small2"}
    assert sorted(selection.rule_tests[GD]["selection"]) == ["A", "B", "C"]
    assert error_of(selection.standard_errors[
        item_key(GD, "selection", "rule_tests", "A")]) == pytest.approx(0.0994, abs=1e-4)


def test_the_larger_polarity_is_cut_to_r_times_the_smaller() -> None:
    """Story 8 acceptance 9: 400 ``fires`` against 100 at r = 2 keep 200."""
    halves = {GD: {"selection": ["s"], "reporting": ["r"]}}
    records, tags = [], {}
    for game in ("s", "r"):
        for polarity, n in ((FIRES, 400), (NOT, 100)):
            for i in range(n):
                record = _Record(f"{game}.{polarity}.{i}", game)
                records.append(record)
                tags[record.record_id] = [_tag("R", polarity)]
    frozen = _freeze(halves, records, tags, r=2)
    sides = frozen.rule_tests[GD]["selection"]["R"]
    assert (len(sides["fires"]), len(sides["does_not_fire"])) == (200, 100)
    again = _freeze(halves, records, tags, r=2).rule_tests[GD]["selection"]["R"]["fires"]
    assert again == sides["fires"]


def test_directed_fork_and_other_ineligible_records_are_never_rule_tests() -> None:
    """Story 3 acceptance 5, FR-072: a directed fork, an aimed record, an
    interventional or synthetic one, polarity ``other`` and a review-list tag
    give no rule-test record."""
    halves = _two_halves(2)
    records, tags = _balanced_records(halves)
    game = halves[GD]["selection"][0]
    odd = [
        _Record("directed", game, fork=True, aimed_at=("R",),
                directed={"variant": "v", "operation": "pt"}),
        _Record("aimed", game, aimed_at=("R",)),
        _Record("forced", game, fork=True, interventional=True),
        _Record("copy~pc0", game, synthetic=True),
    ]
    records += odd
    for record in odd:
        tags[record.record_id] = [_tag("R", FIRES)]
    records += [_Record("other", game), _Record("review", game)]
    tags["other"] = [_tag("R", Polarity.OTHER)]
    tags["review"] = [RuleTag("R", FIRES, observed=False, condition=(), outcome=(), scenario="s")]
    frozen = _freeze(halves, records, tags)
    ids = {r.record_id for half in frozen.rule_tests[GD].values()
           for sides in half.values() for rows in sides.values() for r in rows}
    assert not ids & {"directed", "aimed", "forced", "copy~pc0", "other", "review"}


def _scenario(rule: str, departure: Departure | None) -> Scenario:
    return Scenario(id=f"{rule}-a", unit=rule.split(".")[0], rules=(rule,),
                    record_kind="resolution", variants=(), departure=departure)


def test_a_confirmed_unpatched_departure_keeps_its_rule_out_and_an_accepted_one_in() -> None:
    """Story 1 acceptance 6 and 7."""
    catalogue = ScenarioCatalogue((
        _scenario("U", Departure(DepartureStatus.CONFIRMED, DepartureResolution.UNPATCHED)),
        _scenario("A", Departure(DepartureStatus.CONFIRMED, DepartureResolution.ACCEPTED)),
        _scenario("S", Departure(DepartureStatus.SUSPECTED)),
    ))
    assert blocked_rules(catalogue) == {"U"}
    halves = _two_halves(2)
    records, tags = [], {}
    for by_half in halves.values():
        for games in by_half.values():
            for game in games:
                for k, rule in enumerate(("U", "U", "A", "A")):
                    record = _Record(f"{game}.{k}", game)
                    records.append(record)
                    tags[record.record_id] = [_tag(rule, FIRES if k % 2 else NOT)]
    frozen = _freeze(halves, records, tags, rules=_rules(tags, blocked=blocked_rules(catalogue)))
    for half in ("selection", "reporting"):
        assert sorted(frozen.rule_tests[GD][half]) == ["A"]
        assert frozen.rule_tests[GD][half]["A"]["fires"]


def test_untrained_rules_have_no_tests_and_undetected_rules_are_named_untested() -> None:
    halves = _two_halves(2)
    records, tags = _balanced_records(halves)
    for record_id in list(tags):
        tags[record_id] = [*tags[record_id], _tag("recorded-only", FIRES)]
    rules = _rules(tags, trained={"R", "never-seen"}, declared={"R", "never-seen",
                                                                "recorded-only"})
    frozen = _freeze(halves, records, tags, rules=rules)
    for half in ("selection", "reporting"):
        assert sorted(frozen.rule_tests[GD][half]) == ["R"]
        left = frozen.left_out[GD][half]["rule_tests"]
        assert [(item.item, item.reason) for item in left] == [("never-seen", UNTESTED)]


def test_rule_test_entities_come_from_the_tag() -> None:
    halves = _two_halves(1)
    records, tags = _balanced_records(halves)
    frozen = _freeze(halves, records, tags)
    rows = frozen.rule_tests[GD]["selection"]["R"]["fires"]
    assert {row.entities for row in rows} == {("E1",)}


def test_rule_test_entities_are_the_tag_s_outcome_entities() -> None:
    """Research R1: a rule test reads where the outcome falls, never the condition."""
    halves = _two_halves(1)
    records, tags = _balanced_records(halves)
    tags = {record_id: [replace(tag, condition=("E9",)) for tag in found]
            for record_id, found in tags.items()}
    frozen = _freeze(halves, records, tags)
    rows = frozen.rule_tests[GD]["selection"]["R"]["fires"]
    assert {row.entities for row in rows} == {("E1",)}


def test_a_rule_with_no_effective_output_is_named_untested_in_every_half() -> None:
    halves = _two_halves(2)
    records, tags = _balanced_records(halves)
    with pytest.raises(SelectionSetRefused) as refused:
        _freeze(halves, records, tags, rules=_rules(tags, unreadable={"R"}))
    frozen = refused.value.selection
    for half in ("selection", "reporting"):
        assert frozen.rule_tests[GD].get(half, {}) == {}
        left = frozen.left_out[GD][half]["rule_tests"]
        assert [(item.item, item.reason) for item in left] == [("R", NO_READABLE_OUTPUT)]


# ── copies ──────────────────────────────────────────────────────────────


def test_copies_are_read_once_by_tests_and_probes_and_every_time_by_the_deviance() -> None:
    halves = _two_halves(2)
    records, tags = _balanced_records(halves)
    copied = records[0]
    records.append(copied)
    losses = _Losses()
    frozen = _freeze(halves, records, tags, deviance=losses)
    half = next(h for h, games in halves[GD].items() if copied.game_id in games)
    tests = [r.record_id for sides in frozen.rule_tests[GD][half].values()
             for rows in sides.values() for r in rows]
    assert tests.count(copied.record_id) == 1
    probed = [r.record_id for rows in frozen.probes[GD][half].values() for r in rows]
    assert probed.count(copied.record_id) == 1
    sample = [(r.record_id, r.copy) for r in frozen.deviance_samples[GD][half]]
    assert (copied.record_id, 0) in sample and (copied.record_id, 1) in sample
    assert sum(len(seen) for seen in losses.seen) == len(records)


def test_the_deviance_sample_is_capped_by_a_seeded_draw() -> None:
    halves = _two_halves(2)
    records, tags = _balanced_records(halves, per_game=10)
    frozen = _freeze(halves, records, tags, deviance_cap=7)
    for half in ("selection", "reporting"):
        assert len(frozen.deviance_samples[GD][half]) == 7
    assert frozen.deviance_samples == _freeze(halves, records, tags,
                                              deviance_cap=7).deviance_samples


# ── probes, grids, deviance ─────────────────────────────────────────────


class _TwoProbes:
    """``dies`` on every record, ``rare`` on one record in four."""

    def items(self, records):
        for i, record in enumerate(records):
            yield ProbeItem("dies", "binary", record.record_id, "E1", float(i % 2))
            if i % 4 == 0:
                yield ProbeItem("rare", "amount", record.record_id, None, float(i))


def test_probe_types_are_cut_to_r_times_the_smallest() -> None:
    halves = _two_halves(4)
    records, tags = _balanced_records(halves, per_game=8)
    frozen = _freeze(halves, records, tags, probes=_TwoProbes(), r=2)
    for half in ("selection", "reporting"):
        sizes = {t: len(rows) for t, rows in frozen.probes[GD][half].items()}
        assert sizes["dies"] == 2 * sizes["rare"]
    assert frozen.probe_kinds == {"dies": "binary", "rare": "amount"}


def test_an_empty_grid_refuses_and_a_constant_grid_is_left_out() -> None:
    halves = _two_halves(2)
    records, tags = _balanced_records(halves)

    @dataclass
    class _Never(_Grid):
        def qualify(self, record):
            return None

    with pytest.raises(SelectionSetRefused) as refused:
        _freeze(halves, records, tags, grids=[_Grid(), _Never(name="never")])
    assert {m.detail for m in refused.value.misses} == {"grid never: every cell is empty"}
    frozen = _freeze(halves, records, tags,
                     grids=[_Grid(), _Grid(name="constant", outcomes=(1.0, 1.0))])
    for half in ("selection", "reporting"):
        assert sorted(frozen.grids[GD][half]) == ["combat"]
        left = frozen.left_out[GD][half]["grids"]
        assert [item.item for item in left] == ["constant"]
        grid = frozen.grids[GD][half]["combat"]
        assert [c.outcome for c in grid.cells] == [1.0, 0.0]
        assert len({r.game_id for r in grid.records}) == len(grid.records)


def test_deviance_errors_are_recorded_under_both_masks() -> None:
    halves = _two_halves(2)
    records, tags = _balanced_records(halves)

    class _Masked(_Losses):
        def losses(self, records):
            records = list(records)
            return {"damage": {"hard": [1.0] * len(records), "soft": [1.0] * len(records)},
                    "never": {"hard": [0.0] * len(records), "soft": [0.0] * len(records)}}

    frozen = _freeze(halves, records, tags, deviance=_Masked())
    error = frozen.standard_errors[item_key(GD, "selection", "deviance", "damage")]
    assert set(error) == {"hard", "soft"} and error["hard"] == pytest.approx(1 / math.sqrt(2))
    left = frozen.left_out[GD]["selection"]["deviance"]
    assert [item.item for item in left] == ["never", "never"]
    assert {item.reason.split(" under the ")[1] for item in left} == {
        "hard rule mask", "soft rule mask"}


def test_every_heads_terms_are_deviance_items_judged_together() -> None:
    """The verdict, global and created-objects heads' terms are items beside
    the per-entity fields (``selection_set.deviance_item``): each gets its
    standard error under both masks, one too rare to bound is left out and
    named, and the component's check reads every head's kept items."""
    halves = _two_halves(150)
    records, tags = _balanced_records(halves, per_game=2)
    rare_game = halves[GD]["selection"][0]

    class _EveryHead(_Losses):
        def losses(self, records):
            records = list(records)
            ones = [1.0] * len(records)
            rare = [1.0 if r.game_id == rare_game else 0.0 for r in records]
            return {name: {"hard": values, "soft": values} for name, values in (
                ("gate", ones), ("verdict.can_play", ones), ("created_objects.count", ones),
                ("global.day_night_change", rare))}

    def frozen_with(max_component_error):
        # The toy rule tests, probes and grids miss the strict targets; only
        # the deviance component is looked at, on the set as assembled.
        try:
            return _freeze(halves, records, tags, deviance=_EveryHead(), min_games_per_half=1,
                           max_item_error=0.1, max_component_error=max_component_error), set()
        except SelectionSetRefused as refused:
            return refused.selection, {m for m in refused.misses if m.component == "deviance"}

    frozen, misses = frozen_with(0.05)
    errors = frozen.standard_errors
    for name in ("gate", "verdict.can_play", "created_objects.count"):
        error = errors[item_key(GD, "selection", "deviance", name)]
        assert error["hard"] == pytest.approx(1 / math.sqrt(150))
    rare = errors[item_key(GD, "selection", "deviance", "global.day_night_change")]
    assert rare == {"hard": 1.0, "soft": 1.0}
    left = frozen.left_out[GD]["selection"]["deviance"]
    assert [item.item for item in left] == ["global.day_night_change"] * 2
    # Three kept items at 0.0816 each: the component is √3·0.0816 / 3 = 0.047.
    assert not misses
    _, misses = frozen_with(0.045)
    assert {(m.half, m.detail.split(" over ")[1].split(" under ")[0]) for m in misses} == {
        ("selection", "3 item(s)"), ("reporting", "3 item(s)")}


def test_a_missing_component_refuses_naming_it() -> None:
    halves = _two_halves(2)
    records, tags = _balanced_records(halves)
    with pytest.raises(SelectionSetRefused) as refused:
        _freeze(halves, records, tags, probes=None, grids=[], deviance=None)
    assert _misses(refused) == {(GD, half, component) for half in ("selection", "reporting")
                                for component in ("probes", "grids", "deviance")}


# ── the command and SC-010 ──────────────────────────────────────────────


def test_run_exits_one_naming_each_small_half(tmp_path) -> None:
    from effects.infrastructure.corpus_store import CorpusStore

    store = CorpusStore(tmp_path / "corpus")
    store.directory.mkdir()
    store.manifest_path.write_text(
        __import__("json").dumps(_manifest(_two_halves(3)).as_dict()), encoding="utf-8")
    printed: list[str] = []
    code = run(store.directory, reports_dir=tmp_path / "reports", echo=printed.append)
    assert code == 1
    assert printed[1:] == [
        f"  {GD} selection, games: 3 game(s), fewer than the 300 a half needs",
        f"  {GD} reporting, games: 3 game(s), fewer than the 300 a half needs",
    ]
    assert not (tmp_path / "reports").exists()


def test_sc010_on_the_fixture_every_component_and_balance_holds(tmp_path) -> None:
    """SC-010 over the gen-3 fixture, tagged by the real catalogue: every
    component within its target, every polarity, probe type and grid within r."""
    from effects.infrastructure.converted_trees import SourceTrees
    from effects.infrastructure.rules_catalogue_io import load_scenarios
    from tests.unit.effects.builders import GEN3_SIDECAR_FOLDERS, curated_fixture_records

    records = curated_fixture_records()
    games = sorted({r.game_id for r in records})
    halves = {GD: {"selection": games[0::2], "reporting": games[1::2]}}
    manifest = _manifest(halves)
    lines = SourceTrees.of(GEN3_SIDECAR_FOLDERS, None).sidecar_cache()
    rules = RuleTestRules.of(load_scenarios(), manifest, lines)
    rules = RuleTestRules(rules.tag, rules.declared, rules.declared, rules.blocked)
    limits = dict(min_games_per_half=1, max_item_error=2.0, max_component_error=2.0, r=4)
    frozen = freeze_selection_set(
        manifest, {GD: _reader(records)}, rules=rules, probes=_TwoProbes(),
        grids=[_Grid()], deviance=_Losses(), **limits)
    assert any(frozen.rule_tests[GD][half] for half in ("selection", "reporting"))
    for half in ("selection", "reporting"):
        for component in ("rule_tests", "probes", "grids"):
            items = [error_of(value) for key, value in frozen.standard_errors.items()
                     if key.startswith(f"{GD}/{half}/{component}/")
                     and error_of(value) <= limits["max_item_error"]]
            assert component_standard_error(items) <= limits["max_component_error"]
        for sides in frozen.rule_tests[GD][half].values():
            sizes = [len(rows) for rows in sides.values()]
            assert max(sizes) <= limits["r"] * min(sizes)
        sizes = [len(rows) for rows in frozen.probes[GD][half].values()]
        assert max(sizes) <= limits["r"] * min(sizes)
    path = selection_set_path(tmp_path, manifest.digest())
    from effects.domain.selection_set import write_selection_set

    write_selection_set(path, frozen)
    assert load_selection_set(path, manifest.digest()).rule_tests == frozen.rule_tests


def test_a_grid_freezes_the_fewest_records_its_precision_target_needs() -> None:
    """T160: a grid's records cost a forward per cell every epoch, so it
    keeps the fewest games, by smallest keyed hash, that bring its worst-case
    standard error ``1 / (2 sqrt(N) B)`` to the target; outcomes (1, 0) give
    B = 0.5, so a target of 0.5 needs 4 records and one of 1 needs 1."""
    from effects.domain.seeded_draw import keyed_hash
    from effects.domain.selection_set import grid_records_needed

    assert grid_records_needed([1.0, 0.0], target=0.5) == 4
    assert grid_records_needed([1.0, 0.0], target=1.0) == 1
    assert grid_records_needed([1.0, 0.0]) == 124             # the default target, 0.09
    assert grid_records_needed([1.0, 0.0], target=0.01, cap=50) == 50
    assert grid_records_needed([1.0, 1.0]) == 0
    halves = _two_halves(10)
    records, tags = _balanced_records(halves)
    frozen = _freeze(halves, records, tags, grid_target_error=0.5)
    for half in ("selection", "reporting"):
        grid = frozen.grids[GD][half]["combat"]
        assert len(grid.records) == 4
        games = halves[GD][half]
        first = {r.record_id for r in records if r.game_id in games}
        seed = _manifest(halves).seed
        smallest = sorted(min((keyed_hash(rid, seed), rid) for rid in first
                              if rid.split(".")[0] == game) for game in games)[:4]
        assert sorted(r.record_id for r in grid.records) == sorted(rid for _, rid in smallest)
    assert frozen.settings["grid_target_error"] == 0.5
    assert frozen.settings["grid_record_cap"] == 512


def test_a_capped_grid_is_judged_on_the_records_it_keeps() -> None:
    """FR-068b: the cap lowers a grid's precision, and the freeze's own
    standard error judges what is left: at one record a grid of B = 0.5 has a
    standard error of 1, above the item target, and is left out."""
    halves = _two_halves(10)
    records, tags = _balanced_records(halves)
    with pytest.raises(SelectionSetRefused) as refused:
        _freeze(halves, records, tags, grid_record_cap=1, max_item_error=0.9)
    assert _misses(refused) == {(GD, "selection", "grids"), (GD, "reporting", "grids")}
    frozen = refused.value.selection
    for half in ("selection", "reporting"):
        assert "combat" not in frozen.grids[GD][half]
        left = {item.item: item.reason for item in frozen.left_out[GD][half]["grids"]}
        assert "standard error 1.0000" in left["combat"]

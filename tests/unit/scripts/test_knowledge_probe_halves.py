"""The probing run over the selection set's halves (T168)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
from effect_knowledge_probes import (
    common,
    compare,
    halves,
    labels,
    record_walk,
    run,
    selection,
)
from effect_knowledge_probes.line_labelling import ObservedOutcomes
from effect_knowledge_probes.tables import write_tables

from tests.unit.effects.builders import (
    GEN3_SIDECAR_FOLDERS,
)
from tests.unit.scripts.knowledge_probe_support import probabilities

GD = "game-disjoint"


class _NoDeviance:
    """The deviance sample's losses: one field at a loss of one, both masks."""

    def losses(self, records):
        records = list(records)
        return {"damage_taken": {"hard": [1.0] * len(records), "soft": [1.0] * len(records)}}


@pytest.fixture(scope="module")
def frozen_corpus(tmp_path_factory, records, sidecars, keywords_of):
    """The fixture's games as a curated corpus's game-disjoint stratum, every
    third record repeated as an added copy is, and a selection set frozen over
    it by the probe suite's own ports at thresholds a fixture can meet."""
    from effects.application.freeze_selection_set import (
        RuleTestRules,
        SelectionSetRefused,
        freeze_selection_set,
        stratum_reader,
    )
    from effects.domain.corpus_manifest import ClassCounts, CorpusManifest, SourceShard
    from effects.infrastructure.rules_catalogue_io import load_scenarios
    from effects.infrastructure.shard_io import write_shard

    root = tmp_path_factory.mktemp("frozen-corpus")
    stratum = root / "validation" / GD
    stratum.mkdir(parents=True)
    write_shard(stratum / "shard-00000.jsonl.gz", [*records, *records[::3]])
    (root / "training").mkdir()
    write_shard(root / "training" / "shard-00000.jsonl.gz", [*records[:40], *records[:40]])
    games = sorted({r.game_id for r in records})
    halves_of = {GD: {"selection": tuple(games[0::2]), "reporting": tuple(games[1::2])}}
    manifest = CorpusManifest(
        seed=42, surface="script", vocab_path="vocab-script.txt", variant_scripts="",
        holdout_permille=20, holdout_max_carriers=8, text_cap=200,
        card_disjoint_text_cap=50, training_records=0, class_mix={"rewrite": 1.0},
        delivered_mix={"rewrite": 1.0}, held_out_cards=(), held_out_texts=(),
        card_disjoint_games=(), game_disjoint_games=tuple(games),
        rarity={}, sources=(SourceShard("a.jsonl.gz", 1),),
        per_class={"rewrite": ClassCounts(1, 1, 0, 1)}, per_stratum={}, unique_texts={},
        shortfall={}, halves=halves_of,
    )
    rules = RuleTestRules.of(load_scenarios(), manifest, sidecars)
    rules = RuleTestRules(rules.tag, rules.declared, rules.declared, rules.blocked)
    try:
        frozen = freeze_selection_set(
            manifest, {GD: stratum_reader(stratum)}, rules=rules,
            probes=selection.ProbeItems(
                sidecars, common.acting_text_function(sidecars, "script")),
            grids=selection.grid_sources(sidecars, keywords_of), deviance=_NoDeviance(),
            min_games_per_half=1, max_item_error=10.0, max_component_error=10.0)
    except SelectionSetRefused as refused:
        frozen = refused.selection
    return SimpleNamespace(root=root, stratum=stratum, manifest=manifest,
                           selection=frozen.sealed())


def _context(frozen_corpus, sidecars, probe_model):
    """What a probing run reads of one checkpoint, the model the fixture's."""
    from effects.infrastructure.rules_catalogue_io import load_heads

    root = frozen_corpus.root
    heads = load_heads()
    return SimpleNamespace(
        corpus=root, corpus_digest=frozen_corpus.selection.corpus_digest,
        manifest=SimpleNamespace(
            surface="script", rule_map=heads,
            effective_rule_outputs={rule: tuple(e.outputs) for rule, e in heads.items()}),
        cards_folders=tuple(GEN3_SIDECAR_FOLDERS), variant_scripts=None,
        checkpoint=SimpleNamespace(
            provenance=SimpleNamespace(held_out_cards=()), training_settings={"seed": 1},
            model_config=probe_model.model.config),
        sidecars=lambda: sidecars, stratum_dir=lambda name: root / "validation" / name,
        training_dir=root / "training", model=lambda records, **_: probe_model,
    )


def _assert_items_match_frozen_labels(read, frozen) -> None:
    labelled = {(row.record_id, probe_type, row.entity): row.label
                for probe_type, rows in frozen.probes[GD]["reporting"].items()
                for row in rows}
    found = {(rid, item.target, item.entity): item.label
             for rid, items in read.probe_items.items() for item in items}
    assert found and set(found) <= set(labelled)
    assert all(found[key] == labelled[key] for key in found)


def _assert_read_within_frozen(read, frozen) -> None:
    """The half's read records belong to its games, its items match the frozen labels."""
    games = set(frozen.halves[GD]["reporting"])
    assert all(r.game_id in games for r in read.records.values())
    _assert_items_match_frozen_labels(read, frozen)
    assert set(read.grids) <= set(frozen.grids[GD]["reporting"])
    assert read.conditions or read.targeting
    for rows in read.sweep_rows.values():
        assert all(row["record_id"] in read.records for row in rows)


class TestReadingAHalf:
    def test_repeated_rows_of_one_record_are_read_once(self, records):
        doubled = [*records[:5], *records[:5], records[0]]
        assert [r.record_id for r in halves.distinct(doubled)] == [
            r.record_id for r in records[:5]]

    def test_a_capped_probe_keeps_the_smallest_hashes_per_game_and_in_all(self):
        cap = halves.Capped(limit=3, per_game=2)
        rows = [SimpleNamespace(record_id=f"r{i}", game_id=f"g{i % 2}") for i in range(8)]
        for row in rows:
            cap.add(row, row.record_id)
        order = sorted(rows, key=lambda r: common.stable_hash(r.record_id))
        kept = {r.record_id for g in ("g0", "g1")
                for r in [r for r in order if r.game_id == g][:2]}
        assert cap.values() == [r.record_id for r in order if r.record_id in kept][:3]

    def test_a_half_keeps_its_frozen_items_and_qualifiers(self, frozen_corpus, sidecars,
                                                         keywords_of, probe_model):
        frozen = frozen_corpus.selection
        context = _context(frozen_corpus, sidecars, probe_model)
        seen: list[str] = []
        settings = halves.ReadSettings(probe_types=frozenset(frozen.probe_kinds))
        read = halves.read_half(
            context, frozen, GD, "reporting", settings,
            acting_text=common.acting_text_function(sidecars, "script"),
            keywords_of=keywords_of, observe=lambda r: seen.append(r.record_id))
        assert seen and len(seen) == len(set(seen))
        _assert_read_within_frozen(read, frozen)


class TestRuleTests:
    """A probing run scores a half's rule tests through the control score's
    own scorer (``control_score.score_half``), one rule per item."""

    @pytest.fixture(scope="class")
    def scored(self, frozen_corpus, sidecars, probe_model):
        context = _context(frozen_corpus, sidecars, probe_model)
        return {half: halves.score_rule_tests(context, frozen_corpus.selection, probe_model,
                                              half)
                for half in halves.HALVES}

    def test_every_frozen_rule_is_scored_or_left_out(self, scored, frozen_corpus):
        frozen = frozen_corpus.selection
        for half, by_stratum in scored.items():
            assert set(by_stratum) == {GD}
            tests = by_stratum[GD]
            assert set(tests["rules"]) | set(tests["left_out"]) == set(
                frozen.rule_tests[GD][half])
            assert not set(tests["rules"]) & set(tests["left_out"])

    def test_a_rule_scores_on_the_control_scores_scale(self, scored):
        scores = [value for by_stratum in scored.values()
                  for value in by_stratum[GD]["rules"].values()]
        assert scores and all(-1.0 <= value <= 1.0 for value in scores)
        for by_stratum in scored.values():
            tests = by_stratum[GD]
            if tests["rules"]:
                assert tests["component"] == pytest.approx(
                    np.mean(list(tests["rules"].values())))
            else:
                assert np.isnan(tests["component"])

    def test_the_scores_are_the_scorers_own(self, scored, frozen_corpus, probe_model):
        from effects.application.control_score import score_half
        from effects.application.freeze_selection_set import stratum_reader
        from effects.domain.selection_set import RULE_TESTS
        from effects.infrastructure.rules_catalogue_io import load_heads

        direct = score_half(
            frozen_corpus.selection, {GD: stratum_reader(frozen_corpus.stratum)},
            half="reporting", encoder=probe_model.encoder, model=probe_model.model,
            batcher=probe_model.batcher,
            rule_outputs={rule: tuple(entry.outputs) for rule, entry in load_heads().items()},
            mask=None, stratum_weight=0.5, deviance_floor=0.0, components=(RULE_TESTS,))
        assert dict(direct.strata[GD].items[RULE_TESTS]) == pytest.approx(
            scored["reporting"][GD]["rules"])


class TestGridsOfAHalf:
    def test_every_frozen_grid_is_read_in_its_cells_order(
        self, frozen_corpus, sidecars, keywords_of, probe_model,
    ):
        frozen = frozen_corpus.selection
        context = _context(frozen_corpus, sidecars, probe_model)
        settings = halves.ReadSettings(probe_types=frozenset(), gen3_probes=False,
                                       sweeps=False)
        reads = [halves.read_half(context, frozen, GD, half, settings,
                                  acting_text=lambda key: None, keywords_of=keywords_of)
                 for half in halves.HALVES]
        scored = [(read, halves.score_grids(probe_model, frozen, read)) for read in reads]
        assert any(found for _, found in scored)
        for read, found in scored:
            for name, body in found.items():
                cells = frozen.grids[GD][read.half][name].cells
                record, entities = read.grids[name][0]
                values = halves.grid_predictions(probe_model, name, record, entities)
                assert len(values) == len(cells)
                probabilities(values)
                assert body["records"] == len(read.grids[name])
                assert 0.0 <= body["mae"] <= 1.0


class TestTheProbingRun:
    @pytest.fixture(scope="class")
    def scorecard(self, frozen_corpus, sidecars, probe_model, tmp_path_factory):
        args = run.parse_args(["--checkpoint", "latest.pt", "--no-ablation", "--no-board",
                               "--abilities-root", str(tmp_path_factory.mktemp("cache")),
                               "--reports-dir", "reports"])
        return run.probe(args, _context(frozen_corpus, sidecars, probe_model),
                         frozen_corpus.selection)

    def test_probes_and_rule_tests_are_reported_per_half(self, scorecard, frozen_corpus):
        frozen = frozen_corpus.selection
        assert set(scorecard.halves) == {GD}
        assert set(scorecard.halves[GD]) == set(halves.HALVES)
        for half, body in scorecard.halves[GD].items():
            assert body["games"] == len(frozen.halves[GD][half])
            assert body["probes"] and set(body["probes"]) <= set(frozen.probe_kinds)
            for item in body["probes"].values():
                assert item["kind"] in ("binary", "amount") and item["n"] > 0
            tests = body["rule_tests"]
            assert set(tests["rules"]) | set(tests["left_out"]) == set(
                frozen.rule_tests[GD][half])
            assert "grids" in body
        assert scorecard.selection_set_digest == frozen.digest

    def test_the_ladders_are_filed_per_stratum_and_half(self, scorecard):
        names = {stratum for body in scorecard.families.values()
                 for entry in body["targets"].values() for stratum in entry["strata"]
                 if entry["scope"] in ("act", "entity")}
        assert names <= {f"{GD}/{half}" for half in halves.HALVES}

    def test_the_gen3_probes_are_reported(self, scorecard):
        reported = {name for body in scorecard.halves[GD].values() for name in body}
        assert reported & {"conditions", "anthem", "target_legality"}

    def test_the_scorecard_round_trips_and_writes_its_tables(self, scorecard, tmp_path):
        from effect_knowledge_probes.scorecard import Scorecard

        data = json.loads(json.dumps(scorecard.to_json()))
        again = Scorecard.from_json(data)
        assert again.halves == data["halves"]
        write_tables(tmp_path, again)
        text = (tmp_path / "halves.md").read_text(encoding="utf-8")
        assert "## Rule tests" in text and f"{GD}/reporting" in text

    def test_compare_ranks_each_half_s_scores_apart(self, scorecard):
        ranked = compare.half_ranking([scorecard, scorecard])
        assert ranked
        wheres = {f"{GD}/{half}" for half in halves.HALVES}
        for name in ranked:
            where, score = name.split(" ", 1)
            assert where in wheres
            assert score in ("probes AUC", "probes R²", "grids", "rule tests")


class TestLineProbesReadEachRecordOnce:
    def test_observed_outcomes_drop_repeated_rows(self, records, sidecars):
        from effects.domain.records import Moment, RecordKind

        acting = common.acting_text_function(sidecars, "script")
        record = next(r for r in records if r.kind is RecordKind.RESOLUTION
                      and r.moment is Moment.RESOLUTION and r.ability
                      and acting(r.ability[0]) and labels.effect_profile(r) is not None)
        outcomes = ObservedOutcomes(acting)
        for _ in range(3):
            outcomes.observe(record)
        assert sum(len(v) for v in outcomes.profiles.values()) == 1

    def test_the_training_walk_drops_repeats_unless_told_not_to(self, frozen_corpus):
        directory = frozen_corpus.root / "training"
        ids = [r.record_id for r in record_walk.records([directory])]
        assert len(ids) == len(set(ids)) == 40
        assert len(list(record_walk.records([directory], distinct=False))) == 80

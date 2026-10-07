"""The knowledge-probe suite (FR-089): share, labels, sweeps, folds, digest, compare.

``scripts/`` is not part of the installed package, so the suite's modules are
loaded by file path, through ``run.sibling`` — the same loader the suite uses,
so a test reads the very modules a run would.

Fixtures are real: the gen-1 records and sidecars cut from the corpus by
``scripts/make_fixture_records.py``, and the real gen-2 chained sidecars.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[3]
_SUITE = _ROOT / "scripts" / "effect_knowledge_probes"
_FIXTURES = _ROOT / "tests" / "fixtures" / "effects"


def _load_run():
    spec = importlib.util.spec_from_file_location(
        "effect_knowledge_probes.run", _SUITE / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run = _load_run()
labels = run.sibling("labels")
ladder = run.sibling("ladder")
sweeps = run.sibling("sweeps")
compare = run.sibling("compare")
common = run.sibling("common")


@pytest.fixture(scope="module")
def records():
    from effects.infrastructure.record_io import read_shard

    return list(read_shard(_FIXTURES / "gen1-records.jsonl.gz"))


@pytest.fixture(scope="module")
def sidecars():
    return common.sidecar_cache([
        _FIXTURES / "gen1-sidecars" / "cardsfolder",
        _FIXTURES / "gen1-sidecars" / "tokenscripts",
    ])


def _fixture_lines(tree: str):
    from effects.infrastructure.sidecar_io import read_sidecar

    for path in sorted((_FIXTURES / tree).rglob("*.provenance.json")):
        sidecar = read_sidecar(path)
        for line in sidecar.lines:
            if line.script_text:
                yield line


def _line(tree: str, *needles: str):
    for line in _fixture_lines(tree):
        if all(n in line.script_text for n in needles):
            return line
    raise AssertionError(f"no fixture line holds {needles}")


# ── the share and its minimum gap ───────────────────────────────────────


class TestShare:
    def test_it_is_the_vector_s_part_of_the_gap(self):
        assert ladder.share(0.5, 0.7, 0.9) == pytest.approx(0.5)

    def test_it_is_not_reported_under_the_minimum_gap(self):
        assert ladder.share(0.60, 0.70, 0.60 + ladder.MIN_GAP - 1e-6) is None

    def test_it_is_reported_at_the_minimum_gap(self):
        assert ladder.share(0.60, 0.61, 0.60 + ladder.MIN_GAP + 1e-9) is not None

    def test_a_share_above_one_is_kept(self):
        assert ladder.share(0.5, 0.95, 0.8) > 1.0

    def test_an_undefined_rung_gives_no_share(self):
        assert ladder.share(float("nan"), 0.7, 0.9) is None

    def test_the_bootstrap_interval_brackets_the_estimate(self):
        rng = np.random.default_rng(0)
        n = 400
        groups = np.array([f"t{i // 4}" for i in range(n)], dtype=object)
        y = rng.integers(0, 2, n).astype(float)
        oof0 = rng.normal(size=n)
        oof1 = y + rng.normal(scale=1.0, size=n)
        pred3 = y + rng.normal(scale=0.5, size=n)
        point = ladder.share(ladder.score("binary", y, oof0), ladder.score("binary", y, oof1),
                             ladder.score("binary", y, pred3))
        low, high = ladder.bootstrap_share("binary", y, oof0, oof1, pred3, groups,
                                           resamples=200)
        assert low <= point <= high

    def test_run_ladder_reports_each_probe_type_s_own_share(self):
        rng = np.random.default_rng(1)
        n = 300
        y = rng.integers(0, 2, n).astype(float)
        noise = rng.normal(size=(n, 3))
        data = ladder.LadderInput(
            kind="binary", y=y,
            groups=np.array([f"g{i}" for i in range(n)], dtype=object),
            features={"0": noise, "1": np.column_stack([noise, y + rng.normal(size=n)])},
            rung3=y + rng.normal(scale=0.3, size=n),
        )
        result = ladder.run_ladder(data, rungs=("0", "1"),
                                   mlp_options={"hidden": 8, "epochs": 3, "device": "cpu"},
                                   bootstrap=50)
        assert set(result["rungs"]) == {"0", "1", "3"}
        assert set(result["share"]) == {"linear", "mlp"}
        assert 0.0 < result["share"]["linear"]["value"] < 1.5

    def test_no_share_is_reported_when_the_model_knows_nothing_more(self):
        rng = np.random.default_rng(2)
        n = 1000
        y = rng.integers(0, 2, n).astype(float)
        data = ladder.LadderInput(
            kind="binary", y=y, groups=np.array([f"g{i}" for i in range(n)], dtype=object),
            # The board alone reads the label as well as the model does.
            features={"0": np.column_stack([y + rng.normal(scale=0.3, size=n),
                                            rng.normal(size=n)]),
                      "1": rng.normal(size=(n, 2))},
            rung3=y + rng.normal(scale=0.3, size=n),
        )
        result = ladder.run_ladder(data, rungs=("0", "1"),
                                   mlp_options={"hidden": 16, "epochs": 150, "device": "cpu"},
                                   bootstrap=0)
        assert result["share"] is None


# ── folds ───────────────────────────────────────────────────────────────


class TestFolds:
    def test_no_group_falls_on_both_sides(self):
        groups = [f"text-{i % 37}" for i in range(500)]
        folds = ladder.fold_assignment(groups)
        seen: dict[str, set[int]] = {}
        for group, fold in zip(groups, folds):
            seen.setdefault(group, set()).add(int(fold))
        assert all(len(f) == 1 for f in seen.values())
        assert set(folds.tolist()) == set(range(ladder.FOLDS))

    def test_assignment_is_a_function_of_the_names_alone(self):
        groups = ["b", "a", "c", "a", "d", "e", "f"]
        first = ladder.fold_assignment(groups)
        again = ladder.fold_assignment(list(reversed(groups)))[::-1]
        assert first.tolist() == again.tolist()

    def test_pooled_items_are_grouped_by_card(self, records, sidecars):
        acting = run._acting_text_function(sidecars, "script")
        join = labels.join_triggers(records)
        by_record = {r.record_id: r for r in records}
        for record in records:
            for item in labels.record_items(record, sidecars, join, acting):
                if item.entity is None:
                    continue
                entity = by_record[item.record_id].state.entity(item.entity)
                assert item.group == labels.card_of(entity)

    def test_the_width_control_is_one_vector_per_text(self):
        a = ladder.width_vector("SP$ Draw | NumCards$ 1", 16)
        assert np.array_equal(a, ladder.width_vector("SP$ Draw | NumCards$ 1", 16))
        assert not np.array_equal(a, ladder.width_vector("SP$ Draw | NumCards$ 2", 16))


# ── the probes ──────────────────────────────────────────────────────────


class TestProbes:
    def test_the_mlp_reads_a_non_linear_label_on_cpu(self):
        rng = np.random.default_rng(3)
        X = rng.normal(size=(600, 2))
        y = ((X[:, 0] * X[:, 1]) > 0).astype(float)
        folds = ladder.fold_assignment([str(i) for i in range(600)])
        mlp = ladder.mlp_oof(X, y, folds, "binary", hidden=32, epochs=60, device="cpu")
        linear = ladder.linear_oof(X, y, folds, "binary")
        assert ladder.score("binary", y, mlp) > 0.8
        assert ladder.score("binary", y, linear) < 0.65

    def test_an_mlp_amount_stays_within_a_span_of_the_training_range(self):
        rng = np.random.default_rng(5)
        X = rng.normal(size=(60, 2))
        X[-1] = 1e6
        y = rng.uniform(0, 3, size=60)
        folds = np.array([0] * 59 + [1])
        prediction = ladder.mlp_oof(X, y, folds, "amount", hidden=8, epochs=5, device="cpu")
        low, high = y[:59].min(), y[:59].max()
        span = max(high - low, 1.0)
        assert low - span - 1e-6 <= prediction[-1] <= high + span + 1e-6

    def test_the_linear_probe_reads_an_amount(self):
        rng = np.random.default_rng(4)
        X = rng.normal(size=(300, 3))
        y = 2 * X[:, 0] + rng.normal(scale=0.1, size=300)
        folds = ladder.fold_assignment([str(i) for i in range(300)])
        assert ladder.score("amount", y, ladder.linear_oof(X, y, folds, "amount")) > 0.95

    def test_the_constants_are_the_planned_ones(self):
        assert (ladder.MIN_GAP, ladder.FOLDS, ladder.SEED) == (0.05, 5, 42)
        assert (ladder.MLP_HIDDEN, ladder.MLP_LAYERS, ladder.MLP_EPOCHS) == (256, 2, 30)
        assert (ladder.LINEAR_C, ladder.RIDGE_ALPHA) == (1.0, 1.0)
        assert ladder.BOOTSTRAP_RESAMPLES == 1000


# ── line labels from real scripts ───────────────────────────────────────


class TestLineLabels:
    def test_an_activated_damage_ability(self):
        line = _line("gen1-sidecars", "AB$ DealDamage", "NumDmg$ 3", "Sac<1/CARDNAME")
        found = labels.line_labels(line.script_text, line.line_kind, line.script_api_type)
        assert found["damage"] == 3
        assert found["cost_r"] == 1 and found["cost_generic"] == 4
        assert found["cost_total"] == 5
        assert found["damage_per_mana"] == pytest.approx(3 / 5)
        assert found["sacrifice_cost"] == 1.0 and found["tap_cost"] == 0.0
        assert found["repeatable"] == 0.0
        assert found["instant_speed"] == 1.0

    def test_a_spell_line_has_no_mana_cost_labels(self):
        line = _line("gen1-sidecars", "SP$ DealDamage", "NumDmg$ 3")
        found = labels.line_labels(line.script_text, line.line_kind, line.script_api_type,
                                   "instant")
        assert found["damage"] == 3
        assert "cost_total" not in found and "damage_per_mana" not in found
        assert found["instant_speed"] == 1.0
        assert found["repeatable"] == 0.0

    def test_a_variable_amount_is_left_unlabelled(self):
        line = _line("gen1-sidecars", "NumAtt$ -X")
        found = labels.line_labels(line.script_text, line.line_kind, line.script_api_type)
        assert "power_change" not in found and "toughness_change" not in found
        assert found["until_end_of_turn"] == 1.0

    def test_a_restricted_any_colour_mana_ability(self):
        line = _line("gen1-sidecars", "Produced$ Any", "RestrictValid$")
        found = labels.line_labels(line.script_text, line.line_kind, line.script_api_type)
        assert found["any_colour"] == 1.0 and found["restricted"] == 1.0
        assert found["mana_amount"] == 1.0 and found["tap_cost"] == 1.0
        assert found["produces_g"] == 0.0

    def test_a_two_colour_mana_ability(self):
        line = _line("gen1-sidecars", "Produced$ Combo U R")
        found = labels.line_labels(line.script_text, line.line_kind, line.script_api_type)
        assert found["produces_u"] == found["produces_r"] == 1.0
        assert found["produces_w"] == 0.0 and found["any_colour"] == 0.0

    def test_a_sorcery_speed_activation(self):
        line = _line("gen1-sidecars", "SorcerySpeed$ True", "KW$ Flying")
        found = labels.line_labels(line.script_text, line.line_kind, line.script_api_type)
        assert found["instant_speed"] == 0.0 and found["repeatable"] == 1.0

    def test_a_counter_is_not_until_end_of_turn(self):
        line = _line("gen1-sidecars", "AB$ PutCounter |")
        found = labels.line_labels(line.script_text, line.line_kind, line.script_api_type)
        assert found["as_counter"] == 1.0 and found["until_end_of_turn"] == 0.0

    def test_a_gen2_chain_sums_its_segments(self):
        chained = [
            line for line in _fixture_lines("gen2-sidecars")
            if "[SEG]" in line.script_text and "NumDmg$" in line.script_text
        ]
        if not chained:
            pytest.skip("no chained damage text in the gen-2 fixture")
        line = chained[0]
        found = labels.line_labels(line.script_text, line.line_kind, line.script_api_type)
        segments = labels.segments(line.script_text)
        stated = [s["NumDmg"] for s in segments if "NumDmg" in s]
        if all(v.lstrip("+-").isdigit() for v in stated):
            assert found["damage"] == sum(int(v) for v in stated)
        else:
            assert "damage" not in found

    def test_a_charm_root_states_no_amounts(self):
        root = "CharmNum$ 2 | Choices$ SV1,SV2,SV3,SV4 | SP$ Charm"
        found = labels.line_labels(root, "spell", "Charm")
        assert not {"damage", "power_change", "counters_placed"} & set(found)


# ── record labels from real records ─────────────────────────────────────


class TestRecordLabels:
    def _items(self, records, sidecars):
        acting = run._acting_text_function(sidecars, "script")
        join = labels.join_triggers(records)
        by_id = {r.record_id: r for r in records}
        out = []
        for record in records:
            out += [(by_id[i.record_id], i)
                    for i in labels.record_items(record, sidecars, join, acting)]
        return out

    def test_every_item_names_a_known_target_and_a_finite_label(self, records, sidecars):
        items = self._items(records, sidecars)
        assert items
        for _, item in items:
            assert item.target in labels.TARGETS_BY_NAME
            assert np.isfinite(item.label)

    def test_a_blocker_label_follows_the_legal_blockers(self, records, sidecars):
        blocks = [(r, i) for r, i in self._items(records, sidecars) if i.target == "may_block"]
        assert blocks
        assert any(i.label == 0.0 for _, i in blocks)
        for record, item in blocks:
            assert item.label == float(item.entity in record.payload.legal_blockers)
            assert item.partner_entity == record.payload.anchor_attacker

    def test_a_decision_yields_its_affordable_verdict(self, records, sidecars):
        decisions = [(r, i) for r, i in self._items(records, sidecars)
                     if i.target == "affordable"]
        assert decisions
        for record, item in decisions:
            assert item.label == float(record.payload.candidates[0].affordable)

    def test_a_trigger_yields_whether_it_fired(self, records, sidecars):
        fires = [(r, i) for r, i in self._items(records, sidecars) if i.target == "fires"]
        assert {i.label for _, i in fires} == {0.0, 1.0}
        for record, item in fires:
            assert item.label == float(record.payload.fired)

    def test_affected_items_cover_the_battlefield_of_an_effect_half(self, records, sidecars):
        from effects.domain.effect_targets import derive_targets

        affected = [(r, i) for r, i in self._items(records, sidecars) if i.target == "affected"]
        assert affected
        for record, item in affected:
            entry = derive_targets(record).get(item.entity)
            assert item.label == float(bool(entry and entry.affected))

    def test_the_join_finds_causes_in_the_same_game(self, records):
        result = labels.join_triggers(records)
        by_id = {r.record_id: r for r in records}
        assert 0.0 <= result.join_rate <= 1.0 or np.isnan(result.join_rate)
        for record_id, key in result.cause.items():
            game = by_id[record_id].game_id
            assert any(r.game_id == game and r.ability and r.ability[0] == key
                       for r in records if r.kind.value == "resolution")

    def test_state_dependence_is_the_spread_of_a_text_s_outcomes(self, records):
        profiles = [p for p in (labels.effect_profile(r) for r in records) if p is not None]
        assert len(profiles) >= 2
        same = labels.state_dependence({"t": [profiles[0], profiles[0]]})
        assert same["t"] == (0.0, 2)
        mixed = labels.state_dependence({"t": profiles[:2], "once": profiles[:1]})
        assert "once" not in mixed and mixed["t"][1] == 2
        assert labels.state_weight(5) == pytest.approx(0.5)

    def test_the_mined_pairs_join_a_death_trigger_to_a_sacrifice_outlet(self):
        items = common.line_items([_FIXTURES / "gen1-sidecars" / "cardsfolder"], "script")
        pairs = labels.mined_pairs(items)
        by_key = {item.key: item for item in items}
        assert pairs
        for trigger, cause in pairs:
            assert by_key[trigger].line_kind == "triggered"
        assert len({t for t, _ in pairs}) == len(pairs)


# ── the sweep editor ────────────────────────────────────────────────────


def _differences(before, after) -> set[str]:
    """Which top-level parts of a record's state differ."""
    changed = set()
    if before.state.players != after.state.players:
        changed.add("players")
    if before.state.entities != after.state.entities:
        changed.add("entities")
    if before.payload != after.payload or before.ability != after.ability:
        changed.add("record")
    return changed


class TestSweepEditor:
    def test_toughness_edits_one_entity_only(self, records):
        record = next(r for r in records if any(
            e.pt is not None and e.zone == "battlefield" for e in r.state.entities))
        target = next(e for e in record.state.entities if e.pt is not None)
        edited = sweeps.set_toughness(record, target.id, 7)
        assert edited.state.entity(target.id).pt.total[1] == 7
        assert _differences(record, edited) == {"entities"}
        for before, after in zip(record.state.entities, edited.state.entities):
            if before.id != target.id:
                assert before == after

    def test_production_edits_one_player_only(self, records):
        record = next(r for r in records if r.subkind is not None
                      and r.subkind.value == "decision")
        edited = sweeps.set_production(record, record.actor_player, ["1", "1", "R"], 2)
        actor = next(p for p in edited.state.players if p.id == record.actor_player)
        assert actor.untapped_production == {"R": 1, "C": 1}
        assert actor.floating_mana == {}
        assert _differences(record, edited) == {"players"}
        for before, after in zip(record.state.players, edited.state.players):
            if before.id != record.actor_player:
                assert before == after

    def test_board_size_sets_the_opposing_creature_count(self, records):
        record = next(r for r in records if sweeps.opposing_creatures(r))
        for count in (0, 1, 5):
            edited = sweeps.set_opposing_creatures(record, count)
            assert len(sweeps.opposing_creatures(edited)) == count
            ids = [e.id for e in edited.state.entities]
            assert len(ids) == len(set(ids))
            assert _differences(record, edited) <= {"entities"}

    def test_damage_edits_only_the_amount(self):
        text = "NumDmg$ 3 | SP$ DealDamage | ValidTgts$ Creature [SEG] SV1: DB$ Draw"
        assert sweeps.set_damage(text, 7) == text.replace("NumDmg$ 3", "NumDmg$ 7")

    def test_cost_pips_read_generic_and_coloured_mana(self):
        assert sweeps.cost_pips("{2}{G}") == ["1", "1", "G"]

    def test_sweep_records_are_chosen_from_real_records(self, records, sidecars):
        chosen = sweeps.select_sweep_records(records, sidecars)
        assert set(chosen) == {"toughness", "affordability", "board_size", "damage"}
        assert chosen["affordability"]
        ids = {r.record_id for r in records}
        for rows in chosen.values():
            assert all(row["record_id"] in ids for row in rows)


# ── the probe set's digest ──────────────────────────────────────────────


class TestDigest:
    def test_it_is_stable_across_key_order(self):
        a = {"items": [1, 2], "games": {"x": ["g1"]}, "corpus_digest": "c"}
        b = {"corpus_digest": "c", "games": {"x": ["g1"]}, "items": [1, 2]}
        assert run.canonical_digest(a) == run.canonical_digest(b)

    def test_it_excludes_itself(self):
        payload = {"items": [1]}
        payload["digest"] = run.canonical_digest(payload)
        assert run.canonical_digest(payload) == payload["digest"]

    def test_it_changes_with_a_label(self):
        assert run.canonical_digest({"items": [1]}) != run.canonical_digest({"items": [2]})

    def test_a_probing_run_refuses_a_set_frozen_for_another_corpus(self, tmp_path):
        payload = {"corpus_digest": "aaa", "items": []}
        payload["digest"] = run.canonical_digest(payload)
        path = tmp_path / "set.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(SystemExit, match="frozen against corpus digest"):
            run.load_probe_set(path, "bbb")
        assert run.load_probe_set(path, "aaa")["digest"] == payload["digest"]

    def test_a_probing_run_refuses_without_a_set(self, tmp_path):
        with pytest.raises(SystemExit, match="--freeze-probe-set"):
            run.load_probe_set(tmp_path / "missing.json", "aaa")


# ── compare ─────────────────────────────────────────────────────────────


def _card(name: str, one: float, width: float, digest: str = "d") -> dict:
    return {
        "checkpoint": f"models/effects/runs/{name}/latest.pt",
        "probe_set_digest": digest,
        "families": {"magnitudes": {"targets": {"dies": {
            "scope": "entity", "kind": "binary",
            "strata": {"card-disjoint": {"n": 100, "rungs": {
                "0": {"linear": 0.6, "mlp": 0.6},
                "1": {"linear": one, "mlp": one},
                "1w": {"linear": width, "mlp": width},
                "3": {"model": 0.9},
            }}},
        }}}},
    }


class TestCompare:
    def test_arms_rank_on_rung_one_over_the_width_control(self):
        ranked = compare.ranking([_card("narrow", 0.70, 0.65), _card("wide", 0.80, 0.62)])
        assert [Path(name).parent.name for name, _ in ranked["mlp"]] == ["wide", "narrow"]
        assert ranked["linear"][0][1] == pytest.approx(0.18)

    def test_mismatched_digests_warn(self):
        assert compare.digest_warning([_card("a", 0.7, 0.6, "x"), _card("b", 0.7, 0.6, "y")])
        assert compare.digest_warning([_card("a", 0.7, 0.6), _card("b", 0.7, 0.6)]) is None

    def test_the_table_sets_checkpoints_side_by_side(self, tmp_path):
        paths = []
        for name in ("a", "b"):
            path = tmp_path / f"{name}.json"
            path.write_text(json.dumps(_card(name, 0.7, 0.6)), encoding="utf-8")
            paths.append(path)
        out = tmp_path / "compare.md"
        assert compare.main([str(p) for p in paths] + ["--output", str(out)]) == 0
        text = out.read_text(encoding="utf-8")
        assert "## magnitudes" in text and "a linear" in text and "b mlp" in text
        assert "Ranking on rung 1" in text


# ── one forward pass through a real checkpoint ──────────────────────────

_GEN1 = _ROOT / "models" / "effects" / "runs" / "2026-09-17-full-textless-corpus" / "latest.pt"


@pytest.mark.skipif(not _GEN1.exists(), reason="the gen-1 checkpoint is not on this machine")
class TestExtraction:
    @pytest.fixture(scope="class")
    def probe_model(self, records):
        checkpoint = common.load_checkpoint(_GEN1)
        if not Path(checkpoint.provenance.vocab_path).exists():
            pytest.skip("the gen-1 vocabulary is not on this machine")
        return common.load_probe_model(_GEN1, [
            _FIXTURES / "gen1-sidecars" / "cardsfolder",
            _FIXTURES / "gen1-sidecars" / "tokenscripts",
        ], records, checkpoint=checkpoint)

    def test_every_rung_has_one_width_per_target(self, probe_model, records, sidecars):
        acting = run._acting_text_function(sidecars, "script")
        join = labels.join_triggers(records)
        items: dict[str, list] = {}
        for record in records:
            for item in labels.record_items(record, sidecars, join, acting):
                items.setdefault(record.record_id, []).append(item)
        chosen = [r for r in records if r.record_id in items]
        act_hidden: dict = {}
        extracted = ladder.extract_features(probe_model, chosen, items, batch_size=8,
                                            act_hidden=act_hidden)
        assert extracted and act_hidden
        e_dim = probe_model.e_dim
        for target in {i.target for i in extracted}:
            rows = [i for i in extracted if i.target == target]
            for rung in ladder.PROBED_RUNGS:
                assert len({r.features[rung].shape for r in rows}) == 1
            zero = rows[0].features["0"].shape[0]
            assert rows[0].features["1"].shape[0] == zero + 3 * e_dim
            assert rows[0].features["1w"].shape == rows[0].features["1"].shape
            assert all(np.isfinite(r.rung3) for r in rows)

    def test_a_toughness_sweep_reads_one_prediction_per_value(self, probe_model, records,
                                                             sidecars):
        chosen = sweeps.select_sweep_records(records, sidecars)
        if not chosen["toughness"]:
            pytest.skip("no damage record in the fixture")
        row = chosen["toughness"][0]
        record = next(r for r in records if r.record_id == row["record_id"])
        curve = sweeps.run_sweep("toughness", probe_model, record, row)
        assert [p["value"] for p in curve] == list(sweeps.TOUGHNESS_VALUES)
        assert all(0.0 <= p["prediction"] <= 1.0 for p in curve)

    def test_the_per_layer_view_reads_every_trunk_layer(self, probe_model, records,
                                                        sidecars):
        acting = run._acting_text_function(sidecars, "script")
        join = labels.join_triggers(records)
        record = next(r for r in records
                      if labels.record_items(r, sidecars, join, acting))
        items = {record.record_id: labels.record_items(record, sidecars, join, acting)}
        extracted = ladder.extract_features(probe_model, [record], items, per_layer=True)
        depth = len(probe_model.model.trunk.layers)
        assert all(len(item.layers) == depth for item in extracted)
        assert extracted[0].layers[0].shape == extracted[0].features["2"].shape

    def test_method_c_trains_a_shallow_trunk_beside_the_full_model(self, probe_model,
                                                                  records, tmp_path):
        import shutil
        from types import SimpleNamespace

        (tmp_path / "training").mkdir()
        shutil.copy(_FIXTURES / "gen1-records.jsonl.gz",
                    tmp_path / "training" / "shard-00001.jsonl.gz")
        args = SimpleNamespace(method_c_layers=0, method_c_steps=2)
        result = run.method_c(args, probe_model, tmp_path, records[:16])
        assert result["steps"] == 2 and result["layers"] == 0
        assert set(result["shallow"]) == set(result["full"])
        assert all(np.isfinite(v) for v in result["shallow"].values())

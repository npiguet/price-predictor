"""The control score (FR-068, FR-068c, FR-069; story 8 acceptance 1-3).

The scale arithmetic is checked on numbers chosen by hand; the scorer itself on
the gen-3 fixture records, a selection set frozen over them by hand and a tiny
real model, where the scores are a random model's but every component, the
stratum score, the mix and what the scorer read can be checked.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from effects.application.control_score import (
    CellRead,
    ControlScore,
    ControlScoreError,
    ControlScorer,
    StratumScore,
    TextOverride,
    _OutputReading,
    component_score,
    decoder_of,
    explained_deviance,
    mixed,
    probe_item,
    rule_item,
    selection_records,
)
from effects.application.record_forward import RecordForward
from effects.domain.effect_head_layout import GATE_TERM, ZONE_OUTCOMES, ZoneOutcome
from effects.domain.selection_set import (
    COMPONENTS,
    DEVIANCE,
    GRIDS,
    PROBES,
    RULE_TESTS,
    FrozenGrid,
    GridCell,
    ProbeRecord,
    SampleRecord,
    SelectionSet,
    TaggedRecord,
    item_key,
)
from tests.unit.effects.builders import fixture_run

STRATA = ("card-disjoint", "game-disjoint")


# ── the scale (acceptance 1) ────────────────────────────────────────────


class TestTheScale:
    def test_an_auc_probe_is_rescaled_and_an_amount_read_as_its_r_squared(self):
        labels = [1, 1, 0, 0]
        assert probe_item("binary", labels, [0.9, 0.2, 0.4, 0.1]) == pytest.approx(0.5)
        y = [1.0, 2.0, 3.0, 4.0]
        assert probe_item("amount", y, y) == pytest.approx(1.0)
        assert probe_item("amount", y, [2.5] * 4) == pytest.approx(0.0)

    def test_a_rule_averages_its_outputs_rescaled_aucs(self):
        perfect = ([0.9, 0.8], [0.1, 0.2])
        backwards = ([0.1], [0.9])
        assert rule_item([perfect]) == pytest.approx(1.0)
        assert rule_item([perfect, backwards]) == pytest.approx(0.0)

    def test_items_are_clipped_below_at_minus_one_and_averaged(self):
        assert component_score({"a": -3.0, "b": 0.5}, stratum="s", component=GRIDS) == (
            pytest.approx(-0.25))

    def test_the_stratum_score_is_the_mean_of_its_four_components(self):
        score = StratumScore(dict(zip(COMPONENTS, (0.2, 0.4, 0.6, 0.8))), {})
        assert score.score == pytest.approx(0.5)

    def test_the_control_score_mixes_the_strata_by_the_stratum_weight(self):
        card = StratumScore(dict.fromkeys(COMPONENTS, 0.8), {})
        game = StratumScore(dict.fromkeys(COMPONENTS, 0.2), {})
        control = ControlScore({"card-disjoint": card, "game-disjoint": game}, 0.25)
        assert control.control == pytest.approx(0.25 * 0.8 + 0.75 * 0.2)
        assert mixed(1.0, 0.0, 0.5) == 0.5


# ── what is left out (acceptance 2, 3) ──────────────────────────────────


class TestLeftOut:
    def test_a_field_at_or_below_the_deviance_floor_is_left_out(self):
        parts = {"a": 0.02, "b": 0.03, "c": 0.5}
        floor = {"a": 0.04, "b": 0.05, "c": 1.0}
        got = explained_deviance(parts, floor, kept=("a", "b", "c", "d"), deviance_floor=0.04)
        assert got == pytest.approx({"b": 0.4, "c": 0.5})

    def test_a_field_the_freeze_did_not_keep_is_left_out(self):
        got = explained_deviance({"a": 0.5}, {"a": 1.0}, kept=(), deviance_floor=0.0)
        assert got == {}

    def test_a_rule_with_test_records_of_one_polarity_only_is_left_out(self):
        all_fires = rule_item([([0.9, 0.8, 0.7], [])])
        assert math.isnan(all_fires)
        scored = component_score({"all-fires": all_fires, "two-sided": 0.5},
                                 stratum="s", component=RULE_TESTS)
        assert scored == pytest.approx(0.5)

    def test_a_component_with_nothing_to_score_names_stratum_half_and_component(self):
        with pytest.raises(ControlScoreError) as raised:
            component_score({"only": math.nan}, stratum="game-disjoint", component=PROBES)
        assert (raised.value.stratum, raised.value.half, raised.value.component) == (
            "game-disjoint", "selection", PROBES)


# ── a rule test's readout (E6) ──────────────────────────────────────────


class TestRuleReadout:
    def test_the_category_the_fires_records_show_more_is_read(self):
        reading = _OutputReading(decoder_of("zone_outcome"))
        died = [1.0 if c == ZoneOutcome.DIED else 0.0 for c in ZONE_OUTCOMES]
        stayed = [0.0] * len(ZONE_OUTCOMES)
        for _ in range(3):
            reading.observe("fires", died)
            reading.observe("does_not_fire", stayed)
        assert reading.settle()
        assert ZONE_OUTCOMES[reading.candidate] == ZoneOutcome.DIED
        assert reading.sign == 1.0

    def test_an_absent_outcome_reads_the_complement(self):
        """Indestructible: the creature that fires the rule does not die."""
        reading = _OutputReading(decoder_of("zone_outcome"))
        died = [1.0 if c == ZoneOutcome.DIED else 0.0 for c in ZONE_OUTCOMES]
        reading.observe("fires", [0.0] * len(ZONE_OUTCOMES))
        reading.observe("does_not_fire", died)
        assert reading.settle()
        assert ZONE_OUTCOMES[reading.candidate] == ZoneOutcome.DIED
        assert reading.sign == -1.0

    def test_an_output_no_fires_record_observes_is_not_read(self):
        reading = _OutputReading(decoder_of("can_play"))
        reading.observe("does_not_fire", [1.0])
        assert not reading.settle()

    def test_only_the_shipped_heads_outputs_have_a_readout(self):
        for name in ("gate", "created_objects", "zone_outcome", "life_delta", "can_play",
                     "trigger_fired", "mana_paid", "day_night_change"):
            assert decoder_of(name) is not None, name
        for name in ("new:face_change", "planned:timing", "record:event:counter_change"):
            assert decoder_of(name) is None


# ── the scorer on the fixture ───────────────────────────────────────────


def _selection(records) -> SelectionSet:
    """Both strata hold every fixture game, split in two by game; each half
    holds items over its own games' records."""
    games = sorted({r.game_id for r in records})
    halves = {"selection": games[::2], "reporting": games[1::2]}
    by_half = {half: [r for r in records if r.game_id in set(held)]
               for half, held in halves.items()}
    probes, grids, rules, samples, errors = {}, {}, {}, {}, {}
    for stratum in STRATA:
        probes[stratum], grids[stratum], rules[stratum], samples[stratum] = {}, {}, {}, {}
        for half, held in by_half.items():
            with_entities = [r for r in held if r.state.entities]
            probes[stratum][half] = {
                "dies": tuple(ProbeRecord(r.record_id, r.game_id, r.state.entities[0].id,
                                          float(i % 2)) for i, r in enumerate(with_entities[:12])),
                "affordable": tuple(ProbeRecord(r.record_id, r.game_id, None, float(i % 2))
                                    for i, r in enumerate(held[:10])),
            }
            grids[stratum][half] = {"grid": FrozenGrid(
                (GridCell("low", 0.0), GridCell("high", 1.0)),
                tuple(TaggedRecord(r.record_id, r.game_id, (r.state.entities[0].id,))
                      for r in with_entities[:4]))}
            tagged = [TaggedRecord(r.record_id, r.game_id, (r.state.entities[0].id,))
                      for r in with_entities[:16]]
            rules[stratum][half] = {
                "r.board": {"fires": tuple(tagged[:8]), "does_not_fire": tuple(tagged[8:])},
                "r.unread": {"fires": tuple(tagged[:2]), "does_not_fire": tuple(tagged[2:4])},
            }
            samples[stratum][half] = tuple(SampleRecord(r.record_id, r.game_id, 0)
                                           for r in held[:20])
            for name in (GATE_TERM, "zone_outcome", "damage_taken", "pt_change"):
                errors[item_key(stratum, half, DEVIANCE, name)] = {"hard": 0.05, "soft": 0.05}
            errors[item_key(stratum, half, DEVIANCE, "affected_unkept")] = {
                "hard": 0.5, "soft": 0.5}
    return SelectionSet(
        corpus_digest="fixture", r=4,
        halves={s: {h: tuple(g) for h, g in halves.items()} for s in STRATA},
        probes=probes, probe_kinds={"dies": "binary", "affordable": "binary"},
        grids=grids, rule_tests=rules, deviance_samples=samples,
        standard_errors=errors, left_out={}, settings={"max_item_error": 0.1},
    )


RULE_OUTPUTS = {
    "r.board": ("gate", "zone_outcome", "damage_taken", "can_play", "trigger_fired",
                "created_objects", "new:not_a_head"),
    "r.unread": ("new:not_a_head",),
}


class _Editor:
    """Each grid record edited into each cell unchanged, read for its death."""

    def __init__(self) -> None:
        self.asked: list[str] = []

    def cells(self, grid, record, entities):
        self.asked.append(record.record_id)
        return [CellRead(record, entities[0], "dies"), CellRead(record, entities[0], "dies")]


class _Spy:
    """The run's preparer, recording every record it prepares."""

    def __init__(self, preparer) -> None:
        self.preparer = preparer
        self.config = preparer.config
        self.seen: list[str] = []
        self.batchers: list = []

    def scoring_batcher(self):
        batcher = self.preparer.scoring_batcher()
        self.batchers.append(batcher)
        prepare = batcher.prepare

        def recorded(records, strips=None):
            self.seen += [r.record_id for r in records]
            return prepare(records, strips)

        batcher.prepare = recorded
        return batcher

    def prepare_scoring_step(self, records, fields):
        self.seen += [r.record_id for r in records]
        return self.preparer.prepare_scoring_step(records, fields)


@pytest.fixture(scope="module")
def run(corpus, records, tmp_path_factory):
    return fixture_run(corpus, records, tmp_path_factory.mktemp("control"))


def _prepare(run, records, selection, **overrides):
    spy = _Spy(run.preparer)
    editor = _Editor()
    settings = dict(
        preparer=spy, fields=list(run.fields.values()), rule_outputs=RULE_OUTPUTS,
        mask="hard", stratum_weight=0.3, deviance_floor=0.0, editor=editor,
        chunk_records=8, batch_size=8,
    )
    settings.update(overrides)
    scorer = ControlScorer.prepare(
        selection, dict.fromkeys(STRATA, lambda: [records]), **settings)
    return scorer, spy, editor


def _epoch(run, scorer):
    return scorer.epoch(RecordForward(run.encoder, run.model, run.scoring),
                        _losses(run), run.fields[max(run.fields)])


def _losses(run):
    from effects.application.step_loss import StepLossComputer

    return StepLossComputer(run.spec.config, run.device)


@pytest.fixture(scope="module")
def scored(run, records):
    selection = _selection(records)
    scorer, spy, editor = _prepare(run, records, selection)
    return selection, scorer, spy, editor, _epoch(run, scorer)


class TestTheScorer:
    def test_every_stratum_scores_four_components_and_mixes_them(self, scored):
        _, _, _, _, score = scored
        for stratum in STRATA:
            components = score.strata[stratum].components
            assert set(components) == set(COMPONENTS)
            assert all(-1.0 <= value <= 1.0 for value in components.values())
            assert score.strata[stratum].score == pytest.approx(np.mean(list(components.values())))
        assert score.control == pytest.approx(
            0.3 * score.strata["card-disjoint"].score + 0.7 * score.strata["game-disjoint"].score)

    def test_its_items_are_the_selection_sets_and_the_kept_fields(self, scored):
        _, _, _, _, score = scored
        items = score.strata["card-disjoint"].items
        assert set(items[PROBES]) <= {"dies", "affordable"}
        assert set(items[GRIDS]) == {"grid"}
        assert set(items[RULE_TESTS]) == {"r.board"}
        assert set(items[DEVIANCE]) <= {GATE_TERM, "zone_outcome", "damage_taken", "pt_change"}
        assert GATE_TERM in items[DEVIANCE]

    def test_the_reporting_half_is_never_read(self, scored, records):
        selection, _, spy, editor, _ = scored
        reporting = {r.record_id for r in records
                     if r.game_id in set(selection.halves["card-disjoint"]["reporting"])}
        assert spy.seen and not reporting & set(spy.seen)
        assert editor.asked and not reporting & set(editor.asked)

    def test_the_reporting_halfs_items_move_no_score(self, run, records, scored):
        selection, _, _, _, score = scored
        emptied = replace(selection, probes={
            stratum: {**by_half, "reporting": {}} for stratum, by_half in selection.probes.items()})
        scorer, _, _ = _prepare(run, records, emptied)
        again = _epoch(run, scorer)
        assert again.control == pytest.approx(score.control)

    def test_an_epoch_repeats_on_the_same_weights(self, run, scored):
        _, scorer, _, _, score = scored
        assert _epoch(run, scorer).control == pytest.approx(score.control)

    def test_the_cache_stays_on_the_host_and_packed(self, scored):
        _, scorer, _, _, _ = scored
        assert scorer.host_bytes() > 0
        chunk = scorer.strata["card-disjoint"].chunks[0]
        features = chunk.batch.inputs.slot_features
        assert all(not hasattr(value, "device") for value in features.values())


class TestAnEmptyComponentExitsOne:
    def test_a_selection_half_without_probes_refuses_at_startup(self, run, records):
        selection = _selection(records)
        selection = replace(selection, probes={
            stratum: {**by_half, "selection": {}} for stratum, by_half in selection.probes.items()})
        with pytest.raises(ControlScoreError) as raised:
            _prepare(run, records, selection)
        assert (raised.value.half, raised.value.component) == ("selection", PROBES)

    def test_grids_without_an_editor_refuse_at_startup(self, run, records):
        with pytest.raises(ControlScoreError) as raised:
            _prepare(run, records, _selection(records), editor=None)
        assert raised.value.component == GRIDS

    def test_every_field_below_the_deviance_floor_refuses_the_epoch(self, run, records):
        scorer, _, _ = _prepare(run, records, _selection(records), deviance_floor=1e9)
        with pytest.raises(ControlScoreError) as raised:
            _epoch(run, scorer)
        assert (raised.value.stratum, raised.value.half, raised.value.component) == (
            "card-disjoint", "selection", DEVIANCE)


def test_copies_are_counted_as_the_freeze_counts_them(records):
    first, second = records[0], records[1]
    shards = [[first, second], [first, first]]
    got = list(selection_records(
        lambda: shards, games=frozenset({first.game_id, second.game_id}),
        records=frozenset({first.record_id}), copies=frozenset({(first.record_id, 2)})))
    assert [(r.record_id, distinct, copy) for r, distinct, copy in got] == [
        (first.record_id, True, False), (first.record_id, False, True)]


class _TextEditor(_Editor):
    """The second cell reads the record's acting line under an edited text, a
    key of its own, as the damage grid edits a stated amount."""

    def cells(self, grid, record, entities):
        low, high = super().cells(grid, record, entities)
        if not record.ability:
            return [low, high]
        original = record.ability[0]
        key = replace(original, script_file=f"t158-grid/{record.record_id}")
        edited = replace(record, ability=(key, *record.ability[1:]))
        return [low, CellRead(edited, high.entity, high.readout,
                              (TextOverride(key, "edited text", original),))]


def test_a_grid_cell_can_read_an_edited_ability_text(run, records):
    selection = _selection(records)
    with_ability = [r for r in records if r.ability and r.state.entities
                    and r.game_id in set(selection.halves["card-disjoint"]["selection"])][:3]
    grids = {stratum: {**by_half, "selection": {"grid": FrozenGrid(
        (GridCell("low", 0.0), GridCell("high", 1.0)),
        tuple(TaggedRecord(r.record_id, r.game_id, (r.state.entities[0].id,))
              for r in with_ability))}} for stratum, by_half in selection.grids.items()}
    selection = replace(selection, grids=grids)
    spy = _Spy(run.preparer)
    scorer = ControlScorer.prepare(
        selection, dict.fromkeys(STRATA, lambda: [records]), preparer=spy,
        fields=list(run.fields.values()), rule_outputs=RULE_OUTPUTS, mask="hard",
        stratum_weight=0.5, deviance_floor=0.0, editor=_TextEditor(), chunk_records=8)
    overrides = {key: text for batcher in spy.batchers
                 for key, (text, _line) in batcher._overrides.items()}
    assert {key.script_file for key in overrides} == {
        f"t158-grid/{r.record_id}" for r in with_ability}
    assert set(overrides.values()) == {"edited text"}
    assert set(_epoch(run, scorer).strata["card-disjoint"].items[GRIDS]) == {"grid"}


def test_the_packed_cache_unpacks_bit_for_bit(run, records):
    """Every tensor of a scoring step comes back from its host form exactly."""
    import torch

    from effects.application.host_cache import host_bytes, map_tensors, pack, unpack

    step = run.preparer.prepare_scoring_step(records[:16], tuple(run.fields[max(run.fields)]))
    packed = pack(step)
    assert host_bytes(packed) < host_bytes(step)
    before, after = [], []
    map_tensors(step, lambda t: before.append(t) or t)
    map_tensors(unpack(packed), lambda t: after.append(t) or t)
    assert len(before) == len(after)
    for want, got in zip(before, after):
        assert got.dtype == want.dtype and torch.equal(got, want)


# ── either half, and a saved checkpoint (T169) ──────────────────────────


class TestTheReportingHalf:
    @pytest.fixture(scope="class")
    def reported(self, run, records):
        selection = _selection(records)
        scorer, spy, editor = _prepare(run, records, selection, half="reporting", strict=False)
        return selection, spy, editor, _epoch(run, scorer)

    def test_it_reads_the_reporting_half_alone(self, reported, records):
        selection, spy, editor, _ = reported
        halves = selection.halves["card-disjoint"]
        reporting = {r.record_id for r in records if r.game_id in set(halves["reporting"])}
        selected = {r.record_id for r in records if r.game_id in set(halves["selection"])}
        assert spy.seen and set(spy.seen) <= reporting and not selected & set(spy.seen)
        assert editor.asked and set(editor.asked) <= reporting

    def test_it_scores_every_component_of_both_strata(self, reported):
        _, _, _, score = reported
        for stratum in STRATA:
            assert set(score.strata[stratum].components) == set(COMPONENTS)
            assert set(score.strata[stratum].items[RULE_TESTS]) == {"r.board"}
        assert score.strata["card-disjoint"].left_out[RULE_TESTS] == ("r.unread",)

    def test_a_training_run_still_reads_the_selection_half(self, run, records):
        """The validator names no half: the default is the selection half."""
        import inspect

        from effects.application.validation_pass import Validator

        assert inspect.signature(ControlScorer.prepare).parameters["half"].default == "selection"
        assert "half=" not in inspect.getsource(Validator.prepare)


class TestALenientScore:
    def test_an_empty_component_is_nan_and_named_not_an_error(self, run, records):
        selection = _selection(records)
        selection = replace(selection, probes={
            stratum: {**by_half, "reporting": {}} for stratum, by_half in selection.probes.items()})
        scorer, _, _ = _prepare(run, records, selection, half="reporting", strict=False)
        score = _epoch(run, scorer)
        stratum = score.strata["card-disjoint"]
        assert math.isnan(stratum.components[PROBES]) and math.isnan(stratum.score)
        assert not math.isnan(stratum.components[RULE_TESTS])

    def test_the_deviance_floor_empties_the_component_without_an_error(self, run, records):
        scorer, _, _ = _prepare(run, records, _selection(records), half="reporting",
                                strict=False, deviance_floor=1e9)
        assert math.isnan(_epoch(run, scorer).strata["game-disjoint"].components[DEVIANCE])

    def test_chosen_components_alone_are_prepared_and_scored(self, run, records):
        scorer, spy, editor = _prepare(run, records, _selection(records), half="reporting",
                                       strict=False, components=(RULE_TESTS,))
        stratum = _epoch(run, scorer).strata["card-disjoint"]
        assert set(stratum.components) == {RULE_TESTS} and math.isnan(stratum.score)
        assert not editor.asked
        assert not scorer.strata["card-disjoint"].steps


def _checkpoint_batcher(run):
    """A saved checkpoint's batcher, as ``LoadedModel.batcher`` gives one: the
    run's layout and join with its device half."""
    from effects.application.surface_batching import SurfaceBatcher

    template = run.preparer.scoring_batcher()
    return SurfaceBatcher(stager=template.stager, resolver=template.resolver,
                          spec=template.spec, materializer=run.scoring)


class TestASavedCheckpoint:
    def test_its_scoring_step_is_validations(self, run, records):
        import torch

        from effects.application.control_score import CheckpointPreparer
        from effects.application.host_cache import map_tensors

        fields = tuple(run.fields[max(run.fields)])
        preparer = CheckpointPreparer(_checkpoint_batcher(run), mask="hard",
                                      rule_outputs=run.preparer.rule_outputs)
        ours = preparer.prepare_scoring_step(records[:16], fields)
        theirs = run.preparer.prepare_scoring_step(records[:16], fields)
        got, want = [], []
        map_tensors(ours.targets, lambda t: got.append(t) or t)
        map_tensors(theirs.targets, lambda t: want.append(t) or t)
        assert len(got) == len(want) and all(torch.equal(a, b) for a, b in zip(got, want))

    def test_score_half_scores_the_half_it_names(self, run, records):
        from effects.application.control_score import score_half
        from tests.unit.effects.builders import FIXTURE_RULE_OUTPUTS, fixture_selection_set

        selection = fixture_selection_set(records)
        score = score_half(
            selection, dict.fromkeys(STRATA, lambda: [records]), half="reporting",
            encoder=run.encoder, model=run.model, batcher=_checkpoint_batcher(run),
            rule_outputs=FIXTURE_RULE_OUTPUTS, mask="hard", stratum_weight=0.3,
            deviance_floor=0.0, config=run.spec.config)
        assert set(score.strata) == set(STRATA)
        for stratum in STRATA:
            assert set(score.strata[stratum].items[RULE_TESTS]) == {"r.board"}
            assert set(score.strata[stratum].items[GRIDS]) == {"damage"}
            assert not math.isnan(score.strata[stratum].components[DEVIANCE])
        assert score.stratum_weight == 0.3

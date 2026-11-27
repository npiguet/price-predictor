"""The evaluator's report of the checks it never built (FR-116)."""

from __future__ import annotations

from effects.application.evaluate_effect_model import not_implemented
from effects.domain.evaluation_report import CheckName, CheckStatus, EvaluationReport


class TestUnimplementedChecks:
    """FR-116 names three checks that were never built.

    The corpus holds forks and mana records, so a line blaming missing records
    would send an operator looking for data that is already there.
    """

    def test_each_reports_not_implemented_and_blocks_nothing(self):
        for name in (
            CheckName.MATCHED_REAL_VS_FORK, CheckName.PROBE_DIFF,
            CheckName.ROLE_POLARITY,
        ):
            result = not_implemented(name)
            assert result.status is CheckStatus.NOT_IMPLEMENTED
            assert not result.blocks
            assert result.detail.startswith("not implemented")

    def test_the_report_marks_them_apart_from_a_skip(self):
        report = EvaluationReport()
        report.add(not_implemented(CheckName.PROBE_DIFF))
        assert report.render().startswith("[todo] probe-diff: not implemented")


# ── the reporting half ───────────────────────────────────────────


def _context(tmp_path, selection_digest=None):
    """What ``run_reporting_half`` reads of an evaluation: the manifest's
    digest and map, the checkpoint's recorded flags, the loaded model."""
    from types import SimpleNamespace

    from effects.infrastructure.effect_model_store import ScoringFlags
    from tests.unit.effects.configs import evaluation_config

    loaded = SimpleNamespace(encoder="encoder", model="model", batcher=lambda: "batcher")
    return SimpleNamespace(
        config=evaluation_config(reports_dir=tmp_path),
        manifest=SimpleNamespace(
            digest=lambda: "corpus-digest",
            rule_map={"r.board": SimpleNamespace(outputs=["gate", "zone_outcome"])},
            effective_rule_outputs={"r.board": ("zone_outcome",)}),
        main=SimpleNamespace(scored_under=lambda: ScoringFlags("soft", 0.25, 0.1),
                             control_score=0.4,
                             best_control_score=0.99,
                             selection_set_digest=selection_digest),
        corpus_path=tmp_path / "corpus", shards={"card-disjoint": (), "game-disjoint": ()},
        main_model=lambda: loaded,
    )


def _freeze(tmp_path):
    from effects.domain.selection_set import selection_set_path, write_selection_set
    from tests.unit.effects.builders import curated_fixture_records
    from tests.unit.effects.selection_builders import fixture_selection_set

    selection = fixture_selection_set(curated_fixture_records()[:120],
                                      corpus_digest="corpus-digest").sealed()
    write_selection_set(selection_set_path(tmp_path, "corpus-digest"), selection)
    return selection


class TestTheReportingHalf:
    def test_it_scores_the_reporting_half_under_the_checkpoints_own_flags(
        self, tmp_path, monkeypatch,
    ):
        from effects.application import control_score, evaluate_effect_model
        from tests.unit.effects.selection_builders import control_score_of

        selection = _freeze(tmp_path)
        asked = {}

        def score_half(frozen, checkpoint, flags, scope):
            asked.update(frozen=frozen, checkpoint=checkpoint, flags=flags, scope=scope)
            return control_score_of(0.5, flags.stratum_weight)

        monkeypatch.setattr(control_score, "score_half", score_half)
        checks = evaluate_effect_model.run_reporting_half(
            _context(tmp_path, selection_digest=selection.digest))

        assert asked["scope"].half == "reporting"
        flags = asked["flags"]
        assert (flags.mask, flags.stratum_weight, flags.deviance_floor) == ("soft", 0.25, 0.1)
        checkpoint = asked["checkpoint"]
        assert (checkpoint.encoder, checkpoint.model, checkpoint.batcher) == (
            "encoder", "model", "batcher")
        # The effective outputs, never the map's own list (FR-060a).
        assert checkpoint.rule_outputs == {"r.board": ("zone_outcome",)}
        assert set(asked["frozen"].strata) == set(selection.halves)
        names = [check.name for check in checks]
        assert "reporting-half card-disjoint" in names
        assert "rule-tests game-disjoint reporting" in names
        control = next(check for check in checks if check.name == "reporting-half control")
        # The checkpoint's own epoch's score, not the run's best (13.7).
        assert control.values == {"control": 0.5, "selection_control": 0.4}
        assert all(check.status is CheckStatus.REPORTED for check in checks)

    def test_without_a_selection_set_it_is_skipped_naming_why(self, tmp_path):
        from effects.application import evaluate_effect_model

        checks = evaluate_effect_model.run_reporting_half(_context(tmp_path))
        assert [(c.name, c.status) for c in checks] == [
            ("reporting-half", CheckStatus.SKIPPED)]
        assert "no selection set" in checks[0].detail

    def test_a_set_frozen_again_since_training_is_named(self, tmp_path, monkeypatch):
        from effects.application import control_score, evaluate_effect_model
        from tests.unit.effects.selection_builders import control_score_of

        _freeze(tmp_path)
        monkeypatch.setattr(control_score, "score_half",
                            lambda frozen, checkpoint, flags, scope: control_score_of(0.5))
        checks = evaluate_effect_model.run_reporting_half(
            _context(tmp_path, selection_digest="0" * 64))
        control = next(check for check in checks if check.name == "reporting-half control")
        assert "frozen again since training" in control.detail


def test_the_breakdowns_read_each_distinct_record_once(tmp_path):
    from types import SimpleNamespace

    from effects.application.evaluate_effect_model import breakdown_records
    from effects.domain.strata import Stratum
    from effects.infrastructure.corpus_store import CorpusStore
    from effects.infrastructure.shard_io import write_shard
    from tests.unit.effects.builders import curated_fixture_records

    records = curated_fixture_records()[:6]
    path = CorpusStore(tmp_path).sample_path(Stratum.CARD_DISJOINT)
    path.parent.mkdir(parents=True)
    write_shard(path, [*records, *records[:3]])
    read = breakdown_records(SimpleNamespace(corpus_path=tmp_path))
    assert [r.record_id for r in read[Stratum.CARD_DISJOINT]] == [
        r.record_id for r in records]
    assert read[Stratum.GAME_DISJOINT] == []


# ── the reported checks the base spec lists and nothing builds ──


def test_every_unbuilt_reported_check_of_the_base_spec_is_a_todo_line():
    from types import SimpleNamespace

    from effects.application.evaluate_effect_model import _unmeasured_checks

    main = SimpleNamespace(provenance=SimpleNamespace(withheld_keyword="flying"))
    checks = _unmeasured_checks(main)

    assert {check.status for check in checks} == {CheckStatus.NOT_IMPLEMENTED}
    assert {check.name for check in checks} == {
        "matched-real-vs-fork", "probe-diff", "role-polarity",
        "scaling-calibration", "identity-game-disjoint", "state-only-floor",
        "taxonomy-comparison held-out-prediction",
        "taxonomy-comparison decodability",
        "no-state-comparison decodability", "no-state-comparison ward-canary",
        "no-state-comparison scaling-calibration",
        "shared-text-stratum", "novel-combination-stratum",
        "numeric-extrapolation-stratum",
    }


def test_a_check_the_code_builds_is_not_listed_as_unbuilt():
    from effects.application.unbuilt_checks import NOT_IMPLEMENTED_CHECKS

    for built in ("ward-canary", "nearest-neighbour", "umap", "decodability",
                  "zero-shot-keyword", "comparison"):
        assert built not in NOT_IMPLEMENTED_CHECKS


# ── a reported check that raises is a skip line ──


def test_a_reported_check_that_raises_becomes_a_skip_naming_it_and_the_exception(caplog):
    from effects.application.evaluate_effect_model import attempt

    def draw():
        raise ImportError("no matplotlib")

    checks = attempt(CheckName.UMAP, draw)

    assert [(c.name, c.status) for c in checks] == [("umap", CheckStatus.SKIPPED)]
    assert checks[0].detail == "ImportError: no matplotlib"
    assert "Traceback" in caplog.text


def test_a_reported_check_that_returns_is_passed_through_as_a_list():
    from effects.application.evaluate_effect_model import attempt

    one = not_implemented(CheckName.PROBE_DIFF)
    assert attempt("a", lambda: one) == [one]
    assert attempt("a", lambda: [one, one]) == [one, one]

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


# ── the reporting half (T169) ───────────────────────────────────────────


def _context(tmp_path, *, rule_mask="soft", stratum_weight=0.25, deviance_floor=0.1,
             best=0.4, selection_digest=None):
    """What ``run_reporting_half`` reads of an evaluation: the manifest's
    digest and map, the checkpoint's recorded flags, the loaded model."""
    from types import SimpleNamespace

    from effects.application.evaluation_context import EvaluateEffectModelConfig

    loaded = SimpleNamespace(encoder="encoder", model="model", batcher=lambda: "batcher")
    return SimpleNamespace(
        config=EvaluateEffectModelConfig(reports_dir=tmp_path),
        manifest=SimpleNamespace(
            digest=lambda: "corpus-digest",
            rule_map={"r.board": SimpleNamespace(outputs=["gate", "zone_outcome"])},
            effective_rule_outputs={"r.board": ("zone_outcome",)}),
        main=SimpleNamespace(rule_mask=rule_mask, stratum_weight=stratum_weight,
                             deviance_floor=deviance_floor, best_control_score=best,
                             selection_set_digest=selection_digest),
        corpus_path=tmp_path / "corpus",
        main_model=lambda: loaded,
    )


def _freeze(tmp_path):
    from effects.domain.selection_set import selection_set_path, write_selection_set
    from tests.unit.effects.builders import curated_fixture_records, fixture_selection_set

    selection = fixture_selection_set(curated_fixture_records()[:120],
                                      corpus_digest="corpus-digest").sealed()
    write_selection_set(selection_set_path(tmp_path, "corpus-digest"), selection)
    return selection


class TestTheReportingHalf:
    def test_it_scores_the_reporting_half_under_the_checkpoints_own_flags(
        self, tmp_path, monkeypatch,
    ):
        from effects.application import control_score, evaluate_effect_model
        from tests.unit.effects.builders import control_score_of

        selection = _freeze(tmp_path)
        asked = {}

        def score_half(selection, readers, **kwargs):
            asked.update(kwargs, readers=readers, selection=selection)
            return control_score_of(0.5, kwargs["stratum_weight"])

        monkeypatch.setattr(control_score, "score_half", score_half)
        checks = evaluate_effect_model.run_reporting_half(
            _context(tmp_path, selection_digest=selection.digest))

        assert asked["half"] == "reporting"
        assert (asked["mask"], asked["stratum_weight"], asked["deviance_floor"]) == (
            "soft", 0.25, 0.1)
        assert (asked["encoder"], asked["model"], asked["batcher"]) == (
            "encoder", "model", "batcher")
        # The effective outputs, never the map's own list (FR-060a).
        assert asked["rule_outputs"] == {"r.board": ("zone_outcome",)}
        assert set(asked["readers"]) == set(selection.halves)
        names = [check.name for check in checks]
        assert "reporting-half card-disjoint" in names
        assert "rule-tests game-disjoint reporting" in names
        control = next(check for check in checks if check.name == "reporting-half control")
        assert control.values == {"control": 0.5, "selection_control": 0.4}
        assert all(check.status is CheckStatus.REPORTED for check in checks)

    def test_unrecorded_flags_take_the_training_defaults(self, tmp_path, monkeypatch):
        from effects.application import control_score, evaluate_effect_model
        from tests.unit.effects.builders import control_score_of

        _freeze(tmp_path)
        asked = {}
        monkeypatch.setattr(control_score, "score_half", lambda selection, readers, **kw: (
            asked.update(kw) or control_score_of(0.5)))
        evaluate_effect_model.run_reporting_half(_context(
            tmp_path, rule_mask="", stratum_weight=None, deviance_floor=None, best=None))
        assert (asked["mask"], asked["stratum_weight"], asked["deviance_floor"]) == (
            None, 0.5, 0.04)

    def test_without_a_selection_set_it_is_skipped_naming_why(self, tmp_path):
        from effects.application import evaluate_effect_model

        checks = evaluate_effect_model.run_reporting_half(_context(tmp_path))
        assert [(c.name, c.status) for c in checks] == [
            ("reporting-half", CheckStatus.SKIPPED)]
        assert "no selection set" in checks[0].detail

    def test_a_set_frozen_again_since_training_is_named(self, tmp_path, monkeypatch):
        from effects.application import control_score, evaluate_effect_model
        from tests.unit.effects.builders import control_score_of

        _freeze(tmp_path)
        monkeypatch.setattr(control_score, "score_half",
                            lambda selection, readers, **kw: control_score_of(0.5))
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

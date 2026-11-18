"""The knowledge probes' selection set and scorecard comparison."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from effect_knowledge_probes import (
    common,
    compare,
    run,
)

# ── the selection set a probing run reads ───────────────────────────────


def _selection(corpus_digest: str):
    from effects.domain.selection_set import SelectionSet

    return SelectionSet(
        corpus_digest=corpus_digest, r=4, halves={}, probes={}, probe_kinds={}, grids={},
        rule_tests={}, deviance_samples={}, standard_errors={}, left_out={},
    ).sealed()


class TestTheSelectionSet:
    """A probing run reads the selection set frozen for its checkpoint's
    corpus, and refuses any other (FR-071)."""

    def _write(self, directory, selection, name_digest=None):
        from effects.domain.selection_set import selection_set_path, write_selection_set

        return write_selection_set(
            selection_set_path(directory, name_digest or selection.corpus_digest), selection)

    def test_the_set_frozen_for_the_corpus_is_read(self, tmp_path):
        from effect_knowledge_probes import halves

        frozen = _selection("aaa")
        self._write(tmp_path, frozen)
        found = halves.load_selection(SimpleNamespace(corpus_digest="aaa"), tmp_path)
        assert found.digest == frozen.digest

    def test_a_set_frozen_for_another_corpus_is_refused(self, tmp_path):
        from effect_knowledge_probes import halves

        self._write(tmp_path, _selection("bbb"), name_digest="aaa")
        with pytest.raises(SystemExit, match="frozen against corpus digest bbb"):
            halves.load_selection(SimpleNamespace(corpus_digest="aaa"), tmp_path)

    def test_an_edited_set_is_refused(self, tmp_path):
        from effect_knowledge_probes import halves

        path = self._write(tmp_path, _selection("aaa"))
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["r"] = 2
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(SystemExit, match="does not match its own digest"):
            halves.load_selection(SimpleNamespace(corpus_digest="aaa"), tmp_path)

    def test_a_probing_run_refuses_without_a_set(self, tmp_path, monkeypatch):
        context = SimpleNamespace(corpus_digest="aaa")
        monkeypatch.setattr(common, "probe_context", lambda *a, **k: context)
        with pytest.raises(SystemExit, match="--freeze-probe-set"):
            run.main(["--checkpoint", str(tmp_path / "latest.pt"),
                      "--reports-dir", str(tmp_path)])


# ── compare ─────────────────────────────────────────────────────────────


def _card(name: str, one: float, width: float, digest: str = "d"):
    return _scorecard({
        "checkpoint": f"models/effects/runs/{name}/latest.pt",
        "selection_set_digest": digest,
        "families": {"magnitudes": {"targets": {"dies": {
            "scope": "entity", "kind": "binary",
            "strata": {"card-disjoint": {"n": 100, "rungs": {
                "0": {"linear": 0.6, "mlp": 0.6},
                "1": {"linear": one, "mlp": one},
                "1w": {"linear": width, "mlp": width},
                "3": {"model": 0.9},
            }}},
        }}}},
    })


def _scorecard(data: dict):
    from effect_knowledge_probes.scorecard import Scorecard

    return Scorecard.from_json(data)


class TestCompare:
    def test_arms_rank_on_rung_one_over_the_width_control(self):
        ranked = compare.ranking([_card("narrow", 0.70, 0.65), _card("wide", 0.80, 0.62)])
        assert [Path(name).parent.name for name, _ in ranked["mlp AUC"]] == ["wide", "narrow"]
        assert ranked["linear AUC"][0][1] == pytest.approx(0.18)

    def test_auc_and_r2_gains_are_ranked_apart(self):
        """An AUC point and an R² point are not one unit; averaging them would
        let a large R² swing on one amount decide the ranking."""
        def card(name, auc_gain, r2_gain):
            body = _card(name, 0.6 + auc_gain, 0.6)
            body.families["magnitudes"]["targets"]["damage_taken"] = {
                "scope": "entity", "kind": "amount",
                "strata": {"card-disjoint": {"n": 100, "rungs": {
                    "1": {"linear": 0.2 + r2_gain, "mlp": 0.2 + r2_gain},
                    "1w": {"linear": 0.2, "mlp": 0.2},
                }}},
            }
            return body

        ranked = compare.ranking([card("auc", 0.10, -0.50), card("r2", 0.01, 0.30)])
        assert [Path(n).parent.name for n, _ in ranked["mlp AUC"]] == ["auc", "r2"]
        assert [Path(n).parent.name for n, _ in ranked["mlp R²"]] == ["r2", "auc"]
        assert ranked["linear AUC"][0][1] == pytest.approx(0.10)

    def test_mismatched_digests_warn(self):
        assert compare.digest_warning([_card("a", 0.7, 0.6, "x"), _card("b", 0.7, 0.6, "y")])
        assert compare.digest_warning([_card("a", 0.7, 0.6), _card("b", 0.7, 0.6)]) is None

    def test_the_table_sets_checkpoints_side_by_side(self, tmp_path):
        paths = []
        for name in ("a", "b"):
            path = tmp_path / f"{name}.json"
            path.write_text(json.dumps(_card(name, 0.7, 0.6).to_json()), encoding="utf-8")
            paths.append(path)
        out = tmp_path / "compare.md"
        assert compare.main([str(p) for p in paths] + ["--output", str(out)]) == 0
        text = out.read_text(encoding="utf-8")
        assert "## magnitudes" in text and "a linear" in text and "b mlp" in text
        assert "Ranking on rung 1" in text
        manifest = json.loads((tmp_path / "run.json").read_text(encoding="utf-8"))
        assert manifest["commands"] == [" ".join([
            "python", "scripts/effect_knowledge_probes/compare.py", *map(str, paths),
            "--output", str(out),
        ])]
        assert manifest["inputs"] == [p.resolve().as_posix() for p in paths]
        assert (manifest["status"], manifest["outcome"]) == ("finished", "exit 0")

    def test_the_table_prints_to_a_cp1252_stdout(self, tmp_path, monkeypatch):
        path = tmp_path / "a.json"
        path.write_text(json.dumps(_card("a", 0.7, 0.6).to_json()), encoding="utf-8")
        stdout = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
        monkeypatch.setattr(sys, "stdout", stdout)
        assert compare.main([str(path)]) == 0
        assert not (tmp_path / "run.json").exists()  # printing writes no file
        stdout.flush()
        assert "rung 1 − rung 1w" in stdout.buffer.getvalue().decode("utf-8")

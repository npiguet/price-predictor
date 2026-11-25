"""The knowledge-probe run: probe context, line split, selection-set freeze, report directory."""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from effect_knowledge_probes import (
    combat_grid,
    common,
    labels,
    run,
)
from effect_knowledge_probes.line_ladders import line_split

from tests.unit.effects.builders import GEN3_SIDECAR_FOLDERS
from tests.unit.scripts.knowledge_probe_support import (
    LETHAL_STEP,
    block_pair,
    keyword_set,
    one_on_one,
    outcome_by_the_rule,
)

# ── what a checkpoint's probes read (FR-075) ─────────────────────────────


def _corpus_manifest(**overrides):
    from effects.domain.corpus_manifest import CorpusManifest

    fields = dict(
        seed=1, surface="script", vocab_path="models/effects/vocab-script.txt",
        variant_scripts="", holdout_permille=20, holdout_max_carriers=8,
        text_cap=0, card_disjoint_text_cap=0, training_records=1, class_mix={},
        delivered_mix={}, held_out_cards=(), held_out_texts=(),
        card_disjoint_games=("cg",), game_disjoint_games=("gg",), rarity={},
        sources=(), per_class={}, per_stratum={}, unique_texts={}, shortfall={},
    )
    fields.update(overrides)
    return CorpusManifest(**fields)


class TestProbeContext:
    """Every probe reads the corpus, sidecar trees and variant tree its checkpoint
    trained against, never a corpus or probe set named in the script."""

    def _context(self, tmp_path, monkeypatch, *, recorded_digest=None):
        from effects.domain.split_provenance import SplitProvenance
        from effects.infrastructure.corpus_store import CorpusStore

        variants = tmp_path / "variant-scripts"
        manifest = _corpus_manifest(variant_scripts=str(variants))
        corpus = tmp_path / "corpus"
        CorpusStore(corpus).save(manifest)
        checkpoint = SimpleNamespace(
            provenance=SplitProvenance(
                corpus_path=str(corpus),
                corpus_digest=recorded_digest or manifest.digest(),
            ),
            cards_folders=(str(tmp_path / "cardsfolder"), str(tmp_path / "tokenscripts")),
            epoch=3,
        )
        monkeypatch.setattr(common, "load_checkpoint", lambda path: checkpoint)
        return common.probe_context(tmp_path / "latest.pt"), manifest, corpus, variants

    def test_the_corpus_is_the_one_the_checkpoint_recorded(self, tmp_path, monkeypatch):
        context, manifest, corpus, _ = self._context(tmp_path, monkeypatch)
        assert context.corpus == corpus
        assert context.corpus_digest == manifest.digest()
        assert context.validation_dirs() == [
            corpus / "validation" / "card-disjoint",
            corpus / "validation" / "game-disjoint",
        ]

    def test_the_sidecars_read_the_variant_tree_the_corpus_recorded(
        self, tmp_path, monkeypatch,
    ):
        """Without it a variant card's ability keys resolve to no text."""
        context, _, _, variants = self._context(tmp_path, monkeypatch)
        assert context.variant_scripts == variants
        assert context.sidecars().path_for("variant-scripts/shock_variant.txt").parent == (
            variants
        )
        assert context.cards_folders == (tmp_path / "cardsfolder", tmp_path / "tokenscripts")

    def test_a_corpus_rebuilt_since_training_is_refused(self, tmp_path, monkeypatch):
        with pytest.raises(SystemExit, match="rebuilt"):
            self._context(tmp_path, monkeypatch, recorded_digest="an-older-build")


# ── the line ladder's held-out split ────────────────────────────────────


class TestLineSplit:
    """Every game naming a held-out card is kept out of training, so a text is
    untrained exactly when every card printing it is held out."""

    @pytest.mark.parametrize("carriers, held_out, expected", [
        pytest.param({"Shock", "Lightning Strike"}, {"Shock"}, "trained",
                     id="also printed on a trained card"),
        pytest.param({"Shock", "Spark"}, {"Shock", "Spark", "Bolt"}, "held-out",
                     id="every carrier held out"),
        pytest.param({"Lightning Strike"}, {"Shock"}, "trained", id="no carrier held out"),
    ])
    def test_the_split(self, carriers, held_out, expected):
        assert line_split(carriers, held_out) == expected


# ── the selection-set freeze (T156) ─────────────────────────────────────


class TestFreezeSelectionSet:
    """``--freeze-probe-set --corpus DIR`` freezes the selection set with no
    checkpoint, and the combat and evasion grids read the records it froze."""

    def test_the_freeze_takes_a_corpus_and_refuses_a_checkpoint(self, tmp_path):
        args = run.parse_args(["--freeze-probe-set", "--corpus", str(tmp_path),
                               "--reports-dir", "p"])
        assert args.checkpoint is None and args.corpus == tmp_path
        for argv in (["--freeze-probe-set", "--corpus", "c", "--checkpoint", "x.pt"],
                     ["--freeze-probe-set"], []):
            with pytest.raises(SystemExit):
                run.parse_args(argv)

    def test_main_freezes_from_the_corpus_alone(self, tmp_path, monkeypatch):
        from effects.application import freeze_selection_set
        from effects.infrastructure.corpus_store import CorpusStore

        corpus = tmp_path / "corpus"
        CorpusStore(corpus).save(_corpus_manifest())
        seen = {}

        def fake_run(path, **kwargs):
            seen.update(kwargs, corpus=path)
            return 1

        monkeypatch.setattr(freeze_selection_set, "run", fake_run)
        monkeypatch.setattr(common, "probe_context", lambda *a, **k: pytest.fail("no checkpoint"))
        code = run.main(["--freeze-probe-set", "--corpus", str(corpus),
                         "--probe-set-dir", str(tmp_path / "sets"), "--reports-dir", "p",
                         "--cards-folder", str(GEN3_SIDECAR_FOLDERS[0])])
        assert code == 1
        assert seen["corpus"] == corpus and seen["reports_dir"] == tmp_path / "sets"
        names = [grid.name for grid in seen["grids"]]
        assert names[0] == "damage" and names[-1] == "evasion"
        assert len([n for n in names if n.startswith("combat/")]) == 2 * len(combat_grid.VARIANTS)
        assert callable(seen["deviance"])

    def test_probe_items_are_the_board_targets_labelled_with_their_kind(
        self, records, sidecars, acting,
    ):
        from effect_knowledge_probes.selection import ProbeItems

        items = list(ProbeItems(sidecars, acting).items(records))
        assert items
        for item in items:
            spec = labels.TARGETS_BY_NAME[item.probe_type]
            assert spec.scope in ("act", "entity") and item.kind == spec.kind
            assert np.isfinite(item.label)

    def test_a_combat_grid_qualifies_a_block_with_the_rules_outcomes(
        self, records, monkeypatch,
    ):
        from effect_knowledge_probes import selection

        record, attacker, blocker = one_on_one(records)
        outcome_by_the_rule(monkeypatch, record, attacker, blocker)
        grids = {g.name: g for g in selection.grid_sources(None, keyword_set)}
        found = grids["combat/none/blocker"].qualify(record)
        assert found.entities == (attacker, blocker)
        assert found.outcomes == tuple(LETHAL_STEP.ravel())
        assert len(grids["combat/none/blocker"].cells) == 64
        assert set(grids["combat/attacker-deathtouch/blocker"].qualify(record).outcomes) == {1.0}

    def test_the_evasion_grid_qualifies_a_block_with_the_rules_outcomes(self, records):
        from effect_knowledge_probes.selection import EvasionGrid

        record, attacker, blocker = block_pair(records)
        grid = EvasionGrid(keyword_set)
        found = grid.qualify(record)
        assert found.entities == (attacker, blocker)
        assert len(found.outcomes) == len(grid.cells)
        assert found.outcomes[grid.cells.index("flying|none")] == 0.0
        assert found.outcomes[grid.cells.index("flying|reach")] == 1.0
        assert grid.qualify(replace(record, fork=True)) is None

    def test_the_grids_read_the_frozen_records_each_once(self, tmp_path):
        import gzip

        from effect_knowledge_probes.selection import frozen_grid_records

        from effects.domain.selection_set import (
            FrozenGrid,
            GridCell,
            SelectionSet,
            TaggedRecord,
            selection_set_path,
            write_selection_set,
        )
        from tests.unit.effects.builders import GEN3_RECORDS

        with gzip.open(GEN3_RECORDS, "rt", encoding="utf-8") as handle:
            lines = [line for line in handle if line.strip()][:3]
        stratum = tmp_path / "validation" / "game-disjoint"
        stratum.mkdir(parents=True)
        (stratum / "shard-0.jsonl").write_text("".join(lines + lines[:1]), encoding="utf-8")
        rows = [json.loads(line) for line in lines]
        frozen = lambda ids: FrozenGrid(  # noqa: E731
            (GridCell("1x1", (1.0,) * len(ids)),),
            tuple(TaggedRecord(rows[i]["record_id"], rows[i]["game_id"], ("E1", "E2"))
                  for i in ids))
        selection = SelectionSet(
            corpus_digest="d", r=4, halves={}, probe_kinds={}, rule_tests={},
            probes={}, deviance_samples={}, standard_errors={}, left_out={},
            grids={"game-disjoint": {"reporting": {"evasion": frozen([0]),
                                                    "combat/none/blocker": frozen([1])},
                                     "selection": {"evasion": frozen([2])}}},
        )
        write_selection_set(selection_set_path(tmp_path, "d"), selection)
        context = SimpleNamespace(
            corpus_digest="d",
            stratum_shards=lambda name: sorted((tmp_path / "validation" / name).glob("shard-*")))
        found = frozen_grid_records(context, "evasion", reports_dir=tmp_path)
        assert [(r.record_id, e) for r, e in found] == [(rows[0]["record_id"], ("E1", "E2"))]
        both = frozen_grid_records(context, "evasion", halves=("selection", "reporting"),
                                   reports_dir=tmp_path)
        assert sorted(r.record_id for r, _ in both) == sorted(
            [rows[0]["record_id"], rows[2]["record_id"]])
        with pytest.raises(SystemExit, match="no selection set"):
            frozen_grid_records(SimpleNamespace(corpus_digest="other"), "evasion",
                                reports_dir=tmp_path)


# ── where a probing run's report goes ───────────────────────────────────


class TestTheReportDirectory:
    """``--output DIR`` is the report directory itself, its ``run.json``
    included; without it the report goes to a dated directory under
    ``--reports-dir``."""

    def test_without_output_the_report_is_dated_under_reports_dir(self, tmp_path):
        args = run.parse_args(["--checkpoint", "m/run/latest.pt", "--abilities-root", "cache",
                               "--reports-dir", str(tmp_path)])
        assert args.output is None
        assert run.output_dir(args, "2026-11-20") == (
            tmp_path / "knowledge-probes-run-latest-20261120")

    def test_output_names_the_report_directory(self, tmp_path):
        args = run.parse_args(["--checkpoint", "m/run/latest.pt", "--abilities-root", "cache",
                               "--reports-dir", str(tmp_path), "--output", str(tmp_path / "r")])
        assert run.output_dir(args, "2026-11-20") == tmp_path / "r"

    def test_the_freeze_refuses_it(self):
        with pytest.raises(SystemExit):
            run.parse_args(["--freeze-probe-set", "--corpus", "c", "--reports-dir", "p",
                            "--output", "r"])

    def test_the_manifest_is_kept_in_it(self, tmp_path, monkeypatch):
        report = tmp_path / "reports" / "2026-11-20-rule-mask-hard"
        monkeypatch.setattr(run, "_probe_run", lambda args: 0)
        assert run.main(["--checkpoint", str(tmp_path / "latest.pt"), "--output", str(report),
                         "--abilities-root", str(tmp_path), "--reports-dir", str(tmp_path),
                         "--probe-set-dir", str(tmp_path / "sets")]) == 0
        assert json.loads((report / "run.json").read_text(encoding="utf-8"))["status"] == (
            "finished")

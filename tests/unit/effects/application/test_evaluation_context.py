"""The evaluation context's fail-fast rows: the corpus a checkpoint trained on,
and the baseline checkpoints named beside it.

Each is a case where continuing would score the gates partly on games the model
trained on, or against a baseline that never ran.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from effects.application.evaluation_context import parse_baseline_checkpoint
from effects.domain.baselines import Baseline
from effects.domain.split_provenance import SplitProvenance


class TestCorpusMismatch:
    """``evaluate-effect-model`` refuses a corpus rebuilt since training
    (FR-147) — the same discipline as the vocabulary-hash guard, for the
    dataset rather than the vocabulary file."""

    def test_evaluate_refuses_a_corpus_rebuilt_since_training(self, tmp_path):
        from effects.application.evaluation_context import (
            CorpusMismatchError,
            check_corpus,
        )

        provenance = SplitProvenance(
            corpus_path=str(tmp_path), corpus_digest="trained-against-this",
        )
        with pytest.raises(CorpusMismatchError, match="rebuilt"):
            check_corpus(provenance, actual_digest="something-else")

    def test_matching_digests_pass(self, tmp_path):
        from effects.application.evaluation_context import check_corpus

        provenance = SplitProvenance(
            corpus_path=str(tmp_path), corpus_digest="same",
        )
        check_corpus(provenance, actual_digest="same")  # no raise

    def test_the_error_names_the_corpus_path_and_both_digests(self, tmp_path):
        from effects.application.evaluation_context import (
            CorpusMismatchError,
            check_corpus,
        )

        provenance = SplitProvenance(
            corpus_path=str(tmp_path), corpus_digest="trained-against-this",
        )
        with pytest.raises(CorpusMismatchError) as excinfo:
            check_corpus(provenance, actual_digest="something-else")
        message = str(excinfo.value)
        assert str(tmp_path) in message
        assert "trained-against-this" in message
        assert "something-else" in message


class TestCorpusPathResolution:
    """``--corpus`` on ``evaluate-effect-model`` defaults to what the
    checkpoint recorded, an explicit value overriding it (FR-147), and a
    checkpoint that recorded no corpus is refused."""

    _CHECKPOINT = Path("models/effects/effect-model/latest.pt")

    def test_defaults_to_the_checkpoints_recorded_path(self):
        from effects.application.evaluation_context import resolve_corpus_path

        provenance = SplitProvenance(
            corpus_path="output/effects/corpus", corpus_digest="abc123",
        )
        assert resolve_corpus_path(
            provenance, corpus=None, checkpoint=self._CHECKPOINT,
        ) == Path("output/effects/corpus")

    def test_an_explicit_corpus_overrides(self, tmp_path):
        from effects.application.evaluation_context import resolve_corpus_path

        provenance = SplitProvenance(
            corpus_path="output/effects/corpus", corpus_digest="abc123",
        )
        assert resolve_corpus_path(
            provenance, corpus=tmp_path, checkpoint=self._CHECKPOINT,
        ) == tmp_path

    def test_a_checkpoint_that_recorded_no_corpus_is_refused(self, tmp_path):
        """Nothing else says which games it trained on, --corpus or not."""
        from effects.application.evaluation_context import (
            CorpusMismatchError,
            resolve_corpus_path,
        )

        for corpus in (None, tmp_path):
            with pytest.raises(CorpusMismatchError, match="records no curated corpus"):
                resolve_corpus_path(
                    SplitProvenance(), corpus=corpus, checkpoint=self._CHECKPOINT,
                )


class TestVariantCheckpointFlag:
    def test_it_parses_name_equals_path(self):
        name, path = parse_baseline_checkpoint("identity=models/x/latest.pt")
        assert name == "identity"
        assert path.name == "latest.pt"

    def test_a_bare_path_is_rejected(self):
        with pytest.raises(ValueError, match="NAME=PATH"):
            parse_baseline_checkpoint("models/x/latest.pt")

    def test_an_empty_name_is_rejected(self):
        with pytest.raises(ValueError, match="NAME=PATH"):
            parse_baseline_checkpoint("=models/x/latest.pt")

    def test_a_misspelt_baseline_is_rejected(self):
        """Otherwise gate 1 skips for want of an identity checkpoint."""
        with pytest.raises(ValueError, match="unknown baseline 'identiy'"):
            parse_baseline_checkpoint("identiy=models/x/latest.pt")

    def test_the_shipping_model_is_not_a_baseline(self):
        with pytest.raises(ValueError, match="unknown baseline 'full'"):
            parse_baseline_checkpoint("full=models/x/latest.pt")


class TestVariantCheckpointRefusals:
    """``--variant-checkpoint NAME=PATH`` names a checkpoint trained as NAME on
    the corpus the main one read: ``identity=<a full model>``
    would run as a full model, and gate 1 would print that the encoder beats
    the identity baseline."""

    _MAIN = SplitProvenance(corpus_digest="digest-a", withheld_keyword="flying")

    def _load(self, monkeypatch, *, variant=Baseline.IDENTITY, **provenance):
        from effects.application.evaluation_context import (
            EvaluateEffectModelConfig,
            load_baselines,
        )
        from effects.infrastructure import effect_model_store

        loaded = SimpleNamespace(variant=variant, provenance=replace(self._MAIN, **provenance))
        monkeypatch.setattr(effect_model_store, "load_effect_checkpoint", lambda path: loaded)
        config = EvaluateEffectModelConfig(
            checkpoint=Path("main.pt"), abilities_root=Path("abilities"),
            win_rates=Path("rates"), reports_dir=Path("reports"),
            baseline_checkpoints={Baseline.IDENTITY: Path("identity.pt")},
        )
        main = SimpleNamespace(variant=Baseline.FULL, provenance=self._MAIN)
        return load_baselines(config, main), loaded

    def test_a_checkpoint_of_the_named_variant_on_the_same_corpus_is_kept(self, monkeypatch):
        variants, loaded = self._load(monkeypatch)
        assert variants == {Baseline.IDENTITY: loaded}

    def test_identity_naming_a_full_checkpoint_is_refused(self, monkeypatch):
        with pytest.raises(RuntimeError, match="identity.pt") as excinfo:
            self._load(monkeypatch, variant=Baseline.FULL)
        assert "'full'" in str(excinfo.value)
        assert "'identity'" in str(excinfo.value)

    def test_a_checkpoint_trained_on_another_corpus_is_refused(self, monkeypatch):
        with pytest.raises(RuntimeError, match="corpus digest") as excinfo:
            self._load(monkeypatch, corpus_digest="digest-b")
        assert "digest-a" in str(excinfo.value)
        assert "digest-b" in str(excinfo.value)

    def test_a_checkpoint_withholding_another_keyword_is_refused(self, monkeypatch):
        with pytest.raises(RuntimeError, match="withheld keyword") as excinfo:
            self._load(monkeypatch, withheld_keyword=None)
        assert "flying" in str(excinfo.value)


class TestTheTreesTheCheckpointRecorded:
    """``--cards-folder`` defaults to the trees the checkpoint trained against,
    and trees reconverted since the corpus was built are refused:
    a reconverted tree joins the records' keys to other lines in every gate."""

    @staticmethod
    def _tree(root: Path) -> Path:
        from effects.domain.provenance import ProvenanceKey, ProvenanceSidecar, SidecarLine
        from effects.infrastructure.sidecar_io import sidecar_path_for
        from tests.unit.effects.file_builders import write_sidecar

        script = "cardsfolder/s/shock.txt"
        write_sidecar(ProvenanceSidecar(card="Shock", script_file=script, lines=(
            SidecarLine(line_index=0, line_kind="spell",
                        provenance=(ProvenanceKey(script, 0, "spell", 0),),
                        script_text="SP$ DealDamage | NumDmg$ 2", prose_offset=0),
        )), sidecar_path_for(root / "s" / "shock.txt"))
        return root

    def _resolved(self, tree: Path, *, flag=None, recorded_digest=None):
        from effects.application.evaluation_context import (
            EvaluateEffectModelConfig,
            with_checkpoint_defaults,
        )
        from effects.infrastructure.converted_trees import SourceTrees

        digest = recorded_digest or SourceTrees.of((tree,)).converted_sidecar_digest()
        manifest = SimpleNamespace(sidecar_digest=digest, variant_scripts_or=lambda given: given)
        main = SimpleNamespace(cards_folders=(str(tree),))
        config = EvaluateEffectModelConfig(
            checkpoint=Path("main.pt"), abilities_root=Path("abilities"),
            win_rates=Path("rates"), reports_dir=Path("reports"), cards_folders=flag,
        )
        return with_checkpoint_defaults(config, main, manifest)

    def test_no_flag_reads_the_folders_the_checkpoint_recorded(self, tmp_path):
        tree = self._tree(tmp_path / "cardsfolder")
        assert self._resolved(tree).cards_folders == (tree,)

    def test_a_named_folder_wins(self, tmp_path):
        tree = self._tree(tmp_path / "cardsfolder")
        named = self._tree(tmp_path / "copy" / "cardsfolder")
        resolved = self._resolved(
            tree, flag=(named,),
            recorded_digest=self._digest(named),
        )
        assert resolved.cards_folders == (named,)

    @staticmethod
    def _digest(tree: Path) -> str:
        from effects.infrastructure.converted_trees import SourceTrees

        return SourceTrees.of((tree,)).converted_sidecar_digest()

    def test_a_reconverted_tree_is_refused_naming_both_digests(self, tmp_path):
        from effects.infrastructure.converted_trees import SidecarsChangedError

        tree = self._tree(tmp_path / "cardsfolder")
        recorded = self._digest(tree)
        sidecar = next(tree.rglob("*.provenance.json"))
        sidecar.write_text(sidecar.read_text(encoding="utf-8") + " ", encoding="utf-8")
        with pytest.raises(SidecarsChangedError) as excinfo:
            self._resolved(tree, recorded_digest=recorded)
        assert recorded in str(excinfo.value)
        assert self._digest(tree) in str(excinfo.value)

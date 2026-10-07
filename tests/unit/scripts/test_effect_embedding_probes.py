"""The embedding-probe scripts read any checkpoint's cache (T114; Story 6
scenarios 9 and 10).

``scripts/`` is not part of the installed package, so the modules are loaded by
file path. The sidecars are real gen-1 ones from the fixture tree; the cache
rows beside them are written here, row-aligned, at a width no checkpoint has
shipped, so nothing can pass by assuming 64.
"""

from __future__ import annotations

import datetime
import importlib.util
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_PROBES = Path(__file__).resolve().parents[3] / "scripts" / "effect_embedding_probes"
_FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "effects"


def _load(name: str, *, common_module=None):
    """Load one script by path.

    The scripts import their sibling ``common`` by its bare name, which the
    knowledge-probe suite's ``common`` shares. So ``common`` is registered
    under that name only while a script that imports it loads, and the
    previous holder of the name is put back afterwards: ``build_texts`` binds
    the very module the tests patch, and neither suite sees the other's.
    """
    spec = importlib.util.spec_from_file_location(
        f"effect_embedding_probes_{name}", _PROBES / f"{name}.py",
    )
    module = importlib.util.module_from_spec(spec)
    # A dataclass resolves its annotations through its module's entry.
    sys.modules[spec.name] = module
    previous = sys.modules.get("common")
    if common_module is not None:
        sys.modules["common"] = common_module
    sys.path.insert(0, str(_PROBES))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(_PROBES))
        if common_module is not None:
            if previous is None:
                sys.modules.pop("common", None)
            else:
                sys.modules["common"] = previous
    return module


common = _load("common")
build_texts = _load("build_texts", common_module=common)

_WIDTH = 12


def _provenance(vocab: str = "models/effects/vocab-script.txt", corpus: str = ""):
    return SimpleNamespace(vocab_path=vocab, corpus_path=corpus)


@pytest.fixture
def tree(tmp_path):
    """A copy of the real gen-1 sidecars, with a cache of width 12 beside them."""
    cards = tmp_path / "gen1-cardsfolder"
    shutil.copytree(_FIXTURES / "gen1-sidecars" / "cardsfolder", cards)
    abilities = tmp_path / "abilities"
    rng = np.random.default_rng(0)
    from effects.infrastructure.sidecar_io import read_sidecar

    for sidecar_path in cards.rglob("*.provenance.json"):
        sidecar = read_sidecar(sidecar_path)
        relative = Path(sidecar.script_file)
        target = abilities / relative.parent / f"{relative.stem}.npz"
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target, e=rng.normal(size=(len(sidecar.lines), _WIDTH)).astype(np.float32),
        )
    return SimpleNamespace(cards=cards, abilities=abilities, root=tmp_path)


class TestResolvePaths:
    """Scenario 9: vocabulary, cache and width come from the run, not constants."""

    def test_the_width_of_e_is_read_from_the_cache(self, tree):
        paths = common.paths_for(
            tree.root / "arm" / "latest.pt", tree.abilities, _provenance(),
            cards_folders=[tree.cards],
        )
        assert paths.e_width == _WIDTH

    def test_the_surface_follows_the_checkpoint_s_vocabulary(self, tree):
        script = common.paths_for(
            Path("c.pt"), tree.abilities, _provenance("models/effects/vocab-script.txt"),
        )
        prose = common.paths_for(
            Path("c.pt"), tree.abilities, _provenance("models/effects/vocab.txt"),
        )
        assert (script.surface, prose.surface) == ("script", "prose")

    def test_output_is_named_by_checkpoint_and_date(self, tree):
        paths = common.paths_for(
            tree.root / "2026-10-gen2-a" / "latest.pt", tree.abilities, _provenance(),
            today=datetime.date(2026, 10, 8),
        )
        assert paths.out.name == "embedding-probes-2026-10-gen2-a-latest-20261008"
        assert paths.out.parent == common.REPORTS

    def test_sidecar_trees_default_to_what_the_checkpoint_recorded(self, tree):
        paths = common.paths_for(
            Path("c.pt"), tree.abilities, _provenance(),
            recorded_folders=(str(tree.cards),),
        )
        assert paths.cards_folders == (tree.cards,)
        assert paths.tree_root("cardsfolder") == tree.cards

    def test_the_manifest_comes_from_the_recorded_corpus(self, tree):
        paths = common.paths_for(
            Path("c.pt"), tree.abilities, _provenance(corpus="output/effects/corpus"),
        )
        assert paths.manifest == Path("output/effects/corpus/manifest.json")
        assert common.load_manifest_sets(
            common.paths_for(Path("c.pt"), tree.abilities, _provenance()),
        ) == (set(), {})

    def test_an_empty_cache_root_is_refused(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="encode-abilities"):
            common.paths_for(Path("c.pt"), tmp_path, _provenance())

    def test_a_table_built_on_an_earlier_day_is_found(self, tree, monkeypatch):
        monkeypatch.setattr(common, "REPORTS", tree.root / "reports")
        checkpoint = tree.root / "arm" / "latest.pt"
        yesterday = common.paths_for(
            checkpoint, tree.abilities, _provenance(),
            today=datetime.date(2026, 10, 7),
        )
        yesterday.text_table.parent.mkdir(parents=True)
        yesterday.text_table.write_bytes(b"")
        today = common.paths_for(
            checkpoint, tree.abilities, _provenance(),
            today=datetime.date(2026, 10, 8),
        )
        assert today.reusable("cache/texts.pkl") == yesterday.text_table


class TestTheTextTable:
    """Scenario 9: a cache with no taxonomy baseline still yields rows."""

    @pytest.fixture(autouse=True)
    def _no_variant_tree(self, tree, monkeypatch):
        """The real variant-script tree is tens of thousands of sidecars."""
        monkeypatch.setattr(build_texts, "VARIANT_TREE", tree.root / "no-variants")

    def test_rows_are_built_with_no_taxonomy_cache(self, tree, monkeypatch):
        monkeypatch.setattr(common, "REPORTS", tree.root / "reports")
        paths = common.paths_for(
            tree.root / "arm" / "latest.pt", tree.abilities, _provenance(),
            cards_folders=[tree.cards],
        )
        paths.out.mkdir(parents=True)
        table = build_texts.build(paths)
        assert len(table) > 20
        assert all(len(vector) == _WIDTH for vector in table["e_full"])
        assert not common.has_taxonomy(table)
        assert paths.text_table.exists()

    def test_keys_name_the_source_tree_not_the_kept_aside_folder(self, tree, monkeypatch):
        import pickle

        monkeypatch.setattr(common, "REPORTS", tree.root / "reports")
        paths = common.paths_for(
            tree.root / "arm" / "latest.pt", tree.abilities, _provenance(),
            cards_folders=[tree.cards],
        )
        paths.out.mkdir(parents=True)
        build_texts.build(paths)
        with open(paths.keymap, "rb") as handle:
            keymap = pickle.load(handle)
        assert keymap
        assert all(key[0].startswith("cardsfolder/") for key in keymap)


class TestParticipationRatio:
    """Scenario 10: ``(Σλ)² / Σλ²``."""

    def test_an_even_spectrum_uses_every_dimension(self):
        assert common.participation_ratio(np.full(8, 0.125)) == pytest.approx(8.0)

    def test_one_dominant_direction_uses_one(self):
        assert common.participation_ratio(
            np.array([1.0, 0.0, 0.0]),
        ) == pytest.approx(1.0)

    def test_a_known_spectrum(self):
        # (4 + 2 + 1 + 1)² / (16 + 4 + 1 + 1) = 64 / 22.
        assert common.participation_ratio(
            np.array([4.0, 2.0, 1.0, 1.0]),
        ) == pytest.approx(64 / 22)

    def test_it_reads_the_same_from_shares_as_from_eigenvalues(self):
        eigenvalues = np.array([5.0, 3.0, 2.0])
        assert common.participation_ratio(eigenvalues) == pytest.approx(
            common.participation_ratio(eigenvalues / eigenvalues.sum()),
        )


class TestCheckpointLabel:
    def test_a_rolling_checkpoint_is_named_by_its_run(self):
        assert common.checkpoint_label(
            Path("models/effects/runs/2026-09-17-full-textless-corpus/latest.pt"),
        ) == "2026-09-17-full-textless-corpus-latest"

    def test_a_timestamped_checkpoint_keeps_its_stem(self):
        assert common.checkpoint_label(Path("runs/a/20261008-1200.pt")) == "20261008-1200"

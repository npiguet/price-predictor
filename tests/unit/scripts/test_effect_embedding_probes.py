"""The embedding-probe scripts read any checkpoint's cache (Story 6
scenarios 9 and 10).

``scripts/`` is not part of the installed package; the conftest puts it on the
path, so the battery imports as the package its scripts make it. The sidecars
are real gen-3 ones from the fixture tree; the cache rows beside them are
written here, row-aligned, at a width no checkpoint has shipped, so nothing can
pass by assuming 64.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from effect_embedding_probes import build_texts, common, effect_profiles

from effects.domain.provenance import BuiltTrait, ProvenanceKey
from tests.unit.effects.file_builders import write_vocabulary

_FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "effects"

_WIDTH = 12
#: How many of the fixture's sidecars each test copies.
_SIDECARS = 100


#: ``surface -> vocabulary file`` recording it, written once for the module.
_VOCABULARIES: dict[str, Path] = {}


@pytest.fixture(autouse=True, scope="module")
def _vocabularies(tmp_path_factory):
    root = tmp_path_factory.mktemp("vocabularies")
    for surface in ("script", "prose"):
        _VOCABULARIES[surface] = write_vocabulary(root / "vocab.txt" if surface == "prose"
                                                  else root / "renamed.txt", surface)


def _provenance(surface: str = "script", corpus: str = ""):
    return SimpleNamespace(vocab_path=str(_VOCABULARIES[surface]), corpus_path=corpus)


@pytest.fixture
def tree(tmp_path):
    """A copy of the first real gen-3 sidecars and their converted cards, with a
    cache of width 12 beside them."""
    source = _FIXTURES / "gen3-sidecars" / "cardsfolder"
    cards = tmp_path / "old-cardsfolder"
    for path in sorted(source.rglob("*.provenance.json"))[:_SIDECARS]:
        converted = path.with_name(path.name.removesuffix(".provenance.json") + ".txt")
        for found in (path, converted):
            target = cards / found.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(found, target)
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


_UNUSED_REPORTS = Path("unused-reports")


def _paths_for(checkpoint, abilities, provenance, options=None):
    """``paths_for`` with the reports directory the flag now requires."""
    options = options or common.PathOptions()
    if options.reports_dir is None:
        options = dataclasses.replace(options, reports_dir=_UNUSED_REPORTS)
    return common.paths_for(checkpoint, abilities, provenance, options)


class TestResolvePaths:
    """Scenario 9: vocabulary, cache and width come from the run, not constants."""

    def test_the_width_of_e_is_read_from_the_cache(self, tree):
        paths = _paths_for(
            tree.root / "arm" / "latest.pt", tree.abilities, _provenance(),
            options=common.PathOptions(cards_folders=[tree.cards]),
        )
        assert paths.e_width == _WIDTH

    def test_the_surface_follows_the_checkpoint_s_vocabulary(self, tree):
        script = _paths_for(
            Path("c.pt"), tree.abilities, _provenance("script"),
        )
        prose = _paths_for(
            Path("c.pt"), tree.abilities, _provenance("prose"),
        )
        assert (script.surface, prose.surface) == ("script", "prose")

    def test_output_is_named_by_checkpoint_and_date(self, tree):
        paths = _paths_for(
            tree.root / "2026-10-gen3-a" / "latest.pt", tree.abilities, _provenance(),
            options=common.PathOptions(today=datetime.date(2026, 10, 8)),
        )
        assert paths.out.name == "embedding-probes-2026-10-gen3-a-latest-20261008"
        assert paths.out.parent == _UNUSED_REPORTS

    def test_a_reports_dir_moves_the_output_beside_the_arm(self, tree):
        arm = tree.root / "2026-10-gen3-a"
        paths = _paths_for(
            arm / "latest.pt", tree.abilities, _provenance(),
            options=common.PathOptions(
                today=datetime.date(2026, 10, 8), reports_dir=arm / "reports"),
        )
        assert paths.out == (
            arm / "reports" / "embedding-probes-2026-10-gen3-a-latest-20261008"
        )

    def test_sidecar_trees_default_to_what_the_checkpoint_recorded(self, tree):
        paths = _paths_for(
            Path("c.pt"), tree.abilities, _provenance(),
            options=common.PathOptions(recorded_folders=(str(tree.cards),)),
        )
        assert paths.cards_folders == (tree.cards,)
        assert paths.tree_root("cardsfolder") == tree.cards

    def test_the_manifest_comes_from_the_recorded_corpus(self, tree):
        paths = _paths_for(
            Path("c.pt"), tree.abilities, _provenance(corpus="output/effects/corpus"),
        )
        assert paths.manifest == Path("output/effects/corpus/manifest.json")
        assert common.load_manifest_sets(
            _paths_for(Path("c.pt"), tree.abilities, _provenance()),
        ) == (set(), {})

    def test_an_empty_cache_root_is_refused(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="encode-abilities"):
            _paths_for(Path("c.pt"), tmp_path, _provenance())

    def test_a_table_built_on_an_earlier_day_is_found(self, tree, monkeypatch):
        checkpoint = tree.root / "arm" / "latest.pt"
        yesterday = _paths_for(
            checkpoint, tree.abilities, _provenance(),
            options=common.PathOptions(
                today=datetime.date(2026, 10, 7), reports_dir=tree.root / "reports"),
        )
        yesterday.text_table.parent.mkdir(parents=True)
        yesterday.text_table.write_bytes(b"")
        today = _paths_for(
            checkpoint, tree.abilities, _provenance(),
            options=common.PathOptions(
                today=datetime.date(2026, 10, 8), reports_dir=tree.root / "reports"),
        )
        assert today.reusable("cache/texts.pkl") == yesterday.text_table


class TestTheTextTable:
    """Scenario 9: a cache with no taxonomy baseline still yields rows."""


    def test_rows_are_built_with_no_taxonomy_cache(self, tree, monkeypatch):
        paths = _paths_for(
            tree.root / "arm" / "latest.pt", tree.abilities, _provenance(),
            options=common.PathOptions(
                cards_folders=[tree.cards], reports_dir=tree.root / "reports"),
        )
        paths.out.mkdir(parents=True)
        table = build_texts.build(paths)
        assert len(table) > 20
        assert all(len(vector) == _WIDTH for vector in table["e_full"])
        assert not common.has_taxonomy(table)
        assert paths.text_table.exists()

    def test_keys_name_the_source_tree_not_the_kept_aside_folder(self, tree, monkeypatch):
        import pickle

        paths = _paths_for(
            tree.root / "arm" / "latest.pt", tree.abilities, _provenance(),
            options=common.PathOptions(
                cards_folders=[tree.cards], reports_dir=tree.root / "reports"),
        )
        paths.out.mkdir(parents=True)
        build_texts.build(paths)
        with open(paths.keymap, "rb") as handle:
            keymap = pickle.load(handle)
        assert keymap
        assert all(key.script_file.startswith("cardsfolder/") for key in keymap)


class TestTheKeymap:
    """The keymap reads through the one join, so keys differing only in
    ``built`` are distinct entries."""

    _SAGA = "cardsfolder/b/braidss_frightful_return.txt"

    @pytest.fixture
    def keymap(self, tmp_path, monkeypatch):
        """The real gen-3 sidecars, Braids's Frightful Return among them."""
        gen3 = _FIXTURES / "gen3-sidecars"
        paths = SimpleNamespace(cards_folders=[gen3 / "cardsfolder", gen3 / "tokenscripts"],
                                surface="script", variant_scripts=tmp_path / "no-variants")
        return build_texts.keymap_of(paths)

    def test_each_chapter_of_a_saga_is_its_own_entry(self, keymap):
        chapters = [keymap[ProvenanceKey(self._SAGA, 0, "keyword", 0,
                                         built=BuiltTrait("trigger", n))] for n in range(3)]
        assert [text.split(" |")[0] for text in chapters] == [
            "Chapter$ 1", "Chapter$ 2", "Chapter$ 3"]
        assert ProvenanceKey(self._SAGA, 0, "keyword", 0) not in keymap

    def test_a_record_acting_through_chapter_ii_reads_chapter_ii(self, keymap, monkeypatch):
        monkeypatch.setattr(effect_profiles, "_KEYMAP", keymap)
        chapter = {"script_file": self._SAGA, "face": 0, "trait_kind": "keyword",
                   "index_within_kind": 0, "built": {"kind": "trigger", "index": 1}}
        bare = {key: value for key, value in chapter.items() if key != "built"}
        assert effect_profiles.acting_text({"ability": [chapter]}).startswith("Chapter$ 2 |")
        assert effect_profiles.acting_text({"ability": [bare]}) is None


class TestToMarkdown:
    def test_floats_are_formatted_nan_is_blank_and_pipes_are_escaped(self):
        import pandas as pd

        frame = pd.DataFrame({"card": ["a | b"], "share": [0.12345], "gap": [float("nan")]})
        assert common.to_markdown(frame, ".2f").splitlines() == [
            "| card | share | gap |",
            "|---|---:|---:|",
            "| a \\| b | 0.12 |  |",
        ]


class TestCheckpointLabel:
    def test_a_rolling_checkpoint_is_named_by_its_run(self):
        assert common.checkpoint_label(
            Path("models/effects/runs/2026-09-17-full-textless-corpus/latest.pt"),
        ) == "2026-09-17-full-textless-corpus-latest"

    def test_a_timestamped_checkpoint_keeps_its_stem(self):
        assert common.checkpoint_label(Path("runs/a/20261008-1200.pt")) == "20261008-1200"


class TestRunManifest:
    """Every script of the battery records itself in the report directory's run.json."""

    def test_a_script_run_is_recorded_in_its_report_directory(self, tmp_path, monkeypatch):
        checkpoint = tmp_path / "arm" / "latest.pt"
        argv = ["--checkpoint", str(checkpoint), "--abilities-root", str(tmp_path / "abilities"),
                "--reports-dir", str(tmp_path / "reports")]
        monkeypatch.setattr(sys, "argv", ["build_texts.py", *argv])

        with pytest.raises(SystemExit) as exit_:
            common.main_with_manifest(lambda: None, "scripts/effect_embedding_probes/x.py")

        assert exit_.value.code is None
        out = common.report_dir(checkpoint, tmp_path / "reports")
        manifest = json.loads((out / "run.json").read_text(encoding="utf-8"))
        assert manifest["commands"] == [
            "python scripts/effect_embedding_probes/x.py " + " ".join(argv),
        ]
        assert manifest["status"] == "finished"
        assert out.name == f"embedding-probes-arm-latest-{datetime.date.today():%Y%m%d}"

    def test_output_names_the_report_directory_itself(self, tmp_path, monkeypatch):
        """The layout's report run directory, its manifest and files in it."""
        report = tmp_path / "reports" / "2026-11-20-rule-mask-hard-embedding"
        argv = ["--checkpoint", str(tmp_path / "arm" / "latest.pt"),
                "--abilities-root", str(tmp_path / "abilities"), "--output", str(report)]
        monkeypatch.setattr(sys, "argv", ["build_texts.py", *argv])

        with pytest.raises(SystemExit):
            common.main_with_manifest(lambda: None, "scripts/effect_embedding_probes/x.py")

        assert json.loads((report / "run.json").read_text(encoding="utf-8"))["status"] == (
            "finished")
        args = common.add_probe_arguments(argparse.ArgumentParser()).parse_args(argv)
        assert common.report_dir(args.checkpoint, output=args.output) == report


# ── effect profiles ─────────────────────────────────────────────────────


class TestEffectProfiles:
    """A curated training stratum writes one record up to ``--reuse-cap`` times."""

    @staticmethod
    def _record(record_id: str, game: str = "g1") -> dict:
        return {
            "record_id": record_id, "game_id": game, "kind": "resolution",
            "moment": "resolution", "actor_player": "P0",
            "ability": [{"script_file": "cardsfolder/d/divination.txt", "face": 0,
                         "trait_kind": "spell", "index_within_kind": 0}],
            "state": {"entities": []},
            "payload": {"events": [{"type": "card_drawn", "subjects": ["P0"],
                                    "params": {"count": 2}}]},
        }

    @pytest.mark.parametrize("name", ["shard-00001.jsonl.gz", "shard-00001.jsonl"])
    def test_a_record_written_several_times_is_profiled_once(self, tmp_path, monkeypatch,
                                                             name):
        """Either shard spelling: the corpus keeps plain shards collected before gzip."""
        import gzip
        import json

        profiles = effect_profiles
        monkeypatch.setattr(profiles, "_KEYMAP", {
            ProvenanceKey("cardsfolder/d/divination.txt", 0, "spell", 0): "draw two"})
        shard = tmp_path / name
        opener = gzip.open if name.endswith(".gz") else open
        with opener(shard, "wt", encoding="utf-8") as handle:
            for record in [self._record("r.1")] * 4 + [self._record("r.2", "g2")]:
                handle.write(json.dumps(record) + "\n")

        totals = profiles.ProfileTotals()
        for row in profiles.fold_shard(str(shard)):
            totals.add(*row)
        frame = totals.frame()

        row = frame.set_index("text").loc["draw two"]
        assert row["n_records"] == 2
        assert row["res_n"] == 2
        assert row["n_games"] == 2
        assert row["mean_card_drawn_own"] == 2.0


class TestStoredCorpus:
    """The corpus cache under ``encode_corpus``: read back when current, replaced when stale."""

    @staticmethod
    def _write(cache_file: Path, texts: list[str]) -> None:
        rows = np.ones((len(texts), 3), dtype=np.float32)
        np.savez(cache_file, texts=np.array(texts, dtype=object), plain=rows, forced=rows)

    def test_a_cache_for_other_texts_is_dropped_and_deleted(self, tmp_path):
        from effect_embedding_probes import keyword_expansion

        cache_file = tmp_path / "corpus_encodings.npz"
        self._write(cache_file, ["old text"])
        assert keyword_expansion._stored_corpus(cache_file, [], ["new text"]) is None
        assert not cache_file.exists()

    def test_a_cache_for_the_same_texts_is_read_back(self, tmp_path):
        from effect_embedding_probes import keyword_expansion

        cache_file = tmp_path / "corpus_encodings.npz"
        self._write(cache_file, ["a", "b"])
        corpus = keyword_expansion._stored_corpus(cache_file, [], ["a", "b"])
        assert corpus is not None
        assert corpus.texts == ["a", "b"]
        assert cache_file.exists()


class TestPcaVarianceTable:
    def test_the_participation_ratio_note_is_written_under_the_title(self, tmp_path):
        from effect_embedding_probes import pca_directions

        share = np.array([0.5, 0.3, 0.2])
        paths = SimpleNamespace(out=tmp_path)
        pca_directions.write_variance(
            pca_directions.Decomposition(paths, "full", [1, 2, 3], None, None, share, []))
        text = (tmp_path / "pca_full_variance.md").read_text(encoding="utf-8")
        assert "Participation ratio" in text

"""The embedding-probe scripts read any checkpoint's cache (T114; Story 6
scenarios 9 and 10).

``scripts/`` is not part of the installed package; the conftest puts it on the
path, so the battery imports as the package its scripts make it. The sidecars
are real gen-1 ones from the fixture tree; the cache rows beside them are
written here, row-aligned, at a width no checkpoint has shipped, so nothing can
pass by assuming 64.
"""

from __future__ import annotations

import datetime
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from effect_embedding_probes import build_texts, common, effect_profiles

_FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "effects"

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

    def test_a_reports_dir_moves_the_output_beside_the_arm(self, tree):
        arm = tree.root / "2026-10-gen2-a"
        paths = common.paths_for(
            arm / "latest.pt", tree.abilities, _provenance(),
            today=datetime.date(2026, 10, 8), reports_dir=arm / "reports",
        )
        assert paths.out == (
            arm / "reports" / "embedding-probes-2026-10-gen2-a-latest-20261008"
        )

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
            ("cardsfolder/d/divination.txt", 0, "spell", 0, None): "draw two"})
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

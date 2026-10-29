"""Reading the converted card tree and the MTGJSON printings dump."""

from __future__ import annotations

import json

from effects.infrastructure.card_sources import load_first_printings
from effects.infrastructure.converted_trees import ConvertedTrees


class TestCorpusLoaders:
    def test_card_files_are_keyed_by_the_name_line(self, tmp_path):
        (tmp_path / "s").mkdir()
        (tmp_path / "s" / "serra_angel.txt").write_text(
            "name: Serra Angel\ntypes: creature angel\n", encoding="utf-8",
        )
        assert ConvertedTrees({"cardsfolder": tmp_path}).card_files == {
            "Serra Angel": "cardsfolder/s/serra_angel.txt",
        }

    def test_a_file_without_a_name_line_is_skipped(self, tmp_path):
        (tmp_path / "broken.txt").write_text("types: creature\n", encoding="utf-8")
        assert ConvertedTrees({"cardsfolder": tmp_path}).card_files == {}

    def test_first_printings_take_the_earliest_release(self, tmp_path):
        path = tmp_path / "AllPrintings.json"
        path.write_text(json.dumps({"data": {
            "NEW": {"releaseDate": "2026-01-01", "cards": [{"name": "Shock"}]},
            "OLD": {"releaseDate": "1999-01-01", "cards": [{"name": "Shock"}]},
        }}), encoding="utf-8")
        assert load_first_printings(path)["Shock"] == "1999-01-01"

    def test_face_names_are_indexed_too(self, tmp_path):
        path = tmp_path / "AllPrintings.json"
        path.write_text(json.dumps({"data": {
            "SET": {"releaseDate": "2020-01-01", "cards": [
                {"name": "Front // Back", "faceName": "Front"},
            ]},
        }}), encoding="utf-8")
        printings = load_first_printings(path)
        assert printings["Front // Back"] == "2020-01-01"
        assert printings["Front"] == "2020-01-01"

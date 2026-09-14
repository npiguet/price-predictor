"""The decks file a coverage or variant round hands its workers.

The generated-decks format already read by `GeneratedDecksIndex`, so the worker
needs no new parser. The set code is a sentinel: a coverage deck is drawn from
the whole corpus and belongs to no set, and Task 3's decks-only mode is what
stops anything from trying to resolve it against Forge's set table.
"""

from __future__ import annotations

from effects.infrastructure.deck_file import COVERAGE_SET_CODE, write_deck_file


class TestDeckFile:
    def test_one_line_per_deck_in_the_generated_decks_format(self, tmp_path):
        path = tmp_path / "decks.txt"

        written = write_deck_file(
            [["a", "b"], ["c"]], path, label="coverage", set_code=COVERAGE_SET_CODE,
        )

        assert written == 2
        lines = path.read_text(encoding="utf-8").splitlines()
        assert lines[0] == "coverage;COVERAGE;a|b"
        assert lines[1] == "coverage;COVERAGE;c"

    def test_it_overwrites_rather_than_appends(self, tmp_path):
        """Each round's decks replace the last round's, never join them."""
        path = tmp_path / "decks.txt"
        write_deck_file([["a"]], path, label="coverage", set_code=COVERAGE_SET_CODE)

        write_deck_file([["b"]], path, label="coverage", set_code=COVERAGE_SET_CODE)

        assert path.read_text(encoding="utf-8").splitlines() == ["coverage;COVERAGE;b"]

    def test_no_deck_writes_an_empty_file(self, tmp_path):
        path = tmp_path / "decks.txt"

        assert write_deck_file([], path, label="coverage", set_code=COVERAGE_SET_CODE) == 0
        assert path.read_text(encoding="utf-8") == ""

"""Unit tests for parse_pools()."""

from __future__ import annotations

import pytest

from sealed.infrastructure.pool_file_reader import parse_pools


class TestParsePools:
    def test_parses_single_pool(self, tmp_path):
        f = tmp_path / "pools.txt"
        f.write_text("MH3;Card A|Card B|Card C\n", encoding="utf-8")
        assert parse_pools(f) == [("MH3", ["Card A", "Card B", "Card C"])]

    def test_parses_multiple_pools_with_different_set_codes(self, tmp_path):
        f = tmp_path / "pools.txt"
        f.write_text(
            "MH3;Flare of Denial|Wight of the Reliquary\n"
            "BLB;Moonrise Cleric|Bark-Knuckle Boxer\n",
            encoding="utf-8",
        )
        assert parse_pools(f) == [
            ("MH3", ["Flare of Denial", "Wight of the Reliquary"]),
            ("BLB", ["Moonrise Cleric", "Bark-Knuckle Boxer"]),
        ]

    def test_skips_blank_lines(self, tmp_path):
        f = tmp_path / "pools.txt"
        f.write_text(
            "MH3;Card A|Card B\n"
            "\n"
            "   \n"
            "BLB;Card C\n",
            encoding="utf-8",
        )
        result = parse_pools(f)
        assert len(result) == 2
        assert result[0][0] == "MH3"
        assert result[1][0] == "BLB"

    def test_pool_with_one_card(self, tmp_path):
        f = tmp_path / "pools.txt"
        f.write_text("MH3;OnlyCard\n", encoding="utf-8")
        assert parse_pools(f) == [("MH3", ["OnlyCard"])]

    def test_raises_on_missing_separator(self, tmp_path):
        f = tmp_path / "pools.txt"
        f.write_text("Card A|Card B|Card C\n", encoding="utf-8")
        with pytest.raises(ValueError, match="missing ';'"):
            parse_pools(f)

    def test_empty_file_returns_empty_list(self, tmp_path):
        f = tmp_path / "pools.txt"
        f.write_text("", encoding="utf-8")
        assert parse_pools(f) == []

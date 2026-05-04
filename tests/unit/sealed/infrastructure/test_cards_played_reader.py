"""Unit tests for ``cards_played_reader.iter_rows``."""

from __future__ import annotations

from pathlib import Path

import pytest

from sealed.infrastructure.cards_played_reader import iter_rows


def _line(
    cards_played_a: list[str],
    cards_played_b: list[str],
    cards_not_played_a: list[str],
    cards_not_played_b: list[str],
    winner: str = "A",
    starter: str = "B",
) -> str:
    return ";".join([
        "2026-05-03T14:22:01Z",
        "run-x",
        "BLB",
        "forge-best",
        "forge-3sub",
        "|".join(cards_played_a),
        "|".join(cards_played_b),
        "|".join(cards_not_played_a),
        "|".join(cards_not_played_b),
        winner,
        starter,
    ])


class TestIterRows:
    def test_parses_eleven_field_lines(self, tmp_path: Path):
        text = _line(["Lightning Bolt"], ["Counterspell"], [], [], "A", "B") + "\n"
        path = tmp_path / "cards-played.txt"
        path.write_text(text, encoding="utf-8")
        rows = list(iter_rows(path))
        assert len(rows) == 1
        row = rows[0]
        assert row.run_id == "run-x"
        assert row.set_code == "BLB"
        assert row.cards_played_a == ["Lightning Bolt"]
        assert row.cards_played_b == ["Counterspell"]
        assert row.cards_not_played_a == []
        assert row.cards_not_played_b == []
        assert row.winner == "A"
        assert row.starter == "B"

    def test_pipe_multiplicities_preserved(self, tmp_path: Path):
        text = _line(
            ["LB", "LB", "LB"], ["GB"], ["Fire"], [],
        ) + "\n"
        path = tmp_path / "cards-played.txt"
        path.write_text(text, encoding="utf-8")
        rows = list(iter_rows(path))
        assert rows[0].cards_played_a == ["LB", "LB", "LB"]

    def test_empty_file_yields_no_rows(self, tmp_path: Path):
        path = tmp_path / "cards-played.txt"
        path.write_text("", encoding="utf-8")
        assert list(iter_rows(path)) == []

    def test_trailing_partial_line_tolerated(self, tmp_path: Path):
        # Two complete lines + a third line missing the newline. The third
        # line is intentionally truncated mid-field (mimicking JVM crash).
        complete = _line(["a"], ["b"], [], []) + "\n"
        complete2 = _line(["c"], ["d"], [], []) + "\n"
        partial = "2026-05-03T14:22:01Z;run-y;BLB;forge-best;forge-3sub;a"
        path = tmp_path / "cards-played.txt"
        path.write_text(complete + complete2 + partial, encoding="utf-8")
        rows = list(iter_rows(path))
        assert len(rows) == 2

    def test_mid_file_malformed_line_raises(self, tmp_path: Path):
        good = _line(["a"], ["b"], [], []) + "\n"
        bad = "not;enough;fields\n"
        good2 = _line(["c"], ["d"], [], []) + "\n"
        path = tmp_path / "cards-played.txt"
        path.write_text(good + bad + good2, encoding="utf-8")
        with pytest.raises(ValueError, match="Expected 11"):
            list(iter_rows(path))

    def test_invalid_winner_char_raises(self, tmp_path: Path):
        text = _line(["a"], ["b"], [], [], winner="X") + "\n"
        path = tmp_path / "cards-played.txt"
        path.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError, match="winner"):
            list(iter_rows(path))

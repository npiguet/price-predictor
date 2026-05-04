"""Unit tests for the ``cards-win-rates.txt`` writer (FR-013a)."""

from __future__ import annotations

from pathlib import Path

from sealed.application.train_encoder import CardLabel, _write_win_rates


def _label(name: str, wp: int, wd: int, shrunk: float) -> CardLabel:
    return CardLabel(
        card_name=name,
        wins_when_played=wp,
        wins_when_in_deck=wd,
        shrunk_label=shrunk,
    )


class TestWinRatesWriter:
    def test_header_present(self, tmp_path: Path):
        labels = {"LB": _label("LB", 5, 5, 0.95)}
        path = tmp_path / "cards-win-rates.txt"
        _write_win_rates(labels, path)
        first_line = path.read_text(encoding="utf-8").splitlines()[0]
        assert first_line == (
            "card_name;wins_when_played;wins_when_in_deck;raw_ratio;shrunk_label"
        )

    def test_sorted_by_raw_ratio_desc(self, tmp_path: Path):
        labels = {
            "Mid": _label("Mid", 5, 10, 0.5),
            "Top": _label("Top", 9, 10, 0.9),
            "Bot": _label("Bot", 1, 10, 0.1),
        }
        path = tmp_path / "cards-win-rates.txt"
        _write_win_rates(labels, path)
        lines = path.read_text(encoding="utf-8").splitlines()
        # header + 3 rows
        assert lines[1].startswith("Top;")
        assert lines[2].startswith("Mid;")
        assert lines[3].startswith("Bot;")

    def test_five_decimal_floats(self, tmp_path: Path):
        labels = {"LB": _label("LB", 1, 3, 0.4)}
        path = tmp_path / "cards-win-rates.txt"
        _write_win_rates(labels, path)
        row = path.read_text(encoding="utf-8").splitlines()[1]
        # 1/3 = 0.33333..., shrunk = 0.4 → "LB;1;3;0.33333;0.40000"
        assert row == "LB;1;3;0.33333;0.40000"

    def test_overwrites_existing_file(self, tmp_path: Path):
        path = tmp_path / "cards-win-rates.txt"
        path.write_text("stale\nstale\nstale\n", encoding="utf-8")
        _write_win_rates({"X": _label("X", 1, 1, 1.0)}, path)
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2  # header + one row
        assert "stale" not in lines

    def test_parent_directory_auto_created(self, tmp_path: Path):
        path = tmp_path / "nested" / "cards-win-rates.txt"
        _write_win_rates({"X": _label("X", 1, 1, 1.0)}, path)
        assert path.exists()

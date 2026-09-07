"""Tests for the shared append-only line-completeness rules."""

from __future__ import annotations

from pathlib import Path

from price_predictor.infrastructure.append_only import (
    count_complete_lines,
    count_complete_lines_and_truncate_partial,
    iter_complete_lines,
)


def _write(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "corpus.txt"
    path.write_bytes(content.encode("utf-8"))
    return path


class TestIterCompleteLines:
    def test_yields_every_newline_terminated_line(self, tmp_path):
        path = _write(tmp_path, "a\nb\nc\n")
        assert list(iter_complete_lines(path)) == ["a", "b", "c"]

    def test_drops_a_trailing_partial_line(self, tmp_path):
        path = _write(tmp_path, "a\nb\npartia")
        assert list(iter_complete_lines(path)) == ["a", "b"]

    def test_empty_file_yields_nothing(self, tmp_path):
        path = _write(tmp_path, "")
        assert list(iter_complete_lines(path)) == []

    def test_missing_file_yields_nothing(self, tmp_path):
        assert list(iter_complete_lines(tmp_path / "absent.txt")) == []

    def test_single_partial_line_yields_nothing(self, tmp_path):
        path = _write(tmp_path, "no newline here")
        assert list(iter_complete_lines(path)) == []

    def test_blank_lines_survive_as_empty_strings(self, tmp_path):
        path = _write(tmp_path, "a\n\nb\n")
        assert list(iter_complete_lines(path)) == ["a", "", "b"]

    def test_crlf_terminators_are_stripped(self, tmp_path):
        path = _write(tmp_path, "a\r\nb\r\n")
        assert list(iter_complete_lines(path)) == ["a", "b"]

    def test_does_not_modify_the_file(self, tmp_path):
        path = _write(tmp_path, "a\nb\npartia")
        list(iter_complete_lines(path))
        assert path.read_bytes() == b"a\nb\npartia"


class TestCountCompleteLines:
    def test_counts_lines_ending_exactly_on_a_newline(self, tmp_path):
        assert count_complete_lines(_write(tmp_path, "a\nb\nc\n")) == 3

    def test_excludes_a_trailing_partial(self, tmp_path):
        assert count_complete_lines(_write(tmp_path, "a\nb\npartia")) == 2

    def test_empty_and_missing_files_count_zero(self, tmp_path):
        assert count_complete_lines(_write(tmp_path, "")) == 0
        assert count_complete_lines(tmp_path / "absent.txt") == 0

    def test_blank_lines_count_by_default(self, tmp_path):
        assert count_complete_lines(_write(tmp_path, "a\n\nb\n")) == 3

    def test_skip_blank_excludes_whitespace_only_lines(self, tmp_path):
        path = _write(tmp_path, "a\n\n   \nb\n")
        assert count_complete_lines(path, skip_blank=True) == 2


class TestCountAndTruncate:
    def test_file_ending_on_a_newline_is_left_alone(self, tmp_path):
        path = _write(tmp_path, "a\nb\n")
        assert count_complete_lines_and_truncate_partial(path) == 2
        assert path.read_bytes() == b"a\nb\n"

    def test_trailing_partial_is_truncated_away(self, tmp_path):
        path = _write(tmp_path, "a\nb\npartia")
        assert count_complete_lines_and_truncate_partial(path) == 2
        assert path.read_bytes() == b"a\nb\n"

    def test_single_partial_line_empties_the_file(self, tmp_path):
        path = _write(tmp_path, "no newline here")
        assert count_complete_lines_and_truncate_partial(path) == 0
        assert path.read_bytes() == b""

    def test_empty_and_missing_files_count_zero(self, tmp_path):
        assert count_complete_lines_and_truncate_partial(_write(tmp_path, "")) == 0
        assert count_complete_lines_and_truncate_partial(tmp_path / "absent.txt") == 0

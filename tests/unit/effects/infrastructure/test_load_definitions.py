"""``load_definitions``: a table is a file; ``None`` is no table, a missing path is refused."""

from __future__ import annotations

import pytest

from effects.infrastructure.tokenizer_loading import load_definitions
from tests.unit.effects.builders import FIXTURE_KEYWORD_DEFINITIONS


def test_a_real_file_is_read() -> None:
    assert load_definitions(FIXTURE_KEYWORD_DEFINITIONS)


def test_none_is_no_table() -> None:
    assert load_definitions(None) == {}


def test_a_path_with_no_file_is_refused_naming_it_rather_than_read_as_empty(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="absent.json"):
        load_definitions(tmp_path / "absent.json")


def test_a_directory_is_refused_naming_its_path(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match=str(tmp_path).replace("\\", "\\\\")):
        load_definitions(tmp_path)

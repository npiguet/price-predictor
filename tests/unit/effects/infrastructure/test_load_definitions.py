"""``load_definitions``: a table is a file; absent or ``None`` is no table."""

from __future__ import annotations

import pytest

from effects.infrastructure.tokenizer_loading import load_definitions
from tests.unit.effects.builders import FIXTURE_KEYWORD_DEFINITIONS


def test_a_real_file_is_read() -> None:
    assert load_definitions(FIXTURE_KEYWORD_DEFINITIONS)


def test_an_absent_file_and_none_are_no_table(tmp_path) -> None:
    assert load_definitions(tmp_path / "absent.json") == {}
    assert load_definitions(None) == {}


def test_a_directory_is_refused_naming_its_path(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match=str(tmp_path).replace("\\", "\\\\")):
        load_definitions(tmp_path)

"""A multi-faced script is never varied, whichever section marker it uses, and a rename rewrites
only the first ``Name:`` line."""

from __future__ import annotations

from pathlib import Path

import pytest

from effects.domain.script_variants import is_multi_face, renamed_script

_FORGE_CARDS = Path(__file__).resolve().parents[4].parent / (
    "forge/forge-gui/res/cardsfolder"
)


def _real_script(relative: str) -> list[str]:
    path = _FORGE_CARDS / relative
    if not path.is_file():
        pytest.skip(f"no Forge checkout at {_FORGE_CARDS}")
    return path.read_text(encoding="utf-8").splitlines()


def test_a_specialize_script_is_multi_face():
    # Six faces under SPECIALIZE: sections, with no ALTERNATE line.
    lines = _real_script("a/alora_rogue_companion.txt")
    assert not any(line.strip().startswith("ALTERNATE") for line in lines)
    assert is_multi_face(lines)


def test_an_alternate_script_is_multi_face():
    assert is_multi_face(["Name:A", "ALTERNATE", "Name:B"])


def test_a_single_faced_script_is_not_multi_face():
    assert not is_multi_face(["Name:Serra Angel", "ManaCost:3 W W"])


def test_renaming_rewrites_only_the_first_name_line():
    lines = ["Name:Front", "ManaCost:1", "ALTERNATE", "Name:Back"]
    assert renamed_script(lines, "Front Variant 1") == [
        "Name:Front Variant 1", "ManaCost:1", "ALTERNATE", "Name:Back",
    ]

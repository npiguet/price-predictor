"""Which keyword a sidecar line is, and the line and trait vocabularies."""

from __future__ import annotations

from pathlib import Path

import pytest

from effects.domain.provenance import LineKind, SidecarLine, TraitKind
from effects.infrastructure.sidecar_io import iter_sidecars

_FIXTURES = Path(__file__).parents[3] / "fixtures" / "effects"


def _line(text: str | None, api: str | None = "Keyword") -> SidecarLine:
    return SidecarLine(
        line_index=0, line_kind="static", provenance=(),
        script_api_type=api, script_text=text,
    )


@pytest.mark.parametrize(("text", "display", "bare"), [
    ("First Strike", "First Strike", "first_strike"),
    ("Ward:2", "Ward", "ward"),
    ("Protection from red", "Protection from red", "protection_from_red"),
    # A keyword whose script carries a parameter names its keyword, and is
    # not the keyword alone.
    ("Equip:2:::ReduceCost$ Y:This ability costs {2} less", "Equip", None),
    ("Class:2:1 U:AddTrigger$ TriggerClassLevel", "Class", None),
])
def test_a_keyword_line_names_its_keyword_and_is_it_only_when_bare(text, display, bare):
    line = _line(text)
    assert line.keyword_display_name == display
    assert line.bare_keyword == bare


def test_a_non_keyword_line_names_no_keyword():
    line = _line("AB$ DealDamage | Cost$ T | NumDmg$ 1", api="DealDamage")
    assert line.keyword_display_name is None
    assert line.bare_keyword is None


def test_a_keyword_line_with_no_text_names_no_keyword():
    assert _line(None).keyword_display_name is None
    assert _line("").bare_keyword is None


@pytest.mark.parametrize("tree", ["gen1-sidecars", "gen2-sidecars"])
def test_every_fixture_sidecar_spells_its_kinds_from_the_vocabularies(tree):
    """The enums mirror the converter's ``AbilityType`` and ``TraitKind``, so
    a line or key of a kind they do not list is a converter they predate."""
    lines = keys = 0
    for _, sidecar in iter_sidecars(_FIXTURES / tree):
        for line in sidecar.lines:
            lines += 1
            assert line.line_kind in set(LineKind)
            for key in line.provenance:
                keys += 1
                assert key.trait_kind in set(TraitKind)
        for key in sidecar.dropped_keys:
            assert key.trait_kind in set(TraitKind)
    assert lines and keys

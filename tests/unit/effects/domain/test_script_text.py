"""The script-text grammar every reader of a sidecar's ``script_text`` shares.

The equivalence test pins the parser rule families used to read ``Mode$`` and
``Event$`` with: a family that moved would re-place records a corpus already
placed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from effects.domain.script_text import (
    SEG,
    api_of,
    is_bare_keyword,
    params,
    root,
    root_param,
    segments,
)

_FIXTURES = Path(__file__).parents[3] / "fixtures" / "effects"


def _fixture_script_texts() -> list[str]:
    texts: list[str] = []
    for path in sorted(_FIXTURES.glob("gen*-sidecars/**/*.provenance.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        texts += [line["script_text"] for line in data["lines"] if line.get("script_text")]
    return texts


def _former_root_param(script_text: str | None, name: str) -> str | None:
    if not script_text:
        return None
    for part in script_text.split("[SEG]", 1)[0].split("|"):
        key, dollar, value = part.partition("$")
        if dollar and key.strip() == name:
            return value.strip() or None
    return None


def test_segments_drop_their_labels_and_params_split_at_the_first_dollar():
    parts = segments("SP$ Draw | SubAbility$ SV1 [SEG] SV1: DB$ Pump | NumAtt$ +1")
    assert parts == ["SP$ Draw | SubAbility$ SV1", "DB$ Pump | NumAtt$ +1"]
    assert params("NumDmg$ Count$Valid Creature | SP$ DealDamage") == {
        "NumDmg": "Count$Valid Creature", "SP": "DealDamage",
    }


def test_params_keep_the_first_occurrence_and_an_empty_value():
    assert params("Mode$ A | Mode$ B | Flag$ | no dollar here") == {"Mode": "A", "Flag": ""}


def test_the_root_is_the_text_before_the_first_separator_as_written():
    assert root(f"SV1: DB$ Pump {SEG} SV2: DB$ Draw") == "SV1: DB$ Pump "
    assert root("Flying") == "Flying"


def test_root_param_reads_the_root_segment_only():
    text = "Mode$ ChangesZone | Execute$ SV1 [SEG] SV1: DB$ Draw | Mode$ Other"
    assert root_param(text, "Mode") == "ChangesZone"
    assert root_param(text, "Event") is None
    assert root_param(None, "Mode") is None
    assert root_param("Mode$  | Event$ Moved", "Mode") is None


def test_the_api_is_whichever_api_key_names_one():
    assert api_of(params("NumDmg$ 3 | SP$ DealDamage")) == "DealDamage"
    assert api_of(params("DB$ Draw")) == "Draw"
    assert api_of(params("AB$ | Cost$ T")) is None
    assert api_of(params("Mode$ Continuous")) is None


@pytest.mark.parametrize(("text", "bare"), [
    ("First Strike", True),
    ("Protection from black", True),
    ("Mode$ Continuous | Affected$ Creature", False),
    ("SP$ Draw", False),
    ("a | b", False),
    ("", False),
    (None, False),
])
def test_a_bare_keyword_has_no_parameter(text, bare):
    assert is_bare_keyword(text) is bare


def test_root_param_reads_every_fixture_line_as_its_former_parser_did():
    texts = _fixture_script_texts()
    assert texts
    for text in texts:
        for name in ("Mode", "Event", "SP", "AB", "DB", "Cost", "Execute"):
            assert root_param(text, name) == _former_root_param(text, name), (text, name)

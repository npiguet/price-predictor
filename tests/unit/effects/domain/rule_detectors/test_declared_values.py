"""A declaration's free-text values are checked against their closed lists.

``DetectorIndex.of`` refuses a misspelt value, naming the scenario, and the
lists Forge owns are pinned to ``../forge``'s sources and the connector's.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from effects.domain.rule_detectors import DeclarationError, DetectorIndex
from effects.domain.rule_detectors import declared_values as dv
from tests.unit.effects.domain.rule_detectors.builders import catalogue, declaration, scenario

_ROOT = Path(__file__).resolve().parents[5]
_GAME = _ROOT.parent / "forge" / "forge-game" / "src" / "main" / "java" / "forge" / "game"
_RES = _ROOT.parent / "forge" / "forge-gui" / "res"
_CONNECTOR = (_ROOT / "forge-connector" / "src" / "main" / "java" / "com" / "pricepredictor"
              / "connector" / "effects")


@pytest.mark.parametrize(("family", "kind", "params"), [
    ("outcome_event", "resolution", {"event": "zone_change", "to_zone": "batlefield"}),
    ("outcome_event", "resolution", {"event": "zone_change", "api": "Destory"}),
    ("outcome_event", "resolution", {"event": "zone_change", "source_type": "creture"}),
    ("outcome_event", "resolution", {"event": "zone_change", "subject_face_state": "facedown"}),
    ("outcome_event", "resolution", {"event": "counter_change", "counter_type": "P9P9"}),
    ("outcome_event", "resolution", {"event": "zone_change", "keyword": "Flyng"}),
    ("outcome_event", "resolution", {"event": "zone_change", "min_count": "two"}),
    ("trigger_mode", "trigger", {"mode": "ChangeZone"}),
    ("trigger_mode", "trigger", {"mode": "ChangesZone", "destination": "Batlefield"}),
    ("trigger_mode", "trigger", {"keyword": "Undyng"}),
    ("verdict_bit", "playability", {"bit": "timing", "source_type": "creture"}),
    ("verdict_bit", "playability", {"bit": "timing", "keyword": "Flashbak"}),
    ("verdict_bit", "playability", {"bit": "timing", "source_designation": "ploted"}),
    ("verdict_bit", "playability", {"bit": "timing", "cost_symbol": "phyrexan"}),
    ("verdict_bit", "playability", {"bit": "timing", "fires_when": "yes"}),
    ("legality_membership", "playability", {"list": "legal_blockers", "keyword": "Decayd"}),
    ("legality_membership", "playability", {"list": "legal_blockers", "designation": "suspeted"}),
    ("legality_membership", "playability", {"list": "legal_attackers", "judged_type": "batle"}),
    ("legality_membership", "playability", {"list": "legal_attackers", "min_types": "two"}),
    ("continuous_contribution", "continuous", {"field": "types", "value": "Creture"}),
    ("continuous_contribution", "continuous", {"field": "keywords", "value": "flyng"}),
    ("rewrite_event", "rewrite", {"event": "zone_change", "to_zone": "grave"}),
    ("rewrite_event", "rewrite", {"event": "zone_change", "mode": "Movd"}),
])
def test_a_misspelt_value_is_refused_naming_its_scenario(family, kind, params):
    """V5 #22: a value no closed list holds stops the index, not a record mid-run."""
    bad = scenario("999.1-1", ("999.1",), kind, declaration(family, kind, **params))
    with pytest.raises(DeclarationError, match="999.1-1"):
        DetectorIndex.of(catalogue(bad))


def _enum(path: Path, enum: str) -> set[str]:
    if not path.exists():
        pytest.skip(f"no Forge checkout at {path}")
    source = re.sub(r"//[^\n]*|/\*.*?\*/", "", path.read_text(encoding="utf-8"), flags=re.S)
    body = source.split(f"enum {enum}", 1)[1].split("{", 1)[1].split(";", 1)[0]
    body = re.sub(r"\((?:[^()]|\([^()]*\))*\)", "", body)
    return {name.strip() for name in body.split(",") if name.strip()}


def test_forge_s_names_are_its_enums():
    assert dv.FORGE_API_TYPES == _enum(_GAME / "ability" / "ApiType.java", "ApiType")
    assert dv.FORGE_TRIGGER_MODES == _enum(_GAME / "trigger" / "TriggerType.java", "TriggerType")
    assert dv.FORGE_REPLACEMENT_MODES == _enum(_GAME / "replacement" / "ReplacementType.java",
                                               "ReplacementType")
    keywords = (_GAME / "keyword" / "Keyword.java").read_text(encoding="utf-8")
    assert dv.FORGE_KEYWORDS == set(re.findall(r'^\s*[A-Z_0-9]+\("([^"]+)"', keywords, re.M))


def test_each_keyword_line_beyond_the_enum_is_printed_by_some_script():
    folder = _RES / "cardsfolder"
    if not folder.exists():
        pytest.skip(f"no Forge checkout at {folder}")
    printed: set[str] = set()
    for path in folder.rglob("*.txt"):
        text = path.read_text(encoding="utf-8", errors="replace")
        printed |= {name for name in dv.KEYWORD_LINES_BEYOND_ENUM if f"\nK:{name}" in text}
        if printed == dv.KEYWORD_LINES_BEYOND_ENUM:
            break
    assert printed == dv.KEYWORD_LINES_BEYOND_ENUM


def test_the_role_kinds_are_forge_s_role_tokens():
    folder = _RES / "tokenscripts"
    if not folder.exists():
        pytest.skip(f"no Forge checkout at {folder}")
    assert dv.ROLE_KINDS == {p.stem.removeprefix("role_") for p in folder.glob("role_*.txt")}


def test_the_designations_are_the_connector_s():
    path = _CONNECTOR / "CardFacts.java"
    if not path.exists():
        pytest.skip(f"no connector source at {path}")
    source = path.read_text(encoding="utf-8")
    assert dv.DESIGNATIONS == set(re.findall(r'table\.put\("([a-z_]+)"', source))

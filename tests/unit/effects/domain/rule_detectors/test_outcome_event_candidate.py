"""``outcome_event``'s ``source_keyword`` speaks for a card's cast, and a qualified near miss
needs an event it qualifies (CPC1 D-1, findings 7.1 and 7.3, task FR1).

Real fixture records: Kitesail's Equip activation, which attaches, and the same record
edited to be an Aura's cast (Sun Clasp: Enchant creature).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from effects.domain.provenance import ProvenanceKey
from effects.domain.rule_catalogue import Polarity
from effects.domain.rule_detectors import families
from effects.infrastructure.rules_catalogue_io import load_scenarios

RULES = Path(__file__).parents[5] / "rules"
SUN_CLASP_CAST = ProvenanceKey("cardsfolder/s/sun_clasp.txt", 0, "spell", 0)


def _params(sid: str) -> dict:
    found = next(s.detector for s in load_scenarios(RULES).scenarios if s.id == sid)
    return found.params


def _match(sid: str, record, lines):
    return families()["outcome_event"].match(record, lines, _params(sid))


@pytest.fixture(scope="module")
def equip_activation(gen3_records, gen3_sidecars):
    """An Equip activation's resolution holding an ``attached`` event."""
    return next(r for r in gen3_records if r.kind.value == "resolution"
                and r.ability and r.ability[0].script_file.endswith("kitesail.txt")
                and any(e.type.value == "attached" for e in r.payload.listed_events()))


def test_an_equip_activation_is_no_near_miss_of_an_equipment_entering_unattached(
        equip_activation, gen3_sidecars):
    assert _match("301.5b-1", equip_activation, gen3_sidecars) is None


def test_an_aura_s_cast_resolution_is_still_608_3c(equip_activation, gen3_sidecars):
    record = replace(equip_activation, ability=(SUN_CLASP_CAST,))
    found = _match("608.3c-1", record, gen3_sidecars)
    assert found is not None
    assert found.polarity in (Polarity.FIRES, Polarity.DOES_NOT_FIRE)


def test_an_activated_ability_of_an_aura_is_not_608_3c(equip_activation, gen3_sidecars):
    activated = ProvenanceKey("cardsfolder/s/sun_clasp.txt", 0, "spell", 1)
    assert _match("608.3c-1", replace(equip_activation, ability=(activated,)),
                  gen3_sidecars) is None


def test_a_qualified_near_miss_needs_a_wanted_event(equip_activation, gen3_sidecars):
    """The Equip activation read as the cast, its ``zone_change`` events removed:
    no Equipment entered, so the rule is not shown on either side."""
    record = replace(equip_activation, ability=(SUN_CLASP_CAST,))
    bare = replace(record, payload=replace(record.payload, events=tuple(
        e for e in record.payload.events if e.type.value != "zone_change")))
    params = {"event": "zone_change", "to_zone": "battlefield", "source_keyword": "Enchant",
              "without_event": "attached"}
    assert families()["outcome_event"].match(bare, gen3_sidecars, params) is None


def test_the_cast_rules_declare_the_candidate_they_speak_for():
    """Every ``source_keyword`` declaration but the damage rules of wither and infect."""
    for scenario in load_scenarios(RULES).scenarios:
        found = scenario.detector
        params = getattr(found, "params", {})
        if getattr(found, "family", None) != "outcome_event" or "source_keyword" not in params:
            continue
        expected = None if params["source_keyword"] in {"Wither", "Infect"} else "cast"
        assert params.get("candidate") == expected, scenario.id

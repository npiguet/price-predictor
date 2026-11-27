"""Declarations that tagged the wrong records (7.3, 7.6 / 603.6d).

A cast rule's ``source_keyword`` is the card's, so an Aura's activated or granted
ability was taken for its cast; 306.6 fired on every attacker; 701.5a on every
resolved half. Real fixture records, the candidates' abilities edited to name an
Aura's cast and its activated ability (Sun Clasp: Enchant creature, "{W}: ...").
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from effects.domain.provenance import ProvenanceKey
from effects.domain.records import PlayabilityDecisionPayload
from effects.domain.rule_catalogue import Polarity
from effects.domain.rule_detectors import DeclarationError, DetectorIndex, families
from effects.infrastructure.rules_catalogue_io import load_scenarios
from tests.unit.effects.domain.rule_detectors.builders import catalogue, declaration, scenario

RULES = Path(__file__).parents[5] / "rules"
SUN_CLASP = "cardsfolder/s/sun_clasp.txt"
CAST = ProvenanceKey(SUN_CLASP, 0, "spell", 0)
ACTIVATED = ProvenanceKey(SUN_CLASP, 0, "spell", 1)


def _declared(sid: str):
    return next(s.detector for s in load_scenarios(RULES).scenarios if s.id == sid)


def _decision(by_kind, ability: ProvenanceKey):
    """A real decision record whose one targeting candidate is ``ability``."""
    record = next(r for r in by_kind("playability", "decision")
                  if r.payload.candidates[0].legal_targets)
    candidate = replace(record.payload.candidates[0], ability=(ability,), land_play=False,
                        has_legal_target=True)
    return replace(record, payload=PlayabilityDecisionPayload(candidates=(candidate,)))


def _match(sid: str, record, lines):
    found = _declared(sid)
    return families()[found.family].match(record, lines, found.params)


@pytest.mark.parametrize("sid", ["303.4a-1", "702.5a-2"])
def test_an_auras_activated_ability_is_not_its_cast(by_kind, gen3_sidecars, sid):
    activated = _match(sid, _decision(by_kind, ACTIVATED), gen3_sidecars)
    cast = _match(sid, _decision(by_kind, CAST), gen3_sidecars)
    assert activated is None, "an activated ability read as the Aura's cast"
    assert cast is not None


@pytest.mark.parametrize("sid", ["702.5a-1", "702.5d-1"])
def test_the_enchant_restriction_judges_the_cast_alone(by_kind, gen3_sidecars, sid):
    """Archery Training's granted ability tagged 46 permanents "outside by the
    enchant restriction": an activated candidate judges none."""
    record = _decision(by_kind, ACTIVATED)
    found = _match(sid, record, gen3_sidecars)
    assert found is None or not found.entities, found


def test_the_cast_rules_declare_the_candidate_they_speak_for():
    for scenario_ in load_scenarios(RULES).scenarios:
        found = scenario_.detector
        params = getattr(found, "params", {})
        if "source_keyword" in params and found.family in {"verdict_bit", "legality_membership"}:
            assert params.get("candidate") in {"cast", "activated"}, scenario_.id


@pytest.mark.parametrize("family, params", [
    ("verdict_bit", {"bit": "affordable", "source_keyword": "Convoke"}),
    ("legality_membership", {"list": "legal_targets", "restriction": "selector",
                             "source_keyword": "Enchant"}),
])
def test_a_source_keyword_without_a_candidate_is_refused(family, params):
    index = catalogue(scenario("999.1-1", ("999.1",), "playability",
                               declaration(family, "playability", **params)))
    with pytest.raises(DeclarationError, match="candidate"):
        DetectorIndex.of(index)


def test_no_planeswalker_to_attack_is_not_306_6(by_kind, gen3_sidecars):
    records = by_kind("playability", "attackers")
    bare = [r for r in records if not any(
        "planeswalker" in {t.lower() for t in e.types} for e in r.state.entities)]
    assert bare
    assert all(_match("306.6-1", r, gen3_sidecars) is None
               or _match("306.6-1", r, gen3_sidecars).polarity is not Polarity.FIRES
               for r in bare)


def test_a_resolving_trigger_is_not_a_cast_spell_resolving(by_kind, gen3_sidecars):
    resolved = [r for r in by_kind("resolution") if r.moment and r.moment.value == "resolution"]
    triggers = [r for r in resolved if r.ability and r.ability[0].trait_kind == "trigger"
                and _match("701.5a-1", r, gen3_sidecars)]
    assert not triggers, "a trigger's resolution tagged as a cast spell's"
    assert any(_match("701.5a-1", r, gen3_sidecars) for r in resolved)


def test_603_6d_is_read_on_rewrite_records():
    found = _declared("603.6d-1")
    assert found.record_kind == "rewrite"

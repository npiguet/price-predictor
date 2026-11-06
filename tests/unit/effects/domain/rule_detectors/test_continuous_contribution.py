"""``continuous_contribution`` on real gen-2 continuous records (T029)."""

from __future__ import annotations

from dataclasses import replace

from effects.domain.records import ContinuousPayload
from effects.domain.rule_catalogue import Polarity
from effects.domain.rule_detectors.acting_line import acting_line, source_entities
from effects.domain.rule_detectors.continuous_contribution import (
    FAMILY,
    condition_counter_types,
    conditioned_on_counters,
)


def _trample_grant(by_kind):
    return next(r for r in by_kind("continuous")
                if any("trample" in c.keywords for c in r.payload.contributions))


def test_a_static_granting_the_keyword_fires(by_kind):
    record = _trample_grant(by_kind)
    found = FAMILY.match(record, None, {"field": "keywords", "value": "trample"})
    assert found.polarity is Polarity.FIRES
    # The contributions carrying trample are the outcome; the host the condition.
    assert found.outcome == tuple(dict.fromkeys(
        c.entity for c in record.payload.contributions if "trample" in c.keywords))
    assert found.condition == tuple(
        e.id for e in source_entities(record.state, record.ability))


def test_the_same_static_edited_to_grant_nothing_is_the_near_miss(by_kind):
    record = _trample_grant(by_kind)
    stripped = ContinuousPayload(contributions=tuple(
        replace(c, keywords=()) for c in record.payload.contributions))
    found = FAMILY.match(replace(record, payload=stripped), None,
                         {"field": "keywords", "value": "trample"})
    assert found.polarity is Polarity.DOES_NOT_FIRE and not found.observed


def test_a_pt_boost_fires_the_pt_field(by_kind):
    record = next(r for r in by_kind("continuous")
                  if any(c.pt_boost != (0, 0) for c in r.payload.contributions))
    assert FAMILY.match(record, None, {"field": "pt"}).polarity is Polarity.FIRES


def test_a_tokens_static_shows_by_the_tokens_kind(by_kind):
    """A continuous record names no source: its host is the token the static's
    script file made."""
    record = next(r for r in by_kind("continuous")
                  if r.ability[0].script_file.startswith("tokenscripts/"))
    hosts = source_entities(record.state, record.ability)
    assert hosts and all(h.token_script_id for h in hosts)
    kind = (hosts[0].subtypes or hosts[0].name.split())[0]
    assert FAMILY.match(record, None, {"field": "pt", "source_token_kind": kind.lower()})
    assert FAMILY.match(record, None, {"field": "pt", "source_token_kind": "Nonesuch"}) is None
    card = next(r for r in by_kind("continuous")
                if r.ability[0].script_file.startswith("cardsfolder/"))
    assert FAMILY.match(card, None, {"field": "pt", "source_token_kind": kind}) is None


def test_a_roles_kind_is_its_name(by_kind):
    """Every Role is an Aura Role: a Monster Role is told apart by its name alone."""
    record, host = next(
        (r, h) for r in by_kind("continuous")
        for h in source_entities(r.state, r.ability) if "role" in {s.lower() for s in h.subtypes})
    kind = next(word for word in host.name.split() if word.lower() != "role")
    assert FAMILY.match(record, None, {"field": "pt", "source_token_kind": kind}) is not None
    assert FAMILY.match(record, None, {"field": "pt", "source_token_kind": "Nonesuch"}) is None


def test_a_counter_conditioned_static_shows_by_its_counter_type(by_kind, gen2_sidecars):
    def counter_type(record):
        line = acting_line(record, gen2_sidecars)
        return next(iter(sorted(condition_counter_types(line.script_text if line else None))),
                    None)
    record = next(r for r in by_kind("continuous") if counter_type(r))
    declared = {"field": "pt", "counter_condition": counter_type(record)}
    assert FAMILY.match(record, gen2_sidecars, declared) is not None
    assert FAMILY.match(record, None, declared) is None
    assert FAMILY.match(record, gen2_sidecars,
                        {"field": "pt", "counter_condition": "NONESUCH"}) is None


def test_a_counter_condition_compares_in_one_spelling():
    selector = "Mode$ Continuous | IsPresent$ Card.Self+counters_GE2_LEVEL+counters_LE6_LEVEL"
    assert conditioned_on_counters(selector, "level")
    assert conditioned_on_counters("IsPresent$ Card.Self+counters_GE1_P1P1", "+1/+1")
    assert conditioned_on_counters("CheckSVar$ X [SEG] SV1: Count$CardCounters.LEVEL",
                                   "LEVEL") is False  # a later segment is not the static's
    assert not conditioned_on_counters(selector, "CHARGE")
    # What the static applies to is no condition on whether it applies.
    assert not conditioned_on_counters("Affected$ Creature.YouCtrl+counters_GE1_P1P1", "P1P1")


def test_another_keyword_is_the_near_miss(by_kind):
    found = FAMILY.match(_trample_grant(by_kind), None, {"field": "keywords", "value": "flying"})
    assert found.polarity is Polarity.DOES_NOT_FIRE


def test_a_type_is_compared_without_case(by_kind):
    """The record spells types lower-case; a declaration writes ``Creature``."""
    record = next((r for r in by_kind("continuous")
                   if any(t.lstrip("-") for c in r.payload.contributions for t in c.types)),
                  None)
    if record is None:
        import pytest

        pytest.skip("no type-changing static in the fixture")
    written = next(t for c in record.payload.contributions for t in c.types if t.lstrip("-"))
    for spelled in (written, written.upper(), written.capitalize()):
        found = FAMILY.match(record, None, {"field": "types", "value": spelled})
        assert found.polarity is Polarity.FIRES, spelled


class TestT030Parameters:
    """Source keyword, script parameter, face state and designation (T030 follow-up)."""

    @staticmethod
    def _carriers_as(record, **changes):
        """``record`` with every contribution's entity changed, as gen-3 entities."""
        carrying = {c.entity for c in record.payload.contributions}
        return replace(record, state=replace(record.state, entities=tuple(
            replace(e, lines=(), **changes) if e.id in carrying else e
            for e in record.state.entities)))

    def test_a_script_parameter_and_a_source_keyword_read_the_static(
            self, by_kind, gen2_sidecars):
        record = _trample_grant(by_kind)
        line = acting_line(record, gen2_sidecars)
        key = next(k for k in ("Mode", "Affected", "AddKeyword") if f"{k}$" in line.script_text)
        declared = {"field": "keywords", "value": "trample"}
        assert FAMILY.match(record, gen2_sidecars, {**declared, "script_param": key})
        assert FAMILY.match(record, gen2_sidecars,
                            {**declared, "script_param": "NoSuchKey"}) is None
        assert FAMILY.match(record, gen2_sidecars,
                            {**declared, "source_keyword": "No Such Keyword"}) is None

    def test_a_face_state_narrows_the_contributions(self, by_kind):
        record = _trample_grant(by_kind)
        declared = {"field": "keywords", "value": "trample", "affected_face_state": "flipped"}
        flipped = self._carriers_as(record, face_state="flipped")
        assert FAMILY.match(flipped, None, declared).polarity is Polarity.FIRES
        assert FAMILY.match(self._carriers_as(record, face_state="front"), None, declared) is None
        assert FAMILY.needs(declared) == {"face_state"}

    def test_a_designation_counts_where_it_explains_the_side(self, by_kind):
        record = _trample_grant(by_kind)
        record = replace(record, payload=ContinuousPayload(contributions=tuple(
            c for c in record.payload.contributions if "trample" in c.keywords)))
        declared = {"field": "keywords", "value": "trample", "designation": "suspected"}
        held = self._carriers_as(record, designations=("suspected",))
        found = FAMILY.match(held, None, declared)
        assert found.polarity is Polarity.FIRES
        stripped = replace(record, payload=ContinuousPayload(contributions=tuple(
            replace(c, keywords=()) for c in record.payload.contributions)))
        lacking = self._carriers_as(stripped, designations=())
        assert FAMILY.match(lacking, None, declared).polarity is Polarity.DOES_NOT_FIRE
        # Granted to an entity lacking it, or withheld from one holding it: neither side.
        assert FAMILY.match(self._carriers_as(record, designations=()), None, declared) is None
        assert FAMILY.match(self._carriers_as(stripped, designations=("suspected",)), None,
                            declared) is None
        assert FAMILY.needs(declared) == {"designations"}

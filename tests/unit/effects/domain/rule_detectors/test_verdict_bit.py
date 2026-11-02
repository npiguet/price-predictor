"""``verdict_bit`` on real gen-2 decision records (T028)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from effects.domain.records import FailedCheck, PlayabilityDecisionPayload
from effects.domain.rule_catalogue import Polarity
from effects.domain.rule_detectors.acting_line import line_of, root_params_of
from effects.domain.rule_detectors.verdict_bit import ACT, FAMILY, cost_of


def _unaffordable(by_kind):
    return next(r for r in by_kind("playability", "decision")
                if any(not c.affordable for c in r.payload.candidates))


def test_an_unaffordable_candidate_fires_the_cost_rule(by_kind):
    found = FAMILY.match(_unaffordable(by_kind), None, {"bit": "affordable"})
    assert found.polarity is Polarity.FIRES and found.observed and found.entities == (ACT,)


def test_the_same_candidate_edited_affordable_is_the_near_miss(by_kind):
    record = _unaffordable(by_kind)
    paid = PlayabilityDecisionPayload(candidates=tuple(
        replace(c, affordable=True) for c in record.payload.candidates))
    found = FAMILY.match(replace(record, payload=paid), None, {"bit": "affordable"})
    assert found.polarity is Polarity.DOES_NOT_FIRE and not found.observed


def test_responsible_static_counts_only_restricted_candidates(by_kind):
    assert FAMILY.match(_unaffordable(by_kind), None,
                        {"bit": "affordable", "responsible_static": True}) is None


def test_the_gen3_bits_need_their_gen3_fields():
    assert FAMILY.needs({"bit": "timing"}) == {"timing"}
    assert FAMILY.needs({"bit": "failed_check", "check": "casting_speed"}) == {"failed_checks"}
    assert FAMILY.needs({"bit": "can_play"}) == frozenset()


def test_a_failed_check_reads_the_named_identifier(by_kind):
    """A gen-2 candidate edited to carry gen-3's ``failed_checks``."""
    record = _unaffordable(by_kind)
    candidate = record.payload.candidates[0]
    failing = replace(candidate, failed_checks=(FailedCheck("casting_speed"),))
    passing = replace(candidate, failed_checks=())
    params = {"bit": "failed_check", "check": "casting_speed"}
    for edited, polarity in ((failing, Polarity.FIRES), (passing, Polarity.DOES_NOT_FIRE)):
        one = replace(record, payload=PlayabilityDecisionPayload(candidates=(edited,)))
        assert FAMILY.match(one, None, params).polarity is polarity
    unread = replace(record, payload=PlayabilityDecisionPayload(candidates=(candidate,)))
    assert FAMILY.match(unread, None, params) is None


def test_a_golden_candidate_reads_its_timing_bit():
    from effects.infrastructure.record_codec import record_from_dict
    from tests.unit.effects.builders import java_golden_records

    decisions = [record_from_dict(raw) for raw in java_golden_records()
                 if raw["kind"] == "playability" and raw["subkind"] == "decision"]
    assert decisions
    for record in decisions:
        found = FAMILY.match(record, None, {"bit": "timing"})
        assert found.polarity is Polarity.DOES_NOT_FIRE and found.observed is False


def _decisions(by_kind):
    return by_kind("playability", "decision")


def _line(record, sidecars):
    return line_of(record.payload.candidates[0].ability, sidecars)


class TestCandidateFilters:
    def test_a_keyword_ability_shows_by_its_keyword(self, by_kind, gen2_sidecars):
        record = next(r for r in _decisions(by_kind) if _line(r, gen2_sidecars) is not None
                      and _line(r, gen2_sidecars).keyword_display_name)
        name = _line(record, gen2_sidecars).keyword_display_name
        for spelled in (name, name.lower().replace(" ", "_")):
            assert FAMILY.match(record, gen2_sidecars,
                                {"bit": "can_play", "keyword": spelled}) is not None
        assert FAMILY.match(record, gen2_sidecars,
                            {"bit": "can_play", "keyword": "no such keyword"}) is None
        assert FAMILY.match(record, None, {"bit": "can_play", "keyword": name}) is None

    def test_a_spell_of_a_card_printing_a_keyword_shows_by_its_source(
            self, by_kind, gen2_sidecars):
        def printed(record):
            keys = record.payload.candidates[0].ability
            return [line.keyword_display_name
                    for line in gen2_sidecars.get(keys[0].script_file).lines
                    if line.keyword_display_name]
        record = next(r for r in _decisions(by_kind)
                      if r.payload.candidates[0].ability
                      and r.payload.candidates[0].ability[0].trait_kind == "spell"
                      and printed(r))
        name = printed(record)[0]
        assert FAMILY.match(record, gen2_sidecars,
                            {"bit": "affordable", "source_keyword": name}) is not None
        assert FAMILY.match(record, gen2_sidecars,
                            {"bit": "affordable", "source_keyword": "no such keyword"}) is None

    def test_the_tap_symbol_reads_off_the_lines_cost(self, by_kind, gen2_sidecars):
        record = next(r for r in _decisions(by_kind)
                      if "T" in (cost_of(r.payload.candidates[0], gen2_sidecars)[0] or ()))
        assert FAMILY.match(record, gen2_sidecars,
                            {"bit": "affordable", "cost_symbol": "T"}) is not None
        assert FAMILY.match(record, gen2_sidecars,
                            {"bit": "affordable", "cost_symbol": "Q"}) is None

    def test_a_mana_symbol_reads_off_the_adjusted_cost_whatever_the_line(self, by_kind):
        """No sidecar at all: a spell's mana cost is the candidate's own."""
        record = next(r for r in _decisions(by_kind)
                      if "/" in r.payload.candidates[0].cost_after_adjustment.get("mana", ""))
        symbols, parts = cost_of(record.payload.candidates[0], None)
        assert parts is None
        hybrid = next(s for s in symbols if "/" in s)
        assert FAMILY.match(record, None, {"bit": "affordable", "cost_symbol": hybrid})
        assert FAMILY.match(record, None, {"bit": "affordable", "cost_symbol": "nohybrid"}) is None
        kind = "monohybrid" if hybrid.split("/")[0].isdigit() else "hybrid"
        assert FAMILY.match(record, None, {"bit": "affordable", "cost_symbol": kind})
        assert FAMILY.match(record, None, {"bit": "affordable", "cost_part": "Sac"}) is None

    def test_a_cost_part_reads_by_its_name(self, by_kind, gen2_sidecars):
        record, part = next(
            (r, sorted(parts)[0]) for r in _decisions(by_kind)
            for parts in [cost_of(r.payload.candidates[0], gen2_sidecars)[1]] if parts)
        assert FAMILY.match(record, gen2_sidecars,
                            {"bit": "affordable", "cost_part": part.title()}) is not None
        assert FAMILY.match(record, gen2_sidecars,
                            {"bit": "affordable", "cost_part": "Forage"}) is None

    def test_a_script_parameter_reads_off_the_candidates_line(self, by_kind, gen2_sidecars):
        record = next(r for r in _decisions(by_kind)
                      if "Cost" in (root_params_of(_line(r, gen2_sidecars)) or {}))
        assert FAMILY.match(record, gen2_sidecars,
                            {"bit": "affordable", "script_param": "Cost"}) is not None
        assert FAMILY.match(record, gen2_sidecars,
                            {"bit": "affordable", "script_param": "NoSuchParam"}) is None


def test_other_subkinds_show_nothing_and_a_bad_bit_is_refused(by_kind):
    blockers = by_kind("playability", "blockers")[0]
    assert FAMILY.match(blockers, None, {"bit": "affordable"}) is None
    with pytest.raises(ValueError, match="unknown bit"):
        FAMILY.match(blockers, None, {"bit": "mana_value"})

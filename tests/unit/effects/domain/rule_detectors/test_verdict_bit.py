"""``verdict_bit`` on real gen-2 decision records (T028)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from effects.domain.records import FailedCheck, PlayabilityDecisionPayload
from effects.domain.rule_catalogue import Polarity
from effects.domain.rule_detectors.acting_line import line_of, root_params_of
from effects.domain.rule_detectors.verdict_bit import FAMILY, cost_of


def _unaffordable(by_kind):
    return next(r for r in by_kind("playability", "decision")
                if any(not c.affordable for c in r.payload.candidates))


def test_an_unaffordable_candidate_fires_the_cost_rule(by_kind):
    found = FAMILY.match(_unaffordable(by_kind), None, {"bit": "affordable"})
    record = _unaffordable(by_kind)
    assert found.polarity is Polarity.FIRES and found.observed
    # The candidate's card is the condition; target legality, the verdict's
    # per-entity part, is written nowhere for a candidate without a target.
    assert found.condition == (record.state.refs.source,)
    candidate = record.payload.candidates[0]
    targets = bool(candidate.legal_targets or not candidate.has_legal_target)
    assert bool(found.outcome) is targets


def test_a_targeting_candidate_names_every_card_and_player_as_its_outcome(by_kind):
    record = next(r for r in by_kind("playability", "decision")
                  if r.payload.candidates[0].legal_targets)
    found = FAMILY.match(record, None, {"bit": "has_legal_target", "fires_when": True})
    assert found is not None
    assert set(found.outcome) >= {e.id for e in record.state.entities} | {
        p.id for p in record.state.players}


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


class TestLandPlays:
    """Rule 305's land-play candidates (FR-002a, FR-033a), on the gen-3
    fixture's real land-play decisions (T064c)."""

    LIMIT = {"bit": "failed_check", "check": "land_play_limit", "land_play": True}

    @staticmethod
    def _decisions():
        from effects.infrastructure.record_codec import record_from_dict
        from tests.unit.effects.builders import gen3_fixture_records

        return [record_from_dict(raw) for raw in gen3_fixture_records()
                if raw["kind"] == "playability" and raw["subkind"] == "decision"]

    def _land(self, test):
        found = [r for r in self._decisions()
                 if r.payload.candidates[0].land_play is True and test(r.payload.candidates[0])]
        assert found, "the fixture holds no such land-play candidate"
        return found

    @staticmethod
    def _failed(candidate):
        return {failed.check for failed in candidate.failed_checks}

    def test_a_land_past_the_turn_s_limit_fires_the_limit_rule(self):
        for record in self._land(lambda c: "land_play_limit" in self._failed(c)):
            assert FAMILY.match(record, None, self.LIMIT).polarity is Polarity.FIRES
            assert FAMILY.match(record, None, {**self.LIMIT, "fires_when": True}).polarity is (
                Polarity.DOES_NOT_FIRE)

    def test_a_playable_land_is_the_limit_rule_s_near_miss(self):
        for record in self._land(lambda c: c.can_play):
            assert FAMILY.match(record, None, self.LIMIT).polarity is Polarity.DOES_NOT_FIRE
            assert FAMILY.match(record, None, {**self.LIMIT, "fires_when": True}).polarity is (
                Polarity.FIRES)
            assert FAMILY.match(record, None, {"bit": "timing", "fires_when": True,
                                               "land_play": True}).polarity is Polarity.FIRES

    def test_a_land_off_its_player_s_turn_fails_whose_turn(self):
        params = {"bit": "failed_check", "check": "whose_turn", "land_play": True}
        for record in self._land(lambda c: "whose_turn" in self._failed(c)):
            assert FAMILY.match(record, None, params).polarity is Polarity.FIRES

    def test_the_filter_keeps_land_plays_apart_from_every_other_candidate(self):
        """The fixture's other decisions predate land candidates (the first
        pilot), so the other side is the golden decision's spell."""
        from effects.infrastructure.record_codec import record_from_dict
        from tests.unit.effects.builders import java_golden_records

        others = [record_from_dict(raw) for raw in java_golden_records()
                  if raw.get("subkind") == "decision"]
        assert others and all(r.payload.candidates[0].land_play is False for r in others)
        for record in others:
            assert FAMILY.match(record, None, self.LIMIT) is None
            assert FAMILY.match(record, None, {"bit": "timing", "land_play": False}) is not None
        land = self._land(lambda c: True)[0]
        assert FAMILY.match(land, None, {"bit": "timing", "land_play": False}) is None

    def test_a_candidate_that_says_nothing_of_land_plays_is_on_neither_side(self, by_kind):
        """A gen-2 candidate, and a gen-3 one collected before land candidates."""
        before = [r for r in self._decisions() if r.payload.candidates[0].land_play is None]
        assert before
        for record in (_unaffordable(by_kind), *before):
            assert all(c.land_play is None for c in record.payload.candidates)
            for value in (True, False):
                assert FAMILY.match(record, None, {"bit": "affordable", "land_play": value}) is None

    def test_the_land_play_filter_needs_its_field_and_a_check_must_be_one_collected(
            self, by_kind):
        assert FAMILY.needs(self.LIMIT) == {"failed_checks", "land_play"}
        assert FAMILY.needs({"bit": "timing", "land_play": True}) == {"timing", "land_play"}
        with pytest.raises(ValueError, match="names its check"):
            FAMILY.match(_unaffordable(by_kind), None,
                         {"bit": "failed_check", "check": "land_limit"})

    def test_rule_305_s_declarations_read_the_fixture_s_land_plays(self):
        """Each declared 305 scenario finds both sides among the land plays."""
        import json
        from pathlib import Path

        scenarios = json.loads((Path(__file__).parents[5] / "rules" / "scenarios"
                                / "305.json").read_text(encoding="utf-8"))["scenarios"]
        declared = {s["id"]: s["detector"]["params"] for s in scenarios
                    if s.get("detector", {}).get("family") == "verdict_bit"}
        assert set(declared) == {"305.1-1", "305.2-1", "305.2a-1", "305.2b-1", "305.3-1",
                                 "305.4-1"}
        land = self._land(lambda c: True)
        for scenario, params in declared.items():
            assert params["land_play"] is True, scenario
            sides = {FAMILY.match(record, None, params).polarity for record in land}
            assert sides == {Polarity.FIRES, Polarity.DOES_NOT_FIRE}, scenario


class TestCandidateFace:
    """A candidate acting through its card's first face or another (T030 follow-up)."""

    def test_the_face_is_read_off_the_candidate_key(self, by_kind):
        record = next(r for r in _decisions(by_kind) if r.payload.candidates[0].ability)
        candidate = record.payload.candidates[0]
        back = replace(candidate, ability=tuple(
            replace(key, face=1) for key in candidate.ability))
        front = replace(candidate, ability=tuple(
            replace(key, face=0) for key in candidate.ability))
        declared = {"bit": "can_play", "candidate_face": "alternative"}
        on_back = replace(record, payload=replace(record.payload, candidates=(back,)))
        on_front = replace(record, payload=replace(record.payload, candidates=(front,)))
        assert FAMILY.match(on_back, None, declared) is not None
        assert FAMILY.match(on_front, None, declared) is None
        assert FAMILY.match(on_front, None, {**declared, "candidate_face": "front"}) is not None
        with pytest.raises(ValueError, match="candidate_face"):
            FAMILY.match(record, None, {**declared, "candidate_face": "back"})

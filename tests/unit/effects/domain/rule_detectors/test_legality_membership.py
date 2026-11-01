"""``legality_membership`` on real gen-2 legality records (T029)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from effects.domain.records import ForbiddenEntity
from effects.domain.rule_catalogue import Polarity
from effects.domain.rule_detectors.legality_membership import FAMILY


def _forbidden_block(by_kind):
    return next(r for r in by_kind("playability", "blockers") if r.payload.forbidden)


def test_a_creature_forbidden_to_block_fires_a_cant_block_rule(by_kind):
    record = _forbidden_block(by_kind)
    found = FAMILY.match(record, None, {"list": "legal_blockers"})
    assert found.polarity is Polarity.FIRES
    assert found.entities == tuple(f.entity for f in record.payload.forbidden)


def test_the_same_creature_edited_into_the_legal_list_is_the_near_miss(by_kind):
    record = _forbidden_block(by_kind)
    moved = tuple(f.entity for f in record.payload.forbidden)
    edited = replace(record, payload=replace(record.payload, forbidden=(),
                                             legal_blockers=moved))
    found = FAMILY.match(edited, None, {"list": "legal_blockers"})
    assert found.polarity is Polarity.DOES_NOT_FIRE and found.entities == moved


def test_responsible_static_counts_only_what_a_static_forbids(by_kind):
    record = _forbidden_block(by_kind)
    unattributed = replace(record, payload=replace(record.payload, forbidden=tuple(
        ForbiddenEntity(f.entity) for f in record.payload.forbidden)))
    found = FAMILY.match(unattributed, None,
                         {"list": "legal_blockers", "responsible_static": True})
    assert found is None or found.polarity is Polarity.DOES_NOT_FIRE


def test_inside_reads_the_legal_list(by_kind):
    record = next(r for r in by_kind("playability", "attackers") if r.payload.legal_attackers)
    found = FAMILY.match(record, None, {"list": "legal_attackers", "fires_when": "inside"})
    assert found.polarity is Polarity.FIRES and found.entities == record.payload.legal_attackers


class TestUniverse:
    def test_a_noncreature_the_attackers_record_never_lists_is_outside(self, by_kind):
        record = next(r for r in by_kind("playability", "attackers") if any(
            e.on_battlefield and not e.is_creature and e.controller == r.state.global_.active
            for e in r.state.entities))
        found = FAMILY.match(record, None, {"list": "legal_attackers",
                                            "universe": "battlefield"})
        assert found.polarity is Polarity.FIRES
        outside = [record.state.entity(e) for e in found.entities]
        assert any(not e.is_creature for e in outside)
        assert all(e.controller == record.state.global_.active for e in outside)
        assert not set(found.entities) & set(record.payload.legal_attackers)

    def test_the_blockers_universe_is_the_defending_players(self, by_kind):
        record = by_kind("playability", "blockers")[0]
        found = FAMILY.match(record, None, {"list": "legal_blockers",
                                            "universe": "battlefield"})
        active = record.state.global_.active
        assert all(record.state.entity(e).controller != active for e in found.entities)

    def _targeting(self, by_kind):
        return next(r for r in by_kind("playability", "decision")
                    if 0 < len(r.payload.candidates[0].legal_targets) < 32)

    def test_what_a_targeting_candidate_does_not_list_is_outside(self, by_kind):
        record = self._targeting(by_kind)
        legal = set(record.payload.candidates[0].legal_targets)
        found = FAMILY.match(record, None, {"list": "legal_targets",
                                            "universe": "battlefield_and_players"})
        assert found.polarity is Polarity.FIRES and not set(found.entities) & legal
        players = {p.id for p in record.state.players}
        assert players - legal <= set(found.entities)
        inside = FAMILY.match(record, None, {"list": "legal_targets", "fires_when": "inside",
                                             "universe": "battlefield"})
        assert inside.polarity is Polarity.FIRES and set(inside.entities) == legal

    def test_a_capped_or_untargeted_candidate_speaks_for_no_outside(self, by_kind):
        record = self._targeting(by_kind)
        candidate = record.payload.candidates[0]
        declared = {"list": "legal_targets", "universe": "battlefield"}
        capped = replace(candidate, legal_targets=tuple(f"E{i}" for i in range(32)))
        untargeted = replace(candidate, legal_targets=(), has_legal_target=True)
        for edited in (capped, untargeted):
            payload = replace(record.payload, candidates=(edited,))
            assert FAMILY.match(replace(record, payload=payload), None, declared) is None

    def test_an_unknown_universe_is_refused(self, by_kind):
        with pytest.raises(ValueError, match="universe"):
            FAMILY.match(by_kind("playability", "attackers")[0], None,
                         {"list": "legal_attackers", "universe": "graveyard"})


def test_the_list_must_match_the_subkind(by_kind):
    record = _forbidden_block(by_kind)
    assert FAMILY.match(record, None, {"list": "legal_attackers"}) is None

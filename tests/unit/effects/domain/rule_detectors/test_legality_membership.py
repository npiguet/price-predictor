"""``legality_membership`` on real gen-2 legality records (T029)."""

from __future__ import annotations

from dataclasses import replace

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


def test_the_list_must_match_the_subkind(by_kind):
    record = _forbidden_block(by_kind)
    assert FAMILY.match(record, None, {"list": "legal_attackers"}) is None

"""``acting_line.ability_kind`` on a keyword's own line: an alternative-cost cast is a cast."""

from __future__ import annotations

import pytest

from effects.domain.rule_catalogue import Polarity
from effects.domain.rule_detectors.acting_line import candidate_kind
from effects.domain.rule_detectors.verdict_bit import FAMILY as VERDICT


def _keyword_candidates(by_kind, sidecars):
    """``(record, candidate, keyword display name)`` of every fixture playability
    candidate whose acting key is a keyword's."""
    found = []
    for record in by_kind("playability"):
        for candidate in getattr(record.payload, "candidates", ()):
            if candidate.ability and candidate.ability[0].trait_kind == "keyword":
                line = sidecars.first_line(tuple(candidate.ability))
                found.append((record, candidate, line.keyword_display_name))
    return found


@pytest.mark.parametrize("keyword", ["Morph", "Sneak"])
def test_an_alternative_cost_keyword_line_is_a_cast(by_kind, gen3_sidecars, keyword):
    held = [(r, c) for r, c, name in _keyword_candidates(by_kind, gen3_sidecars) if name == keyword]
    assert held
    for _, candidate in held:
        assert candidate_kind(candidate, gen3_sidecars) == "cast"


def test_an_activated_keyword_line_stays_activated(by_kind, gen3_sidecars):
    held = [c for _, c, name in _keyword_candidates(by_kind, gen3_sidecars) if name == "Cycling"]
    assert held
    assert all(candidate_kind(c, gen3_sidecars) == "activated" for c in held)


def test_a_declaration_for_the_cast_reads_a_keyword_cast_and_one_for_activated_does_not(
        by_kind, gen3_sidecars):
    record, _, _ = next(t for t in _keyword_candidates(by_kind, gen3_sidecars) if t[2] == "Morph")
    base = {"bit": "has_legal_target", "fires_when": True, "source_keyword": "Morph"}
    found = VERDICT.match(record, gen3_sidecars, {**base, "candidate": "cast"})
    assert found is not None and found.polarity in (Polarity.FIRES, Polarity.DOES_NOT_FIRE)
    assert VERDICT.match(record, gen3_sidecars, {**base, "candidate": "activated"}) is None

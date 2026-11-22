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


def test_a_keyword_s_triggered_or_static_line_is_neither_a_cast_nor_activated(
        gen3_records, gen3_sidecars):
    """A keyword's own trigger (ward, prowess, echo, a token's) is no ability a player
    activates: only a cast or an activated line answers."""
    from effects.domain.rule_detectors.acting_line import ability_kind

    held = [r for r in gen3_records if r.ability and r.ability[0].trait_kind == "keyword"
            and (line := gen3_sidecars.first_line(tuple(r.ability))) is not None
            and line.line_kind in ("triggered", "static")]
    assert held
    assert all(ability_kind(r.ability, gen3_sidecars) is None for r in held)


class _KeywordLine:
    """A sidecar lookup whose acting line is one keyword's line of the given converter kind."""

    def __init__(self, script_text: str, line_kind: str) -> None:
        from effects.domain.provenance import KEYWORD_API_TYPE, SidecarLine

        self.line = SidecarLine(line_index=0, line_kind=line_kind, provenance=(),
                                script_api_type=KEYWORD_API_TYPE, script_text=script_text,
                                prose_offset=0)

    def first_line(self, keys):
        return self.line


@pytest.mark.parametrize("script_text, line_kind, expected", [
    # special actions (116.2c, 116.2h, 116.2k): no ability is activated, nothing is cast
    ("Suspend:4 R", "alternate cost", None),
    ("Plot:2 R", "alternate cost", None),
    ("Foretell:1 U", "alternate cost", None),
    # activated abilities from a hand or a graveyard stay activated
    ("Unearth:1 B", "alternate cost", "activated"),
    ("Ninjutsu:1 U", "alternate cost", "activated"),
    # alternative-cost casts the converter files as activated (702.185a, 702.188a)
    ("Warp:1 W", "activated", "cast"),
    ("Web-slinging:2 W", "activated", "cast"),
    ("Equip:2", "activated", "activated"),
    ("Cycling:2", "activated", "activated"),
])
def test_a_keyword_line_reads_by_its_keyword(script_text, line_kind, expected):
    from effects.domain.provenance import ProvenanceKey
    from effects.domain.rule_detectors.acting_line import ability_kind

    key = ProvenanceKey("cardsfolder/x/x.txt", 0, "keyword", 0)
    assert ability_kind((key,), _KeywordLine(script_text, line_kind)) == expected

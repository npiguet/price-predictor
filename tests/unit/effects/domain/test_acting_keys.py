"""``acting_keys``: the keys a record's ``[ACT]`` reads, and ``first_text``, the
text they fold to."""

from __future__ import annotations

from effects.domain.provenance import ProvenanceKey, first_text
from effects.domain.records import (
    Candidate,
    CombatPayload,
    PlayabilityDecisionPayload,
    ResolutionPayload,
    acting_keys,
)

_KEY = ProvenanceKey("cardsfolder/s/shock.txt", 0, "spell", 0)


class _Record:
    def __init__(self, payload, ability=None) -> None:
        self.payload = payload
        self.ability = ability


def _candidate(*keys) -> Candidate:
    return Candidate(ability=keys, can_play=True, affordable=True, has_legal_target=True)


def test_a_resolution_acts_through_its_own_ability():
    assert acting_keys(_Record(ResolutionPayload(), ability=(_KEY,))) == (_KEY,)


def test_a_decision_acts_through_its_first_candidate():
    other = ProvenanceKey("cardsfolder/b/bolt.txt", 0, "spell", 0)
    payload = PlayabilityDecisionPayload(candidates=(_candidate(_KEY), _candidate(other)))
    assert acting_keys(_Record(payload)) == (_KEY,)


def test_a_decision_without_candidates_acts_through_nothing():
    assert acting_keys(_Record(PlayabilityDecisionPayload())) == ()


def test_a_record_with_no_acting_line_acts_through_nothing():
    assert acting_keys(_Record(CombatPayload())) == ()


def test_the_first_key_that_resolves_names_the_text():
    """A line merged from several traits carries several keys; a runtime-only
    one ahead of them names no text and is passed over."""
    runtime = ProvenanceKey("cardsfolder/s/shock.txt", 0, "spell", 9)
    other = ProvenanceKey("cardsfolder/b/bolt.txt", 0, "spell", 0)
    texts = {_KEY: "deal 2 damage", other: "deal 3 damage"}
    assert first_text((runtime, _KEY, other), texts.get) == "deal 2 damage"


def test_keys_that_resolve_to_nothing_have_no_text():
    assert first_text((_KEY,), {}.get) is None
    assert first_text((), {_KEY: "deal 2 damage"}.get) is None

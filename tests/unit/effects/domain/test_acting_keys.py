"""``acting_keys``: the keys a record's ``[ACT]`` reads."""

from __future__ import annotations

from effects.domain.provenance import ProvenanceKey
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

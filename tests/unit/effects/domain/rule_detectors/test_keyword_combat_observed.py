"""``keyword_combat``: a keyword outcome that did not happen is ``observed`` false, a review."""

from __future__ import annotations

from dataclasses import replace

import pytest

from effects.domain.damage_step_keywords import DAMAGE_STEP_KEYWORDS
from effects.domain.event_schema import Event, EventType
from effects.domain.rule_catalogue import Polarity
from effects.domain.rule_detectors.keyword_combat import FAMILY
from tests.unit.effects.domain.rule_detectors.test_keyword_combat import _unblocked_attack


def _with_events(record, *events):
    return replace(record, payload=replace(record.payload, events=tuple(events)))


def _real_fires(by_kind, sidecars):
    """The fixture's real combats a keyword fires on, with that keyword."""
    return [(r, row.keyword) for r in by_kind("combat") if not r.is_probe
            for row in DAMAGE_STEP_KEYWORDS
            if FAMILY.match(r, sidecars, {"keyword": row.keyword}) is not None]


def test_the_fixture_s_fires_combats_show_their_outcome(by_kind, gen3_sidecars):
    fires = _real_fires(by_kind, gen3_sidecars)
    assert fires
    for record, keyword in fires:
        found = FAMILY.match(record, gen3_sidecars, {"keyword": keyword})
        assert found.polarity is Polarity.FIRES and found.observed is True, record.record_id


def test_a_fires_combat_whose_outcome_did_not_happen_is_observed_false(by_kind, gen3_sidecars):
    """The real combats, edited to hold none of their events."""
    for record, keyword in _real_fires(by_kind, gen3_sidecars):
        found = FAMILY.match(_with_events(record), gen3_sidecars, {"keyword": keyword})
        assert found.polarity is Polarity.FIRES and found.observed is False


#: Per keyword: the carrier E1 (P0) attacks P1, blocked by E2 where the row needs a blocker,
#: and the event showing the row's effect happened.
_OUTCOMES = [
    ("lifelink", None, Event(EventType.LIFE_CHANGE, ("P0",), {"delta": 3})),
    ("trample", "E2", Event(EventType.LIFE_CHANGE, ("P1",), {"delta": -2})),
    ("infect", None, Event(EventType.POISON_CHANGE, ("P1",), {"delta": 3})),
]


@pytest.mark.parametrize(("keyword", "blocker", "event"), _OUTCOMES)
def test_each_keyword_reads_its_own_outcome(keyword, blocker, event):
    bare = _unblocked_attack((keyword,), blocker=blocker)
    assert FAMILY.match(bare, None, {"keyword": keyword}).observed is False
    shown = _with_events(bare, event)
    assert FAMILY.match(shown, None, {"keyword": keyword}).observed is True


def test_the_wrong_sign_is_not_the_outcome():
    """A lifelinker's controller losing life is not the lifelink gain."""
    bare = _unblocked_attack(("lifelink",))
    lost = _with_events(bare, Event(EventType.LIFE_CHANGE, ("P0",), {"delta": -3}))
    assert FAMILY.match(lost, None, {"keyword": "lifelink"}).observed is False

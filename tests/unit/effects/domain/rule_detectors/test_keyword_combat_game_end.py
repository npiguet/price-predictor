"""``keyword_combat`` tags no combat that ends the game, on either side
(decision D13).

Forge destroys no creature in a combat whose damage ends the game, so neither
a real combat nor a damage-step probe of one shows what the keyword did. The
detector leaves such a record out by the test ``sba_zone_move`` uses
(``sba_board.game_ends``).

The probe is the fixture's real one: it stripped deathtouch from E7 in a
combat that takes P1 to -2 life. The ``fires`` side is the probe's live
combat, its blocker E56 (1/2, blocking E25 4/6) granted deathtouch (no real
case is in the fixture).
"""

from __future__ import annotations

from effects.domain.board_edits import BoardEdit
from effects.domain.rule_detectors import keyword_combat
from effects.domain.rule_detectors.sba_board import game_ends

_PREFIX = "3550376f-3c30-4231-90cd-cc0f1b308e87.2-mwqvqdxrusnv."
_LIVE, _PROBE = _PREFIX + "425", _PREFIX + "426"


def _by_id(gen3_records, record_id):
    return next(r for r in gen3_records if r.record_id == record_id)


def test_a_probe_of_a_game_ending_combat_is_no_near_miss(gen3_records, gen3_sidecars):
    probe = _by_id(gen3_records, _PROBE)
    assert probe.is_probe and probe.payload.probed_keyword == "deathtouch"
    assert game_ends(probe)
    assert keyword_combat.FAMILY.match(probe, gen3_sidecars, {"keyword": "deathtouch"}) is None


def test_a_carrier_in_a_game_ending_combat_does_not_fire(gen3_records, gen3_sidecars):
    live = BoardEdit(_by_id(gen3_records, _LIVE)).grant_keyword("E56", "deathtouch").record
    assert game_ends(live)
    assert keyword_combat.FAMILY.match(live, gen3_sidecars, {"keyword": "deathtouch"}) is None

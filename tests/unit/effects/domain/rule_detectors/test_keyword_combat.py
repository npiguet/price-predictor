"""``keyword_combat`` on real gen-3 combat records and probes (story 1 acceptance 2)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from effects.domain.rule_catalogue import Polarity
from effects.domain.rule_detectors.keyword_combat import FAMILY
from effects.domain.rule_detectors.sba_board import game_ends


def test_a_deathtouch_probe_is_the_near_miss_naming_its_carrier(by_kind, gen3_sidecars):
    """One whose combat does not end the game: a probe of one that does is
    tagged nothing (``test_keyword_combat_game_end.py``)."""
    probe = next(r for r in by_kind("combat")
                 if r.payload.probed_keyword == "deathtouch" and not game_ends(r))
    found = FAMILY.match(probe, gen3_sidecars, {"keyword": "deathtouch"})
    assert found.polarity is Polarity.DOES_NOT_FIRE and not found.observed
    assert found.condition == (probe.payload.probed_entity,)
    # The stripped keyword's outcome falls on the creature the carrier fights.
    carrier = probe.state.entity(probe.payload.probed_entity)
    fought = carrier.combat.blocked_by or carrier.combat.blocking
    assert found.outcome == fought[:1]


def test_a_probe_of_another_keyword_shows_nothing(by_kind, gen3_sidecars):
    probe = next(r for r in by_kind("combat") if r.payload.probed_keyword == "trample")
    assert FAMILY.match(probe, gen3_sidecars, {"keyword": "deathtouch"}) is None


def _probe_and_firing_combat(by_kind, sidecars):
    """A probe of ``first_strike`` and its real combat, in which the carrier
    kept the keyword and the keyword's effect was observable."""
    combats = by_kind("combat")
    probes = [r for r in combats if r.payload.probed_keyword == "first_strike"]
    for probe in probes:
        real = next((r for r in combats if r.record_id == probe.mirror_of), None)
        if real is not None and FAMILY.match(real, sidecars, {"keyword": "first_strike"}):
            return probe, real
    pytest.fail("no first strike probe in the fixture has a real combat the keyword shows in")


def test_a_real_combat_with_a_qualifying_carrier_fires(by_kind, gen3_sidecars):
    """The probe's mirror is a real combat in which the carrier kept the keyword."""
    probe, real = _probe_and_firing_combat(by_kind, gen3_sidecars)
    found = FAMILY.match(real, gen3_sidecars, {"keyword": "first_strike"})
    assert found.polarity is Polarity.FIRES and probe.payload.probed_entity in found.condition
    assert found.outcome


def test_the_carrier_edited_out_of_combat_shows_nothing(by_kind, gen3_sidecars):
    probe, real = _probe_and_firing_combat(by_kind, gen3_sidecars)
    carrier = probe.payload.probed_entity
    state = replace(real.state, entities=tuple(
        replace(e, combat=None) if e.id == carrier else e for e in real.state.entities))
    found = FAMILY.match(replace(real, state=state), gen3_sidecars, {"keyword": "first_strike"})
    assert found is None or carrier not in found.condition


def test_a_keyword_outside_gate_two_s_table_is_refused(by_kind):
    with pytest.raises(ValueError, match="not a damage-step keyword"):
        FAMILY.validate({"keyword": "flying"})



def _unblocked_attack(keywords: tuple[str, ...], blocker=None):
    """E1 (P0's 3/3 with ``keywords``) attacking P1, blocked by ``blocker`` if given."""
    from effects.domain.records import CombatPayload, RecordKind
    from effects.domain.state_snapshot import (
        CombatStatus,
        GrantedTemporary,
        PlayerState,
        PowerToughness,
    )
    from tests.unit.effects.record_builders import make_entity, make_record, make_snapshot

    carrier = make_entity(
        "E1", "carrier", pt=PowerToughness((3, 3)),
        granted_temporary=GrantedTemporary(keywords=keywords),
        combat=CombatStatus(attacking="P1", blocked_by=(blocker,) if blocker else ()))
    board = [carrier]
    if blocker:
        board.append(make_entity(blocker, "blocker", controller="P1", pt=PowerToughness((1, 1)),
                                 combat=CombatStatus(blocking=("E1",))))
    players = tuple(PlayerState(id=p, life=20, hand=3, library=30, graveyard=0)
                    for p in ("P0", "P1"))
    return make_record(RecordKind.COMBAT, state=make_snapshot(entities=tuple(board),
                                                             players=players),
                       payload=CombatPayload(attackers=("E1",)))


def test_lifelink_s_outcome_is_the_carrier_s_controller():
    found = FAMILY.match(_unblocked_attack(("lifelink",)), None, {"keyword": "lifelink"})
    assert found.polarity is Polarity.FIRES
    assert (found.condition, found.outcome) == (("E1",), ("P0",))


def test_trample_s_outcome_is_the_defending_player():
    found = FAMILY.match(_unblocked_attack(("trample",), blocker="E2"), None,
                         {"keyword": "trample"})
    assert found.polarity is Polarity.FIRES
    assert (found.condition, found.outcome) == (("E1",), ("P1",))

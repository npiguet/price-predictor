"""D4: the blocking, targeting and timing rules each tag their own records.

Before the per-rule rewrite, 16 blocking scenarios, 12 targeting scenarios and
27 timing scenarios each shared one unfiltered declaration, so a blockers
record with one tapped forbidden creature fired every "can't block" rule,
flying included. These boards go through the real catalogue's index.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from effects.domain.provenance import ProvenanceKey
from effects.domain.records import (
    Candidate,
    ForbiddenEntity,
    PlayabilityBlockersPayload,
    PlayabilityDecisionPayload,
    PlayabilitySubkind,
    RecordKind,
)
from effects.domain.rule_catalogue import Polarity
from effects.domain.rule_detectors import DetectorIndex, tag_record
from effects.domain.rule_outcomes import (
    BlockingRestriction,
    Characteristics,
    TargetingRestriction,
    blocking_restrictions,
    may_block,
    targeting_restrictions,
)
from effects.domain.state_snapshot import GrantedTemporary, PlayerState
from effects.infrastructure.rules_catalogue_io import load_scenarios
from tests.unit.effects.builders import make_entity, make_record, make_snapshot

RULES = Path(__file__).parents[5] / "rules"
BLOCKING = {"509.1a", "701.35a", "702.13b", "702.14c", "702.16f", "702.16j", "702.28b",
            "702.31b", "702.9b", "702.36b", "701.60c", "702.147a"}
TARGETING = {"702.11b", "702.11d", "702.16b", "702.16j", "702.18a", "702.5a", "702.6c",
             "702.11e", "702.5d", "702.6e"}


@pytest.fixture(scope="module")
def index():
    return DetectorIndex.of(load_scenarios(RULES))


def _creature(entity_id, *keywords, controller="P1", colors=("G",), tapped=False, **more):
    return make_entity(entity_id, entity_id.lower(), controller=controller, types=("Creature",),
                       colors=colors, tapped=tapped,
                       granted_temporary=GrantedTemporary(keywords=tuple(keywords)), **more)


def _blockers(attacker, legal=(), forbidden=()):
    entities = (attacker, *legal, *forbidden)
    state = make_snapshot(entities=entities, players=(
        PlayerState(id="P0", life=20, hand=0, library=30, graveyard=0),
        PlayerState(id="P1", life=20, hand=0, library=30, graveyard=0)))
    payload = PlayabilityBlockersPayload(
        anchor_attacker=attacker.id, legal_blockers=tuple(e.id for e in legal),
        forbidden=tuple(ForbiddenEntity(e.id) for e in forbidden))
    return make_record(RecordKind.PLAYABILITY, subkind=PlayabilitySubkind.BLOCKERS,
                       state=state, payload=payload, generation=3)


def _sides(index, record, rules):
    return {t.rule: t.polarity for t in tag_record(record, index).tags if t.rule in rules}


def _fires(index, record, rules):
    return {rule for rule, side in _sides(index, record, rules).items() if side is Polarity.FIRES}


def test_one_tapped_forbidden_creature_fires_the_tapped_rule_alone(index):
    """V5 #16: a vanilla ground attacker and one tapped forbidden creature
    fired all fourteen blocking rules, flying included."""
    record = _blockers(_creature("E1", controller="P0"), forbidden=(_creature("E2", tapped=True),))
    assert _sides(index, record, BLOCKING) == {"509.1a": Polarity.FIRES}


def test_a_flier_fires_flying_and_a_reach_blocker_is_its_near_miss(index):
    flier = _creature("E1", "flying", controller="P0", colors=("U",))
    record = _blockers(flier, legal=(_creature("E3", "reach"),),
                       forbidden=(_creature("E2"),))
    assert _fires(index, record, BLOCKING) == {"702.9b"}
    reach_only = _blockers(flier, legal=(_creature("E3", "reach"),))
    assert _sides(index, reach_only, BLOCKING)["702.9b"] is Polarity.DOES_NOT_FIRE
    assert not _fires(index, reach_only, BLOCKING)


def test_shadow_forbids_both_ways_and_intimidate_reads_colours(index):
    shadow = _blockers(_creature("E1", controller="P0"), forbidden=(_creature("E2", "shadow"),))
    assert _sides(index, shadow, BLOCKING) == {"702.28b": Polarity.FIRES}
    intimidating = _creature("E1", "intimidate", controller="P0", colors=("R",))
    record = _blockers(intimidating, legal=(_creature("E3", colors=("R",)),),
                       forbidden=(_creature("E2"),))
    assert _fires(index, record, BLOCKING) == {"702.13b"}


def _decision(source, targets, legal, actor="P0"):
    key = ProvenanceKey("cardsfolder/s/shock.txt", 0, "spell", 0)
    candidate = Candidate(ability=(key,), can_play=True, affordable=True, has_legal_target=True,
                          legal_targets=tuple(legal))
    state = make_snapshot(entities=(source, *targets), players=(
        PlayerState(id="P0", life=20, hand=1, library=30, graveyard=0),
        PlayerState(id="P1", life=20, hand=1, library=30, graveyard=0)))
    state = state.__class__(**{**{f: getattr(state, f) for f in state.__dataclass_fields__},
                               "refs": state.refs.__class__(source=source.id)})
    return make_record(RecordKind.PLAYABILITY, subkind=PlayabilitySubkind.DECISION,
                       state=state, payload=PlayabilityDecisionPayload(candidates=(candidate,)),
                       actor_player=actor, generation=3)


def test_an_opponents_hexproof_creature_fires_hexproof_alone(index):
    """V5 #16: a land outside every legal target list fired hexproof,
    shroud, protection, enchant and equip at once."""
    shock = make_entity("E9", "shock", zone="stack", types=("Instant",), colors=("R",))
    bears = _creature("E2")
    hexproof = _creature("E3", "hexproof")
    land = make_entity("E4", "forest", controller="P1", types=("Land",))
    record = _decision(shock, (bears, hexproof, land), legal=("E2", "P0", "P1"))
    assert _sides(index, record, TARGETING) == {"702.11b": Polarity.FIRES}
    own = _creature("E3", "hexproof", controller="P0")
    mine = _decision(shock, (bears, own), legal=("E2", "E3", "P0", "P1"))
    assert _sides(index, mine, TARGETING) == {"702.11b": Polarity.DOES_NOT_FIRE}


def test_protection_from_red_keeps_a_red_source_out(index):
    shock = make_entity("E9", "shock", zone="stack", types=("Instant",), colors=("R",))
    protected = _creature("E3", "protection_from_red")
    record = _decision(shock, (protected,), legal=("P0", "P1"))
    assert _sides(index, record, TARGETING) == {"702.16b": Polarity.FIRES}


def test_the_restriction_functions():
    assert blocking_restrictions(["flying"], ["U"], [], ["G"], ["creature"]) == {
        BlockingRestriction.FLYING}
    assert not blocking_restrictions(["flying"], ["U"], ["reach"], ["G"], ["creature"])
    assert blocking_restrictions(["shadow", "flying"], [], [], [], []) == {
        BlockingRestriction.SHADOW, BlockingRestriction.FLYING}
    assert may_block([], [], [], [], ["creature"]) is True
    red = Characteristics.made(["instant"], colours=["R"])
    assert targeting_restrictions(["hexproof"], source=red, opponents=True) == {
        TargetingRestriction.HEXPROOF}
    assert not targeting_restrictions(["hexproof"], source=red, opponents=False)
    assert targeting_restrictions(["shroud"], source=red, opponents=False) == {
        TargetingRestriction.SHROUD}
    assert targeting_restrictions(["hexproof_from_red"], source=red, opponents=True) == {
        TargetingRestriction.HEXPROOF_FROM}
    assert not targeting_restrictions(["hexproof_from_blue"], source=red, opponents=True)
    assert targeting_restrictions(["protection_from_everything"], source=red,
                                  opponents=False) == {TargetingRestriction.PROTECTION}

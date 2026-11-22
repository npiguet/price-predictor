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
    Combatant,
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
    def blue(*keywords):
        return Combatant.made(keywords, ["U"])

    def green(*keywords):
        return Combatant.made(keywords, ["G"], ["creature"])

    assert blocking_restrictions(blue("flying"), green()) == {BlockingRestriction.FLYING}
    assert not blocking_restrictions(blue("flying"), green("reach"))
    assert blocking_restrictions(Combatant.made(["shadow", "flying"]), Combatant.made()) == {
        BlockingRestriction.SHADOW, BlockingRestriction.FLYING}
    assert may_block(Combatant.made(), Combatant.made(types=["creature"])) is True
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


def _with_power(entity, power):
    from dataclasses import replace

    from effects.domain.state_snapshot import PowerToughness

    return replace(entity, pt=PowerToughness((power, power)))


def test_skulk_forbids_a_blocker_with_greater_power():
    """702.118b, which blocking_restrictions did not model."""
    from effects.domain.rule_detectors.legality_restrictions import blocker_judgements, sides

    skulker = _with_power(_creature("E1", "skulk", controller="P0"), 2)
    record = _blockers(skulker, legal=(_with_power(_creature("E3"), 1),),
                       forbidden=(_with_power(_creature("E2"), 3),))
    assert sides(blocker_judgements(record, None, "skulk", None), "outside") == (
        ("E2",), ("E3",))


def test_landwalk_needs_the_land_it_names_on_the_defender_s_battlefield():
    """702.14c: swampwalk applies only while the defending player controls a
    Swamp; nonbasic and snow landwalk read their selector."""
    from dataclasses import replace

    from effects.domain.rule_detectors.legality_restrictions import blocker_judgements, sides
    from effects.domain.state_snapshot import CardLine

    def walker(selector):
        line = CardLine(keys=(), kind="keyword", granted=False, fragment=f"Landwalk:{selector}")
        return replace(_creature("E1", "landwalk", controller="P0"), lines=(line,))

    def board(attacker, *lands, forbidden=True):
        record = _blockers(attacker, forbidden=(_creature("E2"),)) if forbidden else _blockers(
            attacker, legal=(_creature("E2"),))
        return replace(record, state=replace(record.state, entities=(
            *record.state.entities, *lands)))

    swamp = make_entity("E5", "swamp", controller="P1", types=("Land",), subtypes=("Swamp",),
                        supertypes=("Basic",))
    forest = make_entity("E6", "forest", controller="P1", types=("Land",),
                         subtypes=("Forest",), supertypes=("Basic",))
    wastes = make_entity("E7", "evolving wilds", controller="P1", types=("Land",))
    fires = sides(blocker_judgements(board(walker("Swamp"), swamp), None, "landwalk", None),
                  "outside")
    assert fires == (("E2",), ())
    near = sides(blocker_judgements(board(walker("Swamp"), forest, forbidden=False), None,
                                    "landwalk", None), "outside")
    assert near == ((), ("E2",))
    nonbasic = board(walker("Land.nonBasic"), forest)
    assert sides(blocker_judgements(nonbasic, None, "landwalk", None), "outside") == ((), ())
    assert sides(blocker_judgements(board(walker("Land.nonBasic"), wastes), None, "landwalk",
                                    None), "outside") == (("E2",), ())


def test_a_changeling_has_every_creature_type():
    """702.73a: protection from Goblins forbids a changeling blocker."""
    from effects.domain.rule_outcomes import ProtectionQuality, QualityKind

    goblins = ProtectionQuality(QualityKind.CREATURE_TYPE, "goblin")
    assert goblins.covers(Characteristics.made(["creature"], ["shapeshifter"], changeling=True))
    assert not goblins.covers(Characteristics.made(["creature"], ["shapeshifter"]))
    changeling = Combatant.made(["changeling"], ["U"], ["creature"], ["shapeshifter"])
    assert BlockingRestriction.PROTECTION in blocking_restrictions(
        Combatant.made(["protection_from_goblin"], ["R"]), changeling)
    assert may_block(Combatant.made(), changeling, attacker_protections=(goblins,)) is False


_PACIFISM = ProvenanceKey("cardsfolder/p/pacifism.txt", 0, "static", 0)
_RESTRICTED = {"509.1b", "701.35a"}


def _forbidding(attacker, legal=(), forbidden=()):
    """A blockers record whose forbidden entries say who forbids each:
    ``(creature, responsible_static, required_elsewhere)``."""
    record = _blockers(attacker, legal, tuple(entry[0] for entry in forbidden))
    entries = tuple(ForbiddenEntity(creature.id, statics, elsewhere)
                    for creature, statics, elsewhere in forbidden)
    return make_record(RecordKind.PLAYABILITY, subkind=PlayabilitySubkind.BLOCKERS,
                       state=record.state, generation=3,
                       payload=PlayabilityBlockersPayload(
                           anchor_attacker=attacker.id,
                           legal_blockers=record.payload.legal_blockers, forbidden=entries))


def test_a_blocker_nothing_on_the_record_explains_is_detain_s(index):
    """WP7: the connector names Pacifism's static, so a forbidden blocker with
    none, untapped and under no evasion, hidden keyword or lure is detained."""
    attacker = _creature("E1", controller="P0")
    record = _forbidding(attacker, legal=(_creature("E3"),),
                         forbidden=((_creature("E2"), (), False),))
    assert _fires(index, record, _RESTRICTED) == {"701.35a"}


def test_a_blocker_a_static_forbids_is_the_static_s_rule_not_detain(index):
    attacker = _creature("E1", controller="P0")
    record = _forbidding(attacker, forbidden=((_creature("E2"), (_PACIFISM,), False),))
    assert _fires(index, record, _RESTRICTED) == {"509.1b"}
    assert "701.35a" not in _sides(index, record, _RESTRICTED)


def test_a_lure_bound_blocker_and_a_hidden_can_t_block_are_no_detain(index):
    attacker = _creature("E1", controller="P0")
    panicked = make_entity("E3", "e3", controller="P1", types=("Creature",), colors=("G",),
                           granted_temporary=GrantedTemporary(
                               hidden_keywords=("CARDNAME can't block.",)))
    record = _forbidding(attacker, forbidden=((_creature("E2"), (), True), (panicked, (), False)))
    assert _sides(index, record, _RESTRICTED) == {}
    assert "701.35a" not in _sides(index, record, _RESTRICTED)


def test_detain_has_no_near_miss(index):
    """No record marks a detained creature: a legal blocker is no near miss."""
    attacker = _creature("E1", controller="P0")
    record = _forbidding(attacker, legal=(_creature("E3"),))
    assert "701.35a" not in _sides(index, record, _RESTRICTED)


_GRAKK = ProvenanceKey("cardsfolder/g/grakk_the_pacifist.txt", 0, "keyword", 0)


def _explained_as_written(attacker, creature, *, statics=(), elsewhere=False, own=False,
                          unresolved=None, extra=()):
    """A blockers record whose one forbidden creature's entry is as the
    connector writes it, with ``extra`` entities on the board."""
    entry = ForbiddenEntity(creature.id, statics, elsewhere, responsible_own=own,
                            responsible_unresolved=unresolved)
    state = make_snapshot(entities=(attacker, creature, *extra), players=(
        PlayerState(id="P0", life=20, hand=0, library=30, graveyard=0),
        PlayerState(id="P1", life=20, hand=0, library=30, graveyard=0)))
    return make_record(RecordKind.PLAYABILITY, subkind=PlayabilitySubkind.BLOCKERS,
                       state=state, generation=3,
                       payload=PlayabilityBlockersPayload(
                           anchor_attacker=attacker.id, forbidden=(entry,)))


def test_only_a_blocker_nothing_explains_tags_detain_through_the_real_catalogue(index):
    """Every board Forge refuses a block on for a reason the record states,
    read through every detector of the catalogue: a second attacker that must
    be blocked if able (``required_elsewhere``, which the connector writes
    for every such requirement), a creature its own printed keyword keeps
    out, one an unkeyed static keeps out, a Pacifism'd one. Only the plain
    unexplained blocker, which Forge refuses only when detained, is 701.35a's."""
    attacker = _creature("E1", controller="P0")
    must = _creature("E9", "cardname must be blocked if able.", controller="P0")
    boards = {
        "must be blocked elsewhere": _explained_as_written(
            attacker, _creature("E2"), elsewhere=True, extra=(must,)),
        "own printed keyword": _explained_as_written(
            attacker, _creature("E2", "cardname can't attack or block."),
            statics=(_GRAKK,), own=True),
        "unkeyed static": _explained_as_written(
            attacker, _creature("E2"), unresolved="engine_effect"),
        "pacifism": _explained_as_written(attacker, _creature("E2"), statics=(_PACIFISM,)),
        "plain unexplained": _explained_as_written(attacker, _creature("E2")),
    }
    tagged = {name: {t.rule for t in tag_record(record, index).tags}
              for name, record in boards.items()}
    assert {name for name, rules in tagged.items() if "701.35a" in rules} == {
        "plain unexplained"}


def test_a_can_t_block_keyword_of_its_own_explains_without_the_record_s_key(index):
    """A creature's own can't-block keyword explains it however the entry
    reads; an "alone" form only while its controller has no other creature."""
    attacker = _creature("E1", controller="P0")
    grakk = _explained_as_written(attacker, _creature("E2", "cardname can't attack or block."))
    alone = _explained_as_written(
        attacker, _creature("E2", "cardname can't attack or block alone."))
    beside = _explained_as_written(
        attacker, _creature("E2", "cardname can't attack or block alone."),
        extra=(_creature("E3"),))
    assert "701.35a" not in _sides(index, grakk, _RESTRICTED)
    assert "701.35a" not in _sides(index, alone, _RESTRICTED)
    assert _fires(index, beside, _RESTRICTED) == {"701.35a"}

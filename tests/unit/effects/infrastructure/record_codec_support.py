"""Records shared by the record codec tests: a gen-3 snapshot exercising
every optional block, a record built on it, and one record per kind."""

from __future__ import annotations

from effects.domain.event_schema import Event, EventType
from effects.domain.provenance import ProvenanceKey
from effects.domain.records import (
    ActivationPayload,
    Candidate,
    CombatPayload,
    ContinuousPayload,
    Contribution,
    Costs,
    EffectRecord,
    ForbiddenEntity,
    Moment,
    PlayabilityAttackersPayload,
    PlayabilityBlockersPayload,
    PlayabilityDecisionPayload,
    PlayabilitySubkind,
    RecordKind,
    ResolutionPayload,
    RewritePayload,
    RewriteResult,
    TriggerPayload,
)
from effects.domain.state_snapshot import (
    CardLine,
    CombatStatus,
    EntityState,
    GlobalState,
    GrantedTemporary,
    InclusionTier,
    PlayerState,
    PowerToughness,
    Refs,
    StackExtras,
    StateSnapshot,
)
from tests.unit.effects.builders import (
    gen3_payload,
)

_KEY = ProvenanceKey("cardsfolder/s/serra_angel.txt", 0, "trigger", 0)
_DONOR = ProvenanceKey("cardsfolder/a/anthem.txt", 0, "static", 1)


def _rich_snapshot() -> StateSnapshot:
    """A gen-3 snapshot exercising every optional block, so nothing round-trips by luck."""
    return StateSnapshot(
        global_=GlobalState(
            turn=7, phase="combat_damage", active="P0", priority="P1",
            stack_size=2, combat_substep="first_strike", emblems=(_DONOR,),
        ),
        players=(
            PlayerState(
                id="P0", life=14, hand=3, library=21, graveyard=9, poison=1,
                energy=2, this_turn={"creatures_died": 1, "spells_cast": 2},
                floating_mana={"R": 1}, untapped_production={"G": 3}, max_land_plays=1,
            ),
            PlayerState(id="P1", life=8, hand=1, library=30, graveyard=4, max_land_plays=1),
        ),
        entities=(
            EntityState(
                id="E12", name="Serra Angel", zone="battlefield", controller="P0",
                types=("Creature",), subtypes=("Angel",), colors=("W",),
                mana_value=5,
                pt=PowerToughness(base=(4, 4), boosts=(1, 0), counters=(0, 1)),
                tapped=True, damage=2, counters={"P1P1": 1},
                combat=CombatStatus(attacking="P1", blocked_by=("E20",),
                                    became_blocked=True),
                granted_temporary=GrantedTemporary(
                    keywords=("flying",), abilities=(_DONOR,),
                ),
                lines=(
                    CardLine(keys=(_KEY,), kind="trigger", granted=False),
                    CardLine(keys=(_DONOR,), kind="static", granted=True,
                             grant="static", active=True),
                ),
            ),
            EntityState(
                id="S1", name="Fireball", zone="stack", controller="P1", lines=(),
                stack_extras=StackExtras(
                    targets=("E12",), per_target_amounts={"E12": 3},
                    up_to_counts={"E12": 1},
                ),
            ),
        ),
        refs=Refs(
            targets=("E12",), targets_on_stack=(), source="S1", modes=("option-0",), x=3,
            choices={"color": "red"},
        ),
        tiers=frozenset({InclusionTier.REFERENCED, InclusionTier.CORE}),
        links=(),
    )


def _record(**overrides) -> EffectRecord:
    defaults: dict = {
        "record_id": "run-uuid.4.10237",
        "run_id": "run-uuid",
        "timestamp": "2026-09-06T15:31:12.152018Z",
        "game_id": "run-uuid.4.812",
        "generation": 3,
        "kind": RecordKind.RESOLUTION,
        "actor_player": "P0",
        "state": _rich_snapshot(),
        "moment": Moment.RESOLUTION,
        "ability": (_KEY,),
        "payload": ResolutionPayload(
            events=(
                Event(
                    type=EventType.DAMAGE_DEALT, subjects=("E12",),
                    params={"amount": 3, "combat": False}, attributed_to="0",
                ),
            )
        ),
    }
    defaults.update(overrides)
    # A gen-3 effect half carries its outcome: the codec refuses one without.
    defaults["payload"] = gen3_payload(defaults["payload"])
    return EffectRecord(**defaults)


ALL_KINDS = {
    "resolution-activation": dict(
        moment=Moment.ACTIVATION,
        payload=ActivationPayload(
            costs=Costs(
                mana_by_color={"R": 1}, tapped=("E12",), life=2,
                sacrificed=("E13",), discarded=("E14",), exiled=("E15",),
            ),
            events=(),
        ),
        link_id="pair-7",
    ),
    "resolution-effect": dict(link_id="pair-7"),
    "rewrite": dict(
        kind=RecordKind.REWRITE, moment=None,
        payload=RewritePayload(
            incoming=Event(type=EventType.DAMAGE_DEALT, params={"amount": 5}),
            outgoing=Event(type=EventType.DAMAGE_DEALT, params={"amount": 1}),
            result=RewriteResult.UPDATED,
        ),
    ),
    # The shape most of the channel actually takes: Forge substitutes by
    # running another ability, so nothing in the event's map is edited.
    "rewrite-substituted": dict(
        kind=RecordKind.REWRITE, moment=None,
        payload=RewritePayload(
            incoming=Event(type=EventType.ZONE_CHANGE, subjects=("E12",),
                           params={"to_zone": "graveyard"}),
            result=RewriteResult.REPLACED,
            replaced_by=(
                ProvenanceKey(
                    script_file="cardsfolder/l/leyline_of_the_void.txt",
                    face=0, trait_kind="replacement", index_within_kind=0,
                ),
            ),
        ),
    ),
    "rewrite-declined": dict(
        kind=RecordKind.REWRITE, moment=None,
        payload=RewritePayload(
            incoming=Event(type=EventType.DAMAGE_DEALT, params={"amount": 5}),
            result=RewriteResult.NOT_REPLACED,
        ),
    ),
    "continuous": dict(
        kind=RecordKind.CONTINUOUS, moment=None,
        payload=ContinuousPayload(
            contributions=(
                Contribution(
                    entity="E12", pt_boost=(1, 1), keywords=("vigilance",),
                    types=("Creature",), colors=("W",), abilities=(), hidden_keywords=(),
                    name=None,
                ),
            ),
            board_hash="ab12cd",
        ),
    ),
    "combat": dict(
        kind=RecordKind.COMBAT, moment=None, ability=None,
        payload=CombatPayload(
            attackers=("E12",), blocks={"E12": ("E20",)},
            assignment_choices={"E12": {"E20": 4}},
            events=(Event(type=EventType.DAMAGE_DEALT,
                          params={"amount": 4, "combat": True}),),
        ),
    ),
    "trigger": dict(
        kind=RecordKind.TRIGGER, moment=None,
        payload=TriggerPayload(
            event=Event(type=EventType.LIFE_CHANGE, params={"delta": 3}),
            fired=True,
        ),
    ),
    "playability-decision": dict(
        kind=RecordKind.PLAYABILITY, moment=None, ability=None,
        subkind=PlayabilitySubkind.DECISION,
        payload=PlayabilityDecisionPayload(
            candidates=(
                Candidate(
                    ability=(_KEY,), can_play=True, affordable=False,
                    has_legal_target=True, legal_targets=("E20",),
                    cost_after_adjustment={"R": 2},
                    responsible_static=(_DONOR,), timing=True, failed_checks=(),
                    land_play=False,
                ),
            )
        ),
    ),
    "playability-attackers": dict(
        kind=RecordKind.PLAYABILITY, moment=None, ability=None,
        subkind=PlayabilitySubkind.ATTACKERS,
        payload=PlayabilityAttackersPayload(
            anchor_attacker="E12", legal_defenders=("P1",), must_attack=False,
            legal_attackers=("E12",),
            forbidden=(ForbiddenEntity(entity="E13", responsible_static=(_DONOR,)),),
        ),
    ),
    "playability-blockers": dict(
        kind=RecordKind.PLAYABILITY, moment=None, ability=None,
        subkind=PlayabilitySubkind.BLOCKERS,
        payload=PlayabilityBlockersPayload(
            anchor_attacker="E12", legal_blockers=("E20",),
            forbidden=(ForbiddenEntity(entity="E21"),), must_block=(), min_blockers=2,
        ),
    ),
}

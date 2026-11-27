"""Record, snapshot and entity builders for the effect-record unit tests.

Each test module used to hand-roll its own ``_record`` / ``_snapshot`` /
``_entity``. These three build the plain case; a test that needs something
specific passes it as an override, or keeps a small local wrapper that does.
"""

from __future__ import annotations

from effects.domain.event_schema import Event, EventType
from effects.domain.records import (
    ActivationPayload,
    CombatPayload,
    ContinuousPayload,
    Costs,
    EffectRecord,
    Moment,
    PlayabilityAttackersPayload,
    PlayabilityBlockersPayload,
    PlayabilityDecisionPayload,
    PlayabilitySubkind,
    RecordKind,
    ResolutionOutcome,
    ResolutionPayload,
    RewritePayload,
    TriggerPayload,
    TurnActionPayload,
    TurnActionSubkind,
)
from effects.domain.split import HeldOutCards
from effects.domain.state_snapshot import (
    CardLine,
    EntityState,
    GlobalState,
    InclusionTier,
    PlayerState,
    Refs,
    StateSnapshot,
)
from effects.domain.text_holdout import holdout_key


def game_end_marker(last: EffectRecord, *, records_withheld: int = 0) -> EffectRecord:
    """The game-end marker a game whose last record is ``last`` writes:
    a ``turn_action`` / ``game_end`` on ``last``'s board, no events."""
    from dataclasses import replace

    return replace(
        last, record_id=f"{last.game_id}.game-end", kind=RecordKind.TURN_ACTION,
        moment=None, subkind=TurnActionSubkind.GAME_END, ability=None,
        ability_unresolved=None, link_id=None, mirror_of=None, variant_of=None,
        interventional=False, fork=False, fork_kind=None, copy_dropped=None, synthetic=False,
        directed=None, curation=None, what_if=None,
        payload=TurnActionPayload(records_withheld=records_withheld),
    )


def with_game_end_markers(records) -> list[EffectRecord]:
    """``records`` with a game-end marker after each game's last record, as
    a collection that ended every game writes its shard."""
    records = list(records)
    last: dict[str, int] = {}
    for index, record in enumerate(records):
        last[record.game_id] = index
    ends = {index: game for game, index in last.items()}
    out: list[EffectRecord] = []
    for index, record in enumerate(records):
        out.append(record)
        if index in ends:
            out.append(game_end_marker(record))
    return out


_EVENT = Event(type=EventType.LIFE_CHANGE, params={"delta": 1})


#: The tiers stage-one collection wrote: what a test snapshot claims by default.
STAGE_ONE_TIERS: frozenset[InclusionTier] = frozenset(
    {InclusionTier.REFERENCED, InclusionTier.CORE}
)


def make_snapshot(**overrides) -> StateSnapshot:
    """An empty turn-1 board collected at the stage-one tiers; ``overrides``
    replace any ``StateSnapshot`` field. Its refs name no target on the stack,
    as the collector writes a refs block that has none: ``[]``, not absent."""
    defaults: dict = {
        "global_": GlobalState(
            turn=1, phase="main1", active="P0", priority="P0", stack_size=0,
        ),
        "refs": Refs(targets_on_stack=()),
        "players": (),
        "entities": (),
        "tiers": STAGE_ONE_TIERS,
        "links": (),
    }
    defaults.update(overrides)
    return StateSnapshot(**defaults)


def make_entity(entity_id: str, name: str, **overrides) -> EntityState:
    """A battlefield permanent controlled by ``P0``; its ``lines`` come from
    :func:`lines_of`."""
    defaults: dict = {
        "id": entity_id, "name": name, "zone": "battlefield", "controller": "P0",
    }
    defaults.update(overrides)
    return EntityState(**defaults)


def lines_of(printed=(), granted=()) -> tuple[CardLine, ...]:
    """A gen-3 entity's lines for its ``printed`` keys and the keys a static
    grants it, one line per key, of the key's trait kind."""
    return (
        *(CardLine(keys=(key,), kind=key.trait_kind, granted=False) for key in printed),
        *(CardLine(keys=(key,), kind=key.trait_kind, granted=True, grant="static")
          for key in granted),
    )


def _default_payload(kind: RecordKind, moment, subkind):
    if kind is RecordKind.RESOLUTION:
        if moment is Moment.ACTIVATION:
            # The cost half: ``{events, costs}``, its outcome on the effect half.
            return ActivationPayload(costs=Costs(), events=())
        return ResolutionPayload(conditions_met=(), outcome=ResolutionOutcome.RESOLVED)
    if kind is RecordKind.PLAYABILITY:
        return {
            PlayabilitySubkind.DECISION: PlayabilityDecisionPayload,
            # Attackers records are one per creature, each anchored.
            PlayabilitySubkind.ATTACKERS: lambda: PlayabilityAttackersPayload(
                anchor_attacker="E1",
            ),
            PlayabilitySubkind.BLOCKERS: lambda: PlayabilityBlockersPayload(
                anchor_attacker="E1", must_block=(),
            ),
        }[subkind]()
    if kind is RecordKind.TURN_ACTION:
        if subkind is TurnActionSubkind.GAME_END:
            return TurnActionPayload(records_withheld=0)
        return TurnActionPayload()
    return {
        RecordKind.COMBAT: CombatPayload,
        RecordKind.CONTINUOUS: ContinuousPayload,
        RecordKind.TRIGGER: lambda: TriggerPayload(event=_EVENT, fired=True),
        RecordKind.REWRITE: lambda: RewritePayload(
            incoming=_EVENT, outgoing=_EVENT,
        ),
    }[kind]()


def gen3_payload(payload):
    """An effect half in the shape the codec reads: it carries its outcome
    (``resolved`` by default) and its conditions. Any other payload, and a
    half already so, as given."""
    from dataclasses import replace

    if isinstance(payload, ResolutionPayload) and (
            payload.conditions_met is None
            or (payload.outcome is None and not payload.outcome_written)):
        return replace(
            payload, conditions_met=payload.conditions_met or (),
            outcome=payload.outcome or (
                None if payload.outcome_written else ResolutionOutcome.RESOLVED))
    return payload


def make_record(kind: RecordKind = RecordKind.RESOLUTION, **overrides) -> EffectRecord:
    """A valid gen-3 record of ``kind`` with every envelope field defaulted.

    ``moment`` (resolution) and ``subkind`` (playability) select the default
    payload; any field, ``payload`` included, can be overridden. A resolution
    half passed in the superseded shape is completed into the current one
    (:func:`gen3_payload`); the other members of a passed payload, state or
    entity are the test's to give.
    """
    game_id = overrides.get("game_id", "g1")
    defaults: dict = {
        "record_id": f"{game_id}.1", "run_id": "run", "timestamp": "t",
        "game_id": game_id, "kind": kind, "actor_player": "P0",
        "state": make_snapshot(), "generation": 3,
    }
    moment = overrides.pop("moment", None)
    subkind = overrides.pop("subkind", None)
    if kind is RecordKind.RESOLUTION:
        moment = moment or Moment.RESOLUTION
    elif kind is RecordKind.PLAYABILITY:
        subkind = subkind or PlayabilitySubkind.DECISION
    elif kind is RecordKind.TURN_ACTION:
        subkind = subkind or TurnActionSubkind.DRAW
    # A discriminator on the wrong kind is forwarded as given, so a test can
    # still build the record the envelope check is meant to refuse.
    if moment is not None:
        defaults["moment"] = moment
    if subkind is not None:
        defaults["subkind"] = subkind
    if "payload" not in overrides:
        defaults["payload"] = _default_payload(kind, moment, subkind)
    else:
        overrides["payload"] = gen3_payload(overrides["payload"])
    defaults.update(overrides)
    return EffectRecord(**defaults)


def named_entity(name: str, **overrides) -> EntityState:
    """:func:`make_entity` with its id derived from ``name``."""
    return make_entity(f"E-{name}", name, **overrides)


def record_on_board(game_id: str, entities=(), ability=None, **overrides) -> EffectRecord:
    """A record of game ``game_id`` whose board holds ``entities``."""
    return make_record(
        game_id=game_id, ability=ability,
        state=make_snapshot(entities=tuple(entities)), **overrides,
    )


def held_out_cards(
    names=(), script_files=(), *, texts=(),
) -> HeldOutCards:
    """A holdout of exactly these cards, script files and ``texts``, each
    text's :meth:`HeldOutCards.holds_text` key derived as
    :func:`text_keyed_holdout` derives it."""
    return HeldOutCards(
        names=frozenset(names), script_files=frozenset(script_files),
        texts=frozenset(texts),
        keys=frozenset(holdout_key(text) for text in texts),
    )


def player_of(snapshot: StateSnapshot, player_id: str) -> PlayerState | None:
    """The snapshot's player of ``player_id``, or None."""
    return next((each for each in snapshot.players if each.id == player_id), None)

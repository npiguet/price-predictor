"""Builders the validate-corpus test modules share: a healthy window and the
records each test perturbs it with.

Each test builds a small shard directory that reproduces one measured defect
from the first collected corpus and asserts the validator names it. The point
is not that the checks exist but that they *fire*: the corpus that motivated
them parsed cleanly, satisfied every schema test, and was worthless, so a
validator that passed it would be worse than none.

The synthetic shards are written through :class:`RawShardWriter` rather than by
hand, so a defect is planted in the records themselves and has to survive a
real serialize/parse round trip to be found — the same path a collector's
output takes.
"""

from __future__ import annotations

from dataclasses import dataclass

from effects.application.corpus_validation.findings import Finding
from effects.domain.event_schema import AttributionKind, Event, EventType
from effects.domain.provenance import ProvenanceKey, ProvenanceSidecar, SidecarLine
from effects.domain.records import (
    ActivationPayload,
    CollectionMode,
    CombatPayload,
    Costs,
    EffectRecord,
    ForkKind,
    Moment,
    RecordKind,
    ResolutionOutcome,
    ResolutionPayload,
    RewritePayload,
    RewriteResult,
    TriggerPayload,
)
from effects.domain.state_snapshot import GlobalState, InclusionTier, StateSnapshot
from effects.infrastructure.sidecar_io import SidecarCache
from tests.unit.effects.file_builders import write_sidecar
from tests.unit.effects.record_builders import make_entity, make_record, make_snapshot

_KEY = ProvenanceKey("cardsfolder/l/lightning_bolt.txt", 0, "spell", 0)
_BEARS = "cardsfolder/g/grizzly_bears.txt"
_STAGE_TWO = frozenset({
    InclusionTier.REFERENCED, InclusionTier.CORE, InclusionTier.UNREFERENCED_STACK,
})


def _snapshot(turn: int = 3, names: tuple[str, ...] = (), tiers=None) -> StateSnapshot:
    return make_snapshot(
        global_=GlobalState(
            turn=turn, phase="main1", active="P0", priority="P0", stack_size=0,
        ),
        entities=tuple(
            make_entity(f"E{index}", name) for index, name in enumerate(names)
        ),
        tiers=_STAGE_TWO if tiers is None else tiers,
    )


def _resolution(record_id: str, game_id: str = "run.0-L1.0", **overrides):
    fields: dict = {
        "record_id": record_id,
        "timestamp": "2026-09-09T10:00:00.000000Z",
        "game_id": game_id,
        "state": _snapshot(),
        "ability": (_KEY,),
        # The mode a run collected under is a property of the checkout, not of
        # the record, so the fixture states the healthy one rather than
        # inheriting the dataclass default: a window of `degraded` records is
        # itself one of the defects under test.
        "mode": CollectionMode.PATCHED,
        "payload": ResolutionPayload(
            events=(
                Event(
                    type=EventType.DAMAGE_DEALT,
                    subjects=("E1",),
                    attributed_to=AttributionKind.ROOT,
                ),
            ),
        ),
    }
    fields.update(overrides)
    if fields.get("fork") and "fork_kind" not in fields:
        probe = fields.get("kind") is RecordKind.COMBAT and not fields.get("interventional")
        fields["fork_kind"] = ForkKind.PROBE if probe else ForkKind.INTERVENTION
    return make_record(**fields)


def _activation(record_id: str, **overrides):
    """A cost half: written at the cast, its events and costs (FR-042a)."""
    fields: dict = {
        "moment": Moment.ACTIVATION,
        "payload": ActivationPayload(
            costs=overrides.pop("costs", Costs(mana_by_color={"R": 1})), events=(),
        ),
    }
    fields.update(overrides)
    return _resolution(record_id, **fields)


@dataclass(frozen=True)
class _TriggerEvent:
    """What a built trigger record's event says: whether the trigger fired,
    and the zone-change mode it was evaluated for."""

    fired: bool = True
    mode: str = "ChangesZone"


def _trigger(record_id: str, ability=(_KEY,), event: _TriggerEvent = _TriggerEvent(),
             **overrides):
    fields: dict = {
        "kind": RecordKind.TRIGGER,
        "moment": None,
        "ability": ability,
        "payload": TriggerPayload(
            event=Event(
                type=EventType.ZONE_CHANGE,
                params={
                    "mode": event.mode, "from_zone": "battlefield",
                    "to_zone": "graveyard",
                },
            ),
            fired=event.fired,
        ),
    }
    fields.update(overrides)
    return _resolution(record_id, **fields)


def _rewrite(record_id: str, ability=(_KEY,), **overrides):
    """A genuine in-place edit: 3 damage down to 1, reported as ``Updated``.

    The healthy default is the *rare* shape deliberately, because it is the one
    that exercises every rewrite field at once. The substitutions and the
    negatives that make up most of the channel are built per test.
    """
    fields: dict = {
        "kind": RecordKind.REWRITE,
        "moment": None,
        "ability": ability,
        "payload": RewritePayload(
            incoming=Event(type=EventType.DAMAGE_DEALT, params={"amount": 3}),
            outgoing=Event(type=EventType.DAMAGE_DEALT, params={"amount": 1}),
            result=RewriteResult.UPDATED,
            replaced_by=(_KEY,),
        ),
    }
    fields.update(overrides)
    return _resolution(record_id, **fields)


#: A creature headed for the graveyard, which a substitution usually rewrites.
_TO_GRAVEYARD = Event(
    type=EventType.ZONE_CHANGE, subjects=("E1",), params={"to_zone": "graveyard"},
)


def _substitution(record_id: str, *, incoming: Event = _TO_GRAVEYARD, **payload_kwargs):
    """A rewrite whose payload holds ``incoming`` and ``payload_kwargs`` alone."""
    return _rewrite(
        record_id, payload=RewritePayload(incoming=incoming, **payload_kwargs),
    )


def _combat(record_id: str, **overrides):
    fields: dict = {
        "kind": RecordKind.COMBAT,
        "moment": None,
        "ability": None,
        "payload": CombatPayload(attackers=("E0",)),
    }
    fields.update(overrides)
    return _resolution(record_id, **fields)


def _healthy() -> list[EffectRecord]:
    """A window nothing is wrong with, for every test to perturb.

    Deliberately varied where a defect would show as sameness: two outcomes,
    two games, paid costs, distinct payloads. Each game ends with its
    game-end marker.
    """
    from tests.unit.effects.record_builders import game_end_marker

    records: list[EffectRecord] = []
    for game in (0, 1):
        gid = f"run.0-L1.{game}"
        base = game * 100
        records += [
            _activation(f"run.0-L1.{base}", link_id=f"link.{game}", game_id=gid),
            _resolution(f"run.0-L1.{base + 1}", link_id=f"link.{game}", game_id=gid),
            _activation(
                f"run.0-L1.{base + 2}", link_id=f"fizzle.{game}",
                game_id=gid, costs=Costs(mana_by_color={"U": 2}, tapped=("E4",)),
            ),
            _resolution(
                f"run.0-L1.{base + 7}", link_id=f"fizzle.{game}", game_id=gid,
                payload=ResolutionPayload(outcome=ResolutionOutcome.FIZZLED),
            ),
            _trigger(f"run.0-L1.{base + 3}", game_id=gid, state=_snapshot(turn=4)),
            # The negative half of the trigger sample. Without it every window
            # reads 100% fired, which is the very skew the check looks for.
            _trigger(
                f"run.0-L1.{base + 6}", game_id=gid, event=_TriggerEvent(fired=False),
                state=_snapshot(turn=4),
            ),
            _rewrite(f"run.0-L1.{base + 4}", game_id=gid, state=_snapshot(turn=5)),
            last := _combat(f"run.0-L1.{base + 5}", game_id=gid, state=_snapshot(turn=6)),
            game_end_marker(last),
        ]
    return records


def _named(findings: list[Finding], fragment: str) -> Finding:
    return next(f for f in findings if fragment in f.name)


def _write_bears_sidecar(root, keyword_indices, spell_indices=(), dropped=()):
    """A sidecar for Grizzly Bears declaring exactly these ordinals.

    ``spell_indices`` exists so a test can declare a spell *range* with a gap
    in it: outside a declared range a key is runtime-only, so a planted
    mismatch of another trait kind has nowhere else to live. ``dropped`` is
    the contentless set, which resolves to no line and still counts as a join.
    """
    sidecar = ProvenanceSidecar(
        card="Grizzly Bears",
        script_file=_BEARS,
        lines=tuple(
            SidecarLine(
                line_index=row,
                line_kind="keyword",
                provenance=(ProvenanceKey(_BEARS, 0, "keyword", index),), prose_offset=0,
            )
            for row, index in enumerate(keyword_indices)
        ) + tuple(
            SidecarLine(
                line_index=len(keyword_indices) + row,
                line_kind="spell",
                provenance=(ProvenanceKey(_BEARS, 0, "spell", index),), prose_offset=0,
            )
            for row, index in enumerate(spell_indices)
        ),
        dropped_keys=tuple(dropped),
    )
    write_sidecar(
        sidecar, root / "cardsfolder" / "g" / "grizzly_bears.provenance.json",
    )
    return SidecarCache({"cardsfolder": root / "cardsfolder"})

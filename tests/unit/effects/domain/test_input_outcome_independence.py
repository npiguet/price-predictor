"""The input does not depend on the outcome (Phase G's shared invariant).

The head predicts what a record's events and outcome are, so its input is the
board as the acting line found it. Built from a fixture record and from the same
record with its events and outcome removed, and with the library cards only
those events put in the snapshot (root spec § 8.4), the surface is the same.
"""

from __future__ import annotations

import random
import zlib
from collections.abc import Callable
from dataclasses import dataclass, replace

import pytest

from effects.domain.effect_head_input import SurfaceBuilder
from effects.domain.records import (
    ActivationPayload,
    CombatPayload,
    ContinuousPayload,
    EffectRecord,
    PlayabilityAttackersPayload,
    PlayabilityBlockersPayload,
    PlayabilityDecisionPayload,
    ResolutionPayload,
    RewritePayload,
    TriggerPayload,
    TurnActionPayload,
)
from effects.domain.state_snapshot import EntityRole
from effects.infrastructure.record_codec import record_from_dict
from tests.unit.effects.builders import gen3_fixture_records

E_DIM = 4


def _e_for(key) -> tuple[float, ...]:
    """A vector of its own for every key, so a slot reading another line shows."""
    seed = zlib.crc32(repr(key).encode())
    return tuple(float((seed >> (8 * i)) & 0xFF) for i in range(E_DIM))


@dataclass(frozen=True)
class _Lines:
    """Every key has a line, is no keyword, has no modes and is one slot."""

    e_for: Callable = _e_for
    has_line: Callable = lambda key: True
    keyword_of: Callable = lambda key: None
    options_for: Callable = lambda key: ()
    board_keys: Callable = lambda key: (key,)


def _no_verdict(candidate):
    return replace(
        candidate, can_play=False, affordable=False, has_legal_target=False,
        legal_targets=(), cost_after_adjustment={}, responsible_static=(),
        timing=None, failed_checks=None, land_play=None,
    )


#: Each payload kind without what happened: its events and its outcome.
_WITHOUT_OUTCOME: dict[type, Callable] = {
    ActivationPayload: lambda p: replace(p, events=()),
    ResolutionPayload: lambda p: replace(p, events=(), conditions_met=None, outcome=None),
    RewritePayload: lambda p: replace(
        p, outgoing=None, result=None, replaced_by=(), buffered=None,
        buffered_ran=None, buffered_events=(),
    ),
    ContinuousPayload: lambda p: replace(p, contributions=()),
    CombatPayload: lambda p: replace(p, events=(), assignment_choices={}),
    TriggerPayload: lambda p: replace(p, fired=False),
    PlayabilityDecisionPayload: lambda p: replace(
        p, candidates=tuple(_no_verdict(c) for c in p.candidates)),
    PlayabilityAttackersPayload: lambda p: replace(
        p, legal_attackers=(), forbidden=(), legal_defenders=(), must_attack=False,
        tapped_by_attack=None,
    ),
    PlayabilityBlockersPayload: lambda p: replace(
        p, legal_blockers=(), forbidden=(), min_blockers=0, must_block=None),
    TurnActionPayload: lambda p: replace(p, events=(), records_withheld=None),
}


def _event_named_library_cards(record: EffectRecord) -> set[str]:
    """The library cards below the top that the record's events name: the
    snapshot holds them only because those events happened."""
    named = {subject for event in record.payload.all_events() for subject in event.subjects}
    return {
        entity.id for entity in record.state.entities
        if entity.role == EntityRole.LIBRARY_CARD and entity.order_rank != 0
        and entity.id in named
    }


def _without_outcome(record: EffectRecord) -> EffectRecord:
    gone = _event_named_library_cards(record)
    state = replace(record.state, entities=tuple(
        entity for entity in record.state.entities if entity.id not in gone))
    payload = _WITHOUT_OUTCOME[type(record.payload)](record.payload)
    return replace(record, state=state, payload=payload)


def _surface(record: EffectRecord):
    return SurfaceBuilder(_Lines(), e_dim=E_DIM, rng=random.Random(0)).build(record)


@pytest.fixture(scope="module")
def records_by_kind() -> dict[type, list[EffectRecord]]:
    grouped: dict[type, list[EffectRecord]] = {kind: [] for kind in _WITHOUT_OUTCOME}
    for record in map(record_from_dict, gen3_fixture_records()):
        grouped[type(record.payload)].append(record)
    return grouped


_KINDS = [
    pytest.param(
        CombatPayload, id="combat",
        marks=pytest.mark.xfail(
            strict=True, reason="G03: the damage assignment is an input (finding 4.4)"),
    ),
    *(pytest.param(kind, id=kind.__name__) for kind in _WITHOUT_OUTCOME
      if kind is not CombatPayload),
]


@pytest.mark.parametrize("kind", _KINDS)
def test_the_input_does_not_depend_on_the_outcome(records_by_kind, kind):
    records = records_by_kind[kind]
    assert records, f"the fixture holds no {kind.__name__} record"
    differing = [
        record.record_id for record in records
        if _surface(record) != _surface(_without_outcome(record))
    ]
    assert differing == []

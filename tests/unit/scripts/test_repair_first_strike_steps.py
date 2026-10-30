"""The repair of a built corpus's first-strike-step combat records.

The records are built with the domain types and written with the collector's
own serializer, so each test reads the shape a real gen-2 shard carries: Forge's
phase name, the ``first_strike`` substep, combat damage events with their cause,
and keywords already normalized as ``record_codec`` parses them.
"""

from __future__ import annotations

import gzip
import importlib.util
import json
from pathlib import Path

from effects.domain.damage_step_keywords import KeywordResolver
from effects.domain.event_schema import Event, EventType
from effects.domain.records import CombatPayload
from effects.domain.state_snapshot import (
    CombatStatus,
    GlobalState,
    GrantedTemporary,
    PlayerState,
)
from effects.infrastructure.record_codec import format_record_line
from effects.infrastructure.shard_io import read_shard
from tests.unit.effects.domain.test_effect_head_input import _entity, _record, _snapshot

_SCRIPT = Path(__file__).parents[3] / "scripts" / "repair_first_strike_steps.py"


def _load():
    spec = importlib.util.spec_from_file_location("repair_first_strike_steps", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


repair_module = _load()


def _attacker(entity_id: str, *keywords: str, blocked_by=()):
    return _entity(
        entity_id, controller="P0",
        granted_temporary=GrantedTemporary(keywords=tuple(keywords)),
        combat=CombatStatus(attacking="P1", blocked_by=tuple(blocked_by)),
    )


def _first_strike_record(entities, events, blocks, *, substep="first_strike", probe=None):
    state = _snapshot(
        global_=GlobalState(
            turn=9, phase="combat_first_strike_damage", active="P0",
            priority="P0", stack_size=0, combat_substep=substep,
        ),
        players=(
            PlayerState(id="P0", life=14, hand=3, library=21, graveyard=9),
            PlayerState(id="P1", life=17, hand=1, library=30, graveyard=4),
        ),
        entities=tuple(entities),
    )
    payload = CombatPayload(
        attackers=tuple(blocks), blocks=dict(blocks), events=tuple(events),
        probed_keyword=probe,
    )
    record = _record(kind=_combat(), state=state, payload=payload)
    return json.loads(format_record_line(record))


def _combat():
    from effects.domain.records import RecordKind

    return RecordKind.COMBAT


def _hit(subject: str, amount: int, cause: str) -> Event:
    return Event(
        type=EventType.DAMAGE_DEALT, subjects=(subject,),
        params={"amount": amount, "combat": True, "cause": cause},
    )


def _keywords():
    return repair_module.ResolvedKeywords(KeywordResolver(None))


def _outcomes(raw):
    return [
        (e["type"], e["subjects"][0], e["params"]["delta"])
        for e in raw["payload"]["events"] if e["type"] != "damage_dealt"
    ]


def test_an_unblocked_first_striker_costs_the_defender_its_damage():
    raw = _first_strike_record(
        [_attacker("E1", "first_strike")], [_hit("P1", 3, "E1")], {"E1": ()},
    )
    outcome, repaired = repair_module.repair(raw, _keywords())
    assert outcome == "repaired"
    assert _outcomes(repaired) == [("life_change", "P1", -3)]


def test_hits_from_several_attackers_sum_into_one_life_change():
    raw = _first_strike_record(
        [_attacker("E1", "first_strike"), _attacker("E2", "double_strike")],
        [_hit("P1", 3, "E1"), _hit("P1", 2, "E2")], {"E1": (), "E2": ()},
    )
    _outcome, repaired = repair_module.repair(raw, _keywords())
    assert _outcomes(repaired) == [("life_change", "P1", -5)]


def test_lifelink_gains_its_controller_life_and_infect_gives_poison():
    raw = _first_strike_record(
        [_attacker("E1", "first_strike", "lifelink"),
         _attacker("E2", "double_strike", "infect")],
        [_hit("P1", 2, "E1"), _hit("P1", 1, "E2")], {"E1": (), "E2": ()},
    )
    _outcome, repaired = repair_module.repair(raw, _keywords())
    assert _outcomes(repaired) == [
        ("life_change", "P0", 2), ("life_change", "P1", -2),
        ("poison_change", "P1", 1),
    ]
    poison = repaired["payload"]["events"][-1]
    assert poison["params"]["cause"] == "P0"


def test_a_blocked_fight_with_a_first_striker_drops_the_record():
    raw = _first_strike_record(
        [_attacker("E1", "first_strike", blocked_by=("E9",)),
         _attacker("E2"), _entity("E9", controller="P1")],
        [_hit("P1", 2, "E2")], {"E1": ("E9",), "E2": ()},
    )
    assert repair_module.repair(raw, _keywords()) == ("dropped", None)


def test_a_blocked_fight_without_one_keeps_the_record():
    # Neither side strikes first, so nothing happened between them this step.
    raw = _first_strike_record(
        [_attacker("E1", "first_strike"), _attacker("E2", blocked_by=("E9",)),
         _entity("E9", controller="P1")],
        [_hit("P1", 3, "E1")], {"E1": (), "E2": ("E9",)},
    )
    outcome, repaired = repair_module.repair(raw, _keywords())
    assert outcome == "repaired"
    assert _outcomes(repaired) == [("life_change", "P1", -3)]


def test_regular_steps_probe_forks_and_complete_records_are_left_alone():
    entities, events = [_attacker("E1", "first_strike")], [_hit("P1", 3, "E1")]
    regular = _first_strike_record(entities, events, {"E1": ()}, substep="regular")
    probe = _first_strike_record(entities, events, {"E1": ()}, probe="first_strike")
    once = repair_module.repair(
        _first_strike_record(entities, events, {"E1": ()}), _keywords(),
    )[1]
    for raw in (regular, probe, once):
        assert repair_module.repair(raw, _keywords()) == ("kept", raw)


def test_a_shard_is_rewritten_and_still_reads(tmp_path: Path):
    kept = _first_strike_record(
        [_attacker("E1", "first_strike")], [_hit("P1", 3, "E1")], {"E1": ()},
        substep="regular",
    )
    repaired = _first_strike_record(
        [_attacker("E1", "first_strike")], [_hit("P1", 3, "E1")], {"E1": ()},
    )
    dropped = _first_strike_record(
        [_attacker("E1", "first_strike", blocked_by=("E9",)),
         _entity("E9", controller="P1")], [], {"E1": ("E9",)},
    )
    shard = tmp_path / "shard-00001.jsonl.gz"
    with gzip.open(shard, "wt", encoding="utf-8") as handle:
        for raw in (kept, repaired, dropped):
            handle.write(json.dumps(raw, separators=(",", ":")) + "\n")

    counts = repair_module.repair_shard(shard, _keywords(), write=True)

    assert (counts["repaired"], counts["dropped"], counts["records"]) == (1, 1, 3)
    records = list(read_shard(shard))
    assert len(records) == 2
    assert any(
        e.type is EventType.LIFE_CHANGE for e in records[1].payload.events
    )

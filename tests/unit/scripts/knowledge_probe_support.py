"""Builders the knowledge-probe tests share: fixture lines, joins, selections, grid records."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from effect_knowledge_probes import combat_grid, labels, sweeps
from effect_knowledge_probes.board_edits import BoardEdit

E_DIM = 8
_ROOT = Path(__file__).resolve().parents[3]
FIXTURES = _ROOT / "tests" / "fixtures" / "effects"


def join_triggers(records):
    join = labels.TriggerJoin()
    for record in records:
        join.observe(record)
    return join.result()


def select_sweep_records(records, sidecars) -> dict:
    selector = sweeps.SweepSelector(sidecars)
    for record in records:
        selector.observe(record)
    return selector.result()


#: The rule's deaths over a power x toughness grid: lethal where power reaches toughness.
LETHAL_STEP = np.array([[float(p >= t) for t in range(1, 9)] for p in range(1, 9)])


def fixture_lines():
    from effects.infrastructure.sidecar_io import read_sidecar

    for path in sorted((FIXTURES / "gen3-sidecars").rglob("*.provenance.json")):
        sidecar = read_sidecar(path)
        for line in sidecar.lines:
            if line.script_text:
                yield line


def fixture_line(*needles: str):
    for line in fixture_lines():
        if all(n in line.script_text for n in needles):
            return line
    raise AssertionError(f"no fixture line holds {needles}")


def _cleared_entity(entity, attacker, blocker):
    """``entity`` with the pair's damage and keywords cleared and their block set."""
    if entity.id in (attacker, blocker):
        entity = replace(entity, damage=0, granted_temporary=replace(
            entity.granted_temporary, keywords=()))
    if entity.id == attacker:
        entity = replace(entity, combat=replace(entity.combat, blocked_by=(blocker,)))
    if entity.id == blocker:
        entity = replace(entity, combat=replace(entity.combat, blocking=(attacker,)))
    return entity


def one_on_one(records):
    """A fixture combat turned into one attacker blocked by one blocker, both
    cleared of damage and keywords."""
    from effects.domain.records import RecordKind

    record = next(r for r in records if r.kind is RecordKind.COMBAT
                  and sum(1 for b in r.payload.blocks.values() if b) == 1)
    attacker, (blocker, *_) = next((a, b) for a, b in record.payload.blocks.items() if b)
    entities = [_cleared_entity(e, attacker, blocker) for e in record.state.entities]
    record = replace(
        record, fork=False, interventional=False, mirror_of=None,
        payload=replace(record.payload, blocks={attacker: (blocker,)}, probed_keyword=None),
        state=replace(record.state, entities=tuple(entities),
                      global_=replace(record.state.global_, combat_substep="regular")))
    return record, attacker, blocker


def keyword_set(entity):
    return set(entity.granted_temporary.keywords)


def outcome_by_the_rule(monkeypatch, record, attacker: str, blocker: str) -> dict:
    """``derive_targets`` patched to report the deaths the damage-step rule gives the
    block; returned so a test can change one creature's outcome."""
    import effects.domain.effect_targets as effect_targets
    from effects.domain.effect_head_layout import ZONE_OUTCOMES

    a, _ = combat_grid.side_of(record, attacker)
    b, _ = combat_grid.side_of(record, blocker)
    died = ZONE_OUTCOMES.index("died")
    outcome = {e: SimpleNamespace(fields={"zone_outcome": died} if dies else {}, unknown=set())
               for e, dies in zip((attacker, blocker), combat_grid.step_deaths(a, b))}
    monkeypatch.setattr(effect_targets, "derive_targets", lambda r: outcome)
    return outcome


def block_pair(records):
    """A fixture block-legality record made to qualify: a red attacker, a green blocker."""
    from effects.domain.records import PlayabilityBlockersPayload

    record = next(r for r in records if isinstance(r.payload, PlayabilityBlockersPayload)
                  and r.payload.legal_blockers)
    attacker, blocker = record.payload.anchor_attacker, record.payload.legal_blockers[0]
    board = (BoardEdit(record).set_colours(attacker, ("R",)).set_colours(blocker, ("G",))
             .edit_entity(blocker, lambda e: replace(e, types=("creature",))))
    for entity in (attacker, blocker):
        board = board.edit_entity(entity, lambda e: replace(
            e, granted_temporary=replace(e.granted_temporary, keywords=())))
    return board.record, attacker, blocker


def differences(before, after) -> set[str]:
    """Which top-level parts of a record's state differ."""
    changed = set()
    if before.state.players != after.state.players:
        changed.add("players")
    if before.state.entities != after.state.entities:
        changed.add("entities")
    if before.payload != after.payload or before.ability != after.ability:
        changed.add("record")
    return changed


def decision_record(records, cost: str):
    """The fixture's decision record whose first candidate costs ``cost``."""
    return next(r for r in records if r.subkind is not None
                and r.subkind.value == "decision"
                and sweeps.candidate_mana(r) == cost)


def probabilities(values) -> None:
    values = np.asarray(values, dtype=float)
    assert values.size and np.all(np.isfinite(values))
    assert np.all((values >= 0.0) & (values <= 1.0))

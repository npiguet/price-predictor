"""Predicted deaths when the anthem keeping damaged creatures alive is destroyed (FR-074).

    python scripts/effect_knowledge_probes/anthem.py CHECKPOINT [--records 400]
        [--shards 200] [--per-game 3]

A static that raises toughness ("creatures you control get +1/+1", "equipped creature
gets +1/+2") keeps a creature alive that has lethal damage marked without it. The model
reads the anthem as a line of its own, the creatures it reaches as that line's
``affected``, and each creature's toughness with the boost already in it; this probe asks
whether it knows what happens to them when the anthem goes. It takes real validation
resolutions and combat damage steps whose board holds such an anthem and at least one
creature it keeps alive that the record saw survive, and destroys the anthem: the
permanent leaves the board with every link naming it, and each creature it reached loses
its boost and the lines it granted. Rule 704.5f-g then decides each creature's death on
the edited board (``rule_outcomes.damage_lethal`` at no further damage): the state-based
check the record's effect half spans puts every creature whose marked damage reaches its
new toughness into the graveyard.

Every creature the anthem reached is an item twice, as collected (the rule: it
survives, as the record saw) and with the anthem destroyed (the rule's answer), read at
its slot as the predicted death. The report gives the AUC over every item against the
rule and the mean predicted death as collected, and after the edit where the rule kills
the creature and where it does not.

Nothing here does work at import time.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from pathlib import Path

if __package__ in (None, ""):
    # Run by path: put scripts/ on the path so the suite imports as a package.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from effect_knowledge_probes import common, readout, record_walk  # noqa: E402
from effect_knowledge_probes.board_edits import BoardEdit  # noqa: E402
from effects.domain.rule_outcomes import damage_lethal  # noqa: E402
from effects.domain.script_text import params, root  # noqa: E402

AS_COLLECTED = "as collected"
DESTROYED = "anthem destroyed"


@dataclass(frozen=True, slots=True)
class Anthem:
    """A permanent's own static raising the toughness of what it affects by
    a stated amount: ``boost`` is ``(power, toughness)``."""

    entity: str
    key: object
    boost: tuple[int, int]
    affected: tuple[str, ...]


def _literal(value: str | None) -> int:
    value = (value or "").strip()
    return int(value) if value.lstrip("+").isdigit() else 0


def _printed_static(line) -> bool:
    """A printed static line in force that affects something."""
    return bool(line.kind == "static" and not line.granted and line.keys
                and line.in_force and line.affected)


def _toughness_params(line, lines) -> dict | None:
    """The ``Continuous`` static's parameters when it adds toughness, else None."""
    sidecar = lines.line_or_none(line.keys[0])
    if sidecar is None or not sidecar.script_text:
        return None
    found = params(root(sidecar.script_text))
    if found.get("Mode") != "Continuous" or _literal(found.get("AddToughness")) <= 0:
        return None
    return found


def _reached_creatures(record, line) -> tuple[str, ...]:
    """The line's affected cards that are creatures with a power and toughness."""
    def creature(ref) -> bool:
        card = record.state.entity(ref)
        return card is not None and card.is_creature and card.pt is not None

    return tuple(ref for ref in line.affected if creature(ref))


def _line_anthem(record, entity, line, lines) -> Anthem | None:
    found = _toughness_params(line, lines)
    if found is None:
        return None
    affected = _reached_creatures(record, line)
    if not affected:
        return None
    boost = (_literal(found.get("AddPower")), _literal(found.get("AddToughness")))
    return Anthem(entity.id, line.keys[0], boost, affected)


def _entity_anthems(record, entity, lines) -> list[Anthem]:
    found = (_line_anthem(record, entity, line, lines)
             for line in entity.lines if _printed_static(line))
    return [anthem for anthem in found if anthem is not None]


def anthems(record, lines) -> list[Anthem]:
    """Every in-force, printed static on the battlefield that adds a stated
    toughness to the creatures it affects, in board order."""
    out: list[Anthem] = []
    for entity in record.state.entities:
        if entity.on_battlefield:
            out += _entity_anthems(record, entity, lines)
    return out


def dies_without(record, anthem: Anthem, entity_id: str) -> bool:
    """Rule 704.5f-g on the board with ``anthem`` gone: whether the creature's
    marked damage reaches its toughness less the boost."""
    creature = record.state.entity(entity_id)
    return damage_lethal(creature.toughness - anthem.boost[1], creature.damage, 0)


def destroyed(record, anthem: Anthem):
    """``record`` with ``anthem``'s permanent gone and every creature it
    reached without its boost and without the lines it granted."""
    def unboosted(entity):
        pt = entity.pt
        boosts = (pt.boosts[0] - anthem.boost[0], pt.boosts[1] - anthem.boost[1])
        lines = tuple(line for line in entity.lines
                      if not (line.granted and anthem.key in line.keys))
        return replace(entity, pt=replace(pt, boosts=boosts), lines=lines)

    board = BoardEdit(record)
    for entity_id in anthem.affected:
        board = board.edit_entity(entity_id, unboosted)
    return board.remove_entity(anthem.entity).record


# ── which records qualify ───────────────────────────────────────────────


def _effect_half_or_combat(record) -> bool:
    """A non-fork, non-interventional resolution's effect half or combat damage step."""
    from effects.domain.records import Moment, RecordKind

    if record.fork or record.interventional:
        return False
    if record.kind is RecordKind.COMBAT:
        return True
    return record.kind is RecordKind.RESOLUTION and record.moment is Moment.RESOLUTION


def _kept_alive(record, anthem: Anthem) -> list[str]:
    """The creatures the anthem keeps alive that carry damage."""
    return [e for e in anthem.affected
            if record.state.entity(e).damage > 0 and dies_without(record, anthem, e)]


def _already_died(targets, anthem: Anthem) -> bool:
    """Whether the record already has one of the anthem's creatures dying."""
    from effects.domain.effect_head_layout import ZONE_OUTCOMES, HeadField, ZoneOutcome

    died = ZONE_OUTCOMES.index(ZoneOutcome.TO_GRAVEYARD)
    return any(targets.get(e) is not None
               and targets[e].fields.get(HeadField.ZONE_OUTCOME) == died
               for e in anthem.affected)


def chosen(record, lines):
    """``(record, anthem)`` for the first anthem keeping a creature alive
    that the record saw survive, on a resolution's effect half or a combat
    damage step that is no fork; None otherwise."""
    from effects.domain.effect_targets import derive_targets

    if not _effect_half_or_combat(record):
        return None
    targets = None
    for anthem in anthems(record, lines):
        if not _kept_alive(record, anthem):
            continue
        targets = targets if targets is not None else derive_targets(record)
        if not _already_died(targets, anthem):
            return record, anthem
    return None


def items(record, anthem: Anthem) -> list[tuple[str, object, str, bool]]:
    """``(edit, record, creature, rule says it dies)`` for every creature the
    anthem reached, as collected and with the anthem destroyed."""
    edited = destroyed(record, anthem)
    return [*((AS_COLLECTED, record, e, False) for e in anthem.affected),
            *((DESTROYED, edited, e, dies_without(record, anthem, e))
              for e in anthem.affected)]


# ── reading the model and scoring ───────────────────────────────────────


def predict_items(probe_model, rows, *, chunk: int = 32) -> list[float]:
    out: list[float] = []
    for start in range(0, len(rows), chunk):
        part = rows[start : start + chunk]
        out += common.predict(probe_model, [row[1] for row in part],
                              readout.at_cards("dies", [row[2] for row in part]))
    return out


def score(rows, predictions) -> dict:
    """The AUC over every item against the rule, and the mean predicted death
    as collected and after the edit, where the rule kills and where it spares."""
    import numpy as np

    labels = [row[3] for row in rows]

    def mean(keep) -> float | None:
        values = [p for row, p in zip(rows, predictions) if keep(row)]
        return float(np.mean(values)) if values else None

    return {
        "creatures": sum(1 for row in rows if row[0] == AS_COLLECTED),
        "auc": float(common.auc(labels, list(predictions))) if rows else float("nan"),
        "as_collected": mean(lambda row: row[0] == AS_COLLECTED),
        "destroyed_dies": mean(lambda row: row[0] == DESTROYED and row[3]),
        "destroyed_survives": mean(lambda row: row[0] == DESTROYED and not row[3]),
        "rule_dies": sum(1 for row in rows if row[0] == DESTROYED and row[3]),
    }


def report(result: dict, records: int) -> None:
    from _lib.markdown import print_table

    print(f"\nthe anthem destroyed: {records} records, {result['creatures']} creatures it "
          f"reached, {result['rule_dies']} of which the rule kills")
    print_table(["AUC against the rule", "predicted death as collected",
                 "destroyed, rule kills", "destroyed, rule spares"],
                [[common.fmt(result["auc"]), common.fmt(result["as_collected"]),
                  common.fmt(result["destroyed_dies"]),
                  common.fmt(result["destroyed_survives"])]], "rrrr")


def main() -> None:
    args, context = common.grid_run(__doc__, records=400, per_game=3)
    sidecars = context.sidecars()
    found = record_walk.qualifying(
        context.validation_shards(args.shards), lambda record: chosen(record, sidecars),
        args.records, per_game=args.per_game)
    rows = [row for record, anthem in found for row in items(record, anthem)]
    probe_model = context.model([row[1] for row in rows])
    common.heading(args, context, f"{len(found)} validation boards with an anthem keeping "
                   "a damaged creature alive")
    report(score(rows, predict_items(probe_model, rows)), len(found))


if __name__ == "__main__":
    main()

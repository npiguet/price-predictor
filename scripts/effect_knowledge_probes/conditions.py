"""Predicted outcomes that turn on whose turn it is and which step the turn is in (FR-074).

    python scripts/effect_knowledge_probes/conditions.py CHECKPOINT [--records 3000]
        [--shards 200] [--per-game 3]

Many abilities work only on their controller's turn, or in one step of it: "activate
only during your turn", "cast this spell only during combat", "during your turn,
CARDNAME gets +2/+2", "if it's your turn, draw a card". The model is told whose turn
it is and the step (the ``[GLOBAL]`` slot), and each line's text; this probe asks
whether it puts the two together. It takes real validation records whose acting line,
decision candidate or board carries such a condition, and reads the record as
collected and edited: whose turn handed to the other player, and the step moved into
or out of the steps the condition names. The rule's answer is computed for every
edited record, so the score holds whatever the real board was. A record qualifies only
where that computation agrees with what Forge recorded for it as collected.

Five subsets, each reading the output its condition decides:

=============  ==============================================  ========================
subset         records                                         reads
=============  ==============================================  ========================
restriction    decisions whose candidate states whose turn     the verdict head's
               (``PlayerTurn$``, ``OpponentTurn$``) or the      ``whose_turn`` and
               steps (``ActivationPhases$``) it may be played   ``phase`` failed checks
casting-speed  decisions on a sorcery-speed play               the ``casting_speed``
                                                               failed check
split-second   decisions with an object on the stack; edited   the ``split_second``
               by giving it split second or taking it away     failed check
static         a ``continuous`` or legality record whose        the affected gate at
               static holds only on a turn or in a step        each creature it
               (``Condition$ PlayerTurn``, ``Phases$``)         affects, or the legality
                                                               of each it keeps out
clause         a resolution whose conditional clause names      the affected gate at
               whose turn or the steps (``ConditionPlayerTurn$``) what that clause did
=============  ==============================================  ========================

An edit that changes whose turn or the step also sets the ``active`` flag of every
static on the board whose condition is one of these to what the edited board makes it,
so the board never contradicts itself: the snapshot states that answer beside the turn.

Each subset is scored as an AUC over its items, the record as collected and both edits
pooled, against the rule's answer; the table beside it gives the mean prediction where
the rule says yes and where it says no, per edit.

Nothing here does work at import time.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path

if __package__ in (None, ""):
    # Run by path: put scripts/ on the path so the suite imports as a package.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from effect_knowledge_probes import common, readout, record_walk  # noqa: E402
from effect_knowledge_probes.board_edits import BoardEdit  # noqa: E402
from effects.domain.rule_outcomes import (  # noqa: E402
    MAIN_PHASES,
    forge_phases,
    sorcery_timing,
    split_second_forbids,
)
from effects.domain.script_text import params, root, segments  # noqa: E402

SUBSETS: tuple[str, ...] = ("restriction", "casting-speed", "split-second", "static", "clause")
AS_COLLECTED = "as collected"
WHOSE_TURN = "whose turn"
PHASE = "phase"
SPLIT_SECOND_ADDED = "split second added"
SPLIT_SECOND_REMOVED = "split second removed"
#: The steps a phase edit moves a record to: none needs a combat on the board.
_QUIET_PHASES: tuple[str, ...] = ("upkeep", "draw", "main1", "combat_begin", "main2",
                                  "end_of_turn")
#: A static's ``Condition$`` values that name whose turn it is.
_TURN_CONDITIONS = {"PlayerTurn": True, "NotPlayerTurn": False}
#: Readouts at an entity's slot; the rest are the record's verdicts.
_ENTITY_READOUTS = frozenset({"affected", "may_block", "may_attack"})


@dataclass(frozen=True, slots=True)
class ConditionItem:
    """One question of the probe: ``record`` (as collected or edited) read
    through ``readout`` at ``entity`` (None for a verdict), and the rule's
    answer for it."""

    subset: str
    record_id: str
    edit: str
    record: object
    readout: str
    entity: str | None
    rule: bool


# ── the conditions ──────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class TurnFacts:
    """What a turn or step condition is checked against."""

    own_turn: bool
    phase: str
    stack_size: int

    @classmethod
    def of(cls, record, holder: str | None) -> TurnFacts:
        state = record.state.global_
        return cls(own_turn=holder is not None and state.active == holder,
                   phase=state.phase, stack_size=state.stack_size)


def whose_turn_holds(found: dict[str, str], own_turn: bool, *,
                     player: str = "PlayerTurn", opponent: str = "OpponentTurn") -> bool | None:
    """Whether a ``PlayerTurn$``/``OpponentTurn$``-style pair holds; None
    where the parameters name neither."""
    if found.get(player, "").lower() == "true":
        return own_turn
    if found.get(opponent, "").lower() == "true":
        return not own_turn
    return None


def phase_holds(value: str | None, phase: str) -> bool | None:
    named = forge_phases(value)
    return None if named is None else phase in named


def static_holds(found: dict[str, str], facts: TurnFacts) -> bool | None:
    """Whether a static's turn or step condition holds under ``facts``; None
    where it states none, or one beyond whose turn and the step."""
    answers = []
    condition = found.get("Condition")
    if condition is not None:
        if condition not in _TURN_CONDITIONS:
            return None
        answers.append(facts.own_turn is _TURN_CONDITIONS[condition])
    turn = whose_turn_holds(found, facts.own_turn)
    if turn is not None:
        answers.append(turn)
    if "Phases" in found:
        held = phase_holds(found["Phases"], facts.phase)
        if held is None:
            return None
        answers.append(held)
    return all(answers) if answers else None


def clause_holds(found: dict[str, str], facts: TurnFacts) -> bool | None:
    """Whether a resolving clause's ``ConditionPlayerTurn$``/
    ``ConditionOpponentTurn$``/``ConditionPhases$`` holds; None where it
    states none, or ``ConditionFirstCombat$`` as well, which no edit reads."""
    if "ConditionFirstCombat" in found:
        return None
    answers = []
    turn = whose_turn_holds(found, facts.own_turn, player="ConditionPlayerTurn",
                            opponent="ConditionOpponentTurn")
    if turn is not None:
        answers.append(turn)
    if "ConditionPhases" in found:
        held = phase_holds(found["ConditionPhases"], facts.phase)
        if held is None:
            return None
        answers.append(held)
    return all(answers) if answers else None


# ── the edits ───────────────────────────────────────────────────────────


def other_player(record, player: str) -> str | None:
    return next((p.id for p in record.state.players if p.id != player), None)


def turn_handed_over(record):
    """``record`` on the other player's turn; None with no other player."""
    active = record.state.global_.active
    other = other_player(record, active)
    return None if other is None else BoardEdit(record).set_active_player(other).record


def phase_moved(record, named: frozenset[str] | None):
    """``record`` moved into ``named`` when outside it, out of it when inside:
    the first step of :data:`_QUIET_PHASES` on the other side. None where no
    such step exists, or ``named`` is unknown."""
    if named is None:
        return None
    inside = record.state.global_.phase in named
    for phase in _QUIET_PHASES:
        if (phase in named) != inside:
            return BoardEdit(record).set_phase(phase).record
    return None


def casting_speed_moved(record):
    """``record`` moved out of its main phase, or into the first one."""
    phase = record.state.global_.phase
    target = "end_of_turn" if phase in MAIN_PHASES else "main1"
    return BoardEdit(record).set_phase(target).record


def resync_statics(record, lines):
    """``record`` with every turn- or step-conditioned static's ``active``
    set to what the board makes it: the snapshot states that answer beside
    whose turn it is, and an edit to one must move the other."""
    edit = BoardEdit(record)
    for entity in record.state.entities:
        for line in entity.lines or ():
            if line.active is None or not line.keys or line.kind != "static":
                continue
            sidecar = lines.line_or_none(line.keys[0])
            if sidecar is None or not sidecar.script_text:
                continue
            held = static_holds(params(root(sidecar.script_text)),
                                TurnFacts.of(record, entity.controller))
            if held is not None and held != line.active:
                edit = edit.set_line_active(entity.id, line.keys[0], held)
    return edit.record


def _edits(record, lines, named_phases) -> list[tuple[str, object]]:
    """The record as collected, on the other player's turn, and in a step on
    the other side of ``named_phases`` (when one exists), statics resynced."""
    out = [(AS_COLLECTED, record)]
    handed = turn_handed_over(record)
    if handed is not None:
        out.append((WHOSE_TURN, resync_statics(handed, lines)))
    moved = phase_moved(record, named_phases) if named_phases is not None else None
    if moved is not None:
        out.append((PHASE, resync_statics(moved, lines)))
    return out


# ── which records qualify ───────────────────────────────────────────────


def _candidate(record):
    from effects.domain.records import PlayabilityDecisionPayload

    payload = record.payload
    if (not isinstance(payload, PlayabilityDecisionPayload) or record.fork
            or not payload.candidates):
        return None
    candidate = payload.candidates[0]
    if candidate.failed_checks is None or not candidate.ability:
        return None
    return candidate


def _failed(candidate) -> set[str]:
    return {str(check.check) for check in candidate.failed_checks or ()}


def restriction_items(record, lines) -> list[ConditionItem]:
    """A decision whose candidate states whose turn or the steps it may be played in."""
    candidate = _candidate(record)
    if candidate is None:
        return []
    line = lines.first_line(candidate.ability)
    if line is None or not line.script_text:
        return []
    found = params(root(line.script_text))
    holder = record.actor_player
    checks = {}
    if whose_turn_holds(found, True) is not None:
        checks["whose_turn"] = lambda r: whose_turn_holds(found, TurnFacts.of(r, holder).own_turn)
    named = forge_phases(found.get("ActivationPhases"))
    if named is not None:
        checks["phase"] = lambda r: r.state.global_.phase in named
    if not checks:
        return []
    failed = _failed(candidate)
    # The rule must say what Forge said of the record as collected.
    if any(holds(record) == (check in failed) for check, holds in checks.items()):
        return []
    items = []
    for edit, edited in _edits(record, lines, named):
        for check, holds in checks.items():
            items.append(ConditionItem("restriction", record.record_id, edit, edited,
                                       f"failed_{check}", None, not holds(edited)))
    return items


def sorcery_speed(record, candidate, line, keywords_of) -> bool | None:
    """Whether a candidate is a sorcery-speed play: a land play (305.1), a
    spell that is no instant and has no flash, or an ability its script makes
    one (``SorcerySpeed$``, a loyalty ability). ``line`` None is the
    permanent spell every permanent carries, which no line prints. None where
    the record cannot say: a land's play on a record without ``land_play``."""
    if candidate.land_play or "casting_speed" in _failed(candidate):
        return True
    card = record.state.entity(record.state.refs.source or "")
    found = params(root(line.script_text or "")) if line is not None else {}
    if line is None or "SP" in found:
        if card is not None and "land" in card.types and candidate.land_play is None:
            return None
        if card is None:
            return None
        return "instant" not in card.types and "flash" not in keywords_of(card)
    return (found.get("SorcerySpeed", "").lower() == "true"
            or found.get("Planeswalker", "").lower() == "true")


def casting_speed_items(record, lines, keywords_of) -> list[ConditionItem]:
    candidate = _candidate(record)
    if candidate is None:
        return []
    line = lines.first_line(candidate.ability)
    if not sorcery_speed(record, candidate, line, keywords_of):
        return []
    holder = record.actor_player

    def fails(r) -> bool:
        facts = TurnFacts.of(r, holder)
        return not sorcery_timing(own_turn=facts.own_turn, phase=facts.phase,
                                  stack_size=facts.stack_size)

    if fails(record) != ("casting_speed" in _failed(candidate)):
        return []
    items = []
    edits = [(AS_COLLECTED, record)]
    handed = turn_handed_over(record)
    if handed is not None:
        edits.append((WHOSE_TURN, resync_statics(handed, lines)))
    edits.append((PHASE, resync_statics(casting_speed_moved(record), lines)))
    for edit, edited in edits:
        items.append(ConditionItem("casting-speed", record.record_id, edit, edited,
                                   "failed_casting_speed", None, fails(edited)))
    return items


def _stack(record) -> list:
    return [e for e in record.state.entities if e.zone == "stack"]


def has_split_second(entity) -> bool:
    return any(line.in_force and line.keyword == "split_second"
               for line in entity.lines or ()) or (
        "split_second" in entity.granted_temporary.keywords)


def without_split_second(entity):
    lines = entity.lines
    if lines is not None:
        lines = tuple(line for line in lines if line.keyword != "split_second")
    granted = entity.granted_temporary
    return replace(entity, lines=lines, granted_temporary=replace(
        granted, keywords=tuple(k for k in granted.keywords if k != "split_second")))


def split_second_items(record) -> list[ConditionItem]:
    """A decision with an object on the stack: as collected, with split second
    given to the stack's first object, and with every split second taken away."""
    candidate = _candidate(record)
    if candidate is None or candidate.land_play:
        return []
    stack = _stack(record)
    if not stack:
        return []

    def forbids(r) -> bool:
        return split_second_forbids(
            split_second_on_stack=any(has_split_second(e) for e in _stack(r)),
            land_play=bool(candidate.land_play))

    if forbids(record) != ("split_second" in _failed(candidate)):
        return []
    edits = [(AS_COLLECTED, record),
             (SPLIT_SECOND_ADDED, BoardEdit(record).grant_keyword(
                 stack[0].id, "split_second").record)]
    if forbids(record):
        board = BoardEdit(record)
        for entity in stack:
            board = board.edit_entity(entity.id, without_split_second)
        edits.append((SPLIT_SECOND_REMOVED, board.record))
    return [ConditionItem("split-second", record.record_id, edit, edited,
                          "failed_split_second", None, forbids(edited))
            for edit, edited in edits]


def _host(record, key):
    for entity in record.state.entities:
        if any(key in line.keys for line in entity.lines or ()):
            return entity
    return None


def static_items(record, lines) -> list[ConditionItem]:
    """A ``continuous`` record of a turn- or step-conditioned static, read at
    each creature it affects (its host where it affects the host alone), or a
    legality record such a static keeps creatures out of, read at each one."""
    from effects.domain.corpus_curation import turn_step_condition
    from effects.domain.records import (
        ContinuousPayload,
        PlayabilityAttackersPayload,
        PlayabilityBlockersPayload,
    )

    if record.fork:
        return []
    condition = turn_step_condition(record, lines)
    if condition is None:
        return []
    key = condition.carrier
    line = lines.line_for(key)
    host = _host(record, key)
    if line is None or not line.script_text or host is None:
        return []
    found = params(root(line.script_text))
    held = static_holds(found, TurnFacts.of(record, host.controller))
    if held is None or held != (str(condition.side) == "met"):
        return []
    payload = record.payload
    named = forge_phases(found.get("Phases"))
    readings: list[tuple[str, str, bool]] = []  # (readout, entity, yes when it holds)
    if isinstance(payload, ContinuousPayload):
        affected = {c.entity for c in payload.contributions}
        if found.get("Affected") == "Card.Self":
            affected.add(host.id)
        elif not held:
            return []  # an unmet static's reach is unknown
        readings = [("affected", entity, True) for entity in sorted(affected)]
    elif isinstance(payload, PlayabilityAttackersPayload | PlayabilityBlockersPayload):
        target = ("may_attack" if isinstance(payload, PlayabilityAttackersPayload)
                  else "may_block")
        # Only where this static alone keeps a creature out does its condition
        # decide the creature's legality.
        kept = sorted({entry.entity for entry in payload.forbidden
                       if tuple(entry.responsible_static) == (key,)})
        readings = [(target, entity, False) for entity in kept]
    items = []
    for edit, edited in _edits(record, lines, named):
        holds = static_holds(found, TurnFacts.of(edited, host.controller))
        if holds is None:
            continue
        for target, entity, when_held in readings:
            items.append(ConditionItem("static", record.record_id, edit, edited, target,
                                       entity, holds == when_held))
    return items


def clause_items(record, lines) -> list[ConditionItem]:
    """A resolution whose conditional clause names whose turn or the steps
    and resolved, read at each entity only that clause's events touched."""
    from effects.domain.records import Moment, RecordKind, ResolutionPayload

    payload = record.payload
    if (record.kind is not RecordKind.RESOLUTION or record.moment is not Moment.RESOLUTION
            or record.fork or not isinstance(payload, ResolutionPayload)
            or not payload.conditions_met or not record.ability):
        return []
    line = lines.first_line(record.ability)
    if line is None or not line.script_text:
        return []
    chain = segments(line.script_text)
    holder = record.actor_player
    for entry in payload.conditions_met:
        index = 0 if entry.segment == "root" else (
            int(entry.segment) + 1 if entry.segment.isdigit() else None)
        if index is None or index >= len(chain) or not entry.met:
            continue
        found = params(chain[index])
        if clause_holds(found, TurnFacts.of(record, holder)) is not True:
            continue
        touched: dict[str, set[str]] = defaultdict(set)
        for event in payload.events:
            for subject in event.subjects:
                touched[subject].add(event.attributed_to or "")
        own = sorted(e for e, by in touched.items() if by == {entry.segment}
                     and record.state.entity(e) is not None)
        if not own:
            continue
        named = forge_phases(found.get("ConditionPhases"))
        items = []
        for edit, edited in _edits(record, lines, named):
            holds = clause_holds(found, TurnFacts.of(edited, holder))
            for entity in own:
                items.append(ConditionItem("clause", record.record_id, edit, edited,
                                           "affected", entity, bool(holds)))
        return items
    return []


def record_items(record, lines, keywords_of) -> list[ConditionItem]:
    """Every subset's items of one record: a decision may sit in three."""
    return [*restriction_items(record, lines), *casting_speed_items(record, lines, keywords_of),
            *split_second_items(record), *static_items(record, lines),
            *clause_items(record, lines)]


# ── reading the model ───────────────────────────────────────────────────


def predict_items(probe_model, items: list[ConditionItem], *, chunk: int = 32) -> list[float]:
    """Each item's reading, ``chunk`` records a forward pass."""
    out: list[float] = []
    for start in range(0, len(items), chunk):
        part = items[start : start + chunk]

        def read(outputs, verdict, surface, row, part=part):
            item = part[row]
            if item.readout in _ENTITY_READOUTS:
                return readout.at_card(item.readout, item.entity)(outputs, verdict, surface, row)
            return readout.read(item.readout, outputs, verdict, row, 0)

        out += common.predict(probe_model, [item.record for item in part], read)
    return out


def score(items: Iterable[ConditionItem], predictions: Iterable[float]) -> dict:
    """``{subset: {"items", "records", "auc", "by_edit": {edit: {...}}}}``:
    each subset's AUC over every item against the rule, and per edit the
    mean prediction where the rule says yes and where it says no."""
    import numpy as np

    grouped: dict[str, list[tuple[ConditionItem, float]]] = defaultdict(list)
    for item, prediction in zip(items, predictions):
        grouped[item.subset].append((item, prediction))
    out: dict = {}
    for subset in SUBSETS:
        rows = grouped.get(subset, [])
        labels = [item.rule for item, _ in rows]
        scores = [p for _, p in rows]
        by_edit: dict = {}
        for edit in dict.fromkeys(item.edit for item, _ in rows):
            yes = [p for item, p in rows if item.edit == edit and item.rule]
            no = [p for item, p in rows if item.edit == edit and not item.rule]
            by_edit[edit] = {"yes": len(yes), "no": len(no),
                             "mean_yes": float(np.mean(yes)) if yes else None,
                             "mean_no": float(np.mean(no)) if no else None}
        out[subset] = {
            "items": len(rows), "records": len({item.record_id for item, _ in rows}),
            "auc": float(common.auc(labels, scores)) if rows else float("nan"),
            "by_edit": by_edit,
        }
    return out


def report(scores: dict) -> None:
    from _lib.markdown import print_table

    print("\neach subset's AUC against the rule, every item pooled")
    print_table(["subset", "records", "items", "AUC"],
                [[s, body["records"], body["items"], common.fmt(body["auc"])]
                 for s, body in scores.items()], "lrrr")
    rows = [[subset, edit, cell["yes"], common.fmt(cell["mean_yes"]), cell["no"],
             common.fmt(cell["mean_no"])]
            for subset, body in scores.items() for edit, cell in body["by_edit"].items()]
    print("\nmean prediction where the rule says yes and where it says no, per edit")
    print_table(["subset", "edit", "yes", "mean", "no", "mean"], rows, "llrrrr")


def main() -> None:
    from effects.domain.damage_step_keywords import KeywordResolver

    args, context = common.grid_run(__doc__, records=3000, per_game=3)
    sidecars = context.sidecars()
    keywords_of = KeywordResolver(sidecars).keywords_of
    chosen = record_walk.qualifying(
        context.validation_dirs(),
        lambda record: record_items(record, sidecars, keywords_of) or None,
        args.records, per_game=args.per_game, shards=args.shards)
    items = [item for found in chosen for item in found]
    probe_model = context.model([item.record for item in items])
    common.heading(args, context, f"{len(chosen)} validation records carrying a turn or "
                   "step condition")
    report(score(items, predict_items(probe_model, items)))


if __name__ == "__main__":
    main()

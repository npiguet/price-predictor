"""Predicted target legality, grouped by the restriction the acting ability states.

    python scripts/effect_knowledge_probes/target_restrictions.py CHECKPOINT
        [--records 1500] [--shards 200] [--per-game 3]

The knowledge probes score one ``legal_target`` question over every targeting
decision. This probe splits it by what the acting ability's script restricts its
targets to: the ``ValidTgts$`` value of its first segment, such as ``Creature.OppCtrl``
or ``Artifact,Enchantment``. A real playability decision record names its candidate
ability and Forge's legal targets for it; every card and every player on its board is
one item, labelled legal or not by Forge, and the model's answer is its ``target_legal``
output at that card's or player's slot (FR-074): gen-3 trains that field on every card
and player of a decision whose candidate targets, 1 at a legal target and 0 elsewhere.
The affected gate, which an earlier probe read because its field was only ever trained
to 1, is reported beside it.

Two breakdowns are reported, each as AUC over the items it holds (``target_legal``,
then the gate):

- by **family**, the kind of condition the restriction states: a card type alone,
  who controls it, a supertype (nonLegendary), a colour, a keyword it has, a number
  it is compared with (power, toughness, mana value), tapped or attacking, or other;
- by **restriction**, every ``ValidTgts$`` value with enough items of both labels,
  numbers masked so ``powerLE2`` and ``powerLE3`` share one row.

Counterfactual flips then take legal battlefield targets of single-condition
restrictions and break exactly that condition: hand the target to the other player
(``OppCtrl``, ``YouCtrl``), make it legendary (``nonLegendary``), or remove the one card
type the restriction names. The rule turns each such target illegal, and the probe
reports the model's mean predicted ``target_legal`` before and after.
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    # Run by path: put scripts/ on the path so the suite imports as a package.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _lib.markdown import print_table  # noqa: E402

from effect_knowledge_probes import common, readout, record_walk  # noqa: E402
from effect_knowledge_probes.board_edits import BoardEdit  # noqa: E402
from effects.domain.script_text import root_param  # noqa: E402

BASE_TYPES = {"Artifact": "artifact", "Creature": "creature", "Enchantment": "enchantment",
              "Land": "land", "Planeswalker": "planeswalker", "Battle": "battle"}
COLOUR_QUALIFIERS = {"White", "Blue", "Black", "Red", "Green", "nonWhite", "nonBlue",
                     "nonBlack", "nonRed", "nonGreen", "Colorless", "nonColorless",
                     "MultiColor", "MonoColor"}


# ── the restriction ─────────────────────────────────────────────────────


def masked(restriction: str) -> str:
    return re.sub(r"\d+", "#", restriction)


def _controller_qualifier(qualifier: str) -> bool:
    return qualifier in ("OppCtrl", "YouCtrl", "YouDontCtrl")


def _supertype_qualifier(qualifier: str) -> bool:
    return qualifier.lower().removeprefix("non") in ("legendary", "basic", "snow")


def _colour_qualifier(qualifier: str) -> bool:
    return qualifier in COLOUR_QUALIFIERS


def _keyword_qualifier(qualifier: str) -> bool:
    return qualifier.startswith(("with", "without"))


def _number_qualifier(qualifier: str) -> bool:
    return bool(re.match(r"(power|toughness|cmc|manaValue)(LT|LE|EQ|GE|GT)", qualifier))


def _combat_qualifier(qualifier: str) -> bool:
    return qualifier in ("tapped", "untapped", "attacking", "blocking", "attackingYou")


#: Each family with the test of one qualifier that puts a restriction in it, in the
#: order the first match is taken.
FAMILY_TESTS = (
    ("controller", _controller_qualifier),
    ("supertype", _supertype_qualifier),
    ("colour", _colour_qualifier),
    ("keyword", _keyword_qualifier),
    ("number", _number_qualifier),
    ("combat or tapped", _combat_qualifier),
)


def family(restriction: str) -> str:
    """The kind of condition a restriction states, first match wins."""
    qualifiers = [q for alt in restriction.split(",") for q in re.split(r"[.+]", alt)[1:]]
    for name, test in FAMILY_TESTS:
        if any(test(q) for q in qualifiers):
            return name
    return "other" if qualifiers else "card type only"


# ── the flips ───────────────────────────────────────────────────────────


def flip_for(restriction: str) -> str | None:
    """The single condition a restriction states that a flip can break, or None."""
    if "," in restriction:
        return None
    base, *qualifiers = re.split(r"[.+]", restriction)
    if qualifiers == ["OppCtrl"]:
        return "to the acting player"
    if qualifiers == ["YouCtrl"]:
        return "to the opponent"
    if qualifiers == ["nonLegendary"]:
        return "made legendary"
    if not qualifiers and base in BASE_TYPES:
        return f"loses {BASE_TYPES[base]}"
    return None


def flipped(record, entity_id: str, flip: str):
    """``record`` with one target changed so the restriction no longer holds."""
    actor = record.actor_player
    board = BoardEdit(record)
    if flip == "to the acting player":
        return board.set_controller(entity_id, actor).record
    if flip == "to the opponent":
        other = next(p.id for p in record.state.players if p.id != actor)
        return board.set_controller(entity_id, other).record
    if flip == "made legendary":
        return board.add_supertype(entity_id, "legendary").record
    if flip.startswith("loses "):
        return board.remove_type(entity_id, flip.removeprefix("loses ")).record
    raise ValueError(flip)


# ── which records qualify ───────────────────────────────────────────────


def _targeting_candidate(record):
    """The first candidate of a non-fork playability decision with legal targets, or None."""
    from effects.domain.records import PlayabilityDecisionPayload

    payload = record.payload
    if not isinstance(payload, PlayabilityDecisionPayload) or record.fork:
        return None
    if not payload.candidates or not payload.candidates[0].legal_targets:
        return None
    return payload.candidates[0]


def _restriction_of(candidate, sidecars):
    """The ``ValidTgts`` value of the candidate's first ability line that has one."""
    restriction = None
    for key in candidate.ability or ():
        line = sidecars.line_or_none(key)
        restriction = root_param(line.script_text if line is not None else None, "ValidTgts")
        if restriction:
            break
    return restriction


def decision_items(record, sidecars):
    """``(restriction, [(entity or player, legal)])`` for one targeting
    decision, or None: every card of the board, in any zone, and every player,
    as ``target_legal`` is trained on them."""
    candidate = _targeting_candidate(record)
    if candidate is None:
        return None
    restriction = _restriction_of(candidate, sidecars)
    if not restriction:
        return None
    legal = set(candidate.legal_targets)
    refs = [e.id for e in record.state.entities] + [p.id for p in record.state.players]
    items = [(ref, ref in legal) for ref in refs]
    return (restriction, items) if items else None


#: The two answers read at every item: the field, then the gate an earlier probe read.
READOUTS: tuple[str, ...] = ("target_legal", "affected")


def read_items(probe_model, records: list) -> dict[str, list[dict[str, float]]]:
    """``{readout: [{entity or player: answer}]}`` per record, both readouts."""
    return {name: common.predict_all(probe_model, records, readout.at_every_entity(name))
            for name in READOUTS}


def _new_cell() -> dict:
    return {r: ([], []) for r in READOUTS}


def _scores_at(readings, row: int, entity: str) -> dict[str, float] | None:
    """Both readouts' answers at ``entity`` of record ``row``, None if either lacks it."""
    if any(entity not in readings[name][row] for name in READOUTS):
        return None
    return {name: readings[name][row][entity] for name in READOUTS}


def _add_item(out: dict, keys, legal: bool, scores: dict[str, float]) -> None:
    for table, key in keys:
        cell = out[table][key]
        for name in READOUTS:
            cell[name][0].append(legal)
            cell[name][1].append(scores[name])


def breakdowns(chosen, readings) -> dict[str, dict]:
    """``{"family" | "restriction": {key: {readout: (labels, scores)}}}``."""
    out: dict[str, dict] = {"family": defaultdict(_new_cell),
                            "restriction": defaultdict(_new_cell)}
    for row, (_, restriction, items) in enumerate(chosen):
        for entity, legal in items:
            scores = _scores_at(readings, row, entity)
            if scores is not None:
                keys = (("family", family(restriction)),
                        ("restriction", masked(restriction)))
                _add_item(out, keys, legal, scores)
    return out


def _legal_on_battlefield(record, entity: str, legal: bool) -> bool:
    if not legal:
        return False
    target = record.state.entity(entity)
    return target is not None and target.zone == "battlefield"


def flip_items(chosen) -> list[tuple[str, str, object, object]]:
    """``(flip, entity, record, flipped record)`` for every legal battlefield
    target of a single-condition restriction."""
    out = []
    for record, restriction, items in chosen:
        flip = flip_for(restriction)
        if flip is not None:
            out += [(flip, entity, record, flipped(record, entity, flip))
                    for entity, legal in items if _legal_on_battlefield(record, entity, legal)]
    return out


# ── the report ──────────────────────────────────────────────────────────


def _largest_first(table):
    return sorted(table.items(), key=lambda kv: -len(kv[1][READOUTS[0]][0]))


def _aucs(by_readout) -> list[str]:
    return [f"{common.auc(*by_readout[name]):.3f}" for name in READOUTS]


def _enough_of_both(cell, min_each: int) -> bool:
    legal = sum(cell[READOUTS[0]][0])
    return min_each <= legal <= len(cell[READOUTS[0]][0]) - min_each


def _print_by_family(tables) -> None:
    print("\nby family")
    print_table(["family", "items", "legal", "AUC, target_legal", "AUC, gate"],
                [[key, len(cell[READOUTS[0]][0]), sum(cell[READOUTS[0]][0]), *_aucs(cell)]
                 for key, cell in _largest_first(tables["family"])], "lrrrr")


def _print_by_restriction(tables, min_each: int) -> None:
    print("\nby restriction")
    print_table(["restriction", "family", "items", "legal", "AUC, target_legal", "AUC, gate"],
                [[f"`{key}`", family(key), len(cell[READOUTS[0]][0]),
                  sum(cell[READOUTS[0]][0]), *_aucs(cell)]
                 for key, cell in _largest_first(tables["restriction"])
                 if _enough_of_both(cell, min_each)], "llrrrr")


def _flip_readings(probe_model, chosen, readings) -> dict:
    """``{flip: ([before], [after])}`` of predicted ``target_legal`` per flipped target."""
    flips = defaultdict(lambda: ([], []))
    index = flip_items(chosen)
    after = common.predict_all(probe_model, [row[3] for row in index],
                               readout.at_every_entity("target_legal"))
    before_of = {c[0].record_id: reading for c, reading in zip(chosen, readings["target_legal"])}
    for (flip, entity, record, _), a in zip(index, after):
        b = before_of[record.record_id]
        if entity in b and entity in a:
            flips[flip][0].append(b[entity])
            flips[flip][1].append(a[entity])
    return flips


def _print_flips(flips) -> None:
    print("\ncounterfactual flips: a legal target changed so the restriction fails "
          "(rule: legal -> illegal), predicted target_legal")
    print_table(["flip", "targets", "predicted legal before", "after"],
                [[flip, len(b), f"{np.mean(b):.3f}", f"{np.mean(a):.3f}"]
                 for flip, (b, a) in sorted(flips.items(), key=lambda kv: -len(kv[1][0]))],
                "lrrr")


def main() -> None:
    args, context = common.grid_run(
        __doc__, records=1500, per_game=3,
        options=lambda parser: parser.add_argument(
            "--min-each", type=int, default=15,
            help="items of each label a restriction row needs"))
    sidecars = context.sidecars()
    chosen = record_walk.qualifying(
        context.validation_dirs(), record_walk.with_record(lambda r: decision_items(r, sidecars)),
        args.records, per_game=args.per_game, shards=args.shards)
    probe_model = context.model([c[0] for c in chosen])
    common.heading(args, context, f"{len(chosen)} validation targeting decisions")

    readings = read_items(probe_model, [c[0] for c in chosen])
    tables = breakdowns(chosen, readings)

    _print_by_family(tables)
    _print_by_restriction(tables, args.min_each)
    _print_flips(_flip_readings(probe_model, chosen, readings))


if __name__ == "__main__":
    main()

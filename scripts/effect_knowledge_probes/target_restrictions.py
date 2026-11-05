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
The affected gate, which gen-2's probe read because its field was only ever trained to
1, is reported beside it.

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


def family(restriction: str) -> str:
    """The kind of condition a restriction states, first match wins."""
    qualifiers = [q for alt in restriction.split(",") for q in re.split(r"[.+]", alt)[1:]]
    if any(q in ("OppCtrl", "YouCtrl", "YouDontCtrl") for q in qualifiers):
        return "controller"
    if any(q.lower().removeprefix("non") in ("legendary", "basic", "snow") for q in qualifiers):
        return "supertype"
    if any(q in COLOUR_QUALIFIERS for q in qualifiers):
        return "colour"
    if any(q.startswith("with") or q.startswith("without") for q in qualifiers):
        return "keyword"
    if any(re.match(r"(power|toughness|cmc|manaValue)(LT|LE|EQ|GE|GT)", q)
           for q in qualifiers):
        return "number"
    if any(q in ("tapped", "untapped", "attacking", "blocking", "attackingYou")
           for q in qualifiers):
        return "combat or tapped"
    if not qualifiers:
        return "card type only"
    return "other"


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


def decision_items(record, sidecars):
    """``(restriction, [(entity or player, legal)])`` for one targeting
    decision, or None: every card of the board, in any zone, and every player,
    as ``target_legal`` is trained on them."""
    from effects.domain.records import PlayabilityDecisionPayload

    payload = record.payload
    if (not isinstance(payload, PlayabilityDecisionPayload) or record.fork
            or not payload.candidates or not payload.candidates[0].legal_targets):
        return None
    candidate = payload.candidates[0]
    restriction = None
    for key in candidate.ability or ():
        line = sidecars.line_or_none(key)
        restriction = root_param(line.script_text if line is not None else None, "ValidTgts")
        if restriction:
            break
    if not restriction:
        return None
    legal = set(candidate.legal_targets)
    refs = [e.id for e in record.state.entities] + [p.id for p in record.state.players]
    items = [(ref, ref in legal) for ref in refs]
    return (restriction, items) if items else None


#: The two answers read at every item: the field, then the gate gen-2 read.
READOUTS: tuple[str, ...] = ("target_legal", "affected")


def read_items(probe_model, records: list) -> dict[str, list[dict[str, float]]]:
    """``{readout: [{entity or player: answer}]}`` per record, both readouts."""
    return {name: common.predict_all(probe_model, records, readout.at_every_entity(name))
            for name in READOUTS}


def breakdowns(chosen, readings) -> dict[str, dict]:
    """``{"family" | "restriction": {key: {readout: (labels, scores)}}}``."""
    out: dict[str, dict] = {"family": defaultdict(lambda: {r: ([], []) for r in READOUTS}),
                            "restriction": defaultdict(
                                lambda: {r: ([], []) for r in READOUTS})}
    for row, (record, restriction, items) in enumerate(chosen):
        for entity, legal in items:
            if any(entity not in readings[name][row] for name in READOUTS):
                continue
            for table, key in (("family", family(restriction)),
                               ("restriction", masked(restriction))):
                for name in READOUTS:
                    out[table][key][name][0].append(legal)
                    out[table][key][name][1].append(readings[name][row][entity])
    return out


def flip_items(chosen) -> list[tuple[str, str, object, object]]:
    """``(flip, entity, record, flipped record)`` for every legal battlefield
    target of a single-condition restriction."""
    out = []
    for record, restriction, items in chosen:
        flip = flip_for(restriction)
        if flip is None:
            continue
        for entity, legal in items:
            target = record.state.entity(entity)
            if legal and target is not None and target.zone == "battlefield":
                out.append((flip, entity, record, flipped(record, entity, flip)))
    return out


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

    def largest_first(table):
        return sorted(table.items(), key=lambda kv: -len(kv[1][READOUTS[0]][0]))

    def aucs(by_readout) -> list[str]:
        return [f"{common.auc(*by_readout[name]):.3f}" for name in READOUTS]

    print("\nby family")
    print_table(["family", "items", "legal", "AUC, target_legal", "AUC, gate"],
                [[key, len(cell[READOUTS[0]][0]), sum(cell[READOUTS[0]][0]), *aucs(cell)]
                 for key, cell in largest_first(tables["family"])], "lrrrr")

    print("\nby restriction")
    print_table(["restriction", "family", "items", "legal", "AUC, target_legal", "AUC, gate"],
                [[f"`{key}`", family(key), len(cell[READOUTS[0]][0]),
                  sum(cell[READOUTS[0]][0]), *aucs(cell)]
                 for key, cell in largest_first(tables["restriction"])
                 if args.min_each <= sum(cell[READOUTS[0]][0])
                 <= len(cell[READOUTS[0]][0]) - args.min_each], "llrrrr")

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
    print("\ncounterfactual flips: a legal target changed so the restriction fails "
          "(rule: legal -> illegal), predicted target_legal")
    print_table(["flip", "targets", "predicted legal before", "after"],
                [[flip, len(b), f"{np.mean(b):.3f}", f"{np.mean(a):.3f}"]
                 for flip, (b, a) in sorted(flips.items(), key=lambda kv: -len(kv[1][0]))],
                "lrrr")


if __name__ == "__main__":
    main()

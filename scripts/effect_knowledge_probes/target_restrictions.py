"""Predicted target legality, grouped by the restriction the acting ability states.

    python scripts/effect_knowledge_probes/target_restrictions.py CHECKPOINT
        [--records 1500] [--shards 200] [--per-game 3]

The knowledge probes score one ``legal_target`` question over every targeting
decision. This probe splits it by what the acting ability's script restricts its
targets to: the ``ValidTgts$`` value of its first segment, such as ``Creature.OppCtrl``
or ``Artifact,Enchantment``. A real playability decision record names its candidate
ability and Forge's legal targets for it; every battlefield permanent on its board is
one item, labelled legal or not by Forge, and the model's answer is the affected
gate at that permanent's slot, the read-out the knowledge probes use.

Two breakdowns are reported, each as AUC over the items it holds:

- by **family**, the kind of condition the restriction states: a card type alone,
  who controls it, a supertype (nonLegendary), a colour, a keyword it has, a number
  it is compared with (power, toughness, mana value), tapped or attacking, or other;
- by **restriction**, every ``ValidTgts$`` value with enough items of both labels,
  numbers masked so ``powerLE2`` and ``powerLE3`` share one row.

Counterfactual flips then take legal targets of single-condition restrictions and
break exactly that condition: hand the target to the other player (``OppCtrl``,
``YouCtrl``), make it legendary (``nonLegendary``), or remove the one card type the
restriction names. The rule turns each such target illegal, and the probe reports
the model's mean predicted legality before and after.
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    # Run by path: put scripts/ on the path so the suite imports as a package.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from effect_knowledge_probes import common, record_walk  # noqa: E402
from effect_knowledge_probes.board_edits import BoardEdit  # noqa: E402

BASE_TYPES = {"Artifact": "artifact", "Creature": "creature", "Enchantment": "enchantment",
              "Land": "land", "Planeswalker": "planeswalker", "Battle": "battle"}
COLOUR_QUALIFIERS = {"White", "Blue", "Black", "Red", "Green", "nonWhite", "nonBlue",
                     "nonBlack", "nonRed", "nonGreen", "Colorless", "nonColorless",
                     "MultiColor", "MonoColor"}
_VALID_TGTS = re.compile(r"ValidTgts\$\s*([^|\[]+)")


# ── the restriction ─────────────────────────────────────────────────────


def restriction_of(script_text: str | None) -> str | None:
    """The ``ValidTgts$`` value of a chain's first segment, or None."""
    if not script_text:
        return None
    first = script_text.split("[SEG]", 1)[0]
    match = _VALID_TGTS.search(first)
    return match.group(1).strip() if match else None


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
    """``(restriction, [(entity, legal)])`` for one targeting decision, or None."""
    from effects.domain.records import PlayabilityDecisionPayload

    payload = record.payload
    if (not isinstance(payload, PlayabilityDecisionPayload) or record.fork
            or not payload.candidates or not payload.candidates[0].legal_targets):
        return None
    candidate = payload.candidates[0]
    restriction = None
    for key in candidate.ability or ():
        line = sidecars.line_or_none(key)
        restriction = restriction_of(line.script_text if line is not None else None)
        if restriction:
            break
    if not restriction:
        return None
    legal = set(candidate.legal_targets)
    items = [(e.id, e.id in legal) for e in record.state.entities if e.zone == "battlefield"]
    return (restriction, items) if items else None


def qualifying_records(directories, count: int, shards: int, per_game: int, sidecars):
    def choose(record):
        found = decision_items(record, sidecars)
        return None if found is None else (record, *found)

    return record_walk.qualifying(
        directories, choose, count, per_game=per_game, shards=shards)


# ── reading the model ───────────────────────────────────────────────────


def _gates(outputs, surface, row) -> dict[str, float]:
    import torch

    from effects.domain.effect_head_layout import GATE_INDEX

    return {slot.entity_id: float(torch.sigmoid(outputs[row, index, GATE_INDEX]))
            for index, slot in enumerate(surface.slots)
            if slot.kind.name == "CARD" and slot.entity_id is not None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--records", type=int, default=1500)
    parser.add_argument("--shards", type=int, default=200)
    parser.add_argument("--per-game", type=int, default=3)
    parser.add_argument("--min-each", type=int, default=15,
                        help="items of each label a restriction row needs")
    args = parser.parse_args()

    context = common.probe_context(args.checkpoint)
    chosen = qualifying_records(context.validation_dirs(), args.records, args.shards,
                                args.per_game, context.sidecars())
    probe_model = context.model([c[0] for c in chosen])
    print(f"{args.checkpoint.parent.name} ({context.checkpoint.epoch}), {len(chosen)} validation "
          f"targeting decisions", flush=True)

    gates = common.predict_all(probe_model, [c[0] for c in chosen],
                               lambda o, v, s, r: _gates(o, s, r))
    by_family, by_restriction = defaultdict(lambda: ([], [])), defaultdict(lambda: ([], []))
    for (record, restriction, items), gate in zip(chosen, gates):
        for entity, legal in items:
            if entity not in gate:
                continue
            for table, key in ((by_family, family(restriction)),
                               (by_restriction, masked(restriction))):
                table[key][0].append(legal)
                table[key][1].append(gate[entity])

    print("\nby family")
    print("| family | items | legal | AUC |")
    print("|---|---:|---:|---:|")
    for key, (labels, scores) in sorted(by_family.items(), key=lambda kv: -len(kv[1][0])):
        print(f"| {key} | {len(labels)} | {sum(labels)} | "
              f"{common.auc(labels, scores):.3f} |")

    print("\nby restriction")
    print("| restriction | family | items | legal | AUC |")
    print("|---|---|---:|---:|---:|")
    for key, (labels, scores) in sorted(by_restriction.items(), key=lambda kv: -len(kv[1][0])):
        positives = sum(labels)
        if positives < args.min_each or len(labels) - positives < args.min_each:
            continue
        print(f"| `{key}` | {family(key)} | {len(labels)} | {positives} | "
              f"{common.auc(labels, scores):.3f} |")

    flips = defaultdict(lambda: ([], []))
    flip_records, flip_index = [], []
    for record, restriction, items in chosen:
        flip = flip_for(restriction)
        if flip is None:
            continue
        for entity, legal in items:
            if legal:
                flip_records.append(flipped(record, entity, flip))
                flip_index.append((flip, entity, record))
    after = common.predict_all(probe_model, flip_records,
                               lambda o, v, s, r: _gates(o, s, r))
    gate_of = {c[0].record_id: gate for c, gate in zip(chosen, gates)}
    before = [gate_of[r.record_id] for _, _, r in flip_index]
    for (flip, entity, _), b, a in zip(flip_index, before, after):
        if entity in b and entity in a:
            flips[flip][0].append(b[entity])
            flips[flip][1].append(a[entity])
    print("\ncounterfactual flips: a legal target changed so the restriction fails "
          "(rule: legal -> illegal)")
    print("| flip | targets | predicted legal before | after |")
    print("|---|---:|---:|---:|")
    for flip, (b, a) in sorted(flips.items(), key=lambda kv: -len(kv[1][0])):
        print(f"| {flip} | {len(b)} | {np.mean(b):.3f} | {np.mean(a):.3f} |", flush=True)


if __name__ == "__main__":
    main()

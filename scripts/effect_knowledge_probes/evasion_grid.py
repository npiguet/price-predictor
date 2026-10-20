"""Predicted block legality under each evasion keyword, against each blocker trait.

    python scripts/effect_knowledge_probes/evasion_grid.py CHECKPOINT [--records 40]
        [--shards 200]

The blocking counterpart of ``combat_grid.py``. A real block-legality record anchors
one attacker and lists the creatures that may block it. A triple qualifies when the
candidate blocker was legal, neither creature carries an evasion keyword or a trait
the grid sets, the blocker is a coloured non-artifact creature that is not black and
shares no colour with the attacker, and the attacker has a colour other than black.
One triple per game.

Each grid cell grants the attacker one evasion keyword and gives the blocker one
trait, both through channels the model reads (the until-end-of-turn keyword overlay,
the type and colour lists), and reads the predicted legality of that blocker for that
attacker (the ``blocker_legal`` field at the blocker's slot). The rule's answer is
computed for every edited record by ``may_block``, so the score holds whatever the
real creatures are.

Protection from a colour is two rows: from the blocker's colour, where the rule
forbids the block, and from another colour, where it does not. The model's keyword
flags carry protection without its colour, so the two rows give it identical inputs
and measure what the missing colour costs.
"""
from __future__ import annotations

import argparse
import importlib.util
from dataclasses import replace
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "effect_knowledge_probes.run", Path(__file__).resolve().parent / "run.py")
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)
common = run.sibling("common")
sweeps = run.sibling("sweeps")

CORPUS = Path("output/effects/corpus-gen2")
COLOUR_WORDS = {"W": "white", "U": "blue", "B": "black", "R": "red", "G": "green"}

#: Attacker rows: the keyword the attacker is granted; ``None`` grants nothing.
#: ``protection-blocker`` and ``protection-other`` resolve against the blocker's colour.
ATTACKER_KEYWORDS: tuple[str | None, ...] = (
    None, "flying", "horsemanship", "fear", "intimidate", "shadow", "unblockable",
    "protection-blocker", "protection-other",
)
#: Blocker columns: the trait the blocker is given; ``None`` changes nothing.
BLOCKER_TRAITS: tuple[str | None, ...] = (
    None, "flying", "reach", "horsemanship", "shadow", "artifact", "black", "shares-colour",
)
#: Keywords and traits that decide block legality; a qualifying pair carries none.
EVASION_KEYWORDS = frozenset({
    "flying", "reach", "horsemanship", "fear", "intimidate", "shadow", "unblockable",
    "skulk", "menace", "landwalk", "defender",
})


# ── the rule ────────────────────────────────────────────────────────────


def may_block(attacker_keywords, attacker_colours, blocker_keywords, blocker_colours,
              blocker_types) -> bool:
    """Whether one creature may block another, as far as these keywords decide.

    Rules 702.9 (flying, reach), 702.31 (horsemanship), 702.36 (fear), 702.13
    (intimidate), 702.28 (shadow), 702.16 (protection) and "can't be blocked".
    """
    a, b = set(attacker_keywords), set(blocker_keywords)
    if "unblockable" in a:
        return False
    if "flying" in a and not b & {"flying", "reach"}:
        return False
    if "horsemanship" in a and "horsemanship" not in b:
        return False
    if "fear" in a and "artifact" not in blocker_types and "B" not in blocker_colours:
        return False
    if ("intimidate" in a and "artifact" not in blocker_types
            and not set(attacker_colours) & set(blocker_colours)):
        return False
    if ("shadow" in a) != ("shadow" in b):
        return False
    for keyword in a:
        if keyword.startswith("protection_from_"):
            colour = keyword.removeprefix("protection_from_")
            if any(COLOUR_WORDS.get(c) == colour for c in blocker_colours):
                return False
    return True


# ── the edits ───────────────────────────────────────────────────────────


def _edit_entity(record, entity_id: str, edit):
    entities = tuple(edit(e) if e.id == entity_id else e for e in record.state.entities)
    return replace(record, state=replace(record.state, entities=entities))


def grant_keyword(record, entity_id: str, keyword: str):
    def edit(entity):
        granted = entity.granted_temporary
        if keyword in granted.keywords:
            return entity
        return replace(entity, granted_temporary=replace(
            granted, keywords=(*granted.keywords, keyword)))

    return _edit_entity(record, entity_id, edit)


def set_colours(record, entity_id: str, colours: tuple[str, ...]):
    return _edit_entity(record, entity_id, lambda e: replace(e, colors=colours))


def add_type(record, entity_id: str, card_type: str):
    def edit(entity):
        if card_type in entity.types:
            return entity
        return replace(entity, types=(*entity.types, card_type))

    return _edit_entity(record, entity_id, edit)


def attacker_keyword(row: str | None, attacker, blocker) -> str | None:
    """The keyword a row grants, with protection resolved to a concrete colour."""
    if row == "protection-blocker":
        return f"protection_from_{COLOUR_WORDS[blocker.colors[0]]}"
    if row == "protection-other":
        taken = {c for c in (*attacker.colors, *blocker.colors)}
        other = next(c for c in COLOUR_WORDS if c not in taken)
        return f"protection_from_{COLOUR_WORDS[other]}"
    return row


def edited(record, attacker_id: str, blocker_id: str, row: str | None,
           column: str | None):
    """One grid cell's record, and the rule's answer for it."""
    attacker = record.state.entity(attacker_id)
    if column in ("flying", "reach", "horsemanship", "shadow"):
        record = grant_keyword(record, blocker_id, column)
    elif column == "artifact":
        record = add_type(record, blocker_id, "artifact")
    elif column == "black":
        record = set_colours(record, blocker_id, ("B",))
    elif column == "shares-colour":
        shared = next(c for c in attacker.colors if c != "B")
        record = set_colours(record, blocker_id, (shared,))
    blocker = record.state.entity(blocker_id)
    keyword = attacker_keyword(row, attacker, blocker)
    if keyword is not None:
        record = grant_keyword(record, attacker_id, keyword)
    a = record.state.entity(attacker_id)
    b = record.state.entity(blocker_id)
    rule = may_block(a.granted_temporary.keywords, a.colors, b.granted_temporary.keywords,
                     b.colors, b.types)
    return record, rule


# ── which records qualify ───────────────────────────────────────────────


def qualifying_pair(record, keywords_of) -> tuple[str, str] | None:
    """``(attacker, blocker)`` for the first qualifying legal blocker of a record."""
    from effects.domain.records import PlayabilityBlockersPayload

    payload = record.payload
    if not isinstance(payload, PlayabilityBlockersPayload) or record.fork:
        return None
    if getattr(record, "what_if", False):
        return None
    attacker = record.state.entity(payload.anchor_attacker)
    if attacker is None or "creature" not in attacker.types:
        return None
    if keywords_of(attacker) & EVASION_KEYWORDS or any(
            k.startswith("protection") for k in keywords_of(attacker)):
        return None
    if not any(c != "B" for c in attacker.colors if c in COLOUR_WORDS):
        return None
    for blocker_id in payload.legal_blockers:
        blocker = record.state.entity(blocker_id)
        if blocker is None or "creature" not in blocker.types or "artifact" in blocker.types:
            continue
        colours = [c for c in blocker.colors if c in COLOUR_WORDS]
        if not colours or "B" in colours or set(colours) & set(attacker.colors):
            continue
        if keywords_of(blocker) & EVASION_KEYWORDS:
            continue
        return attacker.id, blocker.id
    return None


def qualifying_records(directories, count: int, shards: int, keywords_of):
    """The first ``count`` qualifying ``(record, attacker, blocker)`` triples, one per game."""
    from effects.infrastructure.record_io import iter_shards, read_shard

    chosen, games = [], set()
    for directory in directories:
        for shard in iter_shards(directory)[:shards]:
            for record in read_shard(shard):
                if record.game_id in games:
                    continue
                pair = qualifying_pair(record, keywords_of)
                if pair is not None:
                    games.add(record.game_id)
                    chosen.append((record, *pair))
                    if len(chosen) >= count:
                        return chosen
    return chosen


# ── reading the model ───────────────────────────────────────────────────


def _may_block(outputs, surface, row, entity_id) -> float:
    import torch

    from effects.domain.effect_model import FIELD_SLICES

    start, _ = FIELD_SLICES["blocker_legal"]
    for index, slot in enumerate(surface.slots):
        if slot.entity_id == entity_id and slot.kind.name == "CARD":
            return float(torch.sigmoid(outputs[row, index, start]))
    return float("nan")


def label(name: str | None) -> str:
    return "none" if name is None else name


def _mean(values: np.ndarray) -> str:
    return f"{values.mean():.2f}" if len(values) else "—"


def main() -> None:
    from effects.domain.damage_step_keywords import KeywordResolver

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--records", type=int, default=40)
    parser.add_argument("--shards", type=int, default=200)
    args = parser.parse_args()

    sidecars = common.sidecar_cache(common.DEFAULT_CARDS_FOLDERS)
    resolver = KeywordResolver(sidecars)
    directories = [CORPUS / "validation" / "card-disjoint",
                   CORPUS / "validation" / "game-disjoint"]
    triples = qualifying_records(directories, args.records, args.shards, resolver.keywords_of)
    checkpoint = common.load_checkpoint(args.checkpoint)
    probe_model = common.load_probe_model(
        args.checkpoint, list(common.DEFAULT_CARDS_FOLDERS), [r for r, _, _ in triples],
        checkpoint=checkpoint)
    print(f"{args.checkpoint.parent.name} ({checkpoint.epoch}), "
          f"{len(triples)} validation attacker-blocker pairs", flush=True)

    grid = np.zeros((len(ATTACKER_KEYWORDS), len(BLOCKER_TRAITS)))
    rule = np.zeros_like(grid)
    per_row = {row: ([], []) for row in ATTACKER_KEYWORDS}
    for i, row in enumerate(ATTACKER_KEYWORDS):
        for j, column in enumerate(BLOCKER_TRAITS):
            cells = [edited(record, a, b, row, column) for record, a, b in triples]
            predictions = sweeps._predict(
                probe_model, [c[0] for c in cells],
                lambda o, v, s, r, ids=[b for _, _, b in triples]: _may_block(o, s, r, ids[r]))
            grid[i, j] = float(np.mean(predictions))
            rule[i, j] = float(np.mean([c[1] for c in cells]))
            per_row[row][0].extend(c[1] for c in cells)
            per_row[row][1].extend(predictions)
    header = "| attacker keyword \\ blocker | " + " | ".join(
        label(t) for t in BLOCKER_TRAITS) + " |"
    for title, values in (("predicted may block", grid), ("rule may block", rule)):
        print(f"\n{title}: error against the rule {np.abs(grid - rule).mean():.3f}")
        print(header)
        print("|---|" + "---:|" * len(BLOCKER_TRAITS))
        for i, row in enumerate(ATTACKER_KEYWORDS):
            print(f"| {label(row)} | " + " | ".join(f"{values[i, j]:.2f}"
                                                   for j in range(len(BLOCKER_TRAITS))) + " |",
                  flush=True)
    # The absolute level of `blocker_legal` sits low even on legal blocks, so the
    # ranking within a row says more: does the model put the blocks the rule allows
    # above the ones it forbids?
    auc = run.sibling("target_restrictions").auc
    every = ([], [])
    print("\nranking within each attacker row, cells and records pooled")
    print("| attacker keyword | allowed, mean | forbidden, mean | AUC |")
    print("|---|---:|---:|---:|")
    for row, (allowed, scores) in per_row.items():
        allowed, scores = np.asarray(allowed, bool), np.asarray(scores, float)
        every[0].extend(allowed)
        every[1].extend(scores)
        print(f"| {label(row)} | {_mean(scores[allowed])} | {_mean(scores[~allowed])} | "
              f"{auc(allowed, scores):.3f} |")
    print(f"| all rows | | | {auc(*every):.3f} |", flush=True)


if __name__ == "__main__":
    main()

"""Predicted block legality under each evasion keyword, against each blocker trait.

    python scripts/effect_knowledge_probes/evasion_grid.py CHECKPOINT [--half HALF]

The blocking counterpart of ``combat_grid.py``. A real block-legality record anchors
one attacker and lists the creatures that may block it. A triple qualifies when the
candidate blocker was legal, neither creature carries an evasion keyword or a trait
the grid sets, the blocker is a coloured non-artifact creature that is not black and
shares no colour with the attacker, and the attacker has a colour other than black.
One triple per game, read from the corpus's selection set: the records it froze for the
evasion grid in ``--half`` (``reporting`` by default, ``selection`` or ``both``) of both
validation strata.

Each grid cell grants the attacker one evasion keyword and gives the blocker one
trait, both through channels the model reads (a keyword as an effect until end of
turn grants it, a granted keyword line plus its kept flag, ``BoardEdit.grant_keyword``;
the type and colour lists), and reads the predicted legality of that blocker for that
attacker (the ``blocker_legal`` field at the blocker's slot). Fear, intimidate and
unblockable keep no flag, so the model reads them through the line alone. The rule's
answer is computed for every edited record by ``may_block``, so the score holds
whatever the real creatures are.

Protection from a colour is two rows: from the blocker's colour, where the rule
forbids the block, and from another colour, where it does not. The keyword flag
carries protection without its colour; the granted line names the colour, so the
two rows differ in that line's text alone and measure whether the model reads it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    # Run by path: put scripts/ on the path so the suite imports as a package.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _lib.markdown import print_table  # noqa: E402

from effect_knowledge_probes import common, readout  # noqa: E402
from effect_knowledge_probes.board_edits import BoardEdit  # noqa: E402
from effects.domain.rule_outcomes import COLOUR_WORDS, may_block  # noqa: E402

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


# ── the edits ───────────────────────────────────────────────────────────


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
    board = BoardEdit(record)
    if column in ("flying", "reach", "horsemanship", "shadow"):
        board = board.grant_keyword(blocker_id, column)
    elif column == "artifact":
        board = board.add_type(blocker_id, "artifact")
    elif column == "black":
        board = board.set_colours(blocker_id, ("B",))
    elif column == "shares-colour":
        shared = next(c for c in attacker.colors if c != "B")
        board = board.set_colours(blocker_id, (shared,))
    blocker = board.record.state.entity(blocker_id)
    keyword = attacker_keyword(row, attacker, blocker)
    if keyword is not None:
        board = board.grant_keyword(attacker_id, keyword)
    record = board.record
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


def label(name: str | None) -> str:
    return "none" if name is None else name


def main() -> None:

    from effect_knowledge_probes.selection import frozen_grid_records

    def options(parser) -> None:
        parser.add_argument("--half", choices=("reporting", "selection", "both"),
                            default="reporting")
        parser.add_argument("--probe-set-dir", type=Path, default=common.REPORTS,
                            help="where the corpus's selection set is")

    args, context = common.grid_run(__doc__, records=40, options=options)
    halves = ("selection", "reporting") if args.half == "both" else (args.half,)
    triples = [(record, *entities) for record, entities in frozen_grid_records(
        context, "evasion", halves=halves, reports_dir=args.probe_set_dir)]
    probe_model = context.model([r for r, _, _ in triples])
    common.heading(args, context, f"{len(triples)} validation attacker-blocker pairs")

    grid = np.zeros((len(ATTACKER_KEYWORDS), len(BLOCKER_TRAITS)))
    rule = np.zeros_like(grid)
    per_row = {row: ([], []) for row in ATTACKER_KEYWORDS}
    may_block = readout.at_cards("may_block", [b for _, _, b in triples])
    for i, row in enumerate(ATTACKER_KEYWORDS):
        for j, column in enumerate(BLOCKER_TRAITS):
            cells = [edited(record, a, b, row, column) for record, a, b in triples]
            predictions = common.predict(probe_model, [c[0] for c in cells], may_block)
            grid[i, j] = float(np.mean(predictions))
            rule[i, j] = float(np.mean([c[1] for c in cells]))
            per_row[row][0].extend(c[1] for c in cells)
            per_row[row][1].extend(predictions)
    for title, values in (("predicted may block", grid), ("rule may block", rule)):
        print(f"\n{title}: error against the rule {np.abs(grid - rule).mean():.3f}")
        print_table(["attacker keyword \\ blocker", *map(label, BLOCKER_TRAITS)],
                    [[label(row), *(f"{v:.2f}" for v in cells)]
                     for row, cells in zip(ATTACKER_KEYWORDS, values)],
                    "l" + "r" * len(BLOCKER_TRAITS))
    # The absolute level of `blocker_legal` sits low even on legal blocks, so the
    # ranking within a row says more: does the model put the blocks the rule allows
    # above the ones it forbids?
    every = ([], [])
    rows = []
    for row, (allowed, scores) in per_row.items():
        allowed, scores = np.asarray(allowed, bool), np.asarray(scores, float)
        every[0].extend(allowed)
        every[1].extend(scores)
        rows.append([label(row), common.fmt_mean(scores[allowed], 2),
                     common.fmt_mean(scores[~allowed], 2), f"{common.auc(allowed, scores):.3f}"])
    print("\nranking within each attacker row, cells and records pooled")
    print_table(["attacker keyword", "allowed, mean", "forbidden, mean", "AUC"], rows, "lrrr")
    print(f"| all rows | | | {common.auc(*every):.3f} |", flush=True)


if __name__ == "__main__":
    main()

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

Protection takes a row per quality it can name (702.16a, FR-074), each in pairs where
one names the blocker and the other does not: from the blocker's colour and from another
colour, from a creature type the blocker has (another type where it has none) and from
one it lacks, from creatures and from artifacts, from monocoloured and from multicoloured,
from mana value 3 or less, and from everything. The keyword flag carries protection
without its quality, and Forge spells most qualities after a colon
(``Protection:Goblin``); the granted line names the quality, so the rows of a pair differ
in that line's text alone and measure whether the model reads it. The rule's answer
reads the quality the row granted (``rule_outcomes.ProtectionQuality``) against the
edited blocker's colours, types, creature types and mana value.
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
from effects.domain.grid_edits import (  # noqa: E402, F401  (the suite's names for them)
    ATTACKER_KEYWORDS,
    BLOCKER_TRAITS,
    EVASION_KEYWORDS,
    OTHER_CREATURE_TYPES,
    PROTECTION_ROWS,
    label,
    protection_quality,
)
from effects.domain.grid_edits import evasion_cell as edited  # noqa: E402
from effects.domain.grid_edits import evasion_pair as qualifying_pair  # noqa: E402, F401
from effects.domain.rule_outcomes import may_block  # noqa: E402, F401


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
    read_may_block = readout.at_cards("may_block", [b for _, _, b in triples])
    for i, row in enumerate(ATTACKER_KEYWORDS):
        for j, column in enumerate(BLOCKER_TRAITS):
            cells = [edited(record, a, b, row, column) for record, a, b in triples]
            predictions = common.predict(probe_model, [c[0] for c in cells], read_may_block)
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

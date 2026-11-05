"""Predicted death of a damage spell's target, damage 1..8 x toughness 1..8, averaged over records.

    python scripts/effect_knowledge_probes/damage_grid.py CHECKPOINT [--source SOURCE]

Each cell edits the acting text's ``NumDmg$`` through the batcher's text memo, as the
damage sweep does, and the target's toughness, as the toughness sweep does, and reads
the target's predicted death (``halves.grid_predictions``, the cells the probing run
scores).

``--source validation`` (the default) reads the records the selection set froze for the
damage grid, in ``--half`` (reporting by default) of both validation strata.
``--source training`` uses training records that qualify for lethality copies (FR-063e)
with a literal amount: the grid then reads the model on the very records its copies were
drawn from, so a soft diagonal there is a fault of fit rather than of generalization.
``--source card-disjoint`` takes qualifying records from the held-out stratum, the same
selection on texts no training card carries.
"""
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    # Run by path: put scripts/ on the path so the suite imports as a package.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _lib.markdown import print_table  # noqa: E402

from effect_knowledge_probes import common, record_walk  # noqa: E402
from effects.domain.grid_edits import VALUES, damage_target  # noqa: E402, F401


def _options(parser) -> None:
    parser.add_argument("--source", choices=("validation", "training", "card-disjoint"),
                        default="validation")
    parser.add_argument("--half", choices=("reporting", "selection", "both"),
                        default="reporting")
    parser.add_argument("--probe-set-dir", type=Path, default=common.REPORTS,
                        help="where the corpus's selection set is (validation source)")


def main() -> None:
    from effect_knowledge_probes import halves
    from effect_knowledge_probes.selection import frozen_grid_records
    from effects.domain.damage_step_keywords import KeywordResolver

    args, context = common.grid_run(__doc__, records=40, options=_options)
    if args.source == "validation":
        chosen = ("selection", "reporting") if args.half == "both" else (args.half,)
        pairs = [(record, entities[0]) for record, entities in frozen_grid_records(
            context, "damage", halves=chosen, reports_dir=args.probe_set_dir)]
    else:
        directory = (context.training_dir if args.source == "training"
                     else context.stratum_dir("card-disjoint"))
        sidecars = context.sidecars()
        choose = damage_target(sidecars, KeywordResolver(sidecars).keywords_of)
        pairs = record_walk.qualifying([directory], choose, args.records, per_game=None,
                                       shards=args.shards)
    records = [record for record, _ in pairs]
    probe_model = context.model(records)
    common.heading(args, context, f"{len(pairs)} {args.source} records")
    grid = np.zeros((8, 8))
    for record, entity in pairs:
        grid += np.reshape(halves.grid_predictions(probe_model, "damage", record, (entity,)),
                           grid.shape)
    grid /= max(len(pairs), 1)
    print_table(["damage \\ toughness", *VALUES],
                [[amount, *(f"{v:.2f}" for v in row)] for amount, row in zip(VALUES, grid)],
                "l" + "r" * 8)


if __name__ == "__main__":
    main()

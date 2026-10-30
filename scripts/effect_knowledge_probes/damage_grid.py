"""Predicted death of a damage spell's target, damage 1..8 x toughness 1..8, averaged over records.

    python scripts/effect_knowledge_probes/damage_grid.py CHECKPOINT [--source SOURCE]

Each cell edits the acting text's ``NumDmg$`` through the batcher's text memo, as the
damage sweep does, and the target's toughness, as the toughness sweep does, and reads
the target's predicted death.

``--source probe`` (the default) uses the frozen probe set's damage-sweep records.
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

from effect_knowledge_probes import common, readout, record_walk, sweeps  # noqa: E402
from effect_knowledge_probes.board_edits import BoardEdit  # noqa: E402

VALUES = range(1, 9)


def probe_records(context, probe_set, host: str) -> list[tuple[object, str]]:
    """The frozen probe set's damage-sweep records on ``host``, with their targets."""
    rows = [r for r in probe_set.sweeps["damage"] if r["host"] == host]
    ids = {r["record_id"] for r in rows}
    found = {record.record_id: record for stratum in probe_set.games
             for record in context.probe_records(probe_set, stratum, ids)}
    return [(found[r["record_id"]], r["entity"]) for r in rows]


def damage_target(sidecars, keywords_of):
    """``choose`` for single-target damage records with a literal amount: the record and
    the creature it hits."""
    from effects.domain.effect_targets import derive_targets
    from effects.domain.lethality_counterfactuals import qualify

    def choose(record):
        if not record.ability:
            return None
        line = sidecars.line_or_none(record.ability[0])
        if line is None:
            return None
        qualified = qualify(record, line, derive_targets(record), keywords_of)
        if (qualified is None or qualified.effect.kind != "damage"
                or qualified.effect.stated != "text" or len(qualified.hit) != 1):
            return None
        return record, qualified.hit[0]

    return choose


def _options(parser) -> None:
    parser.add_argument("--source", choices=("probe", "training", "card-disjoint"),
                        default="probe")
    parser.add_argument("--host", default="spell", help="probe source: spell or triggered")
    parser.add_argument("--probe-set-dir", type=Path, default=common.REPORTS,
                        help="where the corpus's frozen probe set is (probe source)")


def main() -> None:
    from effects.domain.damage_step_keywords import KeywordResolver

    args, context = common.grid_run(__doc__, records=40, options=_options)
    if args.source == "probe":
        pairs = probe_records(context, context.load_probe_set(args.probe_set_dir), args.host)
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
    batcher = probe_model.batcher
    for record, entity in pairs:
        key = record.ability[0]
        original = batcher.text_of(key)
        edited = [BoardEdit(record).set_toughness(entity, t).record for t in VALUES]
        dies = readout.at_card("dies", entity)
        for i, amount in enumerate(VALUES):
            with batcher.reading_as(key, sweeps.set_amount(original, "NumDmg", amount)):
                grid[i] += common.predict(probe_model, edited, dies)
    grid /= max(len(pairs), 1)
    print_table(["damage \\ toughness", *VALUES],
                [[amount, *(f"{v:.2f}" for v in row)] for amount, row in zip(VALUES, grid)],
                "l" + "r" * 8)


if __name__ == "__main__":
    main()

"""Why a probe on the trunk's vector can beat the model's own answer: three checks.

    python scripts/effect_knowledge_probes/ladder_check.py CHECKPOINT
        [--targets damage_taken,may_block,dies] [--probe-set PATH]

The read-out ladder scores rung 2, a probe fitted on the trunk's vector at an item's
slot, against rung 3, the model's own output head. On some targets rung 2 scores far
higher. Three things in how the two are scored could produce that without the trunk
knowing anything the head does not use, and each check removes one:

1. **Siblings across folds.** The ladder's folds keep each card on one side, but
   items of one record or one game land on both, and siblings share a board and
   often a label (a sweeper's damage). Rung 2 is refitted with folds grouped by
   game, and every score is also taken over items whose record holds no other
   item of the target.
2. **Fitted on the scored stratum.** Rung 2 learns the stratum's own scale and
   base rate; the head does not. The head's answer is recalibrated out of fold by
   a one-variable fit (linear for an amount), and an amount is also scored by
   Spearman's rank correlation, which ignores scale.
3. **Ranking across boards.** A yes/no target's AUC pools items from every record,
   so a head whose level shifts from board to board loses ranking it would keep
   within each board. The mean AUC within records holding both labels is
   reported beside the pooled one.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    # Run by path: put scripts/ on the path so the suite imports as a package.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _lib.markdown import print_table  # noqa: E402
from _lib.stats import spearman  # noqa: E402

from effect_knowledge_probes import common, labels, ladder  # noqa: E402


def recalibrated(prediction, y, folds, kind) -> np.ndarray:
    """The head's answer refitted out of fold by one variable: linear for an amount.

    A yes/no answer is left as it is: a monotone refit of one score cannot change
    its AUC.
    """
    if kind == "binary":
        return prediction
    return ladder.linear_oof(prediction.reshape(-1, 1), y, folds, kind)


def within_record_auc(y, prediction, records) -> float:
    by = defaultdict(list)
    for index, record in enumerate(records):
        by[record].append(index)
    values = []
    for members in by.values():
        labels_here = y[members]
        if 0 < labels_here.sum() < len(members):
            values.append(ladder.score("binary", labels_here, prediction[members]))
    return float(np.nanmean(values)) if values else float("nan")


def check(data, records, games, kind) -> list[tuple[str, dict]]:
    """``[(row, {column: value})]`` for one target in one stratum."""
    card_folds = ladder.fold_assignment(data.groups)
    game_folds = ladder.fold_assignment(np.asarray(games, dtype=object))
    counts = defaultdict(int)
    for record in records:
        counts[record] += 1
    alone = np.array([counts[r] == 1 for r in records])
    y = data.y
    predictions = {
        "rung 2 linear, folds by card (published)":
            ladder.probe(data.features["2"], y, card_folds, kind, None, "linear"),
        "rung 2 MLP, folds by card (published)":
            ladder.probe(data.features["2"], y, card_folds, kind, None, "mlp"),
        "rung 2 linear, folds by game":
            ladder.probe(data.features["2"], y, game_folds, kind, None, "linear"),
        "rung 2 MLP, folds by game":
            ladder.probe(data.features["2"], y, game_folds, kind, None, "mlp"),
        "rung 3, the head (published)": data.rung3,
        "rung 3 recalibrated, folds by game": recalibrated(data.rung3, y, game_folds, kind),
    }
    rows = []
    for name, prediction in predictions.items():
        row = {"all": ladder.score(kind, y, prediction),
               "records with one item": ladder.score(kind, y[alone], prediction[alone])}
        if kind == "binary":
            row["within records"] = within_record_auc(y, prediction, records)
        else:
            row["Spearman"] = spearman(y, prediction)
        rows.append((name, row))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--targets", default="damage_taken,may_block,dies")
    parser.add_argument("--probe-set-dir", type=Path, default=common.REPORTS,
                        help="where the corpus's frozen probe set is")
    args = parser.parse_args()
    context = common.probe_context(args.checkpoint)

    targets = set(args.targets.split(","))
    probe_set = context.load_probe_set(args.probe_set_dir)
    kinds = {s.name: s.kind for s in labels.TARGETS}
    items_by_stratum: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for frozen in probe_set.record_items:
        if frozen.target not in targets:
            continue
        # No partner: none of the targets checked here reads one.
        items_by_stratum[frozen.stratum][frozen.record_id].append(
            replace(frozen, partner=None).record_item())
    records_by_stratum = {}
    for stratum in probe_set.games:
        records_by_stratum[stratum] = context.probe_records(
            probe_set, stratum, set(items_by_stratum[stratum]))
        print(f"{stratum}: {len(records_by_stratum[stratum])} records", flush=True)
    every = [r for rows in records_by_stratum.values() for r in rows]
    probe_model = context.model(every)
    common.heading(args, context, f"{len(every)} probe-set records")

    for stratum, records in records_by_stratum.items():
        game_of = {r.record_id: r.game_id for r in records}
        extracted = ladder.extract_features(probe_model, records, items_by_stratum[stratum])
        by_target = defaultdict(list)
        for item in extracted:
            by_target[item.target].append(item)
        inputs = ladder.ladder_inputs(extracted, kinds)
        for target in sorted(inputs):
            data = inputs[target]
            items = by_target[target]
            record_ids = [i.record_id for i in items]
            games = [game_of[r] for r in record_ids]
            kind = kinds[target]
            rows = check(data, record_ids, games, kind)
            per_record = Counter(record_ids)
            singles = sum(1 for r in record_ids if per_record[r] == 1)
            print(f"\n{target}, {stratum}: {len(items)} items, {len(set(record_ids))} records, "
                  f"{len(set(games))} games, {singles} items alone in their record "
                  f"({'AUC' if kind == 'binary' else 'R²'})")
            columns = list(rows[0][1])
            print_table(["read-out", *columns],
                        [[name, *(f"{row[c]:.3f}" for c in columns)] for name, row in rows],
                        "l" + "r" * len(columns))


if __name__ == "__main__":
    main()

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
import importlib.util
from collections import defaultdict
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "effect_knowledge_probes.run", Path(__file__).resolve().parent / "run.py")
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)
common = run.sibling("common")
labels = run.sibling("labels")
ladder = run.sibling("ladder")

PROBE_SET = Path("output/effects/reports/knowledge-probes-set-b3adb69a2950.json")
CORPUS = Path("output/effects/corpus-gen2")


def _spearman(y, prediction) -> float:
    from scipy.stats import spearmanr

    ok = np.isfinite(y) & np.isfinite(prediction)
    return float(spearmanr(y[ok], prediction[ok]).statistic) if ok.sum() > 2 else float("nan")


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
            row["Spearman"] = _spearman(y, prediction)
        rows.append((name, row))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--targets", default="damage_taken,may_block,dies")
    parser.add_argument("--probe-set", type=Path, default=PROBE_SET)
    args = parser.parse_args()
    import json

    targets = set(args.targets.split(","))
    probe_set = json.loads(args.probe_set.read_text(encoding="utf-8"))
    kinds = {s.name: s.kind for s in labels.TARGETS}
    items_by_stratum: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for row in probe_set["record_items"]:
        if row["target"] not in targets:
            continue
        items_by_stratum[row["stratum"]][row["record_id"]].append(labels.RecordItem(
            record_id=row["record_id"], target=row["target"], label=row["label"],
            entity=row["entity"], partner=None, partner_entity=row["partner_entity"],
            group=row["group"]))
    records_by_stratum = {}
    for stratum, game_ids in probe_set["games"].items():
        shards = [Path(n) if Path(n).is_absolute() else CORPUS / n
                  for n in probe_set["sources"].get(stratum, [])]
        wanted = set(items_by_stratum[stratum])
        records_by_stratum[stratum] = [
            record for _, record in common.read_games(shards, set(game_ids), wanted)]
        print(f"{stratum}: {len(records_by_stratum[stratum])} records", flush=True)
    every = [r for rows in records_by_stratum.values() for r in rows]
    checkpoint = common.load_checkpoint(args.checkpoint)
    probe_model = common.load_probe_model(
        args.checkpoint, list(common.DEFAULT_CARDS_FOLDERS), every, checkpoint=checkpoint)
    print(f"{args.checkpoint.parent.name} ({checkpoint.epoch})", flush=True)

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
            singles = sum(1 for r in record_ids if record_ids.count(r) == 1)
            print(f"\n{target}, {stratum}: {len(items)} items, {len(set(record_ids))} records, "
                  f"{len(set(games))} games, {singles} items alone in their record "
                  f"({'AUC' if kind == 'binary' else 'R²'})")
            columns = list(rows[0][1])
            print("| read-out | " + " | ".join(columns) + " |")
            print("|---|" + "---:|" * len(columns))
            for name, row in rows:
                print(f"| {name} | " + " | ".join(f"{row[c]:.3f}" for c in columns) + " |",
                      flush=True)


if __name__ == "__main__":
    main()

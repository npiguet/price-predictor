"""Why a probe on the trunk's vector can beat the model's own answer: three checks.

    python scripts/effect_knowledge_probes/ladder_check.py CHECKPOINT
        [--targets damage_taken,may_block,dies] [--half reporting] [--probe-set-dir DIR]

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
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    # Run by path: put scripts/ on the path so the suite imports as a package.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _lib.markdown import print_table  # noqa: E402
from _lib.stats import spearman  # noqa: E402

from effect_knowledge_probes import common, halves, labels, ladder  # noqa: E402


def recalibrated(prediction, y, folds, kind) -> np.ndarray:
    """The head's answer refitted out of fold by one variable: linear for an amount.

    A yes/no answer is left as it is: a monotone refit of one score cannot change
    its AUC.
    """
    if kind == "binary":
        return prediction
    return ladder.linear_oof(ladder.ProbeProblem(prediction.reshape(-1, 1), y, folds, kind))


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


def _predictions(data, games, kind) -> dict[str, np.ndarray]:
    """Rung 2 fitted with folds by card and by game, and rung 3 as is and recalibrated."""
    card_folds = ladder.fold_assignment(data.groups)
    game_folds = ladder.fold_assignment(np.asarray(games, dtype=object))
    y = data.y

    def rung2(folds, probe_type):
        problem = ladder.ProbeProblem(data.features["2"], y, folds, kind)
        return ladder.probe(problem, probe_type)

    return {
        "rung 2 linear, folds by card (published)": rung2(card_folds, "linear"),
        "rung 2 MLP, folds by card (published)": rung2(card_folds, "mlp"),
        "rung 2 linear, folds by game": rung2(game_folds, "linear"),
        "rung 2 MLP, folds by game": rung2(game_folds, "mlp"),
        "rung 3, the head (published)": data.rung3,
        "rung 3 recalibrated, folds by game": recalibrated(data.rung3, y, game_folds, kind),
    }


class _Scoring:
    """One target's labels and records, and the scores of a read-out against them."""

    def __init__(self, kind: str, y: np.ndarray, records: list) -> None:
        self.kind = kind
        self.y = y
        self.records = records
        counts = defaultdict(int)
        for record in records:
            counts[record] += 1
        self.alone = np.array([counts[r] == 1 for r in records])

    def row(self, prediction) -> dict:
        row = {"all": ladder.score(self.kind, self.y, prediction),
               "records with one item": ladder.score(
                   self.kind, self.y[self.alone], prediction[self.alone])}
        if self.kind == "binary":
            row["within records"] = within_record_auc(self.y, prediction, self.records)
        else:
            row["Spearman"] = spearman(self.y, prediction)
        return row


def check(data, records, games, kind) -> list[tuple[str, dict]]:
    """``[(row, {column: value})]`` for one target in one stratum."""
    predictions = _predictions(data, games, kind)
    scoring = _Scoring(kind, data.y, records)
    return [(name, scoring.row(prediction)) for name, prediction in predictions.items()]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--targets", default="damage_taken,may_block,dies")
    parser.add_argument("--half", choices=("reporting", "selection", "both"),
                        default="reporting")
    parser.add_argument("--probe-set-dir", type=Path, required=True,
                        help="where the corpus's selection set is")
    return parser.parse_args()


def _read_halves(context, selection, args: argparse.Namespace) -> list:
    """Every chosen half of every stratum, read for the targets asked for."""
    settings = halves.ReadSettings(probe_types=frozenset(args.targets.split(",")),
                                   gen3_probes=False, sweeps=False)
    acting_text = common.acting_text_function(context.sidecars(),
                                             common.manifest_surface(context.manifest))
    chosen = halves.HALVES if args.half == "both" else (args.half,)
    reads = []
    for stratum in sorted(selection.halves):
        for half in chosen:
            request = halves.HalfRequest(stratum, half, settings, acting_text, None)
            read = halves.read_half(context, selection, request)
            print(f"{read.name}: {len(read.probe_items)} records", flush=True)
            reads.append(read)
    return reads


class _ReadReport:
    """The three checks, printed for every target of one half's read."""

    def __init__(self, read, kinds: dict[str, str]) -> None:
        self.read = read
        self.kinds = kinds
        self.records = read.probe_records()
        self.game_of = {r.record_id: r.game_id for r in self.records}

    def run(self, probe_model) -> None:
        extracted = ladder.extract_features(probe_model, self.records, self.read.probe_items)
        by_target = defaultdict(list)
        for item in extracted:
            by_target[item.target].append(item)
        inputs = ladder.ladder_inputs(extracted, self.kinds)
        for target in sorted(inputs):
            self._target(target, inputs[target], by_target[target])

    def _target(self, target: str, data, items: list) -> None:
        record_ids = [i.record_id for i in items]
        games = [self.game_of[r] for r in record_ids]
        kind = self.kinds[target]
        rows = check(data, record_ids, games, kind)
        per_record = Counter(record_ids)
        singles = sum(1 for r in record_ids if per_record[r] == 1)
        print(f"\n{target}, {self.read.name}: {len(items)} items, {len(set(record_ids))} "
              f"records, {len(set(games))} games, {singles} items alone in their record "
              f"({'AUC' if kind == 'binary' else 'R²'})")
        columns = list(rows[0][1])
        print_table(["read-out", *columns],
                    [[name, *(f"{row[c]:.3f}" for c in columns)] for name, row in rows],
                    "l" + "r" * len(columns))


def main() -> None:
    args = _parse_args()
    context = common.probe_context(args.checkpoint)
    selection = halves.load_selection(context, args.probe_set_dir)
    kinds = {s.name: s.kind for s in labels.TARGETS}
    reads = _read_halves(context, selection, args)
    every = [r for read in reads for r in read.probe_records()]
    probe_model = context.model(every)
    common.heading(args, context, f"{len(every)} selection-set records")
    for read in reads:
        _ReadReport(read, kinds).run(probe_model)


if __name__ == "__main__":
    main()

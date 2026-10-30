"""Collect the gen-2 effect-model results into the CSV files beside the gen-2 records.

    python scripts/effect_gen2_report/build_data.py

Reads every gen-2 run directory under ``models/effects/runs/`` (training logs,
evaluation logs, knowledge-probe scorecards, embedding-probe reports, damage and
combat grids, boundary metrics) and writes one CSV per kind of result into
``experiments/2026-09-18-effect-model-gen2/data/``. A run directory that does not
exist yet is skipped, so the script can be re-run as more of the sweep lands.

Values that exist only in the records' text (the arms' damage grids measured before
the grid files were kept, gen-1's reference figures, the noise pilots' caches) live
in ``transcribed.py`` beside this script, and are written out with the rest.
``make_charts.py`` draws every chart from these CSVs alone.
"""

from __future__ import annotations

import csv
import glob
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import transcribed  # noqa: E402
from _lib.paths import MODELS, OUTPUT, REPO  # noqa: E402
from lineage import GEN1, RUN_LIST, Run  # noqa: E402

from effects.application.training_log import EpochSummary, parse_epoch_summaries  # noqa: E402

RUNS = MODELS / "effects" / "runs"
OUT = REPO / "experiments" / "2026-09-18-effect-model-gen2" / "data"


def write(name: str, header: list[str], rows: list[list]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / name, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for row in rows:
            w.writerow(["" if v is None else (f"{v:.6g}" if isinstance(v, float) else v)
                        for v in row])
    print(f"wrote {name}: {len(rows)} rows")


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


# ── training logs ──────────────────────────────────────────────────────────


def epochs_of(run: Run) -> list[EpochSummary]:
    return parse_epoch_summaries(read(RUNS / run.directory / "train.log").splitlines())


@dataclass(frozen=True)
class Checkpoint:
    """A probed checkpoint: a run's best card-disjoint epoch, or its last epoch."""

    id: str
    run: Run
    role: str  # "best" or "last"
    directory: str  # where its probe outputs live

    @property
    def path(self) -> Path:
        return RUNS / self.directory


def checkpoints(epochs: dict[str, list[EpochSummary]]
                ) -> list[tuple[Checkpoint, EpochSummary]]:
    out = []
    for run in RUN_LIST:
        eps = epochs.get(run.name)
        if not eps or "arm complete" not in read(RUNS / run.directory / "arm.log"):
            continue  # still training: its best epoch is not final
        best = min(eps, key=lambda e: e.card_disjoint)
        # A continuation that never beat its starting checkpoint keeps that checkpoint,
        # which its parent already probed; its own lowest epoch is probed in "-best".
        best_dir = (run.directory + "-best" if (RUNS / (run.directory + "-best")).exists()
                    else run.directory)
        out.append((Checkpoint(f"{run.name} best" if run.parent else run.name, run, "best",
                               best_dir), best))
        last_dir = run.directory + "-last"
        if run.parent and (RUNS / last_dir).exists():
            last = max(eps, key=lambda e: e.epoch)
            out.append((Checkpoint(f"{run.name} last", run, "last", last_dir), last))
    return out


# ── evaluation logs ────────────────────────────────────────────────────────


def gate2(ck: Checkpoint) -> list[list]:
    rows = []
    for m in re.finditer(r"^\s+(\w+)\s+(pass|→ probe|too few|under)\S*\s+n=(\d+)\s+(.*)$",
                         read(ck.path / "evaluate.log"), re.M):
        agree = re.search(r"direction agreement ([\d.]+)%", m.group(4))
        verdict = {"pass": "pass", "→ probe": "probe"}.get(m.group(2), "too few records")
        if "under-sampled" in m.group(4):
            verdict = "too few records"
        rows.append([ck.id, m.group(1), int(m.group(3)),
                     float(agree.group(1)) if agree else None, verdict])
    return rows


def gate3(ck: Checkpoint) -> list | None:
    text = read(ck.path / "evaluate.log")
    m = re.search(r"mean_pairwise_cosine=([+-][\d.]+)\s+top_component_share=([+-][\d.]+)", text)
    if not m:
        return None
    pr = None
    for f in glob.glob(str(ck.path / "reports" / "embedding-probes-*" / "pca_full_variance.md")):
        p = re.search(r"Participation ratio .*?: (\d+\.\d+)", read(Path(f)))
        if p:
            pr = float(p.group(1))
    return [ck.id, float(m.group(1)), float(m.group(2)), pr]


# ── probe outputs ──────────────────────────────────────────────────────────


def scorecard(ck: Checkpoint) -> dict | None:
    found = sorted(glob.glob(str(ck.path / "reports" / "knowledge-probes-*" / "scorecard.json")))
    return json.loads(read(Path(found[-1]))) if found else None


def board_scorecard(ck: Checkpoint, card: dict | None) -> dict | None:
    if card and card.get("board_abilities"):
        return card["board_abilities"]
    found = glob.glob(str(ck.path / "reports" / "board-run" / "**" / "scorecard.json"),
                      recursive=True)
    for f in found:
        data = json.loads(read(Path(f)))
        if data.get("board_abilities"):
            return data["board_abilities"]
    return None


def grid_cells(text: str) -> list[list[float]]:
    rows = [r for r in text.splitlines() if re.match(r"\| [1-8] \|", r)][:8]
    return [[float(x) for x in r.strip().strip("|").split("|")[1:]] for r in rows]


def combat_blocks(text: str) -> list[tuple[str, str, list[list[float]]]]:
    out = []
    for block in text.split("\n\n")[1:]:
        lines = block.strip().splitlines()
        m = re.match(r"(\S+), (\w+)'s death", lines[0]) if lines else None
        if m:
            out.append((m.group(1), m.group(2), grid_cells(block)))
    return out


STEPPED = {"attacker-first-strike", "blocker-first-strike", "attacker-double-strike",
           "attacker-double-strike-second-hit"}


def combat(ck: Checkpoint) -> list[list]:
    """Combat-grid cells; first and double strike only from per-step output."""
    rows = []
    main = read(ck.path / "combat-grid.txt")
    steps = read(ck.path / "combat-grid-steps.txt")
    per_step_in_main = "second-hit" in main
    for variant, which, cells in combat_blocks(main):
        if variant in STEPPED and not per_step_in_main:
            continue  # read on regular-step boards before the per-step fix
        rows += cell_rows([ck.id, variant, which], cells)
    for variant, which, cells in combat_blocks(steps):
        rows += cell_rows([ck.id, variant, which], cells)
    return rows


def cell_rows(key: list, cells: list[list[float]]) -> list[list]:
    """An 8×8 grid as one row per cell: ``key``, the row and column from 1, the value."""
    return [[*key, r + 1, c + 1, v] for r, row in enumerate(cells) for c, v in enumerate(row)]


def boundary(run: Run) -> list[list]:
    rows = []
    for m in re.finditer(r"epoch (\d+) best [\d.]+: error vs rule ([\d.]+) \| lethal by 1 ([\d.]+)"
                         r" \| exact ([\d.]+) \| survives by 1 ([\d.]+) \| gap (-?[\d.]+)",
                         read(RUNS / run.directory / "boundary-by-epoch.txt")):
        rows.append([run.name, int(m.group(1)), *map(float, m.groups()[1:])])
    return rows


def damage_grid_files(ck: Checkpoint) -> list[tuple[str, Path]]:
    """``(record set, file)`` for a checkpoint's damage grids that exist on disk."""
    run_dir = RUNS / ck.run.directory
    if ck.run.parent:
        # A run that kept its starting checkpoint probes its own lowest epoch in "-best";
        # the run directory's best-epoch grid is the starting checkpoint's.
        if ck.role == "best":
            return [("probe set", ck.path / "grid-best-epoch.txt")]
        return [("probe set", run_dir / "grid-last-epoch.txt")]
    return [("probe set", run_dir / "grid-probe.txt"),
            ("training records", run_dir / "grid-training.txt")]


def embedding_probes(ck: Checkpoint) -> list[list]:
    rows = []
    for f in glob.glob(str(ck.path / "reports" / "embedding-probes-*" / "probes_script.csv")):
        with open(f, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                rows.append([ck.id, r["target"], r["metric"], float(r["full"]),
                             float(r["baseline"]) if r["baseline"] else None])
    return rows


def _number(cell: str) -> float | None:
    cell = cell.strip()
    try:
        return float(cell)
    except ValueError:
        return None  # "n/a", or the unreadable dash an early run printed


def _table(text: str, after: str) -> list[list[str]]:
    """The data rows of the first markdown table following the line ``after``."""
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith(after)), None)
    if start is None:
        return []
    rows, seen_header = [], False
    for line in lines[start + 1:]:
        if not line.startswith("|"):
            if seen_header:
                break
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if not seen_header:
            seen_header = True
            header = cells
            continue
        if set(line) <= set("|-: "):
            continue
        rows.append(cells)
    return [header, *rows] if seen_header else []


def damage_amount_alone(ck: Checkpoint) -> dict[str, float]:
    """``stratum -> rung 3 of damage_taken``, read as the amount alone.

    The scorecards read the model's damage as the affected gate times the amount.
    Every damage item took damage, so the gate's doubt scaled the answer by noise;
    ``ladder_check.py --amount-alone`` rescored rung 3 on the same items without it.
    """
    text = read(ck.path / "ladder-check-amount-alone.txt")
    out = {}
    for stratum in ("card-disjoint", "game-disjoint"):
        rows = _table(text, f"damage_taken, {stratum}")
        for row in rows[1:]:
            if row[0].startswith("rung 3, the head"):
                out[stratum] = _number(row[1])
    return out


def evasion(ck: Checkpoint) -> tuple[list[list], list[list]]:
    """``(grid cells, per-row ranking)`` from a checkpoint's ``evasion-grid.txt``."""
    text = read(ck.path / "evasion-grid.txt")
    predicted, rule = _table(text, "predicted may block"), _table(text, "rule may block")
    cells = []
    if predicted and rule:
        traits = predicted[0][1:]
        for p_row, r_row in zip(predicted[1:], rule[1:]):
            for trait, p, r in zip(traits, p_row[1:], r_row[1:]):
                cells.append([ck.id, p_row[0], trait, _number(p), _number(r)])
    ranking = [[ck.id, row[0], _number(row[1]), _number(row[2]), _number(row[3])]
               for row in _table(text, "ranking within each attacker row")[1:]]
    return cells, ranking


def targeting(ck: Checkpoint) -> tuple[list[list], list[list], list[list]]:
    """``(by family, by restriction, flips)`` from a checkpoint's ``target-restrictions.txt``."""
    text = read(ck.path / "target-restrictions.txt")
    families = [[ck.id, row[0], int(row[1]), int(row[2]), _number(row[3])]
                for row in _table(text, "by family")[1:]]
    restrictions = [[ck.id, row[0].strip("`"), row[1], int(row[2]), int(row[3]),
                     _number(row[4])] for row in _table(text, "by restriction")[1:]]
    flips = [[ck.id, row[0], int(row[1]), _number(row[2]), _number(row[3])]
             for row in _table(text, "counterfactual flips")[1:]]
    return families, restrictions, flips


def colour_pips(ck: Checkpoint) -> tuple[list[list], list[list], list[list]]:
    """``(by configuration, ranking, by cost)`` from a checkpoint's ``colour-pips.txt``."""
    text = read(ck.path / "colour-pips.txt")
    configurations = [[ck.id, row[0], int(row[1]), _number(row[2]), _number(row[3]),
                       _number(row[4]), _number(row[5])]
                      for row in _table(text, "by configuration")[1:]]
    ranking = [[ck.id, row[0], _number(row[1])]
               for row in _table(text, "ranking: rule-affordable")[1:]]
    by_cost = [[ck.id, row[0], int(row[1]), *(_number(c) for c in row[2:6])]
               for row in _table(text, "pip missing against one short")[1:]]
    return configurations, ranking, by_cost


def zone_effects(ck: Checkpoint) -> dict[str, list[list]]:
    """Each table of a checkpoint's ``zone-effects.txt``, rows prefixed by the checkpoint."""
    text = read(ck.path / "zone-effects.txt")
    out = {}
    for name, after in (("counts", "card counts at each player"),
                        ("moves", "whether a card moves"),
                        ("outcomes", "where a moved card goes"),
                        ("amounts", "the amount: NumCards$"),
                        ("who", "who: the player target")):
        rows = []
        for row in _table(text, after)[1:]:
            cells = [c if _number(c) is None else _number(c) for c in row]
            rows.append([ck.id, *cells])
        out[name] = rows
    return out


# ── main ───────────────────────────────────────────────────────────────────


def main() -> None:
    epochs = {run.name: epochs_of(run) for run in [*RUN_LIST, GEN1]}

    write("runs.csv", ["run", "directory", "e_width", "lethality_copies", "lr_scale",
                       "continues", "first_epoch", "last_epoch"],
          [[r.name, r.directory, r.width, r.copies, r.lr_scale, r.parent or "",
            min(e.epoch for e in epochs[r.name]), max(e.epoch for e in epochs[r.name])]
           for r in [GEN1, *RUN_LIST] if epochs[r.name]])

    write("epochs.csv", ["run", "epoch", "train", "card_disjoint", "game_disjoint", "gate_f1",
                         "zone_acc", "deviance"],
          [[name, e.epoch, e.train, e.card_disjoint, e.game_disjoint, e.gate_f1,
            e.zone_accuracy, e.deviance] for name, eps in epochs.items() for e in eps])

    write("fields_by_epoch.csv", ["run", "epoch", "field", "card_disjoint_loss",
                                  "explained_pct"],
          [[name, e.epoch, f, loss, pct] for name, eps in epochs.items() for e in eps
           for f, (loss, pct) in e.fields.items()])

    cks = checkpoints(epochs)
    write("checkpoints.csv", ["checkpoint", "run", "role", "epoch", "e_width", "lethality_copies",
                              "lr_scale", "card_disjoint", "game_disjoint", "gate_f1", "zone_acc",
                              "deviance", "probe_directory"],
          [[ck.id, ck.run.name, ck.role, e.epoch, ck.run.width, ck.run.copies,
            ck.run.lr_scale, e.card_disjoint, e.game_disjoint, e.gate_f1,
            e.zone_accuracy, e.deviance, ck.directory] for ck, e in cks])

    write("gate2.csv", ["checkpoint", "keyword", "qualifying_records", "agreement_pct",
                        "verdict"], [r for ck, _ in cks for r in gate2(ck)])
    write("gate3.csv", ["checkpoint", "mean_pairwise_cosine", "top_component_share",
                        "participation_ratio"],
          [r for ck, _ in cks if (r := gate3(ck))] + transcribed.GATE3)

    ladder, sweeps, sweep_records, board, ablation = [], [], [], [], []
    gen1_card = (OUTPUT / "effects" / "reports"
                 / "knowledge-probes-2026-09-17-full-textless-corpus-20261009/scorecard.json")
    probed = [(ck.id, ck, scorecard(ck)) for ck, _ in cks]
    if gen1_card.exists():
        probed.insert(0, ("gen-1", None, json.loads(read(gen1_card))))
    for cid, ck, card in probed:
        if card:
            amount_alone = damage_amount_alone(ck) if ck else {}
            for family, body in card["families"].items():
                for target, entry in body.get("targets", {}).items():
                    for stratum, r in entry.get("strata", {}).items():
                        for rung, scores in r.get("rungs", {}).items():
                            for probe, v in scores.items():
                                if (target == "damage_taken" and rung == "3"
                                        and stratum in amount_alone):
                                    v = amount_alone[stratum]
                                ladder.append([cid, family, target, entry.get("kind"),
                                               stratum, r.get("n"), rung, probe, v])
            for sweep, body in card.get("sweeps", {}).items():
                for point in body.get("mean", []):
                    sweeps.append([cid, sweep, point["value"], point["prediction"],
                                   point["n"]])
                for i, curve in enumerate(body.get("records", [])):
                    for point in curve:
                        sweep_records.append([cid, sweep, i, point["value"],
                                              point["prediction"]])
            for field, entry in card.get("ablation", {}).items():
                for replacement in ("noise", "api-mean", "nearest"):
                    for scope in ("all", "act", "cards"):
                        ablation.append([cid, field, entry.get("base_loss"), replacement,
                                         scope, entry.get(replacement, {}).get(scope)])
        for subset, body in ((board_scorecard(ck, card) if ck else None) or {}).items():
            for field, entry in body.get("ablation", {}).items():
                board.append([cid, subset, body.get("n"), field, entry.get("base_loss"),
                              entry.get("noise", {}).get("cards")])
    write("probe_ladder.csv", ["checkpoint", "family", "target", "kind", "stratum", "n", "rung",
                               "probe", "score"], ladder)
    write("sweeps.csv", ["checkpoint", "sweep", "value", "prediction", "records"], sweeps)
    write("sweep_records.csv", ["checkpoint", "sweep", "record", "value", "prediction"],
          sweep_records)
    write("ablation.csv", ["checkpoint", "field", "base_loss", "replacement", "scope",
                           "loss_increase"], ablation)
    write("board_abilities.csv", ["checkpoint", "subset", "records", "field", "loss",
                                  "card_slot_noise_increase"], board)
    write("embedding_probes.csv", ["checkpoint", "target", "metric", "score",
                                   "constant_baseline"],
          [r for ck, _ in cks for r in embedding_probes(ck)] + transcribed.EMBEDDING_GEN1)

    grids = [row for ck, _ in cks for record_set, f in damage_grid_files(ck) if f.exists()
             for row in cell_rows([ck.id, record_set], grid_cells(read(f)))]
    known = {(g[0], g[1]) for g in grids}
    grids += [row for key, cells in transcribed.DAMAGE_GRIDS.items() if key not in known
              for row in cell_rows(list(key), cells)]
    write("damage_grids.csv", ["checkpoint", "record_set", "damage", "toughness",
                               "predicted_death"], grids)

    write("combat_grids.csv", ["checkpoint", "variant", "reads", "striker_power",
                               "struck_toughness", "predicted_death"],
          [r for ck, _ in cks for r in combat(ck)])

    ev = [evasion(ck) for ck, _ in cks]
    write("evasion_grid.csv", ["checkpoint", "attacker_keyword", "blocker_trait",
                               "predicted_may_block", "rule_may_block"],
          [r for cells, _ in ev for r in cells])
    write("evasion_ranking.csv", ["checkpoint", "attacker_keyword", "allowed_mean",
                                  "forbidden_mean", "auc"],
          [r for _, ranking in ev for r in ranking])
    tg = [targeting(ck) for ck, _ in cks]
    write("target_families.csv", ["checkpoint", "family", "items", "legal", "auc"],
          [r for families, _, _ in tg for r in families])
    write("target_restrictions.csv",
          ["checkpoint", "restriction", "family", "items", "legal", "auc"],
          [r for _, restrictions, _ in tg for r in restrictions])
    write("target_flips.csv", ["checkpoint", "flip", "targets", "predicted_legal_before",
                               "predicted_legal_after"], [r for _, _, flips in tg for r in flips])

    cp = [colour_pips(ck) for ck, _ in cks]
    write("colour_pips.csv", ["checkpoint", "configuration", "records", "rule_affordable",
                              "predicted_affordable", "below_exact_mean",
                              "below_exact_share"], [r for rows, _, _ in cp for r in rows])
    write("colour_pips_ranking.csv", ["checkpoint", "items", "auc"],
          [r for _, rows, _ in cp for r in rows])
    write("colour_pips_by_cost.csv", ["checkpoint", "cost", "records", "pip_missing",
                                      "one_short", "pip_missing_plus_2", "all_off_colour"],
          [r for _, _, rows in cp for r in rows])
    ze = [zone_effects(ck) for ck, _ in cks]
    write("zone_counts.csv", ["checkpoint", "field", "player_items", "affected", "auc",
                              "records_naming_api", "one_player_affected",
                              "more_for_that_player", "actual_mean", "predicted_mean"],
          [r for z in ze for r in z["counts"]])
    write("zone_moves.csv", ["checkpoint", "zone_before", "cards", "moved", "auc"],
          [r for z in ze for r in z["moves"]])
    write("zone_outcomes.csv", ["checkpoint", "zone_before", "outcome", "cards",
                                "p_outcome_mean", "top_right", "predicted_instead"],
          [r for z in ze for r in z["outcomes"]])
    write("zone_amounts.csv", ["checkpoint", "api", "records",
                               *(f"amount_{a}" for a in range(1, 7))],
          [r for z in ze for r in z["amounts"]])
    write("zone_who.csv", ["checkpoint", "api", "records", "targeted_before",
                           "targeted_after", "other_before", "other_after"],
          [r for z in ze for r in z["who"]])

    bnd = [r for run in RUN_LIST for r in boundary(run)]
    seen = {(r[0], r[1]) for r in bnd}
    bnd += [r for r in transcribed.BOUNDARY if (r[0], r[1]) not in seen]
    write("boundary_by_epoch.csv", ["run", "epoch", "error_vs_rule", "lethal_by_1",
                                    "exactly_lethal", "short_by_1", "gap"], bnd)

    for name, (header, rows) in transcribed.TABLES.items():
        write(name, header, rows)


if __name__ == "__main__":
    main()

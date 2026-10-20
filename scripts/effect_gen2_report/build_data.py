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
from dataclasses import dataclass
from pathlib import Path

import transcribed

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "models" / "effects" / "runs"
OUT = REPO / "experiments" / "2026-09-18-effect-model-gen2" / "data"


@dataclass(frozen=True)
class Run:
    """One training run: a directory whose ``train.log`` holds its epochs."""

    name: str
    directory: str
    width: int
    copies: int
    lr_scale: float
    parent: str | None = None


RUN_LIST = [
    Run("e64 (earlier inputs)", "2026-10-gen2-e64-earlier-inputs", 64, 0, 1.0),
    Run("e64", "2026-10-gen2-e64", 64, 0, 1.0),
    Run("e128", "2026-10-gen2-e128", 128, 0, 1.0),
    Run("e64-lethality", "2026-10-gen2-e64-lethality", 64, 8, 1.0),
    Run("e128-lethality", "2026-10-gen2-e128-lethality", 128, 8, 1.0),
    Run("e256-lethality", "2026-10-gen2-e256-lethality", 256, 8, 1.0),
    Run("A64", "2026-10-gen2-e64-lethality-ext20", 64, 8, 1.0, "e64-lethality"),
    Run("B64", "2026-10-gen2-e64-lethality-lr0.1", 64, 8, 0.1, "A64"),
    Run("A128", "2026-10-gen2-e128-lethality-ext20", 128, 8, 1.0, "e128-lethality"),
    Run("B128", "2026-10-gen2-e128-lethality-lr0.1", 128, 8, 0.1, "A128"),
    Run("A256", "2026-10-gen2-e256-lethality-ext20", 256, 8, 1.0, "e256-lethality"),
    Run("B256", "2026-10-gen2-e256-lethality-lr0.1", 256, 8, 0.1, "A256"),
]
GEN1 = Run("gen-1", "2026-09-17-full-textless-corpus", 64, 0, 1.0)

EPOCH_LINE = re.compile(
    r"epoch (\d+) \| train ([\d.]+) \| card-disjoint ([\d.]+) \| game-disjoint ([\d.]+)"
    r" \| gate F1 ([\d.]+) \| zone acc ([\d.]+) \| deviance ([\d.]+)")
FIELD_ITEM = re.compile(r"([a-z_0-9]+) ([\d.]+) \((-?\d+|n/a)%?\)")


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


def epochs_of(run: Run) -> list[dict]:
    """Every epoch summary line of a run, with the per-field line that follows it."""
    lines = read(RUNS / run.directory / "train.log").splitlines()
    out = []
    for i, line in enumerate(lines):
        m = EPOCH_LINE.search(line)
        if not m:
            continue
        fields = {}
        if i + 1 < len(lines) and lines[i + 1].strip().startswith("fields:"):
            for name, loss, pct in FIELD_ITEM.findall(lines[i + 1]):
                fields[name] = (float(loss), None if pct == "n/a" else float(pct))
        e = dict(zip(("epoch", "train", "card_disjoint", "game_disjoint", "gate_f1",
                      "zone_acc", "deviance"), (int(m.group(1)), *map(float, m.groups()[1:]))))
        e["fields"] = fields
        out.append(e)
    # A resumed run logs an epoch again after the restart; keep the last line for each.
    return list({e["epoch"]: e for e in out}.values())


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


def checkpoints(epochs: dict[str, list[dict]]) -> list[tuple[Checkpoint, dict]]:
    out = []
    for run in RUN_LIST:
        eps = epochs.get(run.name)
        if not eps or "arm complete" not in read(RUNS / run.directory / "arm.log"):
            continue  # still training: its best epoch is not final
        best = min(eps, key=lambda e: e["card_disjoint"])
        out.append((Checkpoint(f"{run.name} best" if run.parent else run.name, run, "best",
                               run.directory), best))
        last_dir = run.directory + "-last"
        if run.parent and (RUNS / last_dir).exists():
            last = max(eps, key=lambda e: e["epoch"])
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
        rows += _cells(ck.id, variant, which, cells)
    for variant, which, cells in combat_blocks(steps):
        rows += _cells(ck.id, variant, which, cells)
    return rows


def _cells(cid, variant, which, cells):
    return [[cid, variant, which, p + 1, t + 1, v]
            for p, row in enumerate(cells) for t, v in enumerate(row)]


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
        name = "grid-best-epoch.txt" if ck.role == "best" else "grid-last-epoch.txt"
        return [("probe set", run_dir / name)]
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


# ── main ───────────────────────────────────────────────────────────────────


def main() -> None:
    epochs = {run.name: epochs_of(run) for run in [*RUN_LIST, GEN1]}

    write("runs.csv", ["run", "directory", "e_width", "lethality_copies", "lr_scale",
                       "continues", "first_epoch", "last_epoch"],
          [[r.name, r.directory, r.width, r.copies, r.lr_scale, r.parent or "",
            min(e["epoch"] for e in epochs[r.name]), max(e["epoch"] for e in epochs[r.name])]
           for r in [GEN1, *RUN_LIST] if epochs[r.name]])

    write("epochs.csv", ["run", "epoch", "train", "card_disjoint", "game_disjoint", "gate_f1",
                         "zone_acc", "deviance"],
          [[name, e["epoch"], e["train"], e["card_disjoint"], e["game_disjoint"], e["gate_f1"],
            e["zone_acc"], e["deviance"]] for name, eps in epochs.items() for e in eps])

    write("fields_by_epoch.csv", ["run", "epoch", "field", "card_disjoint_loss",
                                  "explained_pct"],
          [[name, e["epoch"], f, loss, pct] for name, eps in epochs.items() for e in eps
           for f, (loss, pct) in e["fields"].items()])

    cks = checkpoints(epochs)
    write("checkpoints.csv", ["checkpoint", "run", "role", "epoch", "e_width", "lethality_copies",
                              "lr_scale", "card_disjoint", "game_disjoint", "gate_f1", "zone_acc",
                              "deviance", "probe_directory"],
          [[ck.id, ck.run.name, ck.role, e["epoch"], ck.run.width, ck.run.copies,
            ck.run.lr_scale, e["card_disjoint"], e["game_disjoint"], e["gate_f1"],
            e["zone_acc"], e["deviance"], ck.directory] for ck, e in cks])

    write("gate2.csv", ["checkpoint", "keyword", "qualifying_records", "agreement_pct",
                        "verdict"], [r for ck, _ in cks for r in gate2(ck)])
    write("gate3.csv", ["checkpoint", "mean_pairwise_cosine", "top_component_share",
                        "participation_ratio"],
          [r for ck, _ in cks if (r := gate3(ck))] + transcribed.GATE3)

    ladder, sweeps, sweep_records, board, ablation = [], [], [], [], []
    gen1_card = (REPO / "output/effects/reports"
                 / "knowledge-probes-2026-09-17-full-textless-corpus-20261009/scorecard.json")
    probed = [(ck.id, ck, scorecard(ck)) for ck, _ in cks]
    if gen1_card.exists():
        probed.insert(0, ("gen-1", None, json.loads(read(gen1_card))))
    for cid, ck, card in probed:
        if card:
            for family, body in card["families"].items():
                for target, entry in body.get("targets", {}).items():
                    for stratum, r in entry.get("strata", {}).items():
                        for rung, scores in r.get("rungs", {}).items():
                            for probe, v in scores.items():
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

    grids = []
    for ck, _ in cks:
        for record_set, f in damage_grid_files(ck):
            if f.exists():
                for d, row in enumerate(grid_cells(read(f))):
                    grids += [[ck.id, record_set, d + 1, t + 1, v] for t, v in enumerate(row)]
    known = {(g[0], g[1]) for g in grids}
    grids += [g for g in transcribed.DAMAGE_GRIDS if (g[0], g[1]) not in known]
    write("damage_grids.csv", ["checkpoint", "record_set", "damage", "toughness",
                               "predicted_death"], grids)

    write("combat_grids.csv", ["checkpoint", "variant", "reads", "striker_power",
                               "struck_toughness", "predicted_death"],
          [r for ck, _ in cks for r in combat(ck)])

    bnd = [r for run in RUN_LIST for r in boundary(run)]
    seen = {(r[0], r[1]) for r in bnd}
    bnd += [r for r in transcribed.BOUNDARY if (r[0], r[1]) not in seen]
    write("boundary_by_epoch.csv", ["run", "epoch", "error_vs_rule", "lethal_by_1",
                                    "exactly_lethal", "short_by_1", "gap"], bnd)

    for name, (header, rows) in transcribed.TABLES.items():
        write(name, header, rows)


if __name__ == "__main__":
    main()

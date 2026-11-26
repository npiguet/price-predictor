"""Collect the gen-2 effect-model results into the CSV files beside the gen-2 records.

    python scripts/effect_gen2_report/build_data.py

Reads every gen-2 run directory under ``models/effects/runs/`` (training logs,
evaluation logs, knowledge-probe scorecards, embedding-probe reports, damage and
combat grids, boundary metrics) and writes one CSV per kind of result into
``experiments/2026-09-18-effect-model-gen2/data/``. A run directory that does not
exist yet is skipped, so the script can be re-run as more of the sweep lands.

Figures that exist only in the records' text (the noise pilots' caches, corpus scans,
the arms' damage grids measured before the grid files were kept, gen-1's reference
figures) are hand-maintained in the CSVs themselves. A CSV of only such figures is
not written here; in a CSV that mixes them with computed rows, the rows whose key the
run directories do not produce are read back from the committed file and written out
again. ``make_charts.py`` draws every chart from these CSVs alone.
"""

from __future__ import annotations

import csv
import glob
import json
import re
import sys
from dataclasses import dataclass
from itertools import dropwhile, takewhile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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


def _is_table_line(line: str) -> bool:
    return line.startswith("|")


def _is_not_table_line(line: str) -> bool:
    return not line.startswith("|")


def _is_separator(line: str) -> bool:
    return set(line) <= set("|-: ")


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _table(text: str, after: str) -> list[list[str]]:
    """The data rows of the first markdown table following the line ``after``.

    The table is the first run of lines starting with ``|``: its first line is the
    header, and its separator lines are dropped from the rows below it.
    """
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith(after)), None)
    if start is None:
        return []
    block = list(takewhile(_is_table_line, dropwhile(_is_not_table_line, lines[start + 1:])))
    if not block:
        return []
    header, *body = block
    return [_cells(header), *(_cells(line) for line in body if not _is_separator(line))]


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

#: Every CSV this script writes, with its header, in writing order.
HEADERS = {
    "runs.csv": ["run", "directory", "e_width", "lethality_copies", "lr_scale", "continues",
                 "first_epoch", "last_epoch"],
    "epochs.csv": ["run", "epoch", "train", "card_disjoint", "game_disjoint", "gate_f1",
                   "zone_acc", "deviance"],
    "fields_by_epoch.csv": ["run", "epoch", "field", "card_disjoint_loss", "explained_pct"],
    "checkpoints.csv": ["checkpoint", "run", "role", "epoch", "e_width", "lethality_copies",
                        "lr_scale", "card_disjoint", "game_disjoint", "gate_f1", "zone_acc",
                        "deviance", "probe_directory"],
    "gate2.csv": ["checkpoint", "keyword", "qualifying_records", "agreement_pct", "verdict"],
    "gate3.csv": ["checkpoint", "mean_pairwise_cosine", "top_component_share",
                  "participation_ratio"],
    "probe_ladder.csv": ["checkpoint", "family", "target", "kind", "stratum", "n", "rung",
                         "probe", "score"],
    "sweeps.csv": ["checkpoint", "sweep", "value", "prediction", "records"],
    "sweep_records.csv": ["checkpoint", "sweep", "record", "value", "prediction"],
    "ablation.csv": ["checkpoint", "field", "base_loss", "replacement", "scope",
                     "loss_increase"],
    "board_abilities.csv": ["checkpoint", "subset", "records", "field", "loss",
                            "card_slot_noise_increase"],
    "embedding_probes.csv": ["checkpoint", "target", "metric", "score", "constant_baseline"],
    "damage_grids.csv": ["checkpoint", "record_set", "damage", "toughness", "predicted_death"],
    "combat_grids.csv": ["checkpoint", "variant", "reads", "striker_power", "struck_toughness",
                         "predicted_death"],
    "evasion_grid.csv": ["checkpoint", "attacker_keyword", "blocker_trait",
                         "predicted_may_block", "rule_may_block"],
    "evasion_ranking.csv": ["checkpoint", "attacker_keyword", "allowed_mean",
                            "forbidden_mean", "auc"],
    "target_families.csv": ["checkpoint", "family", "items", "legal", "auc"],
    "target_restrictions.csv": ["checkpoint", "restriction", "family", "items", "legal", "auc"],
    "target_flips.csv": ["checkpoint", "flip", "targets", "predicted_legal_before",
                         "predicted_legal_after"],
    "colour_pips.csv": ["checkpoint", "configuration", "records", "rule_affordable",
                        "predicted_affordable", "below_exact_mean", "below_exact_share"],
    "colour_pips_ranking.csv": ["checkpoint", "items", "auc"],
    "colour_pips_by_cost.csv": ["checkpoint", "cost", "records", "pip_missing", "one_short",
                                "pip_missing_plus_2", "all_off_colour"],
    "zone_counts.csv": ["checkpoint", "field", "player_items", "affected", "auc",
                        "records_naming_api", "one_player_affected", "more_for_that_player",
                        "actual_mean", "predicted_mean"],
    "zone_moves.csv": ["checkpoint", "zone_before", "cards", "moved", "auc"],
    "zone_outcomes.csv": ["checkpoint", "zone_before", "outcome", "cards", "p_outcome_mean",
                          "top_right", "predicted_instead"],
    "zone_amounts.csv": ["checkpoint", "api", "records",
                         *(f"amount_{a}" for a in range(1, 7))],
    "zone_who.csv": ["checkpoint", "api", "records", "targeted_before", "targeted_after",
                     "other_before", "other_after"],
    "boundary_by_epoch.csv": ["run", "epoch", "error_vs_rule", "lethal_by_1", "exactly_lethal",
                              "short_by_1", "gap"],
}

#: The CSVs that also hold hand-entered rows, with how many leading cells key a row.
#: A row of the committed file whose key the run directories do not produce is kept.
HAND_ENTERED_KEY_CELLS = {"gate3.csv": 1, "embedding_probes.csv": 1, "damage_grids.csv": 2,
                          "boundary_by_epoch.csv": 2}


def hand_entered_rows(name: str, produced: list[list]) -> list[list[str]]:
    """The rows of the committed ``name`` whose key ``produced`` lacks, as the file spells them."""
    path = OUT / name
    if not path.exists():
        return []
    key_cells = HAND_ENTERED_KEY_CELLS[name]
    seen = {tuple(str(v) for v in row[:key_cells]) for row in produced}
    with open(path, encoding="utf-8", newline="") as f:
        committed = list(csv.reader(f))[1:]
    return [row for row in committed if tuple(row[:key_cells]) not in seen]


def _concat(items: list, rows_of) -> list[list]:
    """``rows_of(item)``'s rows for each item in turn, as one list."""
    return [row for item in items for row in rows_of(item)]


def _run_row(run: Run, eps: list[EpochSummary]) -> list:
    return [run.name, run.directory, run.width, run.copies, run.lr_scale, run.parent or "",
            min(e.epoch for e in eps), max(e.epoch for e in eps)]


def _run_rows(epochs: dict[str, list[EpochSummary]]) -> list[list]:
    """A row per run that has epochs, gen-1 first."""
    return [_run_row(r, epochs[r.name]) for r in [GEN1, *RUN_LIST] if epochs[r.name]]


def _epoch_rows(name: str, eps: list[EpochSummary]) -> list[list]:
    return [[name, e.epoch, e.train, e.card_disjoint, e.game_disjoint, e.gate_f1,
             e.zone_accuracy, e.deviance] for e in eps]


def _field_rows(name: str, eps: list[EpochSummary]) -> list[list]:
    return [[name, e.epoch, f, loss, pct] for e in eps for f, (loss, pct) in e.fields.items()]


def _per_run(epochs: dict[str, list[EpochSummary]], rows_of) -> list[list]:
    """``rows_of(name, epochs)``'s rows for each run in turn, as one list."""
    return [row for name, eps in epochs.items() for row in rows_of(name, eps)]


def _checkpoint_row(probed: tuple[Checkpoint, EpochSummary]) -> list:
    ck, e = probed
    return [ck.id, ck.run.name, ck.role, e.epoch, ck.run.width, ck.run.copies,
            ck.run.lr_scale, e.card_disjoint, e.game_disjoint, e.gate_f1,
            e.zone_accuracy, e.deviance, ck.directory]


def log_tables(epochs: dict[str, list[EpochSummary]],
               cks: list[tuple[Checkpoint, EpochSummary]]) -> dict[str, list[list]]:
    """The tables read from training and evaluation logs and boundary files."""
    probes = [ck for ck, _ in cks]
    return {
        "runs.csv": _run_rows(epochs),
        "epochs.csv": _per_run(epochs, _epoch_rows),
        "fields_by_epoch.csv": _per_run(epochs, _field_rows),
        "checkpoints.csv": [_checkpoint_row(probed) for probed in cks],
        "gate2.csv": _concat(probes, gate2),
        "gate3.csv": [r for ck in probes if (r := gate3(ck))],
        "boundary_by_epoch.csv": _concat(RUN_LIST, boundary),
    }


#: gen-1's knowledge-probe scorecard, under ``OUTPUT``.
GEN1_SCORECARD = Path("effects", "reports",
                      "knowledge-probes-2026-09-17-full-textless-corpus-20261009", "scorecard.json")
ABLATION_REPLACEMENTS = ("noise", "api-mean", "nearest")
ABLATION_SCOPES = ("all", "act", "cards")


@dataclass(frozen=True)
class Probed:
    """A scorecard to tabulate: gen-1's (no checkpoint) or a gen-2 checkpoint's (or none)."""

    id: str
    checkpoint: Checkpoint | None
    card: dict | None

    def amount_alone(self) -> dict[str, float]:
        return damage_amount_alone(self.checkpoint) if self.checkpoint else {}

    def board(self) -> dict | None:
        return board_scorecard(self.checkpoint, self.card) if self.checkpoint else None


def probed_scorecards(cks: list[tuple[Checkpoint, EpochSummary]]) -> list[Probed]:
    """Each checkpoint's latest scorecard, after gen-1's when it exists."""
    probed = [Probed(ck.id, ck, scorecard(ck)) for ck, _ in cks]
    gen1_card = OUTPUT / GEN1_SCORECARD
    if gen1_card.exists():
        probed.insert(0, Probed("gen-1", None, json.loads(read(gen1_card))))
    return probed


def _stratum_rows(target: str, entry: dict, overrides: dict) -> list[list]:
    """``[stratum, n, rung, probe, score]`` for each score of one ladder target."""
    return [[stratum, r.get("n"), rung, probe, overrides.get((target, rung, stratum), v)]
            for stratum, r in entry.get("strata", {}).items()
            for rung, scores in r.get("rungs", {}).items()
            for probe, v in scores.items()]


def ladder_rows(p: Probed) -> list[list]:
    """The probe ladder's scores; rung 3 of damage_taken as the amount alone where rescored."""
    overrides = {("damage_taken", "3", stratum): v for stratum, v in p.amount_alone().items()}
    return [[p.id, family, target, entry.get("kind"), *row]
            for family, body in p.card["families"].items()
            for target, entry in body.get("targets", {}).items()
            for row in _stratum_rows(target, entry, overrides)]


def sweep_rows(p: Probed) -> list[list]:
    return [[p.id, sweep, point["value"], point["prediction"], point["n"]]
            for sweep, body in p.card.get("sweeps", {}).items()
            for point in body.get("mean", [])]


def sweep_record_rows(p: Probed) -> list[list]:
    return [[p.id, sweep, i, point["value"], point["prediction"]]
            for sweep, body in p.card.get("sweeps", {}).items()
            for i, curve in enumerate(body.get("records", []))
            for point in curve]


def ablation_rows(p: Probed) -> list[list]:
    return [[p.id, field, entry.get("base_loss"), replacement, scope,
             entry.get(replacement, {}).get(scope)]
            for field, entry in p.card.get("ablation", {}).items()
            for replacement in ABLATION_REPLACEMENTS
            for scope in ABLATION_SCOPES]


def board_rows(p: Probed) -> list[list]:
    return [[p.id, subset, body.get("n"), field, entry.get("base_loss"),
             entry.get("noise", {}).get("cards")]
            for subset, body in (p.board() or {}).items()
            for field, entry in body.get("ablation", {}).items()]


def scorecard_tables(cks: list[tuple[Checkpoint, EpochSummary]]) -> dict[str, list[list]]:
    """The tables read from each checkpoint's knowledge-probe scorecard."""
    probed = probed_scorecards(cks)
    carded = [p for p in probed if p.card]
    return {"probe_ladder.csv": _concat(carded, ladder_rows),
            "sweeps.csv": _concat(carded, sweep_rows),
            "sweep_records.csv": _concat(carded, sweep_record_rows),
            "ablation.csv": _concat(carded, ablation_rows),
            "board_abilities.csv": _concat(probed, board_rows)}


def damage_grid_rows(ck: Checkpoint) -> list[list]:
    """A checkpoint's damage-grid cells, for each of its grid files on disk."""
    return [row for record_set, f in damage_grid_files(ck) if f.exists()
            for row in cell_rows([ck.id, record_set], grid_cells(read(f)))]


def _part(parsed: list, part: int | str) -> list[list]:
    """Part ``part`` of each checkpoint's parsed report, one table's rows in checkpoint order."""
    return [row for tables in parsed for row in tables[part]]


def report_tables(cks: list[tuple[Checkpoint, EpochSummary]]) -> dict[str, list[list]]:
    """The tables read from each checkpoint's probe stdout and report files."""
    probes = [ck for ck, _ in cks]
    ev = list(map(evasion, probes))
    tg = list(map(targeting, probes))
    cp = list(map(colour_pips, probes))
    ze = list(map(zone_effects, probes))
    return {
        "embedding_probes.csv": _concat(probes, embedding_probes),
        "damage_grids.csv": _concat(probes, damage_grid_rows),
        "combat_grids.csv": _concat(probes, combat),
        "evasion_grid.csv": _part(ev, 0),
        "evasion_ranking.csv": _part(ev, 1),
        "target_families.csv": _part(tg, 0),
        "target_restrictions.csv": _part(tg, 1),
        "target_flips.csv": _part(tg, 2),
        "colour_pips.csv": _part(cp, 0),
        "colour_pips_ranking.csv": _part(cp, 1),
        "colour_pips_by_cost.csv": _part(cp, 2),
        "zone_counts.csv": _part(ze, "counts"),
        "zone_moves.csv": _part(ze, "moves"),
        "zone_outcomes.csv": _part(ze, "outcomes"),
        "zone_amounts.csv": _part(ze, "amounts"),
        "zone_who.csv": _part(ze, "who"),
    }


def main() -> None:
    epochs = {run.name: epochs_of(run) for run in [*RUN_LIST, GEN1]}
    cks = checkpoints(epochs)
    tables = {**log_tables(epochs, cks), **scorecard_tables(cks), **report_tables(cks)}
    for name in HAND_ENTERED_KEY_CELLS:
        tables[name] += hand_entered_rows(name, tables[name])
    for name, header in HEADERS.items():
        write(name, header, tables[name])


if __name__ == "__main__":
    main()

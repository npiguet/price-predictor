"""The markdown tables of a probing run: per family, the halves, the sweeps and the ablations.

Nothing here does work at import time.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from _lib.markdown import md_table

from effect_knowledge_probes import ablation, common
from effect_knowledge_probes import board as board_probe
from effect_knowledge_probes.scorecard import Scorecard

fmt = common.fmt


def _bodies(scorecard: Scorecard) -> Iterator[tuple[str, dict]]:
    """``(stratum/half, body)`` for every half the scorecard holds, in sorted order."""
    for stratum, by_half in sorted(scorecard.halves.items()):
        for half, body in sorted(by_half.items()):
            yield f"{stratum}/{half}", body


def _probe_rows(where: str, body: dict) -> list[list]:
    return [[where, name, item["kind"], item["n"], fmt(item.get("score")),
             fmt(item.get("rescaled"))]
            for name, item in sorted(body.get("probes", {}).items())]


def _grid_rows(where: str, body: dict) -> list[list]:
    return [[where, name, item["records"], fmt(item["mae"]), fmt(item["baseline"]),
             fmt(item["score"])]
            for name, item in sorted(body.get("grids", {}).items())]


def _rule_row(where: str, body: dict) -> list:
    tests = body.get("rule_tests") or {}
    return [where, len(tests.get("rules", {})), len(tests.get("left_out", {})),
            fmt(tests.get("component"))]


def _gen3_rows(where: str, body: dict) -> list[list]:
    rows = [[where, f"conditions: {subset}", cell["records"], fmt(cell["auc"])]
            for subset, cell in (body.get("conditions") or {}).items() if cell["items"]]
    if body.get("anthem"):
        rows.append([where, "anthem", body["anthem"]["records"], fmt(body["anthem"]["auc"])])
    legality = body.get("target_legality")
    if legality:
        rows += [[where, f"target legality: {name}", legality["records"], fmt(value)]
                 for name, value in legality["auc"].items()]
    return rows


def _half_tables(scorecard: Scorecard) -> list[str]:
    """Per stratum and half: the probe types, grids, rule tests and gen-3 probes."""
    probes, grids, rules, gen3 = [], [], [], []
    for where, body in _bodies(scorecard):
        probes += _probe_rows(where, body)
        grids += _grid_rows(where, body)
        rules.append(_rule_row(where, body))
        gen3 += _gen3_rows(where, body)
    lines = ["# Per stratum and half", "",
             "## Probe types, rung 3", "",
             md_table(["half", "probe type", "kind", "n", "AUC or R²", "rescaled"], probes,
                      "lllrrr"), "",
             "## Grids", "",
             md_table(["half", "grid", "records", "MAE", "MAE of the grid mean", "score"],
                      grids, "llrrrr"), "",
             "## Rule tests", "",
             md_table(["half", "rules scored", "rules left out", "mean 2·AUC − 1"], rules,
                      "lrrr"), ""]
    if gen3:
        lines += ["## Gen-3 probes", "",
                  md_table(["half", "probe", "records", "AUC"], gen3, "llrr"), ""]
    return lines


def _share_row(target: str, stratum: str, probe_type: str, value: dict) -> list:
    ci = value.get("ci") or [None, None]
    return [target, stratum, probe_type, fmt(value["value"]), fmt(ci[0]), fmt(ci[1])]


def _stratum_rows(target: str, stratum: str, result: dict) -> tuple[list, list]:
    """One target's rung rows and share-of-gap rows in one stratum."""
    rungs = [[target, stratum, result["n"], rung, fmt(scores.get("linear")),
              fmt(scores.get("mlp")), fmt(scores.get("model"))]
             for rung, scores in result["rungs"].items()]
    shares = [_share_row(target, stratum, probe_type, value)
              for probe_type, value in (result.get("share") or {}).items() if value]
    return rungs, shares


def _family_lines(family: str, body: dict) -> list[str]:
    rungs, shares = [], []
    for target, entry in sorted(body["targets"].items()):
        for stratum, result in sorted(entry["strata"].items()):
            more_rungs, more_shares = _stratum_rows(target, stratum, result)
            rungs += more_rungs
            shares += more_shares
    lines = [f"# {family}", "", "## Rungs", "",
             md_table(["target", "stratum", "n", "rung", "linear", "mlp", "model"], rungs,
                      "llrlrrr")]
    if shares:
        lines += ["", "## Share of the gap closed by e", "",
                  md_table(["target", "stratum", "probe", "share", "95% low", "95% high"],
                           shares, "lllrrr")]
    return lines


def _sweep_lines(scorecard: Scorecard) -> list[str]:
    lines = ["# Board sweeps", ""]
    for name, body in scorecard.sweeps.items():
        columns = ["wording", "value", "mean prediction", "n"]
        rows = [[p.get("wording", ""), p["value"], fmt(p["prediction"]), p["n"]]
                for p in body["mean"]]
        lines += [f"## {name}", "", md_table(columns, rows, "lrrr"), ""]
    return lines


def _ablation_lines(scorecard: Scorecard) -> list[str]:
    header = [f"{r} {s}" for r in ablation.REPLACEMENTS for s in ablation.SCOPES]
    rows = [[name, fmt(body["base_loss"]),
             *(fmt(body[r][s]) for r in ablation.REPLACEMENTS for s in ablation.SCOPES)]
            for name, body in sorted(scorecard.ablation.items())]
    return ["# Ablation: loss increase per field", "",
            md_table(["field", "base", *header], rows, "lr" + "r" * len(header))]


def _write(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_tables(out_dir: Path, scorecard: Scorecard) -> None:
    """One markdown table per family, the halves, the sweeps and the ablations."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for family, body in scorecard.families.items():
        _write(out_dir / f"{family}.md", _family_lines(family, body))
    if scorecard.halves:
        _write(out_dir / "halves.md", _half_tables(scorecard))
    if scorecard.sweeps:
        _write(out_dir / "sweeps.md", _sweep_lines(scorecard))
    if scorecard.ablation:
        _write(out_dir / "ablation.md", _ablation_lines(scorecard))
    if scorecard.board_abilities:
        (out_dir / "board-abilities.md").write_text(
            board_probe.table(scorecard.board_abilities), encoding="utf-8")

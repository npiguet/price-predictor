"""Set knowledge-probe scorecards side by side and rank the checkpoints (FR-087).

    python scripts/effect_knowledge_probes/compare.py SCORECARD [SCORECARD ...] [--output PATH]

One table per family: a row per target, stratum (and half) and rung, a column
per checkpoint and probe type. Below it, the arms ranked on rung 1 minus rung
1w per probe type and per score — AUC for yes/no targets, R² for amounts, never
averaged together — the comparison that does not depend on each arm's own
rung 3, so a weaker head cannot buy a larger share. Then, per stratum and half
of the selection set, the arms ranked on what the scorecard reports there:
the rung-3 AUC of the yes/no probe types and the R² of the amount ones, each
averaged apart, the grids' and the rule tests' mean scores. Scorecards read
against different selection sets are compared anyway, with a warning: their
numbers describe different items.

Nothing here does work at import time.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

if __package__ in (None, ""):
    # Run by path: put scripts/ on the path so the suite imports as a package.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _lib.markdown import md_table, utf8_console  # noqa: E402

from effect_knowledge_probes.common import fmt  # noqa: E402
from effect_knowledge_probes.scorecard import Scorecard  # noqa: E402
from price_predictor.infrastructure.run_manifest import run_script  # noqa: E402

PROBE_MODELS = ("linear", "mlp")
#: The score each target kind is read by, which names its ranking.
METRICS = {"binary": "AUC", "amount": "R²"}


def load(paths: list[Path]) -> list[Scorecard]:
    return [Scorecard.from_json(json.loads(Path(p).read_text(encoding="utf-8")))
            for p in paths]


def digest_warning(cards: list[Scorecard]) -> str | None:
    digests = {card.selection_set_digest for card in cards}
    if len(digests) > 1:
        return ("WARNING: the scorecards record different selection-set digests "
                f"({', '.join(sorted(str(d)[:12] for d in digests))}); their "
                "numbers are over different items and compare by direction only.")
    return None


def _cells(card: Scorecard):
    """``(family, target, stratum, rung, probe_type) -> score`` of one card."""
    for family, body in card.families.items():
        for target, entry in body.get("targets", {}).items():
            for stratum, result in entry.get("strata", {}).items():
                for rung, scores in result.get("rungs", {}).items():
                    for probe_type, value in scores.items():
                        yield (family, target, stratum, rung, probe_type), value


def width_gain(card: Scorecard, probe_type: str, kind: str) -> float:
    """Mean of rung 1 − rung 1w over the card's board-dependent ``kind`` results.

    One kind at a time: a gain in AUC and a gain in R² are not on one scale.
    """
    gains = []
    for family, body in card.families.items():
        for entry in body.get("targets", {}).values():
            if entry.get("scope") not in ("act", "entity") or entry.get("kind") != kind:
                continue
            for result in entry.get("strata", {}).values():
                rungs = result.get("rungs", {})
                one = rungs.get("1", {}).get(probe_type)
                width = rungs.get("1w", {}).get(probe_type)
                if one is not None and width is not None and not (
                        math.isnan(one) or math.isnan(width)):
                    gains.append(one - width)
    return sum(gains) / len(gains) if gains else float("nan")


def ranking(cards: list[Scorecard]) -> dict[str, list[tuple[str, float]]]:
    """Per probe type and score (``mlp AUC``, ``linear R²``), the checkpoints
    from largest width gain to smallest."""
    out = {}
    for probe_type in PROBE_MODELS:
        for kind, metric in METRICS.items():
            scored = [(card.checkpoint or f"#{i}", width_gain(card, probe_type, kind))
                      for i, card in enumerate(cards)]
            out[f"{probe_type} {metric}"] = sorted(
                scored, key=lambda kv: -kv[1] if not math.isnan(kv[1]) else math.inf)
    return out


def half_scores(card: Scorecard) -> dict[str, float]:
    """``"<stratum>/<half> <score>" -> value`` of one card: the mean rung-3
    AUC of its yes/no probe types and the mean R² of its amount ones, apart,
    and the grids' and rule tests' mean scores."""
    out: dict[str, float] = {}
    for stratum, by_half in card.halves.items():
        for half, body in by_half.items():
            where = f"{stratum}/{half}"
            for kind, metric in METRICS.items():
                values = [item["score"] for item in body.get("probes", {}).values()
                          if item.get("kind") == kind and item.get("score") is not None
                          and not math.isnan(item["score"])]
                if values:
                    out[f"{where} probes {metric}"] = sum(values) / len(values)
            grids = [item["score"] for item in body.get("grids", {}).values()
                     if not math.isnan(item["score"])]
            if grids:
                out[f"{where} grids"] = sum(grids) / len(grids)
            component = (body.get("rule_tests") or {}).get("component")
            if component is not None and not math.isnan(component):
                out[f"{where} rule tests"] = component
    return out


def half_ranking(cards: list[Scorecard]) -> dict[str, list[tuple[str, float]]]:
    """Per stratum, half and score, the checkpoints from best to worst."""
    scores = [half_scores(card) for card in cards]
    out = {}
    for name in sorted({key for found in scores for key in found}):
        scored = [(card.checkpoint or f"#{i}", found.get(name, math.nan))
                  for i, (card, found) in enumerate(zip(cards, scores))]
        out[name] = sorted(scored, key=lambda kv: -kv[1] if not math.isnan(kv[1]) else math.inf)
    return out


def render(cards: list[Scorecard]) -> str:
    names = [Path(card.checkpoint or f"#{i}").parent.name or card.checkpoint
             for i, card in enumerate(cards)]
    lines = ["# Knowledge-probe comparison", ""]
    warning = digest_warning(cards)
    if warning:
        lines += [warning, ""]
    tables: dict[str, dict[tuple, dict[int, float]]] = {}
    for column, card in enumerate(cards):
        for (family, target, stratum, rung, probe_type), value in _cells(card):
            tables.setdefault(family, {}).setdefault(
                (target, stratum, rung), {})[(column, probe_type)] = value
    for family in sorted(tables):
        lines += [f"## {family}", ""]
        header = ["target", "stratum", "rung"] + [
            f"{name} {probe_type}" for name in names for probe_type in (*PROBE_MODELS, "model")
        ]
        rows = [[target, stratum, rung] + [
                    fmt(values.get((column, probe_type)))
                    for column in range(len(cards))
                    for probe_type in (*PROBE_MODELS, "model")]
                for (target, stratum, rung), values in sorted(tables[family].items())]
        lines += [md_table(header, rows, "lll" + "r" * (len(header) - 3)), ""]
    lines += ["## Ranking on rung 1 − rung 1w", ""]
    for probe_type, ranked in ranking(cards).items():
        lines.append(f"- {probe_type}: " + ", ".join(
            f"{Path(name).parent.name or name} ({fmt(gain)})" for name, gain in ranked))
    by_half = half_ranking(cards)
    if by_half:
        lines += ["", "## Ranking per stratum and half", ""]
        for name, ranked in by_half.items():
            lines.append(f"- {name}: " + ", ".join(
                f"{Path(card).parent.name or card} ({fmt(value)})" for card, value in ranked))
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("scorecards", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    return run_script(
        args.output.parent if args.output else None, lambda: _run(args),
        script=__file__, argv=argv, inputs=args.scorecards,
    )


def _run(args: argparse.Namespace) -> int:
    cards = load(args.scorecards)
    warning = digest_warning(cards)
    if warning:
        print(warning, file=sys.stderr)
    text = render(cards)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        utf8_console()
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

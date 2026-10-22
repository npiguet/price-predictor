"""Set knowledge-probe scorecards side by side and rank the checkpoints (FR-087).

    python scripts/effect_knowledge_probes/compare.py SCORECARD [SCORECARD ...] [--output PATH]

One table per family: a row per target, stratum and rung, a column per
checkpoint and probe type. Below it, the arms ranked on rung 1 minus rung 1w per
probe type and per score — AUC for yes/no targets, R² for amounts, never
averaged together — the comparison that does not depend on each arm's own
rung 3, so a weaker head cannot buy a larger share. Scorecards frozen against different
probe sets are compared anyway, with a warning: their numbers describe
different items.

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

from effect_knowledge_probes.common import fmt  # noqa: E402
from effect_knowledge_probes.scorecard import Scorecard  # noqa: E402

PROBE_TYPES = ("linear", "mlp")
#: The score each target kind is read by, which names its ranking.
METRICS = {"binary": "AUC", "amount": "R²"}


def load(paths: list[Path]) -> list[Scorecard]:
    return [Scorecard.from_json(json.loads(Path(p).read_text(encoding="utf-8")))
            for p in paths]


def digest_warning(cards: list[Scorecard]) -> str | None:
    digests = {card.probe_set_digest for card in cards}
    if len(digests) > 1:
        return ("WARNING: the scorecards record different probe-set digests "
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
    for probe_type in PROBE_TYPES:
        for kind, metric in METRICS.items():
            scored = [(card.checkpoint or f"#{i}", width_gain(card, probe_type, kind))
                      for i, card in enumerate(cards)]
            out[f"{probe_type} {metric}"] = sorted(
                scored, key=lambda kv: -kv[1] if not math.isnan(kv[1]) else math.inf)
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
            f"{name} {probe_type}" for name in names for probe_type in (*PROBE_TYPES, "model")
        ]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] * 3 + ["---:"] * (len(header) - 3)) + "|")
        for (target, stratum, rung), values in sorted(tables[family].items()):
            row = [target, stratum, rung] + [
                fmt(values.get((column, probe_type)))
                for column in range(len(cards))
                for probe_type in (*PROBE_TYPES, "model")
            ]
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    lines += ["## Ranking on rung 1 − rung 1w", ""]
    for probe_type, ranked in ranking(cards).items():
        lines.append(f"- {probe_type}: " + ", ".join(
            f"{Path(name).parent.name or name} ({fmt(gain)})" for name, gain in ranked))
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("scorecards", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    cards = load(args.scorecards)
    warning = digest_warning(cards)
    if warning:
        print(warning, file=sys.stderr)
    text = render(cards)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        # The table holds "−"; a redirected stdout on Windows is cp1252.
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

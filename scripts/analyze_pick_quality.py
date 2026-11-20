"""Was an off-lane pick worth it? Card-quality diagnostics over draft corpora.

``analyze_draft_lanes.py`` shows *how often* an agent drafts outside its colours
and whether it had a choice. This script asks what it got for it: when an agent
declines an on-colour card for an off-colour one, is the off-colour card
actually better?

Per agent it reports

- **best-card rate** — share of picks taking the highest-quality card in the pack;
- **off-lane premium** — for every voluntary off-lane pick (outside the seat's
  eventual top-2 colours, made with an on-colour card still available), the
  taken card's quality minus the best available on-colour card's, summarised as
  mean, median, and the share above zero.

A premium near zero with a share near 50 % means the agent's off-colour picks
are indifferent to card quality — it is not trading colour discipline for
power, it has simply stopped weighing colour.

Two quality scales, because neither alone is conclusive:

``--quality winrates`` (default, preferred)
    ``shrunk_score_play`` from a sealed ``cards-win-rates.txt``. Derived from
    real game outcomes, so it is independent of any agent's pick behaviour.

``--quality pickrate``
    P(taken | available), estimated from ``--reference-agents`` seats only. Needs
    no external file, but the scale is defined by the references' own choices —
    so they score high on *best-card rate* by construction and that column is
    not comparable between a reference and a candidate. The premium column is
    unaffected, since it compares two cards on one scale.

Usage
-----
    G=models/draft/agent/gen3

    python scripts/analyze_pick_quality.py --cards-path output/cardsfolder-512 \\
        --drafts "field at argmax, T3=$G/lr1e-5_t3_20260806_144756-yardstick-drafts.jsonl" \\
        --winrates output/sealed/cards-win-rates.txt

    # no win-rate labels on hand:
    python scripts/analyze_pick_quality.py --drafts corpus.jsonl --quality pickrate

``cards-win-rates.txt`` is rewritten by every ``python -m sealed train-encoder``
run; point ``--winrates`` at whichever snapshot matches the corpus.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median

from _lib.markdown import banner
from draft_corpus_common import (
    ColourResolver,
    Lane,
    parse_drafts_arg,
    read_records,
    score_play,
    seat_decisions,
    top_two_colours,
)

QUALITY_COLUMN = "shrunk_score_play"


def _count_corpus(path: Path, references: tuple[str, ...], seen: Counter,
                  taken: Counter) -> None:
    for record, geometry in read_records(path):
        for index, seat in enumerate(record.seats):
            if seat.agent not in references:
                continue
            for decision in seat_decisions(record, geometry, index):
                taken[decision.taken] += 1
                seen.update(decision.legal)


def build_pickrate_quality(
    corpora: list[tuple[str, Path]], references: tuple[str, ...], min_obs: int
) -> dict[str, float]:
    """P(taken | available) over reference seats, pooled across every corpus."""
    seen: Counter = Counter()
    taken: Counter = Counter()
    for _, path in corpora:
        _count_corpus(path, references, seen, taken)
    return {c: taken[c] / n for c, n in seen.items() if n >= min_obs}


@dataclass(frozen=True)
class Setup:
    """What every corpus is analysed against: colours, the quality scale, the agents."""

    colours: ColourResolver
    quality: dict[str, float]
    agents: tuple[str, ...] | None


class QualityTally:
    """Best-card hits and off-lane premiums, accumulated over one corpus."""

    def __init__(self, setup: Setup) -> None:
        self.setup = setup
        self.best_hit: Counter = Counter()
        self.best_total: Counter = Counter()
        self.premiums: dict[str, list[float]] = defaultdict(list)
        self.scored_slots = [0, 0]

    def add_record(self, record, geometry) -> None:
        quality = self.setup.quality
        for booster in record.boosters:
            self.scored_slots[1] += len(booster.picks)
            self.scored_slots[0] += sum(card in quality for card in booster.picks)
        agents = self.setup.agents
        for index, seat in enumerate(record.seats):
            if agents is None or seat.agent in agents:
                self._add_seat(record, geometry, index)

    def _add_seat(self, record, geometry, index: int) -> None:
        colours = self.setup.colours
        agent = record.seats[index].agent
        lane = Lane(frozenset(top_two_colours(geometry.drafted_pool(record, index),
                                              colours)), colours)
        for decision in seat_decisions(record, geometry, index):
            if decision.taken in self.setup.quality:
                self._add_decision(agent, lane, decision)

    def _add_decision(self, agent: str, lane: Lane, decision) -> None:
        quality = self.setup.quality
        got = decision.taken
        scored = [quality[c] for c in decision.legal if c in quality]
        if len(scored) >= 2:
            self.best_total[agent] += 1
            if quality[got] >= max(scored):
                self.best_hit[agent] += 1
        if not lane.off(got):
            return
        on_lane = [quality[card] for card in decision.legal[1:]
                   if card in quality and lane.on(card)]
        if on_lane:
            self.premiums[agent].append(quality[got] - max(on_lane))

    def row(self, agent: str) -> str:
        """One agent's line of the report."""
        values = self.premiums[agent]
        total = self.best_total[agent]
        rate = 100 * self.best_hit[agent] / total if total else 0
        if not values:
            return f"     {agent:<12} {rate:9.1f}% {0:8d} {'-':>11} {'-':>9} {'-':>9}"
        above = 100 * sum(1 for v in values if v > 0) / len(values)
        return (
            f"     {agent:<12} {rate:9.1f}% {len(values):8d} "
            f"{mean(values):+11.4f} {median(values):+9.4f} {above:8.1f}%"
        )


def analyze(label: str, path: Path, setup: Setup) -> None:
    tally = QualityTally(setup)
    for record, geometry in read_records(path):
        tally.add_record(record, geometry)

    slots = tally.scored_slots
    coverage = 100 * slots[0] / slots[1] if slots[1] else 0
    banner(f"{label}   ({coverage:.1f}% of drafted card slots carry a quality label)", width=96)
    print(
        f"     {'agent':<12} {'best-card':>10} {'n':>8} {'mean prem':>11} "
        f"{'median':>9} {'share>0':>9}"
    )
    for agent in sorted(set(tally.best_total) | set(tally.premiums)):
        print(tally.row(agent))
    print()


def _require_winrates(winrates: Path | None) -> None:
    if winrates is None or not winrates.exists():
        raise SystemExit(
            f"--winrates PATH ({winrates}) not found — pass it, or use "
            f"--quality pickrate to derive a scale from the corpus itself"
        )


def _quality_scale(args: argparse.Namespace, corpora: list[tuple[str, Path]]) -> dict[str, float]:
    """The card -> quality map the chosen ``--quality`` names, announced on stdout."""
    if args.quality == "winrates":
        _require_winrates(args.winrates)
        quality = score_play(args.winrates)
        values = sorted(quality.values())
        print(
            f"quality: {QUALITY_COLUMN} for {len(quality)} cards "
            f"(range {values[0]:+.3f} .. {values[-1]:+.3f}, "
            f"median {values[len(values) // 2]:+.3f})\n"
        )
        return quality
    references = tuple(a.strip() for a in args.reference_agents.split(","))
    quality = build_pickrate_quality(corpora, references, args.min_obs)
    print(
        f"quality: P(taken | available) over {'/'.join(references)} seats, "
        f"{len(quality)} cards with >={args.min_obs} observations\n"
    )
    return quality


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--drafts", action="append", required=True, metavar="LABEL=PATH",
        help="corpus to analyze; repeatable. Bare PATH labels by file stem.",
    )
    parser.add_argument("--cards-path", type=Path, required=True)
    parser.add_argument("--quality", choices=("winrates", "pickrate"), default="winrates")
    parser.add_argument("--winrates", type=Path, default=None,
                        help="Required with --quality winrates")
    parser.add_argument(
        "--reference-agents", default="gen1,forge-full",
        help="seats defining the pickrate scale (--quality pickrate only)",
    )
    parser.add_argument("--min-obs", type=int, default=20)
    parser.add_argument(
        "--agents", default=None,
        help="comma-separated mix labels to report (default: all seen)",
    )
    args = parser.parse_args()

    corpora = [parse_drafts_arg(raw) for raw in args.drafts]
    agents = tuple(a.strip() for a in args.agents.split(",")) if args.agents else None

    quality = _quality_scale(args, corpora)

    colours = ColourResolver(args.cards_path)
    for label, path in corpora:
        analyze(label, path, Setup(colours, quality, agents))


if __name__ == "__main__":
    main()

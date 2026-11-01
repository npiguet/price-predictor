"""``rule-counts``: how many records show each rule, per side and per source (FR-008).

Every record of a records set is tagged by the catalogue's detectors and
counted under ``rule -> polarity -> source``. The source says where the record
came from, which decides what it can be trusted for:

- ``aimed``: the record carries ``aimed_at`` (a directed fork, or a coverage or
  variant run aimed at rules), whatever its game;
- ``random_seat``: a natural record of a game with a random seat;
- ``forge_ai``: a natural record of a game the Forge AI played on both seats.

Gen-2 records carry no ``aimed_at``, so all of them are natural.

Shards are read once each, in parallel; a game never spans shards (a shard is
one worker JVM's lifetime), so whether a game has a random seat is decided at
the end of its shard. Each worker builds the detector index and its sidecar
cache once. Prior art: the ``build-corpus`` survey (``corpus_survey``'s
``_ShardSurveyor``), which reads shards the same way; it is not reused whole
because its read context is built around the corpus's holdout and caps.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from effects.application.parallel_map import run_parallel
from effects.domain.rule_catalogue import NoDetector, Polarity
from effects.domain.rule_detectors import DetectorIndex, tag_record
from effects.domain.shard_generation import ShardGeneration
from effects.domain.tally_merge import merge_into

logger = logging.getLogger(__name__)

#: The generations ``rule-counts`` reads: gen-2's corpus before any gen-3
#: collection, and gen-3's; never a mix (FR-008).
COUNTED_GENERATIONS = (ShardGeneration.GEN_2, ShardGeneration.GEN_3)


class RuleSource(StrEnum):
    RANDOM_SEAT = "random_seat"
    FORGE_AI = "forge_ai"
    AIMED = "aimed"


@dataclass
class RuleCounts:
    """``(rule, polarity, source) -> records``, and what the counts cannot say."""

    counts: Counter = field(default_factory=Counter)
    #: Rules a declared detector needs a field for that the records lack (FR-007).
    not_countable: set = field(default_factory=set)
    #: ``(record_id, scenario)`` of every finding whose outcome disagrees with its side.
    review: set = field(default_factory=set)
    records: int = 0

    def to_json(self, none: dict[str, str]) -> dict:
        by_rule: dict[str, dict[str, dict[str, int]]] = {}
        for (rule, polarity, source), n in sorted(self.counts.items()):
            by_rule.setdefault(rule, {}).setdefault(polarity, {})[source] = n
        return {
            "records": self.records,
            "rules": by_rule,
            "not_countable": sorted(self.not_countable - set(by_rule)),
            "none": dict(sorted(none.items())),
            "review": [{"record_id": r, "scenario": s} for r, s in sorted(self.review)],
        }


@dataclass(frozen=True)
class _CountContext:
    """One worker's read side: the detector index and its sidecar cache, built once."""

    rules_dir: Path
    sidecar_roots: dict

    def __call__(self):
        from effects.infrastructure.converted_trees import process_sidecars
        from effects.infrastructure.rules_catalogue_io import load_scenarios

        return DetectorIndex.of(load_scenarios(self.rules_dir)), process_sidecars(
            self.sidecar_roots)


def count_shard(built, shard: Path) -> RuleCounts:
    """One shard's counts. Tags are held per game until the shard ends, when
    each game's source is known."""
    from effects.infrastructure.shard_io import read_shard

    index, sidecars = built
    out = RuleCounts()
    seat_games: set[str] = set()
    pending: list[tuple[str, bool, object]] = []
    for record in read_shard(shard):
        out.records += 1
        if record.random_seat:
            seat_games.add(record.game_id)
        result = tag_record(record, index, sidecars)
        out.not_countable |= result.not_countable
        out.review |= {(record.record_id, tag.scenario) for tag in result.review}
        if result.tags:
            pending.append((record.game_id, record.aimed_at is not None, result.tags))
    for game_id, aimed, tags in pending:
        source = (RuleSource.AIMED if aimed else RuleSource.RANDOM_SEAT
                  if game_id in seat_games else RuleSource.FORGE_AI)
        for tag in tags:
            out.counts[(tag.rule, str(tag.polarity), str(source))] += 1
    return out


def none_rules(catalogue) -> dict[str, str]:
    """Rules whose every scenario declares no detector, with the first reason given."""
    reasons: dict[str, str] = {}
    declared: set[str] = set()
    for scenario in catalogue.scenarios:
        for rule in scenario.rules:
            if isinstance(scenario.detector, NoDetector):
                reasons.setdefault(rule, scenario.detector.reason)
            elif scenario.detector is not None:
                declared.add(rule)
    return {rule: reason for rule, reason in reasons.items() if rule not in declared}


@dataclass(frozen=True)
class RuleCountsConfig:
    records_dir: Path
    output: Path
    rules_dir: Path
    sidecar_roots: dict
    workers: int | None = None


def count(config: RuleCountsConfig) -> tuple[RuleCounts, dict[str, str]]:
    """Count every shard under ``config.records_dir``; refuse a mix of generations."""
    from effects.infrastructure.rules_catalogue_io import load_scenarios
    from effects.infrastructure.shard_io import NoShardsError, iter_shards, require_generation

    shards = iter_shards(config.records_dir)
    if not shards:
        raise NoShardsError(f"no shards under {config.records_dir}")
    generation = require_generation(shards, *COUNTED_GENERATIONS)
    logger.info("Counting %d %s shard(s) under %s", len(shards), generation,
                config.records_dir)
    total = RuleCounts()
    for found in run_parallel(
        shards, count_shard, workers=config.workers, label="Counted",
        progress_every=50, context=_CountContext(config.rules_dir, config.sidecar_roots),
    ):
        merge_into(total, found)
    return total, none_rules(load_scenarios(config.rules_dir))


def unit_of(rule: str) -> str:
    """A rule's catalogue unit: its section, or its second-level rule in 701 and 702."""
    section, _, rest = rule.partition(".")
    if section in ("701", "702"):
        return f"{section}.{''.join(c for c in rest if c.isdigit())}"
    return section


def report_lines(counts: dict) -> list[str]:
    """The printed table: one line per rule grouped by unit, then the rest."""
    def split(sides: dict, polarity: str) -> str:
        found = sides.get(polarity, {})
        return "/".join(str(found.get(str(s), 0)) for s in RuleSource)

    lines = [f"{'rule':12s} {'fires':>20s} {'does_not_fire':>20s}   "
             f"(each {' / '.join(str(s) for s in RuleSource)})"]
    unit = None
    for rule, sides in sorted(counts["rules"].items(), key=lambda item: _rule_key(item[0])):
        if unit_of(rule) != unit:
            unit = unit_of(rule)
            lines.append(f"-- {unit}")
        lines.append(f"{rule:12s} {split(sides, Polarity.FIRES):>20s} "
                     f"{split(sides, Polarity.DOES_NOT_FIRE):>20s}")
    lines.append(f"rules declared none: {len(counts['none'])}")
    lines += [f"  {rule}: {reason}" for rule, reason in counts["none"].items()]
    lines.append(f"rules not countable on these records: {len(counts['not_countable'])}")
    lines += [f"  {rule}" for rule in counts["not_countable"]]
    lines.append(f"records for review: {len(counts['review'])}")
    return lines


def _rule_key(rule: str):
    import re

    m = re.match(r"(\d+)\.(\d+)([a-z]*)", rule)
    return (int(m.group(1)), int(m.group(2)), m.group(3)) if m else (0, 0, rule)


def run(config: RuleCountsConfig) -> int:
    counts, none = count(config)
    document = counts.to_json(none)
    config.output.parent.mkdir(parents=True, exist_ok=True)
    config.output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    for line in report_lines(document):
        print(line)
    print(f"{counts.records} records; counts written to {config.output}")
    return 0

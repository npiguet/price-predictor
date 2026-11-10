"""The breadth check's pass over a records set (T036): pool and tagged counts per scenario.

Reads shards the way ``rule-counts`` does (``rule_counts.CountContext``: one
detector index and sidecar cache per worker), and counts, per declared
scenario, the ``fires`` records its detector finds and those its family finds
with the rule-specific parameters dropped (``domain.rule_breadth``). Run by
``scripts/rule_breadth.py``; it calls every family twice per record of its
kind, so it is meant for a sample of shards, whose shares speak for the
corpus as well as the whole would.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from effects.application.parallel_map import run_parallel
from effects.application.rule_counts import COUNTED_GENERATIONS, CountContext
from effects.domain.rule_breadth import BreadthRow, breadth_counts, breadth_rows
from effects.domain.rule_detectors import DetectorIndex
from effects.domain.tally_merge import merge_into

logger = logging.getLogger(__name__)


@dataclass
class BreadthTally:
    counts: Counter = field(default_factory=Counter)
    records: int = 0


def breadth_shard(built, shard: Path) -> BreadthTally:
    """One shard's pool and tagged counts."""
    from effects.infrastructure.shard_io import read_shard

    index, sidecars, _ = built
    out = BreadthTally()
    for record in read_shard(shard):
        out.records += 1
        out.counts.update(breadth_counts(record, index, sidecars))
    return out


def breadth(records_dir: Path, rules_dir: Path, sidecar_roots: dict,
            workers: int | None) -> tuple[list[BreadthRow], int]:
    """Every declared scenario's breadth row over the shards under
    ``records_dir``, and the records read; refuses a mix of generations."""
    from effects.infrastructure.rules_catalogue_io import load_scenarios
    from effects.infrastructure.shard_io import NoShardsError, iter_shards, require_generation

    shards = iter_shards(records_dir)
    if not shards:
        raise NoShardsError(f"no shards under {records_dir}")
    generation = require_generation(shards, *COUNTED_GENERATIONS)
    logger.info("Breadth over %d %s shard(s) under %s", len(shards), generation, records_dir)
    total = BreadthTally()
    for found in run_parallel(
        shards, breadth_shard, workers=workers, label="Breadth", progress_every=20,
        context=CountContext(rules_dir, sidecar_roots),
    ):
        merge_into(total, found)
    return breadth_rows(DetectorIndex.of(load_scenarios(rules_dir)), total.counts), total.records

"""Measure how much of the curated corpus the effect model reads as no text.

Two tables. The first is per ability slot on the board: how many of the
tokens the head reads are zero vectors, by trait kind and by why (dropped by
the converter, runtime-only, unconverted script). The second is per
resolution record: whether the acting ability has text, and for the ones
that do not, which scripts they act through.

Run before and after a reconversion or a corpus rebuild. The numbers this was
written against are in docs/superpowers/specs/2026-09-17-textless-abilities.md,
in git history up to commit e475ab70.
"""

from __future__ import annotations

import argparse
import collections
import itertools
import random
import sys
from pathlib import Path

from effects.domain.provenance import KeyResolution
from effects.domain.records import RecordKind
from effects.infrastructure.corpus_store import CorpusStore
from effects.infrastructure.curated_shards import require_listed_shards
from effects.infrastructure.shard_io import read_shard
from effects.infrastructure.sidecar_io import SidecarCache


class Audit:
    """The running counts of one pass over the sampled records."""

    def __init__(self, sidecars: SidecarCache) -> None:
        self.sidecars = sidecars
        self.slots = 0
        self.zero_by = collections.Counter()
        self.records = 0
        self.acting = collections.Counter()
        self.textless_scripts = collections.Counter()

    def add(self, record) -> None:
        self.records += 1
        self._count_slots(record)
        self._count_acting(record)

    def _count_slots(self, record) -> None:
        for entity in record.state.entities:
            for key in entity.ability_keys:
                self.slots += 1
                outcome = self.sidecars.resolution_of(key)
                if outcome is not KeyResolution.LINE:
                    self.zero_by[(key.trait_kind, outcome.value)] += 1

    def _count_acting(self, record) -> None:
        if record.kind is not RecordKind.RESOLUTION or not record.ability:
            return
        outcomes = {self.sidecars.resolution_of(key) for key in record.ability}
        if KeyResolution.LINE in outcomes:
            self.acting["has text"] += 1
        elif KeyResolution.RUNTIME_ONLY in outcomes:
            self.acting["runtime-only key (kept)"] += 1
        else:
            self.acting["no text"] += 1
            self.textless_scripts[record.ability[0].script_file] += 1

    def print_report(self) -> None:
        zero = sum(self.zero_by.values())
        print(f"{self.records} records, {self.slots} ability slots, {zero} zero-vector "
              f"({100.0 * zero / self.slots if self.slots else 0.0:.1f}%)")
        for (kind, why), count in self.zero_by.most_common(10):
            print(f"  {count:9d}  {kind:<12} {why}")
        print("resolution records by acting text:")
        for label, count in self.acting.most_common():
            print(f"  {count:9d}  {label}")
        print("no-text acting scripts, most first:")
        for script, count in self.textless_scripts.most_common(10):
            print(f"  {count:9d}  {script}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--cards-folder", action="append", type=Path, required=True)
    parser.add_argument("--variant-scripts", type=Path)
    parser.add_argument("--shards", type=int, default=10)
    parser.add_argument("--records-per-shard", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def _sidecars(args: argparse.Namespace) -> SidecarCache:
    roots = {folder.name: folder for folder in args.cards_folder}
    if args.variant_scripts:
        roots["variant-scripts"] = args.variant_scripts
    return SidecarCache(roots)


def main() -> int:
    args = _parse_args()
    sidecars = _sidecars(args)

    store = CorpusStore(args.corpus)
    files = list(require_listed_shards(store, store.load())["training"])
    if not files:
        # Zero shards print zero of everything, which reads exactly like a
        # clean corpus. A wrong --corpus is the likeliest reason to see it,
        # and this script exists to be compared across runs.
        print(f"no training shards under {store.training_dir}", file=sys.stderr)
        return 2
    random.seed(args.seed)
    files = random.sample(files, min(args.shards, len(files)))

    audit = Audit(sidecars)
    for path in files:
        for record in itertools.islice(read_shard(Path(path)), args.records_per_shard):
            audit.add(record)
    audit.print_report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

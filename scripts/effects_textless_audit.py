"""Measure how much of the curated corpus the effect model reads as no text.

Two tables. The first is per ability slot on the board: how many of the
tokens the head reads are zero vectors, by trait kind and by why (dropped by
the converter, runtime-only, unconverted script). The second is per
resolution record: whether the acting ability has text, and for the ones
that do not, which scripts they act through.

Run before and after a reconversion or a corpus rebuild. The numbers this was
written against are in docs/superpowers/specs/2026-09-17-textless-abilities.md.
"""

from __future__ import annotations

import argparse
import collections
import glob
import itertools
import random
from pathlib import Path

from effects.domain.provenance import KeyResolution
from effects.domain.records import RecordKind
from effects.infrastructure.record_io import read_shard
from effects.infrastructure.sidecar_io import SidecarCache


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--cards-folder", action="append", type=Path, required=True)
    parser.add_argument("--variant-scripts", type=Path)
    parser.add_argument("--shards", type=int, default=10)
    parser.add_argument("--records-per-shard", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    roots = {folder.name: folder for folder in args.cards_folder}
    if args.variant_scripts:
        roots["variant-scripts"] = args.variant_scripts
    sidecars = SidecarCache(roots)

    files = sorted(glob.glob(str(args.corpus / "training" / "shard-*.jsonl.gz")))
    random.seed(args.seed)
    files = random.sample(files, min(args.shards, len(files)))

    slots = 0
    zero_by = collections.Counter()
    records = 0
    acting = collections.Counter()
    textless_scripts = collections.Counter()
    for path in files:
        for record in itertools.islice(read_shard(Path(path)), args.records_per_shard):
            records += 1
            for entity in record.state.entities:
                for key in (*entity.printed, *entity.granted_attached):
                    slots += 1
                    outcome = sidecars.resolution_of(key)
                    if outcome is not KeyResolution.LINE:
                        zero_by[(key.trait_kind, outcome.value)] += 1
            if record.kind is RecordKind.RESOLUTION and record.ability:
                outcomes = {sidecars.resolution_of(key) for key in record.ability}
                if KeyResolution.LINE in outcomes:
                    acting["has text"] += 1
                elif KeyResolution.RUNTIME_ONLY in outcomes:
                    acting["runtime-only key (kept)"] += 1
                else:
                    acting["no text"] += 1
                    textless_scripts[record.ability[0].script_file] += 1

    zero = sum(zero_by.values())
    print(f"{records} records, {slots} ability slots, {zero} zero-vector "
          f"({100.0 * zero / slots if slots else 0.0:.1f}%)")
    for (kind, why), count in zero_by.most_common(10):
        print(f"  {count:9d}  {kind:<12} {why}")
    print("resolution records by acting text:")
    for label, count in acting.most_common():
        print(f"  {count:9d}  {label}")
    print("no-text acting scripts, most first:")
    for script, count in textless_scripts.most_common(10):
        print(f"  {count:9d}  {script}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

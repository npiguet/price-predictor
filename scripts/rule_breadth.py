"""Flag rule detectors that tag a small share of their family's pool (feature 025, T036).

Per declared scenario, counts the ``fires`` records its detector finds and the
records its family finds with the rule-specific parameters dropped, writes
every row to ``--output`` and prints the flagged ones: a pool of at least
``--min-pool`` records of which the detector tags under ``--max-share``. A
flag is a question for review: a declaration copied from one scenario's
board looks like this, and so does a genuinely narrow rule.

Meant for a sample of shards (it runs every family twice per record):

    python -u scripts/rule_breadth.py --records-dir <sample> --workers 10 \\
        --output output/effects/rule-breadth.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from effects.application.rule_breadth import breadth
from effects.domain.default_paths import DEFAULT_CARDS_FOLDER_PATHS, DEFAULT_VARIANT_SCRIPTS
from effects.domain.rule_breadth import flagged_rows
from effects.infrastructure.converted_trees import SourceTrees
from price_predictor.infrastructure.run_manifest import run_script


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--records-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("output/effects/rule-breadth.json"))
    parser.add_argument("--rules-dir", type=Path, default=Path("rules"))
    parser.add_argument("--cards-folder", dest="cards_folders", type=Path, action="append")
    parser.add_argument("--variant-scripts", type=Path, default=Path(DEFAULT_VARIANT_SCRIPTS))
    parser.add_argument("--workers", type=int, default=None,
                        help="Parallel shard readers; unset uses the CPU count")
    parser.add_argument("--min-pool", type=int, default=200,
                        help="Smallest pool a flag is raised on (default: %(default)s)")
    parser.add_argument("--max-share", type=float, default=0.01,
                        help="Share of the pool below which a detector is flagged "
                             "(default: %(default)s)")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args(argv)
    cards_folders = args.cards_folders or DEFAULT_CARDS_FOLDER_PATHS
    return run_script(
        args.output.parent, lambda: _run(args, cards_folders), script=__file__, argv=argv,
        inputs=[args.records_dir, args.rules_dir, *cards_folders, args.variant_scripts],
    )


def _run(args: argparse.Namespace, cards_folders) -> int:
    roots = SourceTrees.of(cards_folders, args.variant_scripts).sidecar_roots()
    rows, records = breadth(args.records_dir, args.rules_dir, roots, args.workers)
    flagged = flagged_rows(rows, args.min_pool, args.max_share)
    document = {
        "records": records, "min_pool": args.min_pool, "max_share": args.max_share,
        "flagged": [row.scenario for row in flagged],
        "scenarios": [row.to_json() for row in rows],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(f"{'scenario':14s} {'tagged':>8s} {'pool':>9s} {'share':>8s}  dropped")
    for row in flagged:
        print(f"{row.scenario:14s} {row.tagged:8d} {row.pool:9d} {row.share:8.4f}  "
              f"{','.join(row.dropped)}")
    print(f"{len(flagged)} of {len(rows)} scenarios flagged over {records} records; "
          f"rows written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

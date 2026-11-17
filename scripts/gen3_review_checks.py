"""Print the gen-3 review's data checks over a records directory and a converted tree.

One line per check: its name, the count it flags, the total it looked at, and the
count the review reported, for comparison. The script only reports; no count makes it
exit non-zero, and it writes no file. Progress, one line per shard, goes to stderr.

Usage: python -u scripts/gen3_review_checks.py [--records-dir DIR] --converted-dir DIR

Without ``--records-dir`` only the checks over the converted tree run.

``--converted-dir`` holds ``cardsfolder/`` and ``tokenscripts/`` with their sidecars: the
tree the records were collected against.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from gen3_review_checks_parts.model import Line
from gen3_review_checks_parts.reviews import review_for
from gen3_review_checks_parts.runner import run_checks, run_tree_checks, shards_of


def format_line(line: Line) -> str:
    name = f"{line.name:<62}"
    return f"{name} count={line.count:>9} total={line.total:>9} review={review_for(line.name)}"


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--records-dir", type=Path, default=None,
                        help="without it, only the tree checks run")
    parser.add_argument("--converted-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.records_dir is None:
        lines = run_tree_checks(args.converted_dir)
    else:
        lines = run_checks(shards_of(args.records_dir), args.converted_dir)
    for line in lines:
        print(format_line(line))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

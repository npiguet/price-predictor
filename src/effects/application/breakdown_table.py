"""The breakdowns' table file: every field of every breakdown line.

The console line holds a group's total and record count; the per-field means
(over 250 counter fields on a line) are here, one row per line and field.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path

from effects.domain.check_label import CheckLabel
from effects.domain.evaluation_report import CheckResult

BREAKDOWNS_CSV = "breakdowns.csv"
COLUMNS = ("check", "qualifier", "field", "value", "records")


def write_breakdowns_csv(path: Path, checks: Iterable[CheckResult]) -> int:
    """One row per (line, field) of every check carrying a table; the number
    of rows written. ``records`` is how many records the mean read, empty for
    a figure that is not a mean over records."""
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        for check in checks:
            label = CheckLabel.parse(check.name)
            for row in check.table:
                writer.writerow((
                    label.head, label.qualifier, row.field, repr(row.value),
                    "" if row.records is None else row.records,
                ))
                count += 1
    return count

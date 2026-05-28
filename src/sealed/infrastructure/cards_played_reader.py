"""Stream parser for ``output/sealed/cards-played.txt``.

Tolerates a trailing partial line (a JVM crash mid-write may leave the
final line non-newline-terminated). Mid-file malformed lines raise.

Schema (eleven semicolon-separated fields, no trailing ``;``):
``timestamp;run_id;set_code;method_A;method_B;cards_played_A;cards_played_B;
cards_not_played_A;cards_not_played_B;winner;starter``
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from sealed.domain.match import Side
from sealed.infrastructure.delimited import parse_pipe_list, split_record

_EXPECTED_FIELDS: int = 11


@dataclass(frozen=True)
class CardsPlayedRow:
    """One game's per-side card-play record (Python mirror of the Java record)."""

    timestamp: str
    run_id: str
    set_code: str
    method_a: str
    method_b: str
    cards_played_a: list[str]
    cards_played_b: list[str]
    cards_not_played_a: list[str]
    cards_not_played_b: list[str]
    winner: Side
    starter: Side


def _parse_line(line: str) -> CardsPlayedRow:
    fields = split_record(line, _EXPECTED_FIELDS)
    timestamp, run_id, set_code, method_a, method_b = fields[0:5]
    return CardsPlayedRow(
        timestamp=timestamp,
        run_id=run_id,
        set_code=set_code,
        method_a=method_a,
        method_b=method_b,
        cards_played_a=parse_pipe_list(fields[5]),
        cards_played_b=parse_pipe_list(fields[6]),
        cards_not_played_a=parse_pipe_list(fields[7]),
        cards_not_played_b=parse_pipe_list(fields[8]),
        winner=Side.parse(fields[9], label="winner"),
        starter=Side.parse(fields[10], label="starter"),
    )


def iter_rows(path: Path) -> Iterator[CardsPlayedRow]:
    """Stream rows from ``path``.

    Tolerates a trailing partial (non-newline-terminated) line silently —
    that's the JVM-crash-mid-write recovery path (Edge Cases).
    Raises ``ValueError`` on a mid-file malformed line.
    """
    path = Path(path)
    with path.open("rb") as f:
        raw = f.read()
    if not raw:
        return
    text = raw.decode("utf-8")
    has_trailing_newline = text.endswith("\n")
    lines = text.splitlines()
    if not has_trailing_newline and lines:
        # Drop the final non-newline-terminated line silently.
        lines = lines[:-1]
    for line in lines:
        yield _parse_line(line)

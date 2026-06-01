"""JSONL corpus IO for ``output/draft/drafts.jsonl`` (FR-012, FR-013).

One self-contained JSON record per line, append-only. Readers tolerate a
trailing partial final line (supervisor crash mid-write); ``--resume`` counts
complete records toward ``--n-drafts``. Mirrors the partial-line tolerance of
``sealed``'s ``pool_file_reader`` / ``cards_played_reader``.

Serialization lives here (infrastructure); the in-memory shape is the pure
``DraftRecord`` domain dataclass from ``draft_geometry``.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import IO, Any

from draft.domain.draft_geometry import Booster, DraftRecord, Seat

_JSON_SEPARATORS = (",", ":")  # compact, newline-free


def record_to_dict(record: DraftRecord) -> dict[str, Any]:
    """Convert a ``DraftRecord`` to its JSON-serializable dict form."""
    return {
        "draft_id": record.draft_id,
        "run_id": record.run_id,
        "timestamp": record.timestamp,
        "seats": [
            {"agent": s.agent, "deck": list(s.deck), "deck_score": s.deck_score}
            for s in record.seats
        ],
        "boosters": [
            {"set_code": b.set_code, "picks": list(b.picks)}
            for b in record.boosters
        ],
    }


def record_from_dict(data: dict[str, Any]) -> DraftRecord:
    """Parse a JSON dict (one corpus line) into a ``DraftRecord``."""
    seats = [
        Seat(
            agent=s["agent"],
            deck=list(s.get("deck", [])),
            deck_score=s.get("deck_score"),
        )
        for s in data["seats"]
    ]
    boosters = [
        Booster(set_code=b["set_code"], picks=list(b["picks"]))
        for b in data["boosters"]
    ]
    return DraftRecord(
        draft_id=data["draft_id"],
        run_id=data["run_id"],
        timestamp=data["timestamp"],
        seats=seats,
        boosters=boosters,
    )


def format_record_line(record: DraftRecord) -> str:
    """Render one corpus line (compact JSON, no trailing newline)."""
    return json.dumps(record_to_dict(record), separators=_JSON_SEPARATORS)


def append_record(out: IO[str], record: DraftRecord) -> None:
    """Append one record line to an open text handle and flush.

    The caller owns the handle (opened in append mode, line-buffered) so a
    long supervisor run keeps it open across many drafts.
    """
    out.write(format_record_line(record))
    out.write("\n")
    out.flush()


def read_records(path: Path) -> Iterator[DraftRecord]:
    """Yield every complete record in ``path``; skip a trailing partial line.

    Each non-blank, newline-terminated line is parsed. A final line without a
    terminating newline (JVM/supervisor crash mid-write) is dropped. Blank
    lines are ignored.
    """
    if not path.exists():
        return
    content = path.read_bytes()
    if not content:
        return
    # A trailing partial line is one not ending in "\n"; splitlines keeps it as
    # the last element, so drop that element when the file doesn't end in "\n".
    text = content.decode("utf-8")
    lines = text.splitlines()
    drop_last = not text.endswith("\n")
    end = len(lines) - 1 if (drop_last and lines) else len(lines)
    for line in lines[:end]:
        stripped = line.strip()
        if not stripped:
            continue
        yield record_from_dict(json.loads(stripped))


def count_complete_records(path: Path) -> int:
    """Count complete (newline-terminated, non-blank) records for ``--resume``.

    A complete record is a non-blank line that ends in ``\\n``. A trailing
    partial final line is not counted (and left in place; the supervisor
    appends after it on its own line only if the partial is newline-free —
    matching ``count_complete_lines_and_truncate_partial`` semantics minus the
    truncation, since JSON readers already skip the partial).
    """
    if not path.exists():
        return 0
    content = path.read_bytes()
    if not content:
        return 0
    text = content.decode("utf-8")
    lines = text.splitlines()
    drop_last = not text.endswith("\n")
    end = len(lines) - 1 if (drop_last and lines) else len(lines)
    return sum(1 for line in lines[:end] if line.strip())

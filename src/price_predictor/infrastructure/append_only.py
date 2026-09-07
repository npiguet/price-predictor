"""Line-completeness rules shared by every append-only corpus in this repo.

The corpora (`match-outcomes.txt`, `cards-played.txt`, `pools.txt`,
`drafts.jsonl`, and the effect-record shards) are all appended to by Forge
worker JVMs that are expected to crash mid-write, so every reader has to treat a
final non-newline-terminated line as absent rather than as corruption. That rule
is the only thing they share — record parsing stays with each corpus.

Only the completeness rule lives here. A "complete line" is one that ends in a
newline; the trailing partial is dropped by the readers and truncated away by
the two `--resume` counters that append after it.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path


def iter_complete_lines(path: Path) -> Iterator[str]:
    """Yield each newline-terminated line of ``path``, terminator stripped.

    A missing or empty file yields nothing. A final line without a terminating
    newline is dropped silently — that is the crash-mid-write recovery path, not
    an error. Blank lines are yielded as empty strings; filtering them is the
    caller's business, because the corpora disagree on whether they are legal.

    The file is streamed rather than materialized, so a multi-gigabyte corpus
    costs one line of memory.
    """
    path = Path(path)
    if not path.exists():
        return
    # Universal newline mode translates \r\n and lone \r to \n, matching what
    # ``str.splitlines`` did in the readers this replaces.
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.endswith("\n"):
                yield line[:-1]
            # Otherwise this is the final partial line: drop it.


def count_complete_lines(path: Path, *, skip_blank: bool = False) -> int:
    """Count the complete lines of ``path``; 0 if it is missing or empty.

    ``skip_blank`` excludes whitespace-only lines, which is what a corpus whose
    reader ignores blank lines needs its ``--resume`` count to agree with.
    """
    lines = iter_complete_lines(path)
    if skip_blank:
        return sum(1 for line in lines if line.strip())
    return sum(1 for _ in lines)


def count_complete_lines_and_truncate_partial(path: Path) -> int:
    """Count complete lines and truncate ``path`` back to its last newline.

    For the appending callers: a process killed between two writes leaves a
    partial final line, and appending after it would splice two records
    together. Truncating first makes the next append start on a clean line.
    Returns the count of surviving complete lines (0 if the file is missing,
    empty, or holds a single partial line with no newline at all).
    """
    path = Path(path)
    if not path.exists():
        return 0
    content = path.read_bytes()
    if not content:
        return 0
    count = content.count(b"\n")
    if not content.endswith(b"\n"):
        last_newline = content.rfind(b"\n")
        if last_newline == -1:
            path.write_bytes(b"")
            return 0
        path.write_bytes(content[: last_newline + 1])
    return count

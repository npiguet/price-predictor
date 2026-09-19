"""The ``effects`` CLI writes UTF-8 whatever the console's code page.

A report holds card texts and arrows, and a redirected stdout on Windows
defaults to cp1252: the evaluator once scored every gate and then died printing
the result.
"""

from __future__ import annotations

import io

from effects.infrastructure.cli import write_output_as_utf8


def _cp1252_stream() -> io.TextIOWrapper:
    return io.TextIOWrapper(io.BytesIO(), encoding="cp1252")


def test_a_cp1252_stream_accepts_text_it_cannot_encode() -> None:
    stdout, stderr = _cp1252_stream(), _cp1252_stream()

    write_output_as_utf8(stdout, stderr)
    stdout.write("pass → probe, Lim-Dûl")
    stdout.flush()

    assert stdout.buffer.getvalue() == "pass → probe, Lim-Dûl".encode("utf-8")


def test_a_stream_without_reconfigure_is_left_alone() -> None:
    stdout = io.StringIO()

    write_output_as_utf8(stdout, stdout)
    stdout.write("→")

    assert stdout.getvalue() == "→"

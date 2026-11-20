"""A worker's startup line is found whatever precedes it in its log, and a
log that cannot be opened fails the spawn (finding 16.1, 16.5, task L01)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from price_predictor.infrastructure.worker_log_files import WorkerLogFiles
from price_predictor.infrastructure.worker_log_mark import LogMark, past_forge_start
from price_predictor.infrastructure.worker_stops import STARTUP_LINE

MIB = 1024 * 1024
NOISE = "The custom card x_variant was not assigned to any set and will be skipped\n"


def _append(path: Path, text: str) -> None:
    with open(path, "ab") as handle:
        handle.write(text.encode("utf-8"))


def test_a_startup_line_three_mib_into_the_log_reads_as_started(tmp_path):
    log = tmp_path / "run.0.log"
    _append(log, "an earlier lifetime\n")
    mark = LogMark.of(log)
    _append(log, NOISE * (3 * MIB // len(NOISE)))
    assert not past_forge_start(mark)

    _append(log, f"{STARTUP_LINE}...\n")

    assert past_forge_start(mark), "startup line 3 MiB in was not found"


def test_a_startup_line_before_the_mark_is_an_earlier_lifetime_s(tmp_path):
    log = tmp_path / "run.0.log"
    _append(log, f"{STARTUP_LINE}...\n")
    mark = LogMark.of(log)
    _append(log, "Forge loading\n")

    assert not past_forge_start(mark)


def test_a_startup_line_split_across_two_reads_is_found(tmp_path):
    log = tmp_path / "run.0.log"
    mark = LogMark.of(log)
    half = len(STARTUP_LINE) // 2
    _append(log, "noise\n" + STARTUP_LINE[:half])
    assert not past_forge_start(mark)

    _append(log, STARTUP_LINE[half:] + "...\n")

    assert past_forge_start(mark)


def test_once_seen_the_log_is_not_read_again(tmp_path):
    log = tmp_path / "run.0.log"
    mark = LogMark.of(log)
    _append(log, f"{STARTUP_LINE}...\n")
    assert past_forge_start(mark)

    log.unlink()

    assert past_forge_start(mark)


def test_a_log_rotated_at_the_spawn_is_read_from_its_start(tmp_path):
    log = tmp_path / "run.0.log"
    _append(log, "x" * 100)
    mark = LogMark.of(log)
    log.rename(log.with_name("run.0.log.1"))
    _append(log, f"{STARTUP_LINE}...\n")

    assert past_forge_start(mark)


def test_an_unopenable_log_fails_the_spawn_naming_its_path(tmp_path):
    logs = WorkerLogFiles(tmp_path, "run-1")
    with patch.object(Path, "open", side_effect=OSError("boom")):
        with pytest.raises(OSError, match="run-1.0.log") as raised:
            logs.get(0)

    assert "boom" in str(raised.value)
    logs.close_all()

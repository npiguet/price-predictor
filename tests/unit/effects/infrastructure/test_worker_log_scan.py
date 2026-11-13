"""Scanning a run's worker logs for failure lines, a round at a time (V4 #3)."""

from __future__ import annotations

from pathlib import Path

from effects.infrastructure.worker_log_scan import WorkerLogScan


def _append(path: Path, text: str) -> None:
    with open(path, "ab") as handle:
        handle.write(text.encode("utf-8"))


def test_each_scan_counts_only_what_the_logs_gained(tmp_path):
    log = tmp_path / "run.0.log"
    _append(log, "Worker: 10 matches generated\nError generating match: a\n")
    scan = WorkerLogScan()

    assert scan.scan([log]).as_dict() == {"match_error": 1}

    _append(log, "Error generating match: b\nError generating match: c\n")
    assert scan.scan([log]).as_dict() == {"match_error": 2}
    assert not scan.scan([log])


def test_a_line_still_being_written_is_read_whole_next_time(tmp_path):
    log = tmp_path / "run.0.log"
    _append(log, "Error generating match: a\nError genera")
    scan = WorkerLogScan()
    assert scan.scan([log]).as_dict() == {"match_error": 1}

    _append(log, "ting match: b\n")
    assert scan.scan([log]).examples["match_error"] == "Error generating match: b"


def test_a_rotated_log_is_finished_before_the_new_one_is_read(tmp_path):
    log = tmp_path / "run.0.log"
    _append(log, "Error generating match: a\n")
    scan = WorkerLogScan()
    scan.scan([log])
    _append(log, "Error generating match: before the rotation\n")
    log.rename(log.with_name(log.name + ".1"))
    _append(log, "Worker giving up after 50 consecutive failed attempts\n")

    assert scan.scan([log]).as_dict() == {"match_error": 1, "worker_gave_up": 1}


def test_a_log_not_yet_written_is_skipped(tmp_path):
    assert not WorkerLogScan().scan([tmp_path / "run.3.log"])


def test_every_slot_s_log_is_read(tmp_path):
    logs = [tmp_path / f"run.{slot}.log" for slot in range(3)]
    for log in logs:
        _append(log, "Error generating match: x\n")

    assert WorkerLogScan().scan(logs).as_dict() == {"match_error": 3}

"""Scanning a run's worker logs for failure lines, a round at a time (V4 #3)."""

from __future__ import annotations

from pathlib import Path

from price_predictor.infrastructure.worker_log_scan import WorkerLogScan
from tests.unit.worker_status_line import directed_counts, status_line


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


def test_a_log_rotated_twice_between_scans_reports_the_lost_generation(tmp_path):
    """The first generation is gone (the second rotation replaced its .1);
    the second is read whole from .1 and the loss is reported, not hidden."""
    log = tmp_path / "run.0.log"
    rotated = log.with_name(log.name + ".1")
    _append(log, "Error generating match: a\n")
    scan = WorkerLogScan()
    scan.scan([log])
    _append(log, "Error generating match: in the lost generation\n")
    log.rename(rotated)
    _append(log, "Error generating match: second generation\n")
    rotated.unlink()
    log.rename(rotated)
    _append(log, "Worker giving up after 50 consecutive failed attempts\n")

    found = scan.scan([log])

    assert found.as_dict() == {"match_error": 1, "worker_gave_up": 1, "log_unread": 1}
    assert str(log) in found.examples["log_unread"]


def _hooks(n: int) -> dict:
    return {"connector": {"RESOLVE": n}, "forge": {"resolve": n}}


def test_the_status_lines_are_followed_across_scans_lifetimes_and_slots(tmp_path):
    """Counts each worker JVM prints and no failure line (WP7, K01b): the
    run's counts are each lifetime's latest status line, summed over
    lifetimes and slots, a lifetime told by the line's own token."""
    first, second = tmp_path / "run.0.log", tmp_path / "run.1.log"
    _append(first, status_line("a", hook_failures=_hooks(1)) + "\n")
    _append(second, status_line("c", hook_failures=_hooks(4),
                                fork_refusals={"directed dies": 4}) + "\n")
    scan = WorkerLogScan()

    assert not scan.scan([first, second])
    _append(first, status_line("a", hook_failures=_hooks(2), fork_refusals={"directed dies": 1})
            + "\n" + status_line("b") + "\n"
            + status_line("b", hook_failures=_hooks(3), fork_refusals={"directed dies": 2}) + "\n")
    assert not scan.scan([first, second])

    assert scan.hook_failures() == {"connector": {"RESOLVE": 9}, "forge": {"resolve": 9}}
    assert scan.fork_refusals() == {"directed dies": 7}


def test_a_status_line_still_being_written_is_read_whole_next_time(tmp_path):
    log = tmp_path / "run.0.log"
    line = status_line("a", fork_refusals={"directed dies": 2}) + "\n"
    _append(log, line[:40])
    scan = WorkerLogScan()
    scan.scan([log])

    _append(log, line[40:])
    scan.scan([log])

    assert scan.fork_refusals() == {"directed dies": 2}


def test_the_totals_reach_the_manifest_fields(tmp_path):
    """Each lifetime's latest line, summed over lifetimes and slots, no failure line."""
    from price_predictor.infrastructure.forge_jvm import WorkerLogFiles
    from price_predictor.infrastructure.worker_log_scan import WorkerLogReport

    def directed(n: int) -> dict:
        return {"704.5f": directed_counts(matched=n, drawn=1, taken=1, kept=1)}

    logs = WorkerLogFiles(tmp_path, "run")
    _append(logs.path_for(0), "\n".join([
        status_line("a", directed=directed(1)),
        status_line("a", directed=directed(2), hook_failures=_hooks(2)),
        status_line("b", directed=directed(3), fork_refusals={"directed dies": 2}),
    ]) + "\n")
    _append(logs.path_for(1), status_line("c", directed=directed(4)) + "\n")
    report = WorkerLogReport(logs, 2)
    report.scan()

    fields = report.manifest_fields()
    assert fields["directed_fork_tally"] == {"704.5f": directed_counts(
        matched=9, drawn=3, taken=3, kept=3)}
    assert fields["hook_failures"] == {"connector": {"RESOLVE": 2}, "forge": {"resolve": 2}}
    assert fields["fork_refusals"] == {"directed dies": 2}
    assert fields["worker_failures"] == {}


def test_the_first_status_line_stays_the_first_and_later_lifetimes_are_compared(tmp_path):
    """Finding 11.1: the run's engine is its first lifetime's; a later one on
    another engine shows in the verdict."""
    logs = [tmp_path / "run.0.log", tmp_path / "run.1.log"]
    scan = WorkerLogScan()
    assert scan.first_status() is None
    scan.scan(logs)
    assert scan.first_status() is None

    _append(logs[1], status_line("b") + "\n")
    scan.scan(logs)
    _append(logs[0], status_line("c", hooks={"mode": "degraded", "gaps": []}) + "\n")
    scan.scan(logs)

    assert scan.first_status().lifetime == "b"
    verdict = scan.engine_verdict()
    assert verdict.first.lifetime == "b"
    assert set(verdict.changed) == {"c"}

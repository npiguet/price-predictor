"""Scanning a run's worker logs for failure lines, a round at a time (V4 #3)."""

from __future__ import annotations

from pathlib import Path

from price_predictor.infrastructure.worker_log_scan import WorkerLogScan


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


_REPORT = ("effect-record hook failures so far, listener calls that threw and lost their "
           "records: {{RESOLVE={n}}}; contained by Forge, per slot: {{resolve={n}}}\n")


def test_the_hook_report_is_followed_across_scans_lifetimes_and_slots(tmp_path):
    """A count each worker JVM prints and nothing read (WP7): the run's counts
    are each lifetime's latest report, summed over lifetimes and slots, and
    none of it is a failure line."""
    first, second = tmp_path / "run.0.log", tmp_path / "run.1.log"
    _append(first, "Effect records: x/run.0-a.jsonl.gz [mode=patched]\n" + _REPORT.format(n=1))
    _append(second, _REPORT.format(n=4))
    scan = WorkerLogScan()

    assert not scan.scan([first, second])
    _append(first, _REPORT.format(n=2) + "Effect records: x/run.0-b.jsonl.gz [mode=patched]\n"
            + _REPORT.format(n=3))
    scan.scan([first, second])

    assert scan.hook_failures() == {"connector": {"RESOLVE": 9}, "forge": {"resolve": 9}}


def test_the_report_s_totals_reach_the_manifest_fields(tmp_path):
    from price_predictor.infrastructure.forge_jvm import WorkerLogFiles
    from price_predictor.infrastructure.worker_log_scan import WorkerLogReport

    logs = WorkerLogFiles(tmp_path, "run")
    _append(logs.path_for(0), _REPORT.format(n=2))
    report = WorkerLogReport(logs, 1)
    report.scan()

    assert report.manifest_fields()["hook_failures"] == {
        "connector": {"RESOLVE": 2}, "forge": {"resolve": 2},
    }


_REFUSALS = "Effect records: fork refusals so far, expected and no failure: {{directed dies={n}}}\n"


def test_the_fork_refusal_tally_is_followed_across_scans_lifetimes_and_slots(tmp_path):
    """A running map each worker JVM prints at a match end (WP5) and nothing
    read: each lifetime's latest tally, summed over lifetimes and slots, and
    no failure line."""
    first, second = tmp_path / "run.0.log", tmp_path / "run.1.log"
    _append(first, "Effect records: x/run.0-a.jsonl.gz [mode=patched]\n" + _REFUSALS.format(n=1))
    _append(second, _REFUSALS.format(n=4))
    scan = WorkerLogScan()

    assert not scan.scan([first, second])
    _append(first, _REFUSALS.format(n=2) + "Effect records: x/run.0-b.jsonl.gz [mode=patched]\n"
            + _REFUSALS.format(n=3))
    assert not scan.scan([first, second])

    assert scan.fork_refusals() == {"directed dies": 9}


def test_the_fork_refusals_reach_the_manifest_fields(tmp_path):
    from price_predictor.infrastructure.forge_jvm import WorkerLogFiles
    from price_predictor.infrastructure.worker_log_scan import WorkerLogReport

    logs = WorkerLogFiles(tmp_path, "run")
    _append(logs.path_for(0), _REFUSALS.format(n=2))
    report = WorkerLogReport(logs, 1)
    report.scan()

    fields = report.manifest_fields()
    assert fields["fork_refusals"] == {"directed dies": 2}
    assert fields["worker_failures"] == {}


_DIRECTED = (
    'Effect records: directed fork tally so far: {{"704.5f":{{"matched":{n},"drawn":1,'
    '"budget_blocked":0,"taken":1,"kept":1,"precondition_threw":0}}}}\n'
)


def test_the_directed_fork_tally_is_summed_and_reaches_the_manifest_fields(tmp_path):
    """Each lifetime's latest line, summed over lifetimes and slots, no failure line."""
    from price_predictor.infrastructure.forge_jvm import WorkerLogFiles
    from price_predictor.infrastructure.worker_log_scan import WorkerLogReport

    logs = WorkerLogFiles(tmp_path, "run")
    _append(logs.path_for(0), "Effect records: x/run.0-a.jsonl.gz [mode=patched]\n"
            + _DIRECTED.format(n=1) + _DIRECTED.format(n=2)
            + "Effect records: x/run.0-b.jsonl.gz [mode=patched]\n" + _DIRECTED.format(n=3))
    _append(logs.path_for(1), _DIRECTED.format(n=4))
    report = WorkerLogReport(logs, 2)
    report.scan()

    fields = report.manifest_fields()
    assert fields["directed_fork_tally"] == {"704.5f": {
        "matched": 9, "drawn": 3, "budget_blocked": 0, "taken": 3, "kept": 3,
        "precondition_threw": 0}}
    assert fields["worker_failures"] == {}

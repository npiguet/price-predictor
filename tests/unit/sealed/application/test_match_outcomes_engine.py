"""``match-outcomes --effect-records`` refuses a degraded engine (finding 11.1,
task K01c): the log scan at each status line reads the workers' status
lines, asks the pool to stop once the engine is not ``patched`` with every
hook, and ``run`` raises, unless ``allow_degraded``. Without effect records
the workers keep no log and nothing is judged."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from price_predictor.domain.worker_status import WorkerEngineRefused
from price_predictor.infrastructure.run_manifest import RunCommand, run_recorded
from sealed.application.match_outcomes import MatchOutcomeSupervisor, MatchRunOptions
from tests.unit.sealed.conftest import FakeProcess
from tests.unit.worker_status_line import PINNED_PREFERENCES, status_line

DEGRADED = status_line("a", hooks={"mode": "degraded", "gaps": []})
GAP = "EffectRecordListener.onReplacement missing"
PATCHED_WITH_GAP = status_line("a", hooks={"mode": "patched", "gaps": [GAP]})


def _supervisor(tmp_path, *, allow_degraded: bool = False) -> MatchOutcomeSupervisor:
    supervisor = MatchOutcomeSupervisor(
        worker_count=1, output_path=tmp_path / "match-outcomes.txt", best_of=3,
        options=MatchRunOptions(
            effect_records_dir=tmp_path / "records",
            allow_degraded=allow_degraded,
        ),
    )
    supervisor._connector = MagicMock()
    supervisor._connector.start.return_value = FakeProcess(pid=1000)
    return supervisor


def _log(supervisor: MatchOutcomeSupervisor, line: str) -> None:
    log = supervisor._worker_logs.get(0)
    log.write(f"{line}\n".encode())
    log.flush()


def _run(tmp_path, line: str, *, allow_degraded: bool = False) -> MatchOutcomeSupervisor:
    """A whole run through the real pool: its one worker prints ``line`` and
    plays a match; the run then ends."""
    supervisor = _supervisor(tmp_path, allow_degraded=allow_degraded)

    def fake_start_worker(worker_id):
        _log(supervisor, line)
        with open(supervisor._output_path, "a", encoding="utf-8") as out:
            out.write("row\n")
        supervisor._shutdown_event.set()
        return FakeProcess(pid=1000, returncode=0)

    supervisor._start_worker = fake_start_worker
    supervisor.run()
    return supervisor


def _entry(tmp_path) -> dict:
    return json.loads(
        (tmp_path / "records" / "run.json").read_text(encoding="utf-8"))["invocations"][0]


@pytest.mark.parametrize(("line", "named"), [
    (DEGRADED, "mode degraded"), (PATCHED_WITH_GAP, GAP)])
def test_the_status_scan_asks_the_pool_to_stop(tmp_path, capsys, line, named):
    supervisor = _supervisor(tmp_path)
    _log(supervisor, line)

    supervisor._pool._on_status()  # the status line's log scan

    assert supervisor._shutdown_event.is_set()
    assert "Stopping the run: " in capsys.readouterr().out


@pytest.mark.parametrize(("line", "named"), [
    (DEGRADED, "mode degraded"), (PATCHED_WITH_GAP, GAP)])
def test_a_run_on_an_engine_that_is_not_whole_is_refused_naming_what_is_missing(
    tmp_path, line, named,
):
    with pytest.raises(WorkerEngineRefused) as refused:
        _run(tmp_path, line)

    assert named in str(refused.value)
    assert str(tmp_path / "records") in str(refused.value)


def test_the_refused_run_still_records_its_totals_and_engine(tmp_path):
    def body():
        _run(tmp_path, PATCHED_WITH_GAP)
        return 0

    with pytest.raises(WorkerEngineRefused):
        run_recorded(tmp_path / "records", body, RunCommand("python -m sealed match-outcomes"))

    entry = _entry(tmp_path)
    assert entry["worker_engine"] == {
        "mode": "patched", "gaps": [GAP], "pinned_preferences": PINNED_PREFERENCES}
    assert entry["matches"] == 1


@pytest.mark.parametrize("line", [DEGRADED, PATCHED_WITH_GAP])
def test_allow_degraded_lets_it_run_and_run_json_records_the_engine(tmp_path, capsys, line):
    def body():
        _run(tmp_path, line, allow_degraded=True)
        return 0

    run_recorded(tmp_path / "records", body, RunCommand("python -m sealed match-outcomes"))

    assert _entry(tmp_path)["worker_engine"]["gaps"] == list(
        json.loads(line.partition(": status: ")[2])["hooks"]["gaps"])
    out = capsys.readouterr().out
    assert out.count("Warning: --allow-degraded") == 1


def test_a_whole_engine_runs_on(tmp_path):
    supervisor = _run(tmp_path, status_line("a"))
    assert supervisor.collected_nothing is True  # no record, but no refusal


def test_a_plain_run_without_effect_records_never_judges(tmp_path):
    """Its workers print the line too, to a log nothing keeps: informational."""
    supervisor = MatchOutcomeSupervisor(
        worker_count=1, output_path=tmp_path / "match-outcomes.txt", best_of=3,
    )

    def fake_start_worker(worker_id):
        supervisor._shutdown_event.set()
        return FakeProcess(pid=1000, returncode=0)

    supervisor._start_worker = fake_start_worker
    supervisor.run()  # raises nothing

    assert supervisor._log_report is None
    assert supervisor._pool._on_status is None

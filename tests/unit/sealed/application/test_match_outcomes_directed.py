"""DR1 (finding 6.1): the ``match-outcomes`` supervisor recomputes each listed
rule's directed-fork probability at every log scan from the forks its workers
kept, and starts each later worker with those, so a rule whose side fills
within the run stops taking the game's budget."""

from __future__ import annotations

import argparse
import json
from unittest.mock import MagicMock

from sealed.application.match_outcomes import MatchOutcomeSupervisor, MatchRunOptions
from sealed.domain.directed_fork_needs import DirectedForkNeeds, DirectedRuleNeed
from sealed.infrastructure.cli_match_outcomes import _effect_collection_caps
from tests.unit.sealed.conftest import FakeProcess
from tests.unit.worker_status_line import directed_counts, status_line

NEEDS = DirectedForkNeeds(count_per_side=100, rules={
    "118.12b": DirectedRuleNeed("fires", 0),
    "704.5f": DirectedRuleNeed("does_not_fire", 6814),
})

#: 118.12b's side filled by this run's kept forks, in a worker's status line.
TALLY = status_line("a", directed={"118.12b": directed_counts(
    matched=140, drawn=120, budget_blocked=20, taken=100, kept=100)}) + "\n"

PROPERTY = "effect.directed.probability.118.12b"


def _supervisor(tmp_path) -> MatchOutcomeSupervisor:
    caps = _effect_collection_caps(argparse.Namespace(
        directed_fork_rules=tuple(NEEDS.rules),
        directed_rule_probabilities=NEEDS.probabilities(),
    ))
    supervisor = MatchOutcomeSupervisor(
        worker_count=1, output_path=tmp_path / "out.txt", best_of=3,
        options=MatchRunOptions(
            effect_records_dir=tmp_path / "records",
            collection_caps=caps,
        ),
    )
    supervisor.steer_directed_forks(NEEDS)
    supervisor._connector = MagicMock()
    supervisor._connector.start.return_value = FakeProcess(pid=1000)
    return supervisor


def _log_tally(supervisor: MatchOutcomeSupervisor) -> None:
    log = supervisor._worker_logs.get(0)
    log.write(TALLY.encode("utf-8"))
    log.flush()


def test_a_worker_spawned_after_a_scan_takes_the_recomputed_probabilities(tmp_path):
    supervisor = _supervisor(tmp_path)

    supervisor._start_worker(0)
    _log_tally(supervisor)
    supervisor._pool._on_status()  # the status line's log scan
    supervisor._start_worker(0)

    first, second = (call.kwargs["launch"].records.collection_caps
                     for call in supervisor._connector.start.call_args_list)
    assert first[PROPERTY] == "1.0"
    assert second[PROPERTY] == "0.0"
    assert second["effect.directed.fork.rules"] == "118.12b,704.5f"


def test_without_a_scan_every_spawn_keeps_the_starting_probabilities(tmp_path):
    supervisor = _supervisor(tmp_path)
    _log_tally(supervisor)

    supervisor._start_worker(0)

    launch = supervisor._connector.start.call_args.kwargs["launch"]
    assert launch.records.collection_caps[PROPERTY] == "1.0"


def test_the_run_json_keeps_the_starting_and_the_final_probabilities(tmp_path):
    from price_predictor.infrastructure.run_manifest import RunCommand, run_recorded

    records = tmp_path / "records"

    def body():
        supervisor = _supervisor(tmp_path)
        supervisor._pool.run = lambda: _log_tally(supervisor)
        supervisor.run()
        return 0

    run_recorded(records, body, RunCommand("python -m sealed match-outcomes"))

    entry = json.loads((records / "run.json").read_text(encoding="utf-8"))["invocations"][0]
    assert entry["caps"]["directed_rule_probabilities"] == {"118.12b": 1.0, "704.5f": 0.0}
    assert entry["final_directed_rule_probabilities"] == {"118.12b": 0.0, "704.5f": 0.0}


def test_a_run_given_no_counts_keeps_its_caps_probabilities_to_the_end(tmp_path):
    from price_predictor.infrastructure.run_manifest import RunCommand, run_recorded

    records = tmp_path / "records"
    caps = _effect_collection_caps(argparse.Namespace(
        directed_fork_rules=("118.12b",), directed_rule_probabilities={"118.12b": 0.5}))

    def body():
        supervisor = MatchOutcomeSupervisor(
            worker_count=1, output_path=tmp_path / "out.txt", best_of=3,
            options=MatchRunOptions(
                effect_records_dir=records,
                collection_caps=caps,
            ),
        )
        supervisor._pool.run = lambda: None
        supervisor.run()
        return 0

    run_recorded(records, body, RunCommand("python -m sealed match-outcomes"))

    entry = json.loads((records / "run.json").read_text(encoding="utf-8"))["invocations"][0]
    assert entry["final_directed_rule_probabilities"] == {"118.12b": 0.5}

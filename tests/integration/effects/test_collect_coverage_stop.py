"""A real ``collect-coverage`` round ends by the stop protocol (WP12).

One short card round over two wording cards, through the command line as an
operator runs it: ``CollectorSupervisor`` and ``ForgeWorkerPool`` under a
``GracefulStop``, two real Forge workers. The round's budget is two matches, so
the pool asks both workers to stop as soon as the progress file counts two;
each finishes the game it is playing, writes its game-end marker, closes its
shard and exits 0 with the stop line (``contracts/cli.md`` § Match worker stop
protocol). What the module reads afterwards:

- every worker exited 0 with the stop line, or was killed still starting Forge
  (no game to cut); no worker was killed mid-game and no game is cut;
- every gen-3 game in the shards has its marker, and the marker decodes with
  the run's mode, its ``aimed_at`` and a board;
- the run directory is laid out as the artifact layout says: shards at the top,
  every worker log, the round's progress file and nothing else under ``logs/``,
  the decks under ``decks/``, the stop files gone;
- ``run.json``'s entry for the command records the resolved caps, the stop
  settings, the rounds, ``worker_stops``, ``hook_failures``, ``fork_refusals``,
  ``worker_failures`` and ``cut_games``, and what it records agrees with the
  logs and shards.

Skips when the connector JAR is not built.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from effects.domain.records import RecordKind, TurnActionSubkind
from effects.infrastructure.record_codec import record_from_dict
from effects.infrastructure.shard_io import ended_games, iter_shard_lines, iter_shards
from price_predictor.infrastructure.forge_jvm import (
    STARTUP_LINE,
    STOPPED_LINE,
    kill_process_tree,
)
from tests.integration.conftest import assert_no_lost_output, require_jar

pytestmark = pytest.mark.integration

_REPOSITORY = Path(__file__).resolve().parents[3]
#: Two Forge starts, a castability consult, two best-of-one games and the
#: stop; a cold machine takes three to four minutes.
_RUN_SECONDS = 900
_WORKERS = 2
_STOP_TIMEOUT = 300
#: Burn and a creature: a game of a few turns, a record per cast.
_WORDING_CARDS = ("lightning bolt", "goblin piker")
#: ``python -m effects``, its pool's status interval outlasting the run: the
#: periodic kill of the longest-running worker
#: (``ForgeWorkerPool._kill_oldest_worker``, tested on stand-in workers in
#: ``tests/unit/infrastructure/test_worker_stop_protocol.py``) would cut a game
#: of a round slower than a minute, and this module reads the round's end alone.
_LAUNCHER = (
    "import runpy\n"
    "from price_predictor.infrastructure.forge_jvm import ForgeWorkerPool\n"
    f"ForgeWorkerPool.STATUS_INTERVAL = {2 * _RUN_SECONDS}\n"
    "runpy.run_module('effects', run_name='__main__', alter_sys=True)\n"
)


def _gen3_lines(run_dir: Path) -> list[tuple[Path, list[dict]]]:
    return [
        (shard, [json.loads(line) for line in iter_shard_lines(shard)])
        for shard in iter_shards(run_dir)
    ]


@pytest.fixture(scope="module")
def coverage_run(tmp_path_factory):
    """Run one ``collect-coverage`` round to its end; the run directory and
    what the command printed."""
    require_jar()
    root = tmp_path_factory.mktemp("coverage-stop")
    run_dir = root / "run"
    wording = root / "wording-cards.txt"
    wording.write_text("\n".join(_WORDING_CARDS) + "\n", encoding="utf-8")
    holdout = root / "holdout-cards.txt"
    holdout.write_text("", encoding="utf-8")
    console = root / "console.log"
    command = [
        sys.executable, "-u", "-c", _LAUNCHER, "collect-coverage",
        "--effect-records", str(run_dir),
        "--wording-cards", str(wording),
        "--exclude-cards", str(holdout),
        "--workers", str(_WORKERS),
        "--decks-per-round", "2",
        "--target-records", "1",
        "--no-progress-rounds", "1",
        "--stop-timeout", str(_STOP_TIMEOUT),
        "--seed", "7",
    ]
    environment = {**os.environ, "PYTHONPATH": os.pathsep.join(["src", "."])}
    started = time.monotonic()
    with open(console, "wb") as out:
        process = subprocess.Popen(
            command, cwd=_REPOSITORY, env=environment, stdout=out, stderr=subprocess.STDOUT,
        )
        try:
            code = process.wait(timeout=_RUN_SECONDS)
        except subprocess.TimeoutExpired:
            kill_process_tree(process)
            pytest.fail(f"collect-coverage did not end within {_RUN_SECONDS} s; see {console}")
    printed = console.read_text(encoding="utf-8", errors="replace")
    return run_dir, code, printed, time.monotonic() - started


def _entry(run_dir: Path) -> dict:
    document = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    entries = [e for e in document["invocations"] if "collect-coverage" in str(e["command"])]
    assert len(entries) == 1, document["invocations"]
    return entries[0]


def test_the_round_ends_with_exit_0(coverage_run) -> None:
    run_dir, code, printed, _ = coverage_run
    assert code == 0, printed[-4000:]
    assert any(iter_shards(run_dir)), "the round wrote no shard"


def test_every_worker_stopped_at_a_game_end_or_was_still_starting(coverage_run) -> None:
    run_dir, _, _, _ = coverage_run
    entry = _entry(run_dir)
    stops = entry["worker_stops"]
    logs = sorted((run_dir / "logs").glob("*.log"))
    assert len(logs) == _WORKERS, [log.name for log in logs]
    stopped = 0
    for log in logs:
        text = log.read_text(encoding="utf-8", errors="replace")
        line = STOPPED_LINE.search(text)
        if line is not None:
            stopped += 1
            assert line.group("reason").startswith("stop file "), line.group(0)
            assert line.group("reason").endswith(".stop"), line.group(0)
        else:
            # Killed at the stop while starting Forge: no game began.
            assert STARTUP_LINE not in text, f"{log.name} played and never printed the stop line"
    assert stopped >= 1, "no worker stopped at a game end"
    assert stops == {
        "stopped": stopped, "recycled": 0, "killed": 0, "ended_otherwise": 0,
        "crashed": 0, "killed_starting": _WORKERS - stopped, "survived": 0,
    }, stops
    assert entry["worker_survivors"] == []


def test_no_game_is_cut(coverage_run) -> None:
    run_dir, _, _, _ = coverage_run
    assert _entry(run_dir)["cut_games"] == {"before_rounds": 0, "by_round": [0], "total": 0}
    games = 0
    for shard, lines in _gen3_lines(run_dir):
        played = {line["game_id"] for line in lines if line.get("generation") == 3}
        ended = ended_games(shard)
        assert played <= ended, f"{shard.name}: games without a marker {sorted(played - ended)}"
        games += len(played)
    assert games >= 2, f"the round's budget is two matches; the shards hold {games} games"


def test_each_marker_decodes_with_the_runs_stamps(coverage_run) -> None:
    run_dir, _, _, _ = coverage_run
    markers = []
    for _, lines in _gen3_lines(run_dir):
        records = [record_from_dict(line) for line in lines]
        modes = {record.mode for record in records}
        assert len(modes) == 1, modes
        for index, record in enumerate(records):
            if record.subkind is not TurnActionSubkind.GAME_END:
                continue
            markers.append(record)
            assert record.kind is RecordKind.TURN_ACTION
            assert record.mode in modes
            assert record.mode.value == "patched", record.record_id
            # A collect-coverage record is aimed at the run's rules: none here.
            assert record.aimed_at == (), record.aimed_at
            assert record.state is not None and record.state.players, record.record_id
            assert record.payload.records_withheld == 0, record.record_id
            later = [r for r in records[index + 1:] if r.game_id == record.game_id]
            assert not later, f"{record.game_id} wrote records after its marker"
    assert markers, "no game-end marker in the shards"


def test_the_run_directory_follows_the_layout(coverage_run) -> None:
    run_dir, _, _, _ = coverage_run
    entry = _entry(run_dir)
    run_id = entry["run_id"]
    top = sorted(path.name for path in run_dir.iterdir())
    assert set(top) <= {"run.json", "logs", "decks"} | {
        shard.name for shard in iter_shards(run_dir)
    }, top
    for shard in iter_shards(run_dir):
        assert shard.name.startswith(f"{run_id}."), shard.name
    logs = sorted(path.name for path in (run_dir / "logs").iterdir())
    assert logs == sorted(
        [f"{run_id}.{slot}.log" for slot in range(_WORKERS)]
        + [f"{run_id}.round-0001.progress.txt"]
    ), logs
    assert [path.name for path in (run_dir / "decks").iterdir()] == [
        f"coverage-decks.{run_id}.txt"
    ]
    progress = (run_dir / "logs" / f"{run_id}.round-0001.progress.txt").read_text(
        encoding="utf-8",
    )
    # At least the budget; past it, what finished between two polls of the
    # progress file (every 2 s: a burn game lasts a few seconds) and the game
    # each worker was playing when asked. Each line is a best-of-one match,
    # one game, every one of them ended with its marker.
    ended = set().union(*(ended_games(shard) for shard in iter_shards(run_dir)))
    assert len(progress.splitlines()) == len(ended) >= 2, progress


def test_run_json_records_the_runs_settings_and_counts(coverage_run) -> None:
    run_dir, _, _, _ = coverage_run
    document = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert document["status"] == "finished", document.get("outcome")
    entry = _entry(run_dir)
    assert entry["stop_timeout"] == _STOP_TIMEOUT
    assert entry["recycle_seconds"] == 600
    assert entry["rounds"] == 1
    caps = entry["caps"]
    assert caps["aimed_at"] == [] and caps["directed_fork_rules"] == [], caps
    assert caps["max_records_per_game"] == 5000, caps
    assert set(entry["hook_failures"]) == {"connector", "forge"}, entry["hook_failures"]
    assert isinstance(entry["fork_refusals"], dict)
    assert isinstance(entry["worker_failures"], dict)
    assert_no_lost_output(entry)

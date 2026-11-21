"""A real match worker recycles itself at a game end, under the real supervisor (WP12).

``MatchOutcomeSupervisor`` and its ``ForgeWorkerPool`` run one Forge worker the
way ``sealed match-outcomes --effect-records`` runs it, inside the command's own
``run.json`` (``run_recorded``) after the command's conversion check against the
gen-3 converted tree, which records the conversion there, except for the
recycle time: the command line refuses one under 300 s, so the supervisor is
handed a short one directly, and
for the status interval: the pool's periodic kill of its longest-running
worker (``ForgeWorkerPool._kill_oldest_worker``, tested on stand-in workers in
``tests/unit/infrastructure/test_worker_stop_protocol.py``) would end the one
worker about when its recycle time comes, so the interval outlasts the run. The
worker stops itself at its first game end past that time, printing the stop
line; the pool starts its successor, a new JVM lifetime with a shard of its
own; once that successor has finished a game the run is ended as Ctrl-C ends
it, and the successor too stops at its game's end.

The run also carries the whole directed-fork list (``rules/directed-fork-rules.txt``)
at probability 1, after its startup check, so a directed fork is taken wherever
a listed rule's board precondition holds in these few games.

Read afterwards: how the workers ended (``worker_stops``, the stop lines), the
two lifetimes' shards (every record decodes, every game has its marker, ids
unique across lifetimes), ``run.json``'s entry and the logs under ``logs/``,
any directed fork's record, and the Java-written shards through the Python
pipeline: ``build-corpus``, ``validate-corpus`` and ``rule-counts``. Those three
report: what a few games cannot satisfy (a stratum's size, a rule's count) is
read, not asserted.

Skips when the connector JAR is not built.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from effects.domain.fork_operation import ForkOperation
from effects.domain.records import TurnActionSubkind
from effects.infrastructure.record_codec import record_from_dict
from effects.infrastructure.shard_io import ended_games, iter_shard_lines, iter_shards
from price_predictor.infrastructure.forge_worker_pool import ForgeWorkerPool
from price_predictor.infrastructure.run_manifest import RunCommand, run_recorded
from price_predictor.infrastructure.worker_stops import STOPPED_LINE, WorkerStopSettings
from sealed.application.match_outcomes import MatchOutcomeSupervisor, MatchRunOptions
from sealed.infrastructure.cli import build_parser
from sealed.infrastructure.cli_match_outcomes import _effect_collection_caps
from sealed.infrastructure.collection_conversion import check_conversion
from sealed.infrastructure.directed_fork_check import directed_fork_error
from tests.integration.conftest import assert_no_lost_output, require_jar
from tests.unit.effects.builders import FIXTURE_KEYWORD_DEFINITIONS, write_vocabulary

pytestmark = pytest.mark.integration

_REPOSITORY = Path(__file__).resolve().parents[3]
#: Below the command line's floor (300 s): long enough for one Forge start and
#: a game, short enough that the module stays in minutes.
_RECYCLE_SECONDS = 60
_STOP_TIMEOUT = 300
#: The whole run's bound: two Forge starts, the recycle time, a few games.
_RUN_SECONDS = 600
_POLL_SECONDS = 2
#: The gen-3 converted tree the records' keys join against, as the command's
#: ``--cards-folder``: its conversion is checked against the connector JAR and
#: recorded in the run's ``run.json``, which ``build-corpus`` reads back.
_CARDS_FOLDER = _REPOSITORY / "output" / "effects" / "gen3" / "converted" / "cardsfolder"


def _directed_rules() -> list[str]:
    text = (_REPOSITORY / "rules" / "directed-fork-rules.txt").read_text(encoding="utf-8")
    return [line.strip() for line in text.splitlines()
            if line.strip() and not line.startswith("#")]


def _lifetime(shard: Path) -> str:
    """``{run_id}.{slot}-{lifetime}.jsonl.gz``: the slot and lifetime part."""
    return shard.name.removesuffix(".jsonl.gz").rsplit(".", 1)[-1]


@pytest.fixture(scope="module")
def recycled_run(tmp_path_factory):
    require_jar()
    root = tmp_path_factory.mktemp("recycle")
    run_dir = root / "run"
    rules = _directed_rules()
    args = build_parser().parse_args([
        "match-outcomes", "--effect-records", str(run_dir), "--workers", "1",
        "--best-of", "1", "--directed-fork-rules", ",".join(rules),
        "--directed-fork-probability", "1", "--cards-folder", str(_CARDS_FOLDER),
    ])
    # The command's startup check, over the whole list: one DirectedRulesMain run.
    startup_error = directed_fork_error(args.directed_fork_rules, args.directed_fork_probability)
    assert startup_error is None, startup_error
    with pytest.MonkeyPatch.context() as patch:
        # Read by the pool at its construction (module docstring).
        patch.setattr(ForgeWorkerPool, "STATUS_INTERVAL", 2 * _RUN_SECONDS)
        supervisor = MatchOutcomeSupervisor(
            worker_count=1, output_path=root / "match-outcomes.txt", best_of=1,
            options=MatchRunOptions(
                effect_records_dir=run_dir,
                collection_caps=_effect_collection_caps(args),
                worker_stop=WorkerStopSettings(
                    timeout=_STOP_TIMEOUT, recycle_seconds=_RECYCLE_SECONDS,
                ),
            ),
        )
    successor_played = threading.Event()

    def end_once_the_successor_played() -> None:
        deadline = time.monotonic() + _RUN_SECONDS
        while time.monotonic() < deadline:
            time.sleep(_POLL_SECONDS)
            if _successor_has_played(supervisor, run_dir):
                successor_played.set()
                break
        supervisor._pool.request_shutdown()

    watcher = threading.Thread(target=end_once_the_successor_played, daemon=True)
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    started = time.monotonic()
    watcher.start()

    def collect() -> int:
        # The command's conversion check, inside the run's manifest as the
        # command runs it: refused or recorded before any worker starts.
        check_conversion(args, run_dir)
        supervisor.run()
        return 0

    try:
        code = run_recorded(run_dir, collect, RunCommand("sealed match-outcomes (test)"))
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    watcher.join(timeout=10)
    return {
        "run_dir": run_dir, "root": root, "code": code, "supervisor": supervisor,
        "successor_played": successor_played.is_set(), "rules": rules,
        "seconds": time.monotonic() - started,
    }


def _successor_has_played(supervisor, run_dir: Path) -> bool:
    """A worker was recycled and the newest shard holds an ended game."""
    shards = sorted(iter_shards(run_dir), key=lambda shard: shard.stat().st_ctime)
    if supervisor._pool.stops.recycled < 1 or len(shards) < 2:
        return False
    return bool(ended_games(shards[-1]))


def _entry(run_dir: Path) -> dict:
    document = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert len(document["invocations"]) == 1, document["invocations"]
    return document["invocations"][0]


def _records(run_dir: Path) -> list:
    return [
        record_from_dict(json.loads(line))
        for shard in iter_shards(run_dir) for line in iter_shard_lines(shard)
    ]


def test_a_worker_recycles_at_a_game_end_and_its_successor_continues(recycled_run) -> None:
    assert recycled_run["successor_played"], "no successor finished a game in the window"
    supervisor = recycled_run["supervisor"]
    stops = supervisor._pool.stops
    assert stops.recycled >= 1, stops
    assert stops.stopped == 1, stops
    assert (stops.killed, stops.crashed, stops.ended_otherwise, stops.killed_starting) == (
        0, 0, 0, 0,
    ), stops
    run_id = supervisor.run_id
    log = (recycled_run["run_dir"] / "logs" / f"{run_id}.0.log").read_text(
        encoding="utf-8", errors="replace",
    )
    reasons = [line.group("reason") for line in STOPPED_LINE.finditer(log)]
    assert reasons[:-1] == [f"recycling after {_RECYCLE_SECONDS} s"] * stops.recycled, reasons
    assert reasons[-1].startswith("stop file "), reasons
    games = [int(line.group("games")) for line in STOPPED_LINE.finditer(log)]
    assert all(count >= 1 for count in games), games


def test_both_lifetimes_shards_decode_and_no_game_is_cut(recycled_run) -> None:
    run_dir = recycled_run["run_dir"]
    shards = list(iter_shards(run_dir))
    lifetimes = {_lifetime(shard) for shard in shards}
    assert len(lifetimes) >= 2, [shard.name for shard in shards]
    assert {lifetime.split("-", 1)[0] for lifetime in lifetimes} == {"0"}
    ids: set[str] = set()
    for shard in shards:
        shard_ids = _assert_shard_whole(shard)
        assert not ids & set(shard_ids), "a successor reissued a record id"
        ids |= set(shard_ids)


def _assert_game_end_markers(records: list, played: set[str], shard: Path) -> None:
    markers = [r for r in records if r.subkind is TurnActionSubkind.GAME_END]
    assert len(markers) == len(played), shard.name
    for marker in markers:
        assert marker.mode.value == "patched" and marker.state is not None
        # match-outcomes aims no record of a live game.
        assert marker.aimed_at is None, marker.aimed_at


def _assert_shard_whole(shard: Path) -> list[str]:
    """Every record of the shard decodes, every game has its marker; returns the record ids."""
    records = [record_from_dict(json.loads(line)) for line in iter_shard_lines(shard)]
    assert records, f"{shard.name} is empty"
    assert {record.generation for record in records} == {3}, shard.name
    played = {record.game_id for record in records}
    assert played == ended_games(shard), f"{shard.name}: a game without its marker"
    assert all(record.game_id.split(".")[-2] == _lifetime(shard) for record in records)
    _assert_game_end_markers(records, played, shard)
    shard_ids = [record.record_id for record in records]
    assert len(shard_ids) == len(set(shard_ids))
    return shard_ids


def test_run_json_and_the_logs(recycled_run) -> None:
    run_dir = recycled_run["run_dir"]
    supervisor = recycled_run["supervisor"]
    assert recycled_run["code"] == 0
    document = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert document["status"] == "finished", document.get("outcome")
    entry = _entry(run_dir)
    stops = supervisor._pool.stops
    assert entry["run_id"] == supervisor.run_id
    assert entry["stop_timeout"] == _STOP_TIMEOUT
    assert entry["recycle_seconds"] == _RECYCLE_SECONDS
    assert entry["worker_stops"] == stops.as_dict(survived=0)
    assert entry["worker_survivors"] == []
    assert set(entry["hook_failures"]) == {"connector", "forge"}, entry["hook_failures"]
    assert isinstance(entry["fork_refusals"], dict)
    assert_no_lost_output(entry)
    outcome_rows = (recycled_run["root"] / "match-outcomes.txt").read_text(
        encoding="utf-8",
    ).splitlines()
    assert entry["matches"] == len(outcome_rows) >= 1
    run_id = supervisor.run_id
    progress = (run_dir / "logs" / f"{run_id}.progress.txt").read_text(encoding="utf-8")
    assert entry["matches_played"] == len(progress.splitlines()) >= entry["matches"]
    # Logs under logs/ only; the stop file is gone once the run has ended.
    assert sorted(path.name for path in (run_dir / "logs").iterdir()) == sorted(
        [f"{run_id}.0.log", f"{run_id}.progress.txt"],
    )
    shards = {shard.name for shard in iter_shards(run_dir)}
    assert {path.name for path in run_dir.iterdir()} == shards | {"run.json", "logs"}


def test_a_directed_fork_record_decodes_with_its_rule(recycled_run) -> None:
    rules = set(recycled_run["rules"])
    directed = [record for record in _records(recycled_run["run_dir"]) if record.directed]
    if not directed:
        pytest.skip(
            "the startup check passed over the whole list (asserted when the run "
            "started), but no listed rule's precondition held in these games"
        )
    for record in directed:
        assert record.fork and not record.interventional, record.record_id
        assert record.mirror_of is None, record.record_id
        assert len(record.aimed_at) == 1 and record.aimed_at[0] in rules, record.aimed_at
        assert ForkOperation(record.directed.operation), record.directed


def _cli(*argv: str, timeout: int = 600) -> subprocess.CompletedProcess:
    environment = {**os.environ, "PYTHONPATH": os.pathsep.join(["src", "."])}
    return subprocess.run(
        [sys.executable, "-u", *argv], cwd=_REPOSITORY, env=environment,
        capture_output=True, text=True, timeout=timeout,
    )


def test_rule_counts_reads_the_java_written_shards(recycled_run) -> None:
    run_dir = recycled_run["run_dir"]
    output = recycled_run["root"] / "rule-counts.json"
    result = _cli("-m", "effects", "rule-counts", "--records-dir", str(run_dir),
                  "--output", str(output), "--variant-scripts", "", "--workers", "2")
    assert result.returncode == 0, (result.stdout + result.stderr)[-4000:]
    counts = json.loads(output.read_text(encoding="utf-8"))
    assert counts["generation"] == "gen-3"
    assert counts["cut_games"] == 0, counts["cut_games"]
    assert counts["records"] > 0


def test_validate_corpus_reports_on_the_java_written_shards(recycled_run) -> None:
    """A report: its verdict on a few games is printed, not asserted; what is
    asserted is that it read every line and found every game ended."""
    result = _cli("-m", "effects", "validate-corpus", "--effect-records",
                  str(recycled_run["run_dir"]), "--variant-scripts", "")
    printed = result.stdout + result.stderr
    print(printed)
    assert result.returncode in (0, 1), printed[-4000:]
    assert "[PASS] every shard line decodes" in printed, printed[-4000:]
    # Watched, so never failed by the check itself; every game here ended.
    assert "games end with their game-end marker: 0 of " in printed, printed[-4000:]


#: build-corpus's refusals of a corpus a few games cannot fill: the two
#: validation strata gate 1 and the trainer read.
_TOO_FEW_GAMES = (
    "No game is admitted to the card-disjoint stratum",
    "the game-disjoint validation sample is empty",
    "the card-disjoint validation sample is empty",
)


def test_build_corpus_reads_the_java_written_shards(recycled_run) -> None:
    """A report on a few games: the survey reads every record of every game,
    none of them cut, and the build then either writes the corpus or refuses
    one it could not use, saying why: a tiny run leaves a validation stratum
    empty (no held-out game admitted, or no game-disjoint record sampled),
    which no trainer can validate on, exit 1."""
    run_dir = recycled_run["run_dir"]
    records = _records(run_dir)
    games = {record.game_id for record in records}
    output = recycled_run["root"] / "corpus"
    result = _cli("-m", "effects", "build-corpus", "--records-dir", str(run_dir),
                  "--cards-folder", "output/effects/gen3/converted/cardsfolder/",
                  "--cards-folder", "output/effects/gen3/converted/tokenscripts/",
                  "--vocab-path", str(write_vocabulary(
                      recycled_run["root"] / "vocab-script.txt", "script")),
                  "--keyword-definitions", str(FIXTURE_KEYWORD_DEFINITIONS),
                  "--output", str(output), "--workers", "2")
    printed = result.stdout + result.stderr
    print(printed)
    assert "Traceback" not in printed, printed[-4000:]
    assert f"Surveyed {len(records)} record(s) in {len(games)} game(s)" in printed, (
        printed[-4000:]
    )
    manifest = output / "manifest.json"
    if result.returncode == 0:
        document = json.loads(manifest.read_text(encoding="utf-8"))
        assert document["cut_games"] == 0, document["cut_games"]
    else:
        assert result.returncode == 1, printed[-4000:]
        assert any(reason in printed for reason in _TOO_FEW_GAMES), printed[-4000:]
        assert not manifest.exists()

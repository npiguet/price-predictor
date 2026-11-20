"""The match workers' stop protocol: its ``-D`` properties and printed
lines, the ``--stop-timeout`` and ``--recycle-seconds`` flags, the settings
and how a pool's workers ended."""

from __future__ import annotations

import dataclasses
import os
import re
import subprocess
from collections.abc import Callable
from pathlib import Path

# ── the match worker's stop protocol ────────────────────────────────────
#
# specs/025-ability-effect-model-gen3/contracts/cli.md § Match worker stop
# protocol. The Java side is WorkerStop, WorkerLoop and MatchWorkerMain; the
# literals below are pinned to them by tests/unit/infrastructure/
# test_worker_stop_protocol.py.

#: The ``-D`` property naming the file whose existence asks a match worker to
#: stop at its game's end (``WorkerStop.STOP_FILE``).
STOP_FILE_PROPERTY = "worker.stop.file"
#: The ``-D`` property after which many seconds of its JVM's life a match
#: worker stops itself at a game end (``WorkerStop.RECYCLE_SECONDS``).
RECYCLE_SECONDS_PROPERTY = "worker.recycle.seconds"
#: What a worker that stopped at a game end prints last, before exiting 0
#: (``WorkerLoop.stoppedLine``): its reason, the matches its JVM finished and
#: the games it played to their end (a match it left unfinished included).
STOPPED_LINE = re.compile(
    r"Worker: stopped at a game end \((?P<reason>.*)\); (?P<matches>\d+) matches "
    r"generated in this JVM; (?P<games>\d+) games played; output closed, exiting 0"
)
#: What a worker given a stop prints once Forge is initialised and its shard
#: open, before its first match (``WorkerStop.describe``): a worker whose log
#: lacks it since its spawn is still starting.
STARTUP_LINE = "Graceful stop: at a game end, when "

#: Seconds a pool waits, after asking a worker to stop, before it kills it.
#: A game has no time bound of its own. Unrecorded sealed best-of-one
#: matches, 974,000 of them on the NAS, last 4 s at the median, 116 s at the
#: 99th percentile, 343 s at the 99.9th and 561 s at the 99.99th, and 63
#: lasted 600 s or more. Gen-3 recorded games, 1,946 of them measured on
#: 2026-11-15, last 4 s at the median, 44 s at the 99th percentile and 88 s
#: at the 99.5th, and about one in 400 runs past 5 minutes. The 60 s kill-recycle
#: (``ForgeWorkerPool._kill_oldest_worker``) lets no JVM live much past
#: ``workers × 60`` s, ten to twelve minutes at the usual 10 to 12 workers,
#: so no game longer than that is ever finished. Ten minutes lets nearly
#: every game in progress at a round's end finish. Too short a value costs
#: a cut game per worker it kills: its records stay without their
#: end-of-game ones and without a game-end marker, so ``build-corpus`` drops
#: the game whole.
DEFAULT_STOP_TIMEOUT = 600
MAX_STOP_TIMEOUT = 3600
#: Seconds of its JVM's life after which a worker stops itself at its next
#: game end and the pool starts a fresh one (``worker.recycle.seconds``).
#: Recycling bounds what Forge accumulates late in a JVM's life (heap growth,
#: loops that do not end; ``match_worker_connector.WORKER_XMX``). The
#: pool also kills its longest-running worker every 60 s
#: (``ForgeWorkerPool._kill_oldest_worker``), cutting its game, so a JVM
#: lives about ``workers × 60`` s, six to twelve minutes for 6 to 12 workers:
#: that kill ends a crawling game, which a recycle at a game end never
#: reaches (about one game in 400 takes 8 to 20 minutes, Forge's AI taking
#: seconds per decision on a huge board). Ten minutes matches that lifetime,
#: so the self-recycle rarely comes first; a worker that reaches it finishes
#: its game before it goes. A worker that is still running ``recycle + stop
#: timeout`` after its start is killed.
DEFAULT_RECYCLE_SECONDS = 600
#: The recycle time counts from the JVM's start, before Forge is initialised,
#: and twelve JVMs starting Forge at once take minutes: below five minutes a
#: worker could reach its recycle time before its first match and stop
#: there, every slot failing to start until the run ends with
#: :class:`WorkerStartError`.
MIN_RECYCLE_SECONDS = 300
MAX_RECYCLE_SECONDS = 86400


def stop_timeout(text: str) -> float:
    """``--stop-timeout`` as argparse reads it: 1 to :data:`MAX_STOP_TIMEOUT`
    seconds. Past an hour a round's end could wait an hour on one worker."""
    import argparse

    value = int(text)
    if not 1 <= value <= MAX_STOP_TIMEOUT:
        raise argparse.ArgumentTypeError(
            f"must be between 1 and {MAX_STOP_TIMEOUT} seconds, got {value}"
        )
    return value


def recycle_seconds(text: str) -> int:
    """``--recycle-seconds`` as argparse reads it: :data:`MIN_RECYCLE_SECONDS`
    to :data:`MAX_RECYCLE_SECONDS`."""
    import argparse

    value = int(text)
    if not MIN_RECYCLE_SECONDS <= value <= MAX_RECYCLE_SECONDS:
        raise argparse.ArgumentTypeError(
            f"must be between {MIN_RECYCLE_SECONDS} and {MAX_RECYCLE_SECONDS} "
            f"seconds, got {value}"
        )
    return value


#: ``--stop-timeout``'s help, on every command that has the flag.
STOP_TIMEOUT_HELP = (
    "Seconds a match worker asked to stop (a round's end, the run's end, a "
    "first Ctrl-C) may take to finish its game before it is killed, 1 to "
    f"{MAX_STOP_TIMEOUT} (default: {DEFAULT_STOP_TIMEOUT}). A game has no time "
    "bound: a worker killed mid-game cuts it, and build-corpus drops a cut game "
    "whole. Ctrl-C reaches the supervisor alone: a first asks every worker to "
    "stop at its game's end, a second kills them at once, cutting their games"
)
#: ``--recycle-seconds``' help, on every command that has the flag.
RECYCLE_SECONDS_HELP = (
    "Seconds of its JVM's life after which a match worker stops itself at its "
    "next game end and a fresh one starts, bounding what Forge accumulates in a "
    f"long-lived JVM, {MIN_RECYCLE_SECONDS} to {MAX_RECYCLE_SECONDS} (default: "
    f"{DEFAULT_RECYCLE_SECONDS}); counted from the JVM's start, so it must "
    "outlast a Forge start beside eleven others; a worker still playing "
    "--stop-timeout seconds past it is killed and its game cut. Under best of N "
    "a recycle abandons the match in flight: no outcome row, its finished games "
    "keep their records"
)


def add_worker_stop_flags(parser, *, only_with: str | None = None) -> None:
    """``--stop-timeout`` and ``--recycle-seconds``, on every command whose
    match workers honour the stop protocol. ``only_with`` names the flag
    without which they do not apply; both then default to None, so the
    command can refuse one given without it."""
    unset = only_with is not None
    applies = f"; only with {only_with}" if unset else ""
    parser.add_argument(
        "--stop-timeout", type=stop_timeout,
        default=None if unset else DEFAULT_STOP_TIMEOUT,
        help=STOP_TIMEOUT_HELP + applies,
    )
    parser.add_argument(
        "--recycle-seconds", type=recycle_seconds,
        default=None if unset else DEFAULT_RECYCLE_SECONDS,
        help=RECYCLE_SECONDS_HELP + applies,
    )


@dataclasses.dataclass(frozen=True)
class WorkerStopSettings:
    """How a pool stops and recycles workers that honour the stop protocol."""

    #: Seconds a worker asked to stop has to end its game; then it is killed.
    timeout: float = DEFAULT_STOP_TIMEOUT
    #: Its JVM's age, in seconds, at which a worker stops itself at a game end.
    recycle_seconds: int = DEFAULT_RECYCLE_SECONDS

    def manifest_fields(self) -> dict:
        """The resolved values as ``run.json`` records them."""
        return {"stop_timeout": self.timeout, "recycle_seconds": self.recycle_seconds}


@dataclasses.dataclass(frozen=True)
class GracefulStop:
    """The stop protocol of a pool's workers: where each slot's stop file
    and log are, and the settings. A pool without one kills its workers."""

    stop_file_for: Callable[[int], Path]
    log_path_for: Callable[[int], Path]
    settings: WorkerStopSettings = WorkerStopSettings()

    def system_properties(self, worker_id: int) -> dict[str, str]:
        """The ``-D`` properties that give slot ``worker_id``'s worker the protocol."""
        return {
            STOP_FILE_PROPERTY: str(self.stop_file_for(worker_id)),
            RECYCLE_SECONDS_PROPERTY: str(self.settings.recycle_seconds),
        }


def graceful_popen_options() -> dict:
    """The ``subprocess.Popen`` options that start a worker under the stop
    protocol in a process group of its own, so the console's Ctrl-C reaches
    the supervisor alone and the supervisor's handler decides (a first asks
    every worker to stop at its game's end, a second kills them): on Windows
    ``CREATE_NEW_PROCESS_GROUP``, which also disables Ctrl-C in the group; on
    POSIX a session of its own. The job object (:func:`bind_to_this_process`)
    still ends the workers if the supervisor dies."""
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


@dataclasses.dataclass
class WorkerStops:
    """How a pool's workers ended, by how.

    ``stopped``: asked to stop, ended at a game end with the stop line.
    ``recycled``: ended themselves at their recycle time, with the stop line.
    ``killed``: the pool killed them mid-game (the periodic recycle of the
    longest-running worker, past the stop timeout, past their recycle time
    by more than it, a second Ctrl-C, or, without the protocol, every
    shutdown): a game cut each, under the protocol. ``killed_starting``:
    under the protocol, killed at a stop while still starting Forge (no
    startup line in their log yet): no game cut.
    ``ended_otherwise``: asked, and ended without the stop line (a crash in
    the last game). ``crashed``: ended unasked without the stop line. A
    survivor never ends: see :attr:`ForgeWorkerPool.survivors`.
    """

    stopped: int = 0
    recycled: int = 0
    killed: int = 0
    ended_otherwise: int = 0
    crashed: int = 0
    killed_starting: int = 0

    def add(self, other: WorkerStops) -> None:
        for spec in dataclasses.fields(self):
            setattr(self, spec.name, getattr(self, spec.name) + getattr(other, spec.name))

    def as_dict(self, survived: int = 0) -> dict[str, int]:
        """The counts as ``run.json``'s ``worker_stops`` records them."""
        return {**dataclasses.asdict(self), "survived": survived}

    def render(self) -> str:
        return (
            f"{self.stopped} stopped at a game end, {self.recycled} recycled "
            f"themselves, {self.killed} killed mid-game, {self.killed_starting} "
            f"killed while starting Forge, {self.ended_otherwise} ended "
            f"otherwise when asked, {self.crashed} crashed"
        )

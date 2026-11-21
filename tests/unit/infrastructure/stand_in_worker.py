"""A stand-in for the Java match worker's stop protocol, for pool tests.

Run as ``python stand_in_worker.py -Dkey=value ...``, the properties a
``MatchWorkerMain`` JVM is given. It honours the protocol as the Java worker
does (``WorkerStop``, ``WorkerLoop``, ``MatchWorkerMain``): the recycle time
counts from its start, before "Forge" initialises; the stop file is first
looked at once initialisation is over, then before each match and at each
game end; it prints the startup line and a whole engine's status line
(``tests.unit.worker_status_line``) after initialisation and the stop line
when it stops, and exits 0. Its literals are pinned to the Java sources by
``test_worker_stop_protocol.py``, so it cannot drift from the real worker.

Test-only properties (``standin.*``) shape it:

- ``standin.init.seconds``: how long "Forge" takes to initialise (default 0);
- ``standin.game.seconds``: how long a game takes (default 0.1);
- ``standin.games.per.match``: the games a match takes (default 1); a stop
  asked mid-match leaves it unfinished, counted as no match;
- ``standin.behaviour``: ``honour`` (default); ``ignore`` never looks at a
  stop; ``exit0-without-line`` stops when asked but exits 0 printing nothing;
- ``standin.crash.after.games``: exit 1, as an uncaught failure, after that
  many games;
- ``standin.die.file``: once this file exists, exit 130 at once, as a JVM
  ends that crashed or was killed mid-game (no stop line);
- ``standin.ready.file``: a file it creates once its Ctrl-C handler is in
  place, before "Forge" initialises.

A console Ctrl-C (``CTRL_C_EVENT``, SIGINT) that reaches it ends it at once
with 130, as it ends a JVM (its shutdown hook closes the shard, no stop line);
under the stop protocol the supervisor starts it in a process group of its
own, which no console Ctrl-C reaches.

``sealed.progress.file`` gets a line per finished match, as the worker's
progress writer appends one.
"""

from __future__ import annotations

import contextlib
import os
import signal
import sys
import time
from pathlib import Path

# Run by path: the project root on sys.path, for the status line's spelling.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tests.unit.worker_status_line import status_line  # noqa: E402

#: ``WorkerStop.STOP_FILE`` and ``WorkerStop.RECYCLE_SECONDS``.
STOP_FILE = "worker.stop.file"
RECYCLE_SECONDS = "worker.recycle.seconds"
#: ``MatchWorkerMain.PROGRESS_FILE``.
PROGRESS_FILE = "sealed.progress.file"
#: ``WorkerLoop.STOPPED`` and the rest of ``WorkerLoop.stoppedLine``, with
#: ``MatchWorkerMain``'s progress noun.
STOPPED = "Worker: stopped at a game end"
IN_THIS_JVM = " in this JVM; "
GAMES_TAIL = " games played; output closed, exiting 0"
PROGRESS_NOUN = "matches generated"
#: ``WorkerStop.reason``.
REASON_STOP_FILE = "stop file "
REASON_RECYCLING = "recycling after "
#: ``WorkerStop.describe``.
DESCRIBE = "Graceful stop: at a game end, when "
DESCRIBE_EXISTS = " exists"
DESCRIBE_OR = " or "
DESCRIBE_RUN = "the JVM has run "
#: ``WorkerStop.fromSystemProperties``'s refusal, ``ExitCode.BAD_CONFIG``.
RECYCLE_REFUSAL = "must be >= 1"
BAD_CONFIG = 2


def _properties(argv: list[str]) -> dict[str, str]:
    found = {}
    for arg in argv:
        if arg.startswith("-D") and "=" in arg:
            key, _, value = arg[2:].partition("=")
            found[key] = value
    return found


def _say(text: str, stream=None) -> None:
    stream = stream or sys.stdout
    stream.write(text + "\n")
    stream.flush()


def _append_line(path: Path, line: str) -> None:
    """Append ``line`` under a lock file: Windows emulates append mode with a
    seek, so two workers appending at once can overwrite each other's line."""
    lock = path.with_name(path.name + ".lock")
    handle = None
    give_up = time.monotonic() + 2  # a holder killed mid-append leaves it behind
    while handle is None and time.monotonic() < give_up:
        try:
            handle = os.open(lock, os.O_CREAT | os.O_EXCL)
        except OSError:  # held, or (Windows) still being deleted by its holder
            time.sleep(0.001)
    try:
        with open(path, "a", encoding="utf-8") as out:
            out.write(line)
    finally:
        if handle is not None:
            os.close(handle)
            with contextlib.suppress(OSError):
                os.unlink(lock)


class _Crashed(Exception):
    """The stand-in's uncaught failure: exit 1."""


class _StandIn:
    """One stand-in worker run: its settings, and one method per protocol step."""

    def __init__(self, started: float, props: dict[str, str], recycle: int | None):
        self.started = started
        self.recycle = recycle
        self.stop_file = Path(props[STOP_FILE]) if STOP_FILE in props else None
        self.behaviour = props.get("standin.behaviour", "honour")
        self.game_seconds = float(props.get("standin.game.seconds", "0.1"))
        self.games_per_match = int(props.get("standin.games.per.match", "1"))
        self.crash_after = int(props.get("standin.crash.after.games", "-1"))
        die = props.get("standin.die.file")
        self.die_file = Path(die) if die is not None else None
        progress = props.get(PROGRESS_FILE)
        self.progress = Path(progress) if progress is not None else None
        self.init_seconds = float(props.get("standin.init.seconds", "0"))
        self.completed = 0
        self.games = 0

    def dying(self) -> bool:
        return self.die_file is not None and self.die_file.exists()

    def pause(self, seconds: float) -> None:
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            if self.dying():
                sys.exit(130)
            time.sleep(0.01)

    def reason(self) -> str | None:
        if self.behaviour == "ignore":
            return None
        if self.stop_file is not None and self.stop_file.exists():
            return REASON_STOP_FILE + str(self.stop_file)
        if self.recycle is not None and time.monotonic() - self.started >= self.recycle:
            return REASON_RECYCLING + str(self.recycle) + " s"
        return None

    def announce(self) -> None:
        self.pause(self.init_seconds)  # Forge initialising
        _say(status_line(str(os.getpid())), sys.stderr)
        if self.stop_file is not None or self.recycle is not None:
            _say(
                DESCRIBE
                + ("" if self.stop_file is None else str(self.stop_file) + DESCRIBE_EXISTS
                   + ("" if self.recycle is None else DESCRIBE_OR))
                + ("" if self.recycle is None else DESCRIBE_RUN + str(self.recycle) + " s")
            )

    def finish_match(self) -> str | None:
        self.completed += 1
        if self.progress is not None:
            _append_line(self.progress, f"{os.getpid()}\n")
        if self.completed == self.crash_after:
            _say("Exception in thread \"main\" java.lang.IllegalStateException",
                 sys.stderr)
            raise _Crashed
        return self.reason()

    def play_match(self) -> str | None:
        """Play one match; the reason it stopped, or None when it finished."""
        why = self.reason()  # before each match
        for _ in range(self.games_per_match if why is None else 0):
            self.pause(self.game_seconds)  # one game
            self.games += 1
            why = self.reason()  # at the game's end
            if why is not None:
                return why
        return self.finish_match() if why is None else why

    def stop(self, why: str) -> int:
        if self.behaviour == "exit0-without-line":
            return 0
        _say(f"{STOPPED} ({why}); {self.completed} {PROGRESS_NOUN}{IN_THIS_JVM}"
             f"{self.games}{GAMES_TAIL}")
        return 0

    def run(self) -> int:
        self.announce()
        try:
            while True:
                why = self.play_match()
                if why is not None:
                    return self.stop(why)
        except _Crashed:
            return 1


def _recycle_seconds(props: dict[str, str]) -> int | None:
    if RECYCLE_SECONDS not in props:
        return None
    try:
        return int(props[RECYCLE_SECONDS])
    except ValueError:
        return 0


def main(argv: list[str]) -> int:
    started = time.monotonic()
    signal.signal(signal.SIGINT, lambda *_: os._exit(130))
    props = _properties(argv)
    if "standin.ready.file" in props:
        Path(props["standin.ready.file"]).write_text("ready", encoding="utf-8")
    recycle = _recycle_seconds(props)
    if recycle is not None and recycle < 1:
        _say(f"{RECYCLE_SECONDS}: {RECYCLE_REFUSAL}", sys.stderr)
        return BAD_CONFIG
    return _StandIn(started, props, recycle).run()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

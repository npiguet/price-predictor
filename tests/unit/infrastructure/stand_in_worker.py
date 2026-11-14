"""A stand-in for the Java match worker's stop protocol, for pool tests.

Run as ``python stand_in_worker.py -Dkey=value ...``, the properties a
``MatchWorkerMain`` JVM is given. It honours the protocol as the Java worker
does (``WorkerStop``, ``WorkerLoop``, ``MatchWorkerMain``): the recycle time
counts from its start, before "Forge" initialises; the stop file is first
looked at once initialisation is over, then before each match and at each
game end; it prints the startup line after initialisation and the stop line
when it stops, and exits 0. Its literals are pinned to the Java sources by
``test_worker_stop_protocol.py``, so it cannot drift from the real worker.

Test-only properties (``standin.*``) shape it:

- ``standin.init.seconds``: how long "Forge" takes to initialise (default 0);
- ``standin.game.seconds``: how long a game takes (default 0.1);
- ``standin.behaviour``: ``honour`` (default); ``ignore`` never looks at a
  stop; ``exit0-without-line`` stops when asked but exits 0 printing nothing;
- ``standin.crash.after.games``: exit 1, as an uncaught failure, after that
  many games;
- ``standin.die.file``: once this file exists, exit 130 at once, as a JVM a
  console Ctrl-C reached does (its shutdown hook closes the shard; no stop
  line).

One game is one match (best of one). ``sealed.progress.file`` gets a line per
match, as the worker's progress writer appends one.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

#: ``WorkerStop.STOP_FILE`` and ``WorkerStop.RECYCLE_SECONDS``.
STOP_FILE = "worker.stop.file"
RECYCLE_SECONDS = "worker.recycle.seconds"
#: ``MatchWorkerMain.PROGRESS_FILE``.
PROGRESS_FILE = "sealed.progress.file"
#: ``WorkerLoop.STOPPED`` and the rest of ``WorkerLoop.stoppedLine``, with
#: ``MatchWorkerMain``'s progress noun.
STOPPED = "Worker: stopped at a game end"
STOPPED_TAIL = " in this JVM; output closed, exiting 0"
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
        except FileExistsError:
            time.sleep(0.001)
    try:
        with open(path, "a", encoding="utf-8") as out:
            out.write(line)
    finally:
        if handle is not None:
            os.close(handle)
            os.unlink(lock)


def main(argv: list[str]) -> int:
    started = time.monotonic()
    props = _properties(argv)
    stop_file = Path(props[STOP_FILE]) if STOP_FILE in props else None
    recycle = None
    if RECYCLE_SECONDS in props:
        try:
            recycle = int(props[RECYCLE_SECONDS])
        except ValueError:
            recycle = 0
        if recycle < 1:
            _say(f"{RECYCLE_SECONDS}: {RECYCLE_REFUSAL}", sys.stderr)
            return BAD_CONFIG
    behaviour = props.get("standin.behaviour", "honour")
    game_seconds = float(props.get("standin.game.seconds", "0.1"))
    crash_after = int(props.get("standin.crash.after.games", "-1"))
    die_file = Path(props["standin.die.file"]) if "standin.die.file" in props else None
    progress = Path(props[PROGRESS_FILE]) if PROGRESS_FILE in props else None

    def dying() -> bool:
        return die_file is not None and die_file.exists()

    def pause(seconds: float) -> None:
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            if dying():
                sys.exit(130)
            time.sleep(0.01)

    def reason() -> str | None:
        if behaviour == "ignore":
            return None
        if stop_file is not None and stop_file.exists():
            return REASON_STOP_FILE + str(stop_file)
        if recycle is not None and time.monotonic() - started >= recycle:
            return REASON_RECYCLING + str(recycle) + " s"
        return None

    pause(float(props.get("standin.init.seconds", "0")))  # Forge initialising
    if stop_file is not None or recycle is not None:
        _say(
            DESCRIBE
            + ("" if stop_file is None else str(stop_file) + DESCRIBE_EXISTS
               + ("" if recycle is None else DESCRIBE_OR))
            + ("" if recycle is None else DESCRIBE_RUN + str(recycle) + " s")
        )
    completed = 0
    while True:
        why = reason()  # before each match
        if why is None:
            pause(game_seconds)  # one game, one match
            completed += 1
            if progress is not None:
                _append_line(progress, f"{os.getpid()}\n")
            if completed == crash_after:
                _say("Exception in thread \"main\" java.lang.IllegalStateException", sys.stderr)
                return 1
            why = reason()  # at the game's end
        if why is not None:
            if behaviour == "exit0-without-line":
                return 0
            _say(f"{STOPPED} ({why}); {completed} {PROGRESS_NOUN}{STOPPED_TAIL}")
            return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

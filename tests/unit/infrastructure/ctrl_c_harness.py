"""Sends a real console Ctrl-C to two children, in a console of its own.

Run as ``python ctrl_c_harness.py <result file> <creationflags> <child argv as
JSON>`` in a new console (``CREATE_NEW_CONSOLE``), so the Ctrl-C it generates
reaches no process but its own console's: never the test runner's. It starts
the child twice, once with ``creationflags`` (the flags a supervisor gives a
worker under the stop protocol) and once without, each given
``-Dstandin.ready.file=<path>``, waits until both have written it, sends
``CTRL_C_EVENT`` to every process of its console, ignoring it itself, and
writes to the result file, as JSON, each child's exit code a few seconds on
(``null`` while it still runs). It ends both children before it exits.
"""

from __future__ import annotations

import ctypes
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def _ready(path: Path, child: subprocess.Popen, seconds: float = 30) -> bool:
    deadline = time.monotonic() + seconds
    while not path.exists():
        if child.poll() is not None or time.monotonic() > deadline:
            return False
        time.sleep(0.05)
    return True


def main(argv: list[str]) -> int:
    result, flags, command = Path(argv[0]), int(argv[1]), json.loads(argv[2])
    # Whoever started the test runner may have started it with Ctrl-C
    # disabled, which every descendant inherits: enable it, as a user's
    # console has it, so the plain child shows the signal arrives. The
    # harness itself then ignores it through its own handler.
    ctypes.WinDLL("kernel32").SetConsoleCtrlHandler(None, False)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    children: dict[str, subprocess.Popen] = {}
    report: dict = {}
    try:
        for name, creationflags in (("grouped", flags), ("plain", 0)):
            ready = result.with_name(f"{result.name}.{name}.ready")
            children[name] = subprocess.Popen(
                [*command, f"-Dstandin.ready.file={ready}"], creationflags=creationflags,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            report[f"{name}_ready"] = _ready(ready, children[name])
        os.kill(0, signal.CTRL_C_EVENT)
        deadline = time.monotonic() + 5
        while children["plain"].poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        time.sleep(1)
        for name, child in children.items():
            report[name] = child.poll()
    finally:
        for child in children.values():
            if child.poll() is None:
                child.kill()
            child.wait(timeout=10)
        tmp = result.with_name(result.name + ".tmp")
        tmp.write_text(json.dumps(report), encoding="utf-8")
        os.replace(tmp, result)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

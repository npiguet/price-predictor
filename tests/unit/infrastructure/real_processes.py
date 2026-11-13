"""Real child processes a test starts and owns, for the few tests that must
see a process die: a sleeping child, one with a grandchild, and a liveness
probe by PID that never signals anything.

Every PID these helpers name belongs to a process the calling test started;
:func:`reap` ends a leftover one so no test leaves a process behind.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import time
from pathlib import Path

_SYNCHRONIZE = 0x00100000
_PROCESS_TERMINATE = 0x0001
_QUERY_LIMITED = 0x1000
_WAIT_TIMEOUT = 0x102

SLEEPER = [sys.executable, "-c", "import time; time.sleep(60)"]

#: A child that starts a sleeping grandchild, writes its PID to argv[1], and sleeps.
_WITH_GRANDCHILD = (
    "import subprocess, sys, time\n"
    "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
    "open(sys.argv[1] + '.tmp', 'w').write(str(child.pid))\n"
    "import os; os.replace(sys.argv[1] + '.tmp', sys.argv[1])\n"
    "time.sleep(60)\n"
)


def is_alive(pid: int, *, within: float = 0.0) -> bool:
    """Whether ``pid`` is still running after waiting up to ``within`` seconds
    for it to end. Opens the process for waiting only; sends nothing."""
    if os.name != "nt":
        deadline = time.monotonic() + within
        while True:
            try:
                os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                pass
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return False
            if time.monotonic() >= deadline:
                return True
            time.sleep(0.05)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    handle = kernel32.OpenProcess(_SYNCHRONIZE | _QUERY_LIMITED, False, pid)
    if not handle:
        return False
    try:
        return kernel32.WaitForSingleObject(handle, int(within * 1000)) == _WAIT_TIMEOUT
    finally:
        kernel32.CloseHandle(handle)


def reap(pid: int) -> None:
    """End ``pid``, a process the test started, if it is still running."""
    if not is_alive(pid):
        return
    if os.name != "nt":
        os.kill(pid, 9)
        return
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    handle = kernel32.OpenProcess(_PROCESS_TERMINATE, False, pid)
    if handle:
        kernel32.TerminateProcess(handle, 1)
        kernel32.CloseHandle(handle)


def start_with_grandchild(pid_file: Path, **popen) -> tuple[subprocess.Popen, int]:
    """A child holding a sleeping grandchild; returns both, the grandchild by PID."""
    child = subprocess.Popen([sys.executable, "-c", _WITH_GRANDCHILD, str(pid_file)], **popen)
    deadline = time.monotonic() + 30
    while not pid_file.exists():
        if child.poll() is not None or time.monotonic() > deadline:
            child.kill()
            raise RuntimeError("the child never started its grandchild")
        time.sleep(0.05)
    return child, int(pid_file.read_text(encoding="utf-8"))

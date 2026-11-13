"""Real child processes a test starts and owns, for the few tests that must
see a process die: a sleeping child, one with a grandchild, and a handle on
a process the test did not start directly (a grandchild) through which it is
watched and ended.

A grandchild is known only by the PID its parent reports. The handle is
opened at once, while the grandchild is certainly running (it sleeps for a
minute), and every later wait or termination goes through that handle,
never through the PID again: once the process ends, Windows may hand its PID
to an unrelated process, and the handle still names the one the test owns.
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


class OwnedProcess:
    """A process the test started indirectly, held by a handle opened while
    it was known to be running. Windows only; elsewhere, by PID through
    ``os.kill`` with signal 0, which tests nothing but existence."""

    def __init__(self, pid: int) -> None:
        self.pid = pid
        self._handle = None
        if os.name == "nt":
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            self._handle = kernel32.OpenProcess(
                _SYNCHRONIZE | _PROCESS_TERMINATE | _QUERY_LIMITED, False, pid,
            )
            if not self._handle:
                raise RuntimeError(f"PID {pid} is not running; nothing to own")

    def is_alive(self, *, within: float = 0.0) -> bool:
        """Whether the process is still running after waiting up to ``within``
        seconds for it to end."""
        if self._handle is None:
            deadline = time.monotonic() + within
            while True:
                try:
                    os.kill(self.pid, 0)
                except ProcessLookupError:
                    return False
                if time.monotonic() >= deadline:
                    return True
                time.sleep(0.05)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        return kernel32.WaitForSingleObject(self._handle, int(within * 1000)) == _WAIT_TIMEOUT

    def end(self) -> None:
        """End the process if it still runs, and release the handle."""
        if self._handle is None:
            if self.is_alive():
                os.kill(self.pid, 9)
            return
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        if self.is_alive():
            kernel32.TerminateProcess(self._handle, 1)
        kernel32.CloseHandle(self._handle)
        self._handle = None


def end_tree(owner: subprocess.Popen) -> None:
    """End ``owner`` and every process it started, then reap it.

    By ``owner``'s PID, which is safe only because ``owner`` is a ``Popen``
    this test holds and has not reaped: Windows never hands out the PID of a
    process while a handle to it is open. A tree kill reaches a grandchild
    whose PID the test never learned, or learned and could not own."""
    if owner.poll() is None:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(owner.pid)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
            )
        else:
            try:
                os.killpg(os.getpgid(owner.pid), 9)
            except (ProcessLookupError, PermissionError):
                pass
    owner.kill()
    owner.wait()


def wait_for_file(path: Path, owner: subprocess.Popen, *, seconds: float = 30) -> str:
    """The text ``owner`` writes to ``path``; ``owner`` killed and an error
    raised when it dies or the wait runs out first."""
    deadline = time.monotonic() + seconds
    while not path.exists():
        if owner.poll() is not None or time.monotonic() > deadline:
            end_tree(owner)
            raise RuntimeError(f"{owner.args!r} never wrote {path}")
        time.sleep(0.05)
    return path.read_text(encoding="utf-8")


def start_with_grandchild(pid_file: Path, **popen) -> tuple[subprocess.Popen, OwnedProcess]:
    """A child holding a sleeping grandchild; returns both, the grandchild owned
    through a handle opened while it runs. On any failure here the child's
    whole tree, the grandchild included, is ended before the error
    propagates."""
    child = subprocess.Popen([sys.executable, "-c", _WITH_GRANDCHILD, str(pid_file)], **popen)
    try:
        grandchild = OwnedProcess(int(wait_for_file(pid_file, child)))
    except BaseException:
        end_tree(child)
        raise
    return child, grandchild

"""Building a ``java`` command, running a one-shot forge-connector main and
killing a process tree."""

from __future__ import annotations

import dataclasses
import functools
import os
import platform
import re
import signal
import subprocess
from collections.abc import Mapping

from price_predictor.infrastructure.custom_card_staging import clear_stale_staging
from price_predictor.infrastructure.forge_classpath import build_forge_classpath
from price_predictor.infrastructure.job_objects import bind_to_this_process

#: The opt-in that keeps JDK 24 and later (JEP 498) from printing a
#: deprecation warning to every worker's stderr the first time Guava's
#: ``AbstractFuture`` calls ``sun.misc.Unsafe``. Guava is the version Forge
#: ships with and the calls are still supported, so the warning says nothing an
#: operator can act on. JDK 17 refuses the option outright, so it is passed only
#: to a JVM new enough to warn; ``forge-connector/pom.xml`` does the same for
#: the build.
UNSAFE_MEMORY_ACCESS_OPTION = "--sun-misc-unsafe-memory-access=allow"

#: The first JDK that warns; the option itself exists from JDK 23.
_FIRST_WARNING_JDK = 24

_JAVA_VERSION = re.compile(r'version "(\d+)(?:\.(\d+))?')


def parse_java_feature_version(version_output: str) -> int | None:
    """The feature release ``java -version`` reports, or None if it names none.

    ``1.8.0_x`` is Java 8; from Java 9 on the first number is the release.
    """
    match = _JAVA_VERSION.search(version_output)
    if match is None:
        return None
    major = int(match.group(1))
    if major == 1 and match.group(2) is not None:
        return int(match.group(2))
    return major


@functools.cache
def java_feature_version() -> int | None:
    """The feature release of the ``java`` on PATH, asked once per process.

    None when it cannot be asked or does not answer, which leaves the worker
    command as it is on a JDK that predates the option: a JVM that cannot be
    identified is never handed an option it might refuse.
    """
    try:
        result = subprocess.run(
            ["java", "-version"],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    output = result.stderr
    return parse_java_feature_version(output) if isinstance(output, str) else None


@dataclasses.dataclass(frozen=True)
class JvmOptions:
    """A JVM's own options, which go before its classpath: its system
    properties (each ``-Dkey=value``, in order) and its maximum heap
    (``-Xmx``)."""

    system_properties: Mapping[str, str] | None = None
    xmx: str | None = None

    def flags(self) -> list[str]:
        """The options as ``java`` arguments."""
        flags = [f"-D{key}={value}" for key, value in (self.system_properties or {}).items()]
        if self.xmx:
            flags.append(f"-Xmx{self.xmx}")
        return flags


def build_jvm_command(
    *,
    main_class: str,
    classpath: str,
    options: JvmOptions | None = None,
    main_args: list[str] | None = None,
) -> list[str]:
    """Construct a ``java`` subprocess command list. The first one a process
    builds clears stale variant staging first (:func:`clear_stale_staging`):
    every Forge JVM of every command is built here."""
    clear_stale_staging()
    cmd: list[str] = ["java"]
    version = java_feature_version()
    if version is not None and version >= _FIRST_WARNING_JDK:
        cmd.append(UNSAFE_MEMORY_ACCESS_OPTION)
    cmd.extend((options or JvmOptions()).flags())
    cmd.extend(["-cp", classpath, main_class])
    if main_args:
        cmd.extend(main_args)
    return cmd


class ForgeWorkerTimeout(TimeoutError):
    """A one-shot forge-connector main ran past its timeout and was killed.

    A ``TimeoutError``, so an ``OSError``: a caller that degrades when the JVM
    cannot run at all (the castability consult) degrades on a hung one too.
    """


@dataclasses.dataclass(frozen=True)
class OneShotRun:
    """How :func:`run_forge_worker` runs its JVM.

    Attributes:
        xmx: the JVM's maximum heap; None leaves the JVM's default.
        include_dependency_glob: passed to :func:`build_forge_classpath`.
        timeout: seconds after which the JVM's tree is killed and
            :class:`ForgeWorkerTimeout` raised; None waits for ever.
    """

    xmx: str | None = None
    include_dependency_glob: bool = False
    timeout: float | None = None


def run_forge_worker(
    main_class: str,
    *,
    main_args: list[str] | None = None,
    input_text: str | None = None,
    run: OneShotRun | None = None,
) -> subprocess.CompletedProcess:
    """Run a one-shot forge-connector main class and return the result.

    The JVM runs as ``run`` says (:class:`OneShotRun`). It is bound to this
    process (:func:`bind_to_this_process`), so a hard-killed caller does not
    leave it running. Past ``run.timeout`` seconds its process tree is
    killed and :class:`ForgeWorkerTimeout` raised: a Forge startup can hang,
    and nothing else would end the command.

    Raises FileNotFoundError if ``java`` is not on PATH.
    """
    run = run or OneShotRun()
    timeout = run.timeout
    cmd = build_jvm_command(
        main_class=main_class,
        classpath=build_forge_classpath(
            include_dependency_glob=run.include_dependency_glob,
        ),
        main_args=main_args,
        options=JvmOptions(xmx=run.xmx),
    )
    capture = input_text is not None
    pipe = subprocess.PIPE if capture else None
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=pipe,
            stdout=pipe,
            stderr=pipe,
            text=capture,
            encoding="utf-8" if capture else None,
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "Java not found. Ensure java is on PATH."
        ) from exc
    bind_to_this_process(proc)
    try:
        stdout, stderr = proc.communicate(input=input_text, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        kill_process_tree(proc)
        raise ForgeWorkerTimeout(
            f"{main_class} ran past its {timeout:.0f} s timeout and was killed"
        ) from exc
    except BaseException:
        kill_process_tree(proc)
        raise
    return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)


#: Seconds :func:`kill_process_tree` waits for the tree kill, and again for
#: the fallback kill, before calling the process a survivor.
KILL_WAIT_SECONDS = 10.0


def kill_process_tree(proc: subprocess.Popen, *, timeout: float = KILL_WAIT_SECONDS) -> bool:
    """Kill a subprocess and its entire child tree; True once it is gone.

    On Windows, ``proc.terminate()`` only kills the parent — child JVM
    processes survive as orphans. ``taskkill /F /T`` walks the PID tree
    and forces termination. On POSIX, ``os.killpg`` does the same when
    the process was launched with ``start_new_session=True``.

    A process still running ``timeout`` seconds after the tree kill is killed
    again directly; one that outlives that too is reported by PID and False
    returned, never counted as dead: a surviving worker keeps appending to
    its shard and to the progress file it was started with.
    """
    if proc.poll() is not None:
        return True
    if _tree_killed(proc, timeout):
        return True
    try:
        proc.kill()
    except Exception:  # noqa: BLE001, S110 — the wait below decides
        pass
    if _waited(proc, timeout):
        return True
    print(
        f"Worker PID {proc.pid} survived the kill; it may still be writing its "
        "shard and progress file"
    )
    return False


def _tree_killed(proc: subprocess.Popen, timeout: float) -> bool:
    """Kill ``proc``'s tree; whether it ended within ``timeout`` seconds. A
    tree kill that raises is reported and counts as not ended."""
    try:
        _kill_tree(proc)
        return _waited(proc, timeout)
    except Exception as exc:  # noqa: BLE001 — the direct kill still runs
        print(f"Tree kill of worker PID {proc.pid} failed ({exc}); killing it directly")
        return False


def _kill_tree(proc: subprocess.Popen) -> None:
    """``taskkill /F /T`` on Windows; elsewhere kill the process group, or
    the process alone when its group cannot be signalled."""
    if platform.system() == "Windows":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        proc.kill()


def _waited(proc: subprocess.Popen, timeout: float) -> bool:
    """Whether ``proc`` ended within ``timeout`` seconds."""
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        return False
    except Exception:  # noqa: BLE001 — a wait that cannot be made proves nothing
        return proc.poll() is not None
    return True

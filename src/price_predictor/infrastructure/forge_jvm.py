"""Shared helpers for invoking forge-connector and Forge JVM main classes.

Centralises the four classpath/JAR/JVM-launching duplications that previously
lived in `run_convert` and the three sealed connectors, plus the
Windows-vs-POSIX process-tree termination helper.
"""

from __future__ import annotations

import dataclasses
import functools
import os
import platform
import re
import shutil
import signal
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import IO

CONNECTOR_JAR_NAME = "forge-connector-1.0.0-SNAPSHOT-jar-with-dependencies.jar"

#: Fallback Forge version, used only when the built JAR cannot be found and so
#: no version can be read off disk. `forge-connector/pom.xml` pins the real one
#: in its `forge.version` property — Maven's `systemPath` cannot glob, so that
#: pin has to be updated by hand on a Forge upgrade.
FORGE_VERSION = "2.0.15-SNAPSHOT"

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

#: Classifier suffixes Maven also emits into `target/`; never the module JAR.
_NON_MODULE_JAR_SUFFIXES = ("-sources", "-javadoc", "-tests", "-jar-with-dependencies")


def project_root() -> Path:
    """Locate the price-predictor project root from this file."""
    return Path(__file__).resolve().parents[3]


def forge_dir() -> Path:
    """Return the sibling Forge checkout directory (../forge)."""
    return project_root().parent / "forge"


def forge_module_jar(module: str) -> Path:
    """Return the path to a Forge module JAR (e.g. ``forge-game``).

    The version is read off the built JAR rather than pinned here, so a Forge
    upgrade needs no edit on the Python side. When no JAR is present the path
    is constructed from ``FORGE_VERSION`` instead, leaving the resulting
    classpath entry to fail at the JVM the way it always did.
    """
    target = forge_dir() / module / "target"
    candidates = [
        jar
        for jar in sorted(target.glob(f"{module}-*.jar"))
        if not jar.stem.endswith(_NON_MODULE_JAR_SUFFIXES)
    ]
    if candidates:
        return candidates[-1]
    return target / f"{module}-{FORGE_VERSION}.jar"


def resolve_connector_jar() -> Path:
    """Return the forge-connector fat JAR path. Raises if not built yet."""
    jar = project_root() / "forge-connector" / "target" / CONNECTOR_JAR_NAME
    if not jar.exists():
        raise FileNotFoundError(
            f"Connector JAR not found at {jar}\n"
            "Build it first: cd forge-connector && mvn install -DskipTests"
        )
    return jar


def check_worker_prerequisites(
    *, files: dict[str, Path | None] | None = None,
) -> None:
    """Refuse a worker pool that could not start one worker, before any starts.

    Checks the connector JAR, ``java`` on PATH and each named input file
    (``{"--exclude-cards": path}``; a None path is skipped). Raises
    ``FileNotFoundError`` naming the first missing one. Without this a missing
    JAR or ``java`` surfaced only inside each worker's spawn, which the pool
    retries until :class:`ForgeWorkerPool` gives up.
    """
    resolve_connector_jar()
    if shutil.which("java") is None:
        raise FileNotFoundError("Java not found. Ensure java is on PATH.")
    for flag, path in (files or {}).items():
        if path is not None and not Path(path).is_file():
            raise FileNotFoundError(f"{flag}: no such file {path}")


def build_forge_classpath(
    *,
    include_full_runtime: bool = True,
    include_dependency_glob: bool = False,
) -> str:
    """Build the classpath needed to run a forge-connector main class.

    Args:
        include_full_runtime: include forge-gui and forge-ai in addition to
            forge-game and forge-core. Required by every entry point that
            calls ``ForgeEnvironmentInitializer.initialize()`` (which uses
            ``GuiBase`` / ``GuiHeadless`` / ``FModel``) — i.e. all of them,
            including ``ConvertMain``.
        include_dependency_glob: append ``forge-game/target/dependency/*``
            (the maven-dependency-plugin output) — required by ``ConvertMain``.
    """
    parts: list[str] = [
        str(resolve_connector_jar()),
        str(forge_module_jar("forge-game")),
        str(forge_module_jar("forge-core")),
    ]
    if include_full_runtime:
        parts.append(str(forge_module_jar("forge-gui")))
        parts.append(str(forge_module_jar("forge-ai")))
    if include_dependency_glob:
        parts.append(str(forge_dir() / "forge-game" / "target" / "dependency" / "*"))
    return os.pathsep.join(parts)


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


#: The list a variant staging writes of the scripts it copied into Forge's
#: custom-cards directory (``ForgeEnvironmentInitializer.STAGED_MANIFEST``):
#: not a ``.txt``, because Forge reads every ``.txt`` there as a card.
STAGED_MANIFEST = ".staged-by-effects.list"


def custom_cards_dir() -> Path:
    """Forge's ``USER_CUSTOM_CARDS_DIR`` as a worker resolves it: the worker
    makes ``forge-gui/`` the assets dir, and a headless GUI's user dir is
    ``<assets>/data/``."""
    return forge_dir() / "forge-gui" / "data" / "custom" / "cards"


#: Beside the manifest: the process that staged, as ``{"pid", "started"}``.
#: Written before anything is staged and removed with the staging, so a
#: staging another running process still uses is never taken from under it.
STAGING_OWNER = ".staged-by-effects.owner"


def process_start_time(pid: int | None = None) -> int | None:
    """When process ``pid`` (this process when None) started, in the
    platform's own units, or None when it is not running or cannot be read.
    A PID and its start time together name one process: Windows and Linux
    reuse PIDs, never both."""
    if platform.system() == "Windows":
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.c_void_p] * 4
        kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.c_void_p]
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        own = pid is None
        handle = kernel32.GetCurrentProcess() if own else kernel32.OpenProcess(0x1000, False, pid)
        if not handle:
            return None
        try:
            code = wintypes.DWORD()
            if not own and kernel32.GetExitCodeProcess(handle, ctypes.byref(code)) and (
                code.value != 259  # STILL_ACTIVE
            ):
                return None
            times = [ctypes.c_uint64() for _ in range(4)]
            if not kernel32.GetProcessTimes(handle, *(ctypes.byref(t) for t in times)):
                return None
            return times[0].value
        finally:
            if not own:
                kernel32.CloseHandle(handle)
    try:
        stat = Path(f"/proc/{os.getpid() if pid is None else pid}/stat").read_text()
    except OSError:
        return None
    return int(stat.rsplit(")", 1)[1].split()[19])


def claim_staging(directory: Path | None = None) -> None:
    """Record this process as the owner of the staging in ``directory``
    (default :func:`custom_cards_dir`), before it stages anything."""
    import json

    directory = custom_cards_dir() if directory is None else Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / STAGING_OWNER).write_text(
        json.dumps({"pid": os.getpid(), "started": process_start_time()}), encoding="utf-8",
    )


def staging_owner(directory: Path) -> int | None:
    """The PID of the running process that owns the staging in ``directory``,
    or None when no owner is recorded or it is no longer running: the
    recorded PID must name a process started at the recorded time, so a
    reused PID is no owner."""
    import json

    try:
        owner = json.loads((Path(directory) / STAGING_OWNER).read_text(encoding="utf-8"))
        pid, started = int(owner["pid"]), owner["started"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return pid if started is not None and process_start_time(pid) == started else None


def unstage_custom_cards(directory: Path | None = None) -> int:
    """Delete every file the last staging copied into ``directory`` (default
    :func:`custom_cards_dir`), as its manifest names them, empty the manifest
    and drop the owner record; returns how many were deleted. A card a person
    put there is not in the manifest and stays, and so does any entry that
    resolves outside ``directory`` (reported). Raises ``OSError`` or
    ``ValueError`` on a manifest that cannot be read."""
    directory = custom_cards_dir() if directory is None else Path(directory)
    manifest = directory / STAGED_MANIFEST
    if not manifest.is_file():
        (directory / STAGING_OWNER).unlink(missing_ok=True)
        return 0
    root = directory.resolve()
    removed = 0
    for name in manifest.read_text(encoding="utf-8").splitlines():
        name = name.strip()
        if not name:
            continue
        target = (directory / name).resolve()
        if not target.is_relative_to(root) or target == root:
            print(f"Left {name!r} alone: the staging list names it outside {directory}")
            continue
        if target.is_file():
            target.unlink()
            removed += 1
    manifest.write_text("", encoding="utf-8")
    (directory / STAGING_OWNER).unlink(missing_ok=True)
    return removed


_stale_staging_checked = False


def clear_stale_staging() -> None:
    """Before this process's first JVM, take out the variant scripts a
    ``collect-variants`` left staged when it was killed too hard to unstage
    them, saying so; every Forge start on the machine would load them.

    A staging whose owner (:func:`claim_staging`) still runs is another
    command's live run, or this one's, and stays, with a notice. Once per
    process. A manifest that cannot be read is reported and left; it never
    stops the command."""
    global _stale_staging_checked
    if _stale_staging_checked:
        return
    _stale_staging_checked = True
    directory = custom_cards_dir()
    owner = staging_owner(directory)
    if owner is not None:
        if owner != os.getpid():
            print(
                f"Left the variant scripts staged in {directory} in place: "
                f"collect-variants (PID {owner}) still runs and uses them"
            )
        return
    try:
        removed = unstage_custom_cards(directory)
    except (OSError, ValueError) as exc:
        print(f"Could not clear the variant scripts staged in {directory}: {exc}")
        return
    if removed:
        print(
            f"Removed {removed} variant script(s) a collect-variants run left "
            f"staged in {directory} (it ended without unstaging them)"
        )


def build_jvm_command(
    *,
    main_class: str,
    classpath: str,
    system_properties: dict[str, str] | None = None,
    xmx: str | None = None,
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
    if system_properties:
        for key, value in system_properties.items():
            cmd.append(f"-D{key}={value}")
    if xmx:
        cmd.append(f"-Xmx{xmx}")
    cmd.extend(["-cp", classpath, main_class])
    if main_args:
        cmd.extend(main_args)
    return cmd


class ForgeWorkerTimeout(TimeoutError):
    """A one-shot forge-connector main ran past its timeout and was killed.

    A ``TimeoutError``, so an ``OSError``: a caller that degrades when the JVM
    cannot run at all (the castability consult) degrades on a hung one too.
    """


def run_forge_worker(
    main_class: str,
    *,
    main_args: list[str] | None = None,
    xmx: str | None = None,
    input_text: str | None = None,
    include_dependency_glob: bool = False,
    timeout: float | None = None,
) -> subprocess.CompletedProcess:
    """Run a one-shot forge-connector main class and return the result.

    ``include_dependency_glob`` is passed to :func:`build_forge_classpath`.
    The JVM is bound to this process (:func:`bind_to_this_process`), so a
    hard-killed caller does not leave it running. Past ``timeout`` seconds
    (None waits for ever) its process tree is killed and
    :class:`ForgeWorkerTimeout` raised: a Forge startup can hang, and nothing
    else would end the command.

    Raises FileNotFoundError if ``java`` is not on PATH.
    """
    cmd = build_jvm_command(
        main_class=main_class,
        classpath=build_forge_classpath(
            include_dependency_glob=include_dependency_glob,
        ),
        main_args=main_args,
        xmx=xmx,
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
    try:
        if platform.system() == "Windows":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        else:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()
        if _waited(proc, timeout):
            return True
    except Exception as exc:  # noqa: BLE001 — the direct kill below still runs
        print(f"Tree kill of worker PID {proc.pid} failed ({exc}); killing it directly")
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


def _waited(proc: subprocess.Popen, timeout: float) -> bool:
    """Whether ``proc`` ended within ``timeout`` seconds."""
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        return False
    except Exception:  # noqa: BLE001 — a wait that cannot be made proves nothing
        return proc.poll() is not None
    return True


# ── binding workers to this process ─────────────────────────────────────

_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000

#: The real ``Popen`` class, taken at import: a test that patches
#: ``subprocess.Popen`` still hands back instances of it.
_POPEN = subprocess.Popen

_job_lock = threading.Lock()
#: This process's job handle, made on first use and never closed: Windows
#: closes it when this process ends, however it ends, and then kills every
#: process still in the job.
_job_handle: int | None = None
_job_unavailable = False


def _create_kill_on_close_job() -> int | None:
    """A Windows job object that kills its processes when its last handle
    closes; None when one cannot be made."""
    import ctypes
    from ctypes import wintypes

    class BasicLimits(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class IoCounters(ctypes.Structure):
        _fields_ = [(name, ctypes.c_uint64) for name in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount",
        )]

    class ExtendedLimits(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", BasicLimits),
            ("IoInfo", IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD,
    ]
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        return None
    limits = ExtendedLimits()
    limits.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not kernel32.SetInformationJobObject(
        job, _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
        ctypes.byref(limits), ctypes.sizeof(limits),
    ):
        kernel32.CloseHandle(job)
        return None
    return job


def bind_to_this_process(proc: subprocess.Popen) -> bool:
    """Have Windows kill ``proc``, and every process it starts, when this
    process ends, however it ends; True when it is bound.

    A worker JVM otherwise outlives a supervisor killed hard (Stop-Process,
    ``taskkill`` without ``/T``, a closed console), which runs no ``finally``:
    it keeps appending records under a run whose manifest says the run
    ended. The binding is a job object with kill-on-close, whose one handle
    this process holds until it exits. Only a real ``Popen`` is bound, by its
    own process handle, never by PID: a test's fake or mock process carries a
    made-up PID or handle that could name an unrelated process or handle.
    Elsewhere than on Windows nothing is bound, and a failure to bind is
    reported and leaves the worker running unbound.
    """
    global _job_handle, _job_unavailable
    if not isinstance(proc, _POPEN) or platform.system() != "Windows":
        return False
    handle = getattr(proc, "_handle", None)
    if handle is None:
        return False
    import ctypes
    from ctypes import wintypes

    with _job_lock:
        if _job_handle is None and not _job_unavailable:
            _job_handle = _create_kill_on_close_job()
            if _job_handle is None:
                _job_unavailable = True
                print("Could not create a job object; workers outlive a hard-killed supervisor")
        if _job_handle is None:
            return False
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        if kernel32.AssignProcessToJobObject(_job_handle, int(handle)):
            return True
    print(
        f"Could not bind worker PID {proc.pid} to this process "
        f"(error {ctypes.get_last_error()}); it outlives a hard-killed supervisor"
    )
    return False


class WorkerLogFiles:
    """Persistent, size-capped stdout/stderr destinations for a worker pool.

    Built for the two supervisors whose workers can trip one of this repo's
    five latched effect-record failure reporters (``ApiEvents.
    reportEmitterFailure`` and its four siblings) -- lines that must reach an
    operator, because a Forge upgrade that silently breaks an emitter would
    otherwise look identical to a mechanic sealed play never uses.
    ``MatchWorkerConnector.start`` already threads a ``log_file`` through to
    ``subprocess.Popen`` (mirroring ``evaluation_connector.py``'s own
    ``_spawn``); this class is what supplies one, since neither supervisor's
    ``spawn_worker`` callback otherwise carries state across
    :class:`ForgeWorkerPool`'s restarts -- it is called fresh, with only a
    worker index, every time a slot is (re)spawned.

    One file per worker slot, not per JVM lifetime: the pool restarts a slot
    many times over a long run (hundreds of JVM lifetimes over an
    eight-hour collection run, each recycled after ``--recycle-seconds`` --
    see ``ForgeWorkerPool``), and a reporter line from any one
    of them has to survive to the end of the run, so the file is opened once
    and reused across every restart of that slot rather than reopened per
    JVM -- the same "open once, reuse across restarts" shape
    ``evaluation_connector.py`` uses for its own per-split worker logs.

    Reuse is still bounded, because unlike a fixed number of validation
    splits, a collection run has no natural end: reused forever would let
    ordinary per-match chatter (a "N matches generated" line every ten
    matches, a full stack trace on each non-fatal match error) grow without
    limit across hundreds of restarts. Past ``max_bytes`` the current file is
    rotated to ``.1`` (replacing any previous one) and a fresh file opened,
    which bounds one slot to ``2 * max_bytes`` on disk instead of an
    open-ended append.
    """

    #: Twelve workers at two generations of the default cap is 480 MiB at the
    #: extreme -- a sliver of what one collection run's shards already
    #: occupy ("7,000 games is under a gigabyte" of compressed shards, per
    #: specs/023-ability-effect-model/quickstart.md) -- and the ordinary case
    #: (a handful of restarts' worth of startup banners and progress lines)
    #: uses far less.
    DEFAULT_MAX_BYTES = 20 * 1024 * 1024

    def __init__(
        self,
        directory: Path,
        run_id: str,
        *,
        max_bytes: int = DEFAULT_MAX_BYTES,
    ) -> None:
        self._directory = Path(directory)
        self._run_id = run_id
        self._max_bytes = max_bytes
        self._open: dict[int, IO[bytes]] = {}

    def path_for(self, worker_id: int) -> Path:
        """Where this worker's current log lives, whether or not it exists yet.

        ``{run_id}.{worker}.log`` in the directory this class was given,
        which every caller makes its run directory's ``logs/``
        (``.claude/skills/artifact-layout/SKILL.md``). One component short of
        a shard's own ``{run_id}.{worker}-{lifetime}``: the lifetime token is
        minted inside the JVM, so a file opened before spawning it can never
        carry one.
        """
        return self._directory / f"{self._run_id}.{worker_id}.log"

    def stop_path_for(self, worker_id: int) -> Path:
        """The file whose existence asks this slot's worker to stop at its
        game's end (``-Dworker.stop.file``): ``{run_id}.{worker}.stop`` beside
        the slot's log, under the run directory's ``logs/``."""
        return self._directory / f"{self._run_id}.{worker_id}.stop"

    def graceful_stop(self, settings: WorkerStopSettings) -> GracefulStop:
        """The stop protocol for a pool whose workers log here: each slot's
        stop file beside its log, its stop line read from that log."""
        return GracefulStop(
            stop_file_for=self.stop_path_for, log_path_for=self.path_for, settings=settings,
        )

    def get(self, worker_id: int) -> IO[bytes]:
        """The open handle for ``worker_id``, rotating first past the cap.

        The rotation degrades quietly (its own ``close()`` and rename are
        guarded: a log that stays over the cap is better than a worker
        lost). A log that cannot be opened raises :class:`OSError` naming the
        path, which fails the spawn: ``ForgeWorkerPool._monitor_worker`` counts
        it a failed start and backs off, so a slot never runs a worker whose
        startup cannot be seen.
        """
        path = self.path_for(worker_id)
        handle = self._open.get(worker_id)
        if handle is not None:
            try:
                too_big = path.stat().st_size > self._max_bytes
            except OSError:
                too_big = False
            if not too_big:
                return handle
            try:
                handle.close()
            except OSError:
                # Already unusable either way -- close() is documented to
                # release the underlying resource even when the OS call
                # itself fails -- so this is not worth degrading over; fall
                # through to reopen below.
                pass
            # Dropped before rotation is attempted, not after (final-fix-
            # 3.md item 5): a handle this method has already closed must
            # never be handed back out of ``_open`` on some later call, which
            # is what a raise between here and the reassignment below used to
            # risk -- close() succeeding (or failing; see above) is what
            # makes the handle unusable, not whether the rename that follows
            # does.
            del self._open[worker_id]
            try:
                rotated = path.with_name(path.name + ".1")
                rotated.unlink(missing_ok=True)
                path.rename(rotated)
            except OSError as exc:
                # A logging failure must degrade quietly, never take down
                # collection (item 5): on Windows especially, a rename can
                # lose to a concurrent reader with the file still open. Falls
                # through to reopen ``path`` below either way: rotated or
                # not, it is where this worker's next lines belong.
                print(f"Could not rotate worker log {path}: {exc}")

        try:
            self._directory.mkdir(parents=True, exist_ok=True)
            handle = path.open("ab")
        except OSError as exc:
            # The directory could not be created, or the file could not be
            # opened (permissions, another process holding it). A worker with
            # no log cannot be told from one still starting Forge, so the
            # spawn fails: the pool counts it a failed start and retries
            # after a backoff.
            raise OSError(f"Could not open worker log {path}: {exc}") from exc
        self._open[worker_id] = handle
        return handle

    def close_all(self) -> None:
        """Close every handle opened so far. Safe to call more than once."""
        for handle in self._open.values():
            try:
                handle.close()
            except OSError:
                pass
        self._open.clear()


#: The most workers a pool may run: the machine runs at most twelve CPU-busy
#: processes, and a Forge worker is one.
MAX_WORKERS = 12


def worker_count(text: str) -> int:
    """``--workers`` as argparse reads it: 1 to :data:`MAX_WORKERS`. With
    none the pool never starts a worker and never stops; past the cap the
    machine is oversubscribed."""
    import argparse

    value = int(text)
    if not 1 <= value <= MAX_WORKERS:
        raise argparse.ArgumentTypeError(
            f"must be between 1 and {MAX_WORKERS}, got {value}"
        )
    return value


class WorkerStartError(RuntimeError):
    """Every slot of a :class:`ForgeWorkerPool` failed to start a worker
    ``max_failed_starts`` times in a row: the run cannot collect anything."""


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
#: loops that do not end; ``MatchWorkerConnector.start``'s ``-Xmx``). The
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

#: How much of a slot's log is read back for a worker's stop line: the line
#: is the last its worker prints, with at most the hook-failure report and a
#: few stderr lines after it.
_LOG_TAIL_BYTES = 64 * 1024
#: The most a startup scan reads at once: Forge prints a line per custom card
#: before its startup line, megabytes with a staged variant tree, so the scan
#: reads on from where it stopped, in chunks, and holds none of the log.
_SCAN_CHUNK_BYTES = 1024 * 1024


def _seconds(value: float) -> str:
    """A duration as a report prints it: ``600s``, ``0.5s``."""
    return f"{value:g}s"


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


@dataclasses.dataclass
class _LogMark:
    """Where a slot's log stood when a worker was spawned: its identity and
    size, so what that worker wrote can be read back. The mark also keeps the
    worker's startup scan: how far the log has been searched for the startup
    line (``scanned``, None before the first search), the tail of the last
    chunk (a line split across two reads) and whether the line was seen."""

    path: Path
    identity: int | None
    size: int
    scanned: int | None = None
    carry: bytes = b""
    started: bool = False

    @classmethod
    def of(cls, path: Path) -> _LogMark:
        try:
            stat = os.stat(path)
        except OSError:
            return cls(path, None, 0)
        return cls(path, stat.st_ino, stat.st_size)

    def written_since(self, *, head: int | None = None, tail: int | None = None) -> str:
        """What the log gained since the mark: its first ``head`` bytes, or
        its last ``tail``. A log rotated at the spawn is read from its start."""
        try:
            stat = os.stat(self.path)
            start = self.size if stat.st_ino == self.identity else 0
            with open(self.path, "rb") as handle:
                if tail is not None:
                    start = max(start, stat.st_size - tail)
                handle.seek(start)
                chunk = handle.read(head if head is not None else -1)
        except OSError:
            return ""
        return chunk.decode("utf-8", errors="replace")


def _past_forge_start(mark: _LogMark | None) -> bool:
    """Whether the worker whose log ``mark`` marks has finished starting
    Forge: its startup line is anywhere in what it wrote since its spawn.
    Each call reads only what the log gained since the last; once the line has
    been seen the log is not read again."""
    if mark is None:
        return False
    if not mark.started:
        _scan_for_startup(mark)
    return mark.started


def _scan_for_startup(mark: _LogMark) -> None:
    """Search the log's new bytes for the startup line."""
    try:
        size = os.stat(mark.path).st_size
        if mark.scanned is None or size < mark.scanned:
            mark.scanned = mark.size if _same_file(mark) else 0
            mark.carry = b""
        with open(mark.path, "rb") as handle:
            handle.seek(mark.scanned)
            _read_through(handle, mark)
    except OSError:
        return


def _same_file(mark: _LogMark) -> bool:
    """Whether the log is still the file the mark was taken of."""
    return os.stat(mark.path).st_ino == mark.identity


def _read_through(handle: IO[bytes], mark: _LogMark) -> None:
    """Read ``handle`` to its end a chunk at a time, advancing the mark's
    scan and setting ``started`` on the startup line. The last bytes of a
    chunk carry over, for a line split across two reads."""
    marker = STARTUP_LINE.encode("utf-8")
    while not mark.started and (chunk := handle.read(_SCAN_CHUNK_BYTES)):
        mark.scanned += len(chunk)
        window = mark.carry + chunk
        mark.started = marker in window
        mark.carry = window[-(len(marker) - 1):]


def stop_line_of(text: str) -> re.Match[str] | None:
    """The last stop line in ``text``, or None."""
    found = None
    for found in STOPPED_LINE.finditer(text):  # noqa: B007 — the last one
        pass
    return found


class ForgeWorkerPool:
    """Supervise a pool of Forge JVM workers that append to one output file.

    Spawns ``worker_count`` workers, monitors each in its own thread, restarts
    the ones that end, reports status every ``status_interval`` seconds and
    shuts down cleanly on SIGINT/SIGTERM.

    Workers are recycled, so no JVM lives long (``DEFAULT_RECYCLE_SECONDS``
    says why): every status interval the pool kills the longest-running
    worker (:meth:`_kill_oldest_worker`), and the monitor thread starts a
    fresh one. Beyond that, the pool works in one of two ways:

    - With ``graceful`` (:class:`GracefulStop`, the match workers' stop
      protocol, ``contracts/cli.md`` § Match worker stop protocol), the
      periodic kill cuts the game its worker plays and passes over a worker
      still starting Forge; each worker also stops itself at the first game
      end past its recycle time, printing the stop line and exiting 0, and
      the pool starts a fresh one, unless the run's stop condition already
      holds; a worker ``recycle + timeout`` seconds old, its game not
      ending, is killed and its game cut. To stop the run (its target
      reached, a shutdown, a first Ctrl-C, which reaches the supervisor
      alone: :func:`graceful_popen_options`) the pool creates every running
      worker's stop file and waits for each to finish its game, killing one
      only past the timeout (no periodic kill meanwhile), and kills at once
      one still starting Forge, which has no game to cut; a second Ctrl-C
      kills every worker still finishing its game. ``run`` returns once
      every worker has ended or been killed, since a supervisor that exits
      takes its workers with it (:func:`bind_to_this_process`).
    - Without, the pool kills every worker at the end, mid-game.

    Progress is measured by counting lines in ``output_path`` rather than by
    workers reporting back, so a worker needs no channel to the supervisor.

    A failed start (``spawn_worker`` raising, a worker exiting within
    ``quick_exit_seconds`` of its start without the pool having killed it, or
    one recycling itself before it played a game) is retried after a backoff
    that doubles from ``restart_backoff``; the first retry is immediate, as
    after an ordinary crash. Once every slot has failed ``max_failed_starts``
    starts in a row, ``run`` ends with :class:`WorkerStartError`: a missing
    JAR, a card database that stops every Forge startup or an unreadable
    input file would otherwise be retried forever, collecting nothing.

    Used by ``sealed match-outcomes``, the ``effects`` collectors and ``draft
    play-draft-games`` (whose workers have no stop protocol); the printed
    lines are part of their operator contract and must not drift.
    """

    #: Seconds between status reports, and how often the longest-running
    #: worker is recycled — see
    #: :meth:`_kill_oldest_worker`, which explains what that recycle defends
    #: against. Do not lengthen it to win throughput.
    STATUS_INTERVAL = 60

    #: Consecutive failed starts in every slot that end the run.
    MAX_FAILED_STARTS = 5
    #: The second retry's delay in seconds; each later one doubles it.
    RESTART_BACKOFF = 1.0
    MAX_RESTART_BACKOFF = 30.0
    #: A worker exiting this soon after its start, unkilled, failed to start:
    #: a Forge start takes tens of seconds, and a worker plays until recycled.
    QUICK_EXIT_SECONDS = 120.0
    #: How often a graceful stop looks at the workers it waits for.
    STOP_POLL_SECONDS = 0.05

    def __init__(
        self,
        *,
        worker_count: int,
        spawn_worker: Callable[[int], subprocess.Popen],
        output_path: Path,
        status_interval: int | None = None,
        should_stop: Callable[[int], bool] | None = None,
        on_status: Callable[[], None] | None = None,
        max_failed_starts: int | None = None,
        restart_backoff: float | None = None,
        quick_exit_seconds: float | None = None,
        graceful: GracefulStop | None = None,
    ) -> None:
        """
        Args:
            worker_count: how many workers to keep alive.
            spawn_worker: called with a worker index, returns a started process.
                Under ``graceful`` it must give the worker
                ``graceful.system_properties(index)``.
            output_path: the file workers append to; its line count is progress.
            status_interval: seconds between status lines (default 60).
            should_stop: called with the current line count after each status
                line; returning True ends the run. None means run until signalled.
            on_status: called after each status line (a log scan); what it
                raises is printed and the run goes on.
            max_failed_starts, restart_backoff, quick_exit_seconds: the
                failed-start rule (class docstring); default the class constants.
            graceful: the workers' stop protocol; None kills them instead.
        """
        self._worker_count = worker_count
        self._spawn_worker = spawn_worker
        self._output_path = Path(output_path)
        self._status_interval = (
            self.STATUS_INTERVAL if status_interval is None else status_interval
        )
        self._should_stop = should_stop
        self._on_status = on_status
        self._graceful = graceful
        self._shutdown_event = threading.Event()
        #: Set by a second interrupt: every worker still asked to stop is
        #: killed at once.
        self._hurry = threading.Event()
        self._interrupted = False
        self._processes: list[subprocess.Popen] = []
        self._start_times: dict[subprocess.Popen, float] = {}
        self._processes_lock = threading.Lock()
        self._max_failed_starts = (
            self.MAX_FAILED_STARTS if max_failed_starts is None else max_failed_starts
        )
        self._restart_backoff = (
            self.RESTART_BACKOFF if restart_backoff is None else restart_backoff
        )
        self._quick_exit_seconds = (
            self.QUICK_EXIT_SECONDS if quick_exit_seconds is None else quick_exit_seconds
        )
        #: Processes the pool killed itself: their exit is no failure.
        self._killed: set[subprocess.Popen] = set()
        #: Of those, the ones killed at a stop while still starting Forge.
        self._killed_starting: set[subprocess.Popen] = set()
        self._failed_starts: dict[int, int] = {}
        self._start_failure: str | None = None
        #: Every worker the pool has killed or tried to, in order: each is
        #: killed once, however often the pool terminates.
        self._kill_attempted: list[subprocess.Popen] = []
        #: Under ``graceful``: each running worker's slot and where its log
        #: stood at its spawn, and when each worker was asked to stop and by
        #: when it must have ended.
        self._slots: dict[subprocess.Popen, int] = {}
        self._log_marks: dict[subprocess.Popen, _LogMark] = {}
        self._deadlines: dict[subprocess.Popen, float] = {}
        #: How the workers ended so far.
        self._stops = WorkerStops()

    @property
    def interrupted(self) -> bool:
        """True when the run ended on SIGINT/SIGTERM rather than by itself.

        Callers need this to tell a clean finish from a Ctrl-C, which decide
        different exit codes.
        """
        return self._interrupted

    @property
    def survivors(self) -> list[int]:
        """PIDs of the workers the pool killed, recycled or at shutdown, that
        are still running when asked.

        Each may still append to its shard and to the progress file it was
        started with, so a caller counting either as final should say so. A
        kill that timed out on a process that ended later leaves no survivor.
        """
        with self._processes_lock:
            attempted = list(self._kill_attempted)
        return [proc.pid for proc in attempted if proc.poll() is None]

    @property
    def stops(self) -> WorkerStops:
        """How the run's workers ended, by how, so far."""
        with self._processes_lock:
            return dataclasses.replace(self._stops)

    def run(self) -> None:
        """Start all workers and block until shutdown.

        Returns only once every worker has ended or been killed (a graceful
        stop waits for each to finish its game). Raises
        :class:`WorkerStartError` when every slot failed to start a worker
        ``max_failed_starts`` times in a row.
        """
        self._output_path.parent.mkdir(parents=True, exist_ok=True)

        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

        print(f"Starting {self._worker_count} workers...")

        monitor_threads = []
        try:
            for i in range(self._worker_count):
                t = threading.Thread(target=self._monitor_worker, args=(i,), daemon=True)
                t.start()
                monitor_threads.append(t)

            self._supervisor_loop()
        finally:
            # However the loop ended: the monitor threads are daemons, so a
            # raise here would otherwise leave every worker running. Under
            # the protocol this waits out every graceful stop, before the
            # supervisor can exit and its job object kill the workers.
            self._shutdown_event.set()
            self._terminate_all()
            for t in monitor_threads:
                t.join(timeout=10)
            self._remove_stop_files()

        if self._start_failure is not None:
            raise WorkerStartError(
                f"every one of {self._worker_count} worker slot(s) failed to "
                f"start {self._max_failed_starts} time(s) in a row; last: "
                f"{self._start_failure}"
            )
        print("Done.")

    def _remove_stop_files(self) -> None:
        """Under the protocol, once every worker has ended: delete each slot's
        stop file, which the run's end created and nothing reads any more.
        Kept while a worker the kill could not end survives, which may still
        read its own."""
        if self._graceful is None or self.survivors:
            return
        for slot in range(self._worker_count):
            stop_file = Path(self._graceful.stop_file_for(slot))
            try:
                stop_file.unlink(missing_ok=True)
            except OSError as exc:
                print(f"Could not delete worker {slot}'s stop file {stop_file}: {exc}")

    def output_line_count(self) -> int:
        """Count lines in the output file (= number of completed matches)."""
        if not self._output_path.exists():
            return 0
        try:
            with open(self._output_path, "rb") as f:
                return sum(1 for _ in f)
        except OSError:
            return 0

    def request_shutdown(self) -> None:
        """Ask the pool to stop; the supervisor loop exits at its next tick."""
        self._shutdown_event.set()

    def _monitor_worker(self, worker_id: int) -> None:
        """Monitor one worker, restarting it when it ends until shutdown.

        The restart reuses the same ``worker_id``, because the id names a pool
        slot and not a process. Together with recycling that makes one slot
        hundreds of JVMs over a run, so nothing a worker counts from zero may
        be namespaced by the slot alone.
        """
        first = True
        while not self._shutdown_event.is_set():
            if not first and self._stop_condition_holds():
                # A worker respawned now would only start Forge to be stopped.
                self._shutdown_event.set()
                return
            first = False
            mark = self._prepare_slot(worker_id)
            try:
                proc = self._spawn_worker(worker_id)
            except Exception as exc:
                if self._shutdown_event.is_set():
                    return
                print(f"Monitor error for worker {worker_id}: {exc}")
                self._note_failed_start(worker_id, f"{type(exc).__name__}: {exc}")
                continue

            started = time.monotonic()
            bind_to_this_process(proc)
            with self._processes_lock:
                self._processes.append(proc)
                self._start_times[proc] = started
                self._slots[proc] = worker_id
                if mark is not None:
                    self._log_marks[proc] = mark
            # _terminate_all may have taken its snapshot before the append; a
            # worker spawned while the pool shuts down is killed here, or it
            # outlives the run. It is still starting Forge: no game is cut.
            if self._shutdown_event.is_set():
                self._kill(proc, starting=self._graceful is not None)

            try:
                proc.wait()
            except Exception as exc:
                print(f"Monitor error for worker {worker_id}: {exc}")

            lifetime = time.monotonic() - started
            ending, matches, games = self._ending(proc)
            # A worker that recycled itself played until its recycle time,
            # however short: it failed to start only if it played no game (a
            # best-of-N match spanning the recycle time finishes no match).
            failed = (
                games == 0 if ending == "recycled"
                else self._failed_start(proc, lifetime=lifetime)
            )
            with self._processes_lock:
                if proc in self._processes:
                    self._processes.remove(proc)
                # A killed worker that has exited is no survivor; holding its
                # Popen for the pool's life would hold a process handle per
                # recycle. One still running stays, to be reported.
                if proc.poll() is not None:
                    self._kill_attempted = [
                        attempted for attempted in self._kill_attempted
                        if attempted is not proc
                    ]
                self._start_times.pop(proc, None)
                self._killed.discard(proc)
                self._killed_starting.discard(proc)
                self._slots.pop(proc, None)
                self._log_marks.pop(proc, None)
                self._deadlines.pop(proc, None)
                if proc.poll() is not None:
                    setattr(self._stops, ending, getattr(self._stops, ending) + 1)

            if self._shutdown_event.is_set():
                return

            if ending == "recycled":
                print(
                    f"Worker {worker_id} recycled itself at a game end after "
                    f"{lifetime:.0f}s ({matches} matches, {games} games), restarting..."
                )
            else:
                print(f"Worker {worker_id} exited (code {proc.returncode}), restarting...")
            if failed:
                self._note_failed_start(
                    worker_id,
                    f"worker recycled itself {lifetime:.0f}s after starting without "
                    "playing a game (is --recycle-seconds shorter than a Forge start?)"
                    if ending == "recycled" else
                    f"worker exited (code {proc.returncode}) {lifetime:.0f}s "
                    "after starting",
                )
            else:
                with self._processes_lock:
                    self._failed_starts[worker_id] = 0

    def _prepare_slot(self, worker_id: int) -> _LogMark | None:
        """Under the protocol, before a slot's worker is spawned: delete its
        stop file, which an earlier worker's stop left there and which would
        stop the new one before its first match, and mark where its log
        stands. None without the protocol."""
        if self._graceful is None:
            return None
        stop_file = Path(self._graceful.stop_file_for(worker_id))
        try:
            stop_file.unlink(missing_ok=True)
        except OSError as exc:
            print(f"Could not delete worker {worker_id}'s stop file {stop_file}: {exc}")
        return _LogMark.of(Path(self._graceful.log_path_for(worker_id)))

    def _ending(self, proc: subprocess.Popen) -> tuple[str, int | None, int | None]:
        """How an ended worker ended (a :class:`WorkerStops` field) and, for
        one that printed the stop line, the matches its JVM finished and the
        games it played.

        Exit 0 with the stop line since its spawn is a stop at a game end:
        ``stopped`` when the pool asked, ``recycled`` when it did not. Exit 0
        without the line, or any other code, is no stop: ``killed_starting``
        or ``killed`` when the pool killed it, ``ended_otherwise`` when the
        pool had asked, else ``crashed``."""
        with self._processes_lock:
            asked = proc in self._deadlines
            killed = proc in self._killed
            starting = proc in self._killed_starting
            mark = self._log_marks.get(proc)
        if getattr(proc, "returncode", None) == 0 and mark is not None:
            line = stop_line_of(mark.written_since(tail=_LOG_TAIL_BYTES))
            if line is not None:
                return (("stopped" if asked else "recycled"), int(line.group("matches")),
                        int(line.group("games")))
        if killed:
            return ("killed_starting" if starting else "killed"), None, None
        return ("ended_otherwise" if asked else "crashed"), None, None

    def _failed_start(self, proc: subprocess.Popen, *, lifetime: float) -> bool:
        """Whether a worker that exited after ``lifetime`` seconds failed to
        start: it exited that soon and the pool did not kill it."""
        with self._processes_lock:
            killed = proc in self._killed
        return not killed and lifetime < self._quick_exit_seconds

    def _note_failed_start(self, worker_id: int, reason: str) -> None:
        """Count one failed start of ``worker_id``. Once every slot has failed
        ``max_failed_starts`` in a row the run ends; otherwise wait out the
        backoff."""
        with self._processes_lock:
            failures = self._failed_starts.get(worker_id, 0) + 1
            self._failed_starts[worker_id] = failures
            every_slot_failed = all(
                self._failed_starts.get(slot, 0) >= self._max_failed_starts
                for slot in range(self._worker_count)
            )
        if every_slot_failed:
            self._start_failure = reason
            self._shutdown_event.set()
            return
        if failures > 1:
            delay = min(
                self._restart_backoff * 2 ** (failures - 2), self.MAX_RESTART_BACKOFF,
            )
            self._shutdown_event.wait(timeout=delay)

    def _stop_condition_holds(self) -> bool:
        """Whether the run's stop condition holds now; False without one, or
        when asking it raises (the supervisor loop meets the same raise)."""
        if self._should_stop is None:
            return False
        try:
            return bool(self._should_stop(self.output_line_count()))
        except Exception:  # noqa: BLE001 — the supervisor loop reports it
            return False

    #: How often the stop condition is polled, in seconds. Far shorter than the
    #: status interval: a run with a small target would otherwise overshoot it by
    #: everything the workers finish before the next status line.
    STOP_POLL_INTERVAL = 2

    def _supervisor_loop(self) -> None:
        """Report status every status interval, and at each report, under
        the protocol, kill a worker whose game has not ended long past its
        recycle time, then recycle the oldest worker.

        The stop condition is polled far more often than that, so a small
        ``--n-pairings`` stops near its target instead of at the next status tick.
        """
        start_time = time.monotonic()
        # The output file is append-only and usually holds earlier runs' rows:
        # the status line counts this run's, and the first rate is not inflated
        # by the whole file.
        baseline = self.output_line_count()
        last_count = baseline
        last_time = start_time
        next_status = start_time + self._status_interval

        while not self._shutdown_event.is_set():
            if self._should_stop is not None and self._should_stop(
                self.output_line_count()
            ):
                self._shutdown_event.set()
                break

            timeout = self._sleep_slice(next_status)
            self._shutdown_event.wait(timeout=timeout)
            if self._shutdown_event.is_set():
                break

            now = time.monotonic()
            if now < next_status:
                continue  # a stop-poll slice, not a status tick

            last_count, last_time = self._report_status(
                start_time, now, last_count, last_time, baseline,
            )
            if self._on_status is not None:
                try:
                    self._on_status()
                except Exception as exc:  # noqa: BLE001 — a report never stops a run
                    print(f"Status report failed: {exc}")
            next_status = now + self._status_interval
            if self._should_stop is not None and self._should_stop(last_count):
                self._shutdown_event.set()
                break
            if self._graceful is not None:
                self._kill_overdue_workers()
            self._kill_oldest_worker()

        if self._graceful is None:
            print(f"Shutting down, terminating {self._worker_count} workers...")
        else:
            print(
                f"Shutting down, asking {self._worker_count} workers to stop at "
                "their game's end..."
            )
        self._terminate_all()

    def _sleep_slice(self, next_status: float) -> float:
        """How long to wait next: the stop-poll slice, or the rest of the interval."""
        remaining = next_status - time.monotonic()
        if self._should_stop is None:
            return max(0.0, remaining)
        return max(0.0, min(float(self.STOP_POLL_INTERVAL), remaining))

    def _report_status(
        self,
        start_time: float,
        now: float,
        last_count: int,
        last_time: float,
        baseline: int = 0,
    ) -> tuple[int, float]:
        """Print one status line and return the file's line count and ``now``.

        ``baseline`` is the line count when the run started; the line reports
        the matches completed since, while the returned count stays the whole
        file's, which is what ``should_stop`` is handed.
        """
        elapsed = now - start_time
        interval = now - last_time

        count = self.output_line_count()
        delta = count - last_count
        rate = delta / interval * 60 if interval > 0 else 0.0

        with self._processes_lock:
            alive = sum(1 for p in self._processes if p.poll() is None)

        print(
            f"[{elapsed:.0f}s] {count - baseline} matches completed"
            f" | {rate:.1f} matches/min"
            f" | {alive}/{self._worker_count} workers alive"
        )
        return count, now

    def _kill_oldest_worker(self) -> None:
        """Kill the longest-running worker so the monitor thread restarts it
        fresh: the recycle of every pool, each status interval.

        Why: Forge workers sometimes get stuck in infinite or near-infinite loops,
        holding a slot indefinitely, and about one game in 400 crawls (Forge's
        AI taking seconds per decision on a huge board, the game lasting 8 to
        20 minutes). Recycling the oldest one each status interval bounds how
        long any single match can occupy a worker. Introduced in commit
        eda4505.

        Under the stop protocol the kill cuts the game the worker plays (no
        game-end marker: ``build-corpus`` drops it), the worker counts as
        ``killed``, and three kinds of worker are passed over: one asked to
        stop (a graceful stop is finishing its game), one the pool already
        tried to kill, and one still starting Forge (no startup line in its
        log since its spawn). Killing that last would only start Forge again,
        and twelve JVMs starting at once take minutes: a pool that killed its
        oldest each minute regardless could kill every worker before its
        first game. When every worker is still starting, none is killed; one
        that never finishes starting is killed by :meth:`_kill_overdue_workers`.
        No kill happens once the pool shuts down (:meth:`_stop_all` asks
        every worker to finish its game).

        This is containment, not tuning, and it is one measure with the worker's
        small ``-Xmx`` (see MatchWorkerConnector.start): Forge degrades late in a
        JVM's life, and a small heap plus a hard recycle bounds that instead of
        letting one worker consume a run. So restarts are the normal operating
        condition, not an accident: anything a worker counts from zero must be
        namespaced by the JVM lifetime rather than by the worker slot;
        effect-record ids are, via the lifetime token in RecordShardWriter. Do
        not lengthen or remove this interval to win throughput: what it
        contains costs far more than the restarts do.
        """
        if self._graceful is not None:
            self._kill_oldest_playing_worker()
            return
        with self._processes_lock:
            alive = [(p, t) for p, t in self._start_times.items() if p.poll() is None]
            if not alive:
                return
            oldest_proc, oldest_time = min(alive, key=lambda x: x[1])
            age = time.monotonic() - oldest_time
            self._killed.add(oldest_proc)

        self._kill(oldest_proc)
        print(f"Recycled longest-running worker (PID {oldest_proc.pid}, age {age:.0f}s)")

    def _kill_oldest_playing_worker(self) -> None:
        """:meth:`_kill_oldest_worker` under the stop protocol: the oldest
        worker past its Forge start, neither asked to stop nor already killed."""
        if self._shutdown_event.is_set():
            return
        with self._processes_lock:
            candidates = sorted(
                (
                    (started, proc, self._slots.get(proc, -1), self._log_marks.get(proc))
                    for proc, started in self._start_times.items()
                    if proc.poll() is None and proc not in self._deadlines
                    and not any(proc is attempted for attempted in self._kill_attempted)
                ),
                key=lambda candidate: candidate[0],
            )
        for started, proc, slot, mark in candidates:
            if not _past_forge_start(mark):
                continue
            age = time.monotonic() - started
            self._kill(proc)
            print(
                f"Recycled longest-running worker {slot} (PID {proc.pid}, age "
                f"{age:.0f}s); the game it was playing is cut (no game-end "
                "marker: build-corpus drops it)"
            )
            return

    def _kill_overdue_workers(self) -> None:
        """Kill every worker still running ``recycle + timeout`` seconds after
        its start: its recycle time asked it to stop at its game's end, and
        that game has not ended within the stop timeout. Under the periodic
        recycle (:meth:`_kill_oldest_worker`) a worker rarely lives that long;
        this ends one that does, such as one still starting Forge, which
        that recycle passes over. A game has no time bound, so this cuts a game
        that is merely long: the line says so, the worker counts as
        ``killed`` and its game, without a marker, among the round's cut
        games."""
        settings = self._graceful.settings
        limit = settings.recycle_seconds + settings.timeout
        now = time.monotonic()
        with self._processes_lock:
            overdue = [
                (proc, self._slots.get(proc, -1), now - started)
                for proc, started in self._start_times.items()
                if proc.poll() is None and now - started > limit
                and proc not in self._deadlines
                and not any(proc is attempted for attempted in self._kill_attempted)
            ]
        for proc, slot, age in overdue:
            print(
                f"Killing worker {slot} (PID {proc.pid}), {age:.0f}s old: its game "
                f"has not ended {_seconds(settings.timeout)} past its "
                f"{_seconds(settings.recycle_seconds)} recycle time; the game it "
                "was playing is cut (no game-end marker: build-corpus drops it)"
            )
            self._kill(proc)

    def _terminate_all(self) -> None:
        """End every running worker: without the protocol, kill it; with it,
        ask it to stop at its game's end and wait (:meth:`_stop_all`)."""
        with self._processes_lock:
            procs = list(self._processes)

        if self._graceful is None:
            for proc in procs:
                self._kill(proc)
            return
        self._stop_all(procs)

    def _stop_all(self, procs: list[subprocess.Popen]) -> None:
        """Ask every running worker of ``procs`` to stop at its game's end,
        then wait until each has ended, killing one ``timeout`` seconds after
        the request. A worker whose log shows it still starting Forge (no
        startup line since its spawn) is killed at once, counted as
        ``killed_starting``: its stop file is created first, so one that
        prints its startup line meanwhile stops before its first match, and
        none has a game to cut. One that has already ended is not waited
        for; one asked before keeps its deadline, so a second call waits out
        no timeout twice; a second interrupt kills at once."""
        settings = self._graceful.settings
        now = time.monotonic()
        asked: list[tuple[subprocess.Popen, int]] = []
        with self._processes_lock:
            for proc in procs:
                if proc.poll() is not None or proc in self._deadlines:
                    continue
                self._deadlines[proc] = now + settings.timeout
                asked.append((proc, self._slots.get(proc, -1)))
        for proc, slot in asked:
            stop_file = Path(self._graceful.stop_file_for(slot))
            try:
                stop_file.parent.mkdir(parents=True, exist_ok=True)
                stop_file.touch()
            except OSError as exc:
                print(f"Could not ask worker PID {proc.pid} to stop ({stop_file}: {exc}); "
                      "killing it")
                self._kill(proc)
        starting = []
        for proc, slot in asked:
            with self._processes_lock:
                mark = self._log_marks.get(proc)
            if not _past_forge_start(mark):
                starting.append(proc)
                print(f"Killing worker {slot} (PID {proc.pid}): still starting Forge, "
                      "no game to cut")
                self._kill(proc, starting=True)
        if len(asked) > len(starting):
            print(
                f"Asked {len(asked) - len(starting)} worker(s) to stop at their game's "
                f"end; each is killed {_seconds(settings.timeout)} after the request"
            )
        while True:
            with self._processes_lock:
                waiting = [
                    (proc, deadline) for proc, deadline in self._deadlines.items()
                    if proc.poll() is None
                    and not any(proc is attempted for attempted in self._kill_attempted)
                ]
            if not waiting:
                return
            now = time.monotonic()
            for proc, deadline in waiting:
                if self._hurry.is_set():
                    print(f"Killing worker PID {proc.pid}: interrupted again")
                    self._kill(proc)
                elif now >= deadline:
                    print(
                        f"Killing worker PID {proc.pid}: it did not stop within "
                        f"{_seconds(settings.timeout)} of the request; its game is cut"
                    )
                    self._kill(proc)
            self._hurry.wait(timeout=self.STOP_POLL_SECONDS)

    def _kill(self, proc: subprocess.Popen, *, starting: bool = False) -> None:
        """Kill one worker's tree, once: a second call for the same worker
        (the loop's shutdown, then ``run``'s ``finally``) does nothing, so a
        survivor's timed-out kill is not waited out twice. The single kill
        path of the pool. ``starting``: a worker still starting Forge, whose
        kill cuts no game (``killed_starting``)."""
        with self._processes_lock:
            if any(proc is attempted for attempted in self._kill_attempted):
                return
            self._kill_attempted.append(proc)
            self._killed.add(proc)
            if starting:
                self._killed_starting.add(proc)
        kill_process_tree(proc)

    def _handle_signal(self, signum, frame) -> None:
        """Signal handler: set shutdown event to stop workers. A signal while
        the pool already shuts down kills every worker still finishing its
        game. Under the protocol the workers run in process groups of their
        own (:func:`graceful_popen_options`), so a console Ctrl-C reaches
        this handler alone and the workers end only as it decides."""
        if self._shutdown_event.is_set():
            self._hurry.set()
        self._interrupted = True
        self._shutdown_event.set()

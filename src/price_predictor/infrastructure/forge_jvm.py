"""Shared helpers for invoking forge-connector and Forge JVM main classes.

Centralises the four classpath/JAR/JVM-launching duplications that previously
lived in `run_convert` and the three sealed connectors, plus the
Windows-vs-POSIX process-tree termination helper.
"""

from __future__ import annotations

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
    many times over a long run (roughly 530 JVM lifetimes across 6 slots
    over an eight-hour collection run -- see
    ``ForgeWorkerPool._kill_oldest_worker``), and a reporter line from any one
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

    def get(self, worker_id: int) -> IO[bytes]:
        """The open handle for ``worker_id``, rotating first past the cap.

        Never raises (final-fix-3.md item 5, widened final-fix-4.md item 6):
        every filesystem call in here -- the rotation's own ``close()``, the
        rename, ``mkdir`` and the final ``open`` -- is guarded, because any of
        them raising uncaught used to propagate out of ``spawn_worker`` into
        ``ForgeWorkerPool._monitor_worker``'s bare ``except Exception``, which
        respawns immediately with no backoff: a tight respawn/print loop
        instead of a log that degraded. ``open("ab")`` is at least as likely
        to fail as the rename on Windows (a directory permission problem, or
        another process holding the path), so it needed the same guard, not
        just the rename.
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
            # opened (permissions, another process holding it, a path that
            # briefly disappeared under a concurrent rotation elsewhere).
            # Degrading to a destination that always exists and is always
            # writable -- matching the DEVNULL every worker's stdio wrote to
            # before F4 gave it a real one -- is what keeps this from taking
            # the whole worker down; a worker whose log is lost is still a
            # worker collecting real records, and that is the failure this
            # class exists to avoid trading away.
            print(f"Could not open worker log {path}: {exc}; "
                  f"this worker's log is discarded for this handle")
            handle = open(os.devnull, "ab")  # noqa: SIM115 - lifetime is _open's
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


class ForgeWorkerPool:
    """Supervise a pool of Forge JVM workers that append to one output file.

    Spawns ``worker_count`` workers, monitors each in its own thread, restarts
    the ones that die, reports status every ``status_interval`` seconds, recycles
    the longest-running worker on each of those ticks, and shuts down cleanly on
    SIGINT/SIGTERM.

    Recycling is a liveness measure, not a throughput one: Forge workers
    sometimes hang in near-infinite loops, and killing the oldest each tick
    bounds how long one can occupy a slot.

    Progress is measured by counting lines in ``output_path`` rather than by
    workers reporting back, so a worker needs no channel to the supervisor.

    A failed start (``spawn_worker`` raising, or a worker exiting within
    ``quick_exit_seconds`` of its start without the pool having killed it) is
    retried after a backoff that doubles from ``restart_backoff``; the first
    retry is immediate, as after an ordinary crash. Once every slot has failed
    ``max_failed_starts`` starts in a row, ``run`` ends with
    :class:`WorkerStartError`: a missing JAR, a card database that stops every
    Forge startup or an unreadable input file would otherwise be retried
    forever, collecting nothing.

    Used by ``sealed match-outcomes`` and ``draft play-draft-games``; the printed
    lines are part of both commands' operator contract and must not drift.
    """

    #: Seconds between status reports, and therefore how often the
    #: longest-running worker is recycled — see :meth:`_kill_oldest_worker`,
    #: which explains what that recycle defends against. Do not lengthen it to
    #: win throughput.
    STATUS_INTERVAL = 60

    #: Consecutive failed starts in every slot that end the run.
    MAX_FAILED_STARTS = 5
    #: The second retry's delay in seconds; each later one doubles it.
    RESTART_BACKOFF = 1.0
    MAX_RESTART_BACKOFF = 30.0
    #: A worker exiting this soon after its start, unkilled, failed to start:
    #: a Forge start takes tens of seconds, and a worker plays until recycled.
    QUICK_EXIT_SECONDS = 120.0

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
    ) -> None:
        """
        Args:
            worker_count: how many workers to keep alive.
            spawn_worker: called with a worker index, returns a started process.
            output_path: the file workers append to; its line count is progress.
            status_interval: seconds between status lines (default 60).
            should_stop: called with the current line count after each status
                line; returning True ends the run. None means run until signalled.
            on_status: called after each status line (a log scan); what it
                raises is printed and the run goes on.
            max_failed_starts, restart_backoff, quick_exit_seconds: the
                failed-start rule (class docstring); default the class constants.
        """
        self._worker_count = worker_count
        self._spawn_worker = spawn_worker
        self._output_path = Path(output_path)
        self._status_interval = (
            self.STATUS_INTERVAL if status_interval is None else status_interval
        )
        self._should_stop = should_stop
        self._on_status = on_status
        self._shutdown_event = threading.Event()
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
        #: Processes the pool killed itself (recycled): their exit is no failure.
        self._killed: set[subprocess.Popen] = set()
        self._failed_starts: dict[int, int] = {}
        self._start_failure: str | None = None
        #: Every worker the pool has killed or tried to, in order: each is
        #: killed once, however often the pool terminates.
        self._kill_attempted: list[subprocess.Popen] = []

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

    def run(self) -> None:
        """Start all workers and block until shutdown.

        Raises :class:`WorkerStartError` when every slot failed to start a
        worker ``max_failed_starts`` times in a row.
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
            # raise here would otherwise leave every worker running.
            self._shutdown_event.set()
            self._terminate_all()
            for t in monitor_threads:
                t.join(timeout=10)

        if self._start_failure is not None:
            raise WorkerStartError(
                f"every one of {self._worker_count} worker slot(s) failed to "
                f"start {self._max_failed_starts} time(s) in a row; last: "
                f"{self._start_failure}"
            )
        print("Done.")

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
        """Monitor one worker, restarting it on crash until shutdown.

        The restart reuses the same ``worker_id``, because the id names a pool
        slot and not a process. Together with the timed recycle that makes one
        slot hundreds of JVMs over a run, so nothing a worker counts from zero
        may be namespaced by the slot alone.
        """
        while not self._shutdown_event.is_set():
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
            # _terminate_all may have taken its snapshot before the append; a
            # worker spawned while the pool shuts down is killed here, or it
            # outlives the run.
            if self._shutdown_event.is_set():
                self._kill(proc)

            try:
                proc.wait()
            except Exception as exc:
                print(f"Monitor error for worker {worker_id}: {exc}")

            lifetime = time.monotonic() - started
            failed = self._failed_start(proc, lifetime=lifetime)
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

            if self._shutdown_event.is_set():
                return

            print(f"Worker {worker_id} exited (code {proc.returncode}), restarting...")
            if failed:
                self._note_failed_start(
                    worker_id,
                    f"worker exited (code {proc.returncode}) {lifetime:.0f}s "
                    "after starting",
                )
            else:
                with self._processes_lock:
                    self._failed_starts[worker_id] = 0

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

    #: How often the stop condition is polled, in seconds. Far shorter than the
    #: status interval: a run with a small target would otherwise overshoot it by
    #: everything the workers finish before the next status line.
    STOP_POLL_INTERVAL = 2

    def _supervisor_loop(self) -> None:
        """Report status and recycle the oldest worker every status interval.

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
            self._kill_oldest_worker()

        self._terminate_all()
        print(f"Shutting down, terminating {self._worker_count} workers...")

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
        """Kill the longest-running worker so the monitor thread restarts it fresh.

        Why: Forge workers sometimes get stuck in infinite or near-infinite loops,
        holding a slot indefinitely. Recycling the oldest one each status interval
        bounds how long any single match can occupy a worker. Introduced in commit
        eda4505.

        This is containment, not tuning, and it is one measure with the worker's
        small ``-Xmx`` (see MatchWorkerConnector.start): Forge degrades late in a
        JVM's life, and a small heap plus a hard recycle bounds that instead of
        letting one worker consume a run. So restarts are the normal operating
        condition, not an accident — an eight-hour collection run is roughly 530
        JVM lifetimes across six slots, some of them killed mid-game. Anything a
        worker counts from zero must therefore be namespaced by the JVM lifetime
        rather than by the worker slot; effect-record ids are, via the lifetime
        token in RecordShardWriter. Do not lengthen or remove this interval to
        win throughput: what it contains costs far more than the restarts do.
        """
        with self._processes_lock:
            alive = [(p, t) for p, t in self._start_times.items() if p.poll() is None]
            if not alive:
                return
            oldest_proc, oldest_time = min(alive, key=lambda x: x[1])
            age = time.monotonic() - oldest_time
            self._killed.add(oldest_proc)

        self._kill(oldest_proc)
        print(f"Recycled longest-running worker (PID {oldest_proc.pid}, age {age:.0f}s)")

    def _terminate_all(self) -> None:
        """Terminate all running worker processes."""
        with self._processes_lock:
            procs = list(self._processes)

        for proc in procs:
            self._kill(proc)

    def _kill(self, proc: subprocess.Popen) -> None:
        """Kill one worker's tree, once: a second call for the same worker
        (the loop's shutdown, then ``run``'s ``finally``) does nothing, so a
        survivor's timed-out kill is not waited out twice."""
        with self._processes_lock:
            if any(proc is attempted for attempted in self._kill_attempted):
                return
            self._kill_attempted.append(proc)
        kill_process_tree(proc)

    def _handle_signal(self, signum, frame) -> None:
        """Signal handler: set shutdown event to stop workers."""
        self._interrupted = True
        self._shutdown_event.set()

"""Binding worker JVMs to this process with a Windows job object, so a
hard-killed supervisor takes its JVMs with it."""

from __future__ import annotations

import platform
import subprocess
import threading

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
    if not isinstance(proc, _POPEN) or platform.system() != "Windows":
        return False
    handle = getattr(proc, "_handle", None)
    if handle is None:
        return False
    import ctypes

    with _job_lock:
        job = _this_process_job()
        if job is None:
            return False
        if _assign_to_job(job, handle):
            return True
    print(
        f"Could not bind worker PID {proc.pid} to this process "
        f"(error {ctypes.get_last_error()}); it outlives a hard-killed supervisor"
    )
    return False


def _this_process_job() -> int | None:
    """Under ``_job_lock``: this process's job handle, made on first use;
    None once making it has failed."""
    global _job_handle, _job_unavailable
    if _job_handle is None and not _job_unavailable:
        _job_handle = _create_kill_on_close_job()
        if _job_handle is None:
            _job_unavailable = True
            print("Could not create a job object; workers outlive a hard-killed supervisor")
    return _job_handle


def _assign_to_job(job: int, handle) -> bool:
    """Whether Windows put the process ``handle`` names into ``job``."""
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    return bool(kernel32.AssignProcessToJobObject(job, int(handle)))

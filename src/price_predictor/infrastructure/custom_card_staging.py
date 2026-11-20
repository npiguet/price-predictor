"""The variant scripts a ``collect-variants`` stages in Forge's custom-cards
directory: who owns the staging, taking it back out, and clearing a
staging a killed run left behind."""

from __future__ import annotations

import os
import platform
from pathlib import Path

from price_predictor.infrastructure.forge_classpath import forge_dir

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

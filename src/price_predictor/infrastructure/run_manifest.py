"""The run manifest: ``run.json``, the provenance record of one run directory.

Every command that writes files keeps one in its run directory — the
command's output directory argument, or the directory holding its output
file argument — as ``.claude/skills/artifact-layout/SKILL.md`` lays it out:
written at the command's start with ``status: "running"``, completed at its
end with ``finished`` or ``failed``, and on a resumed or restarted command
(a ``run.json`` already there) extended with the new command line rather
than replaced. ``purpose``, ``continues``, ``superseded_by`` and the
operator's statuses (``keep``, ``scratch``, ``superseded``) belong to the
operator: a command never writes them, and closing a run re-reads the file
so an edit made while it ran survives. A command that fails having written
nothing — a flag refused at startup, a missing input — leaves no trace: the
manifest goes, and so do the directories made for it.

The manifest is bookkeeping, never a reason to stop: every failure to read
or write it is logged as a warning and the command goes on. Writes go to a
temporary file renamed over ``run.json``, so a reader never sees half of one.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from collections.abc import Callable, Iterable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MANIFEST_NAME = "run.json"
#: The statuses a command writes; any other was set by the operator.
COMMAND_STATUSES = frozenset({"running", "finished", "failed"})
#: The exit code a command returns when Ctrl-C stopped it.
INTERRUPTED_EXIT = 130
_GIT_TIMEOUT_SECONDS = 60
_SUMMARY_LIMIT = 300

PathLike = str | os.PathLike[str]


def repository_root() -> Path:
    """The checkout this package runs from, else the working directory."""
    candidate = Path(__file__).resolve().parents[3]
    return candidate if (candidate / ".git").exists() else Path.cwd()


def _git(repository: Path, *args: str) -> str | None:
    """``git`` output in ``repository``, or None when git fails or is absent."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repository), *args], capture_output=True, text=True,
            timeout=_GIT_TIMEOUT_SECONDS, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def read_commits(repo_root: Path, forge_dir: Path | None = None) -> dict[str, Any]:
    """``{"repo", "forge", "dirty"}``: each checkout's short HEAD, ``forge``
    None when ``../forge`` is no checkout, and ``dirty`` true when either
    working tree has uncommitted changes to tracked files."""
    forge_dir = Path(repo_root).parent / "forge" if forge_dir is None else Path(forge_dir)
    checkouts = [Path(repo_root)] + ([forge_dir] if (forge_dir / ".git").exists() else [])
    shas = [_git(checkout, "rev-parse", "--short", "HEAD") for checkout in checkouts]
    dirty = any(
        bool(_git(checkout, "status", "--porcelain", "--untracked-files=no"))
        for checkout, sha in zip(checkouts, shas, strict=True) if sha is not None
    )
    return {"repo": shas[0], "forge": shas[1] if len(shas) > 1 else None, "dirty": dirty}


def _quote(argument: str) -> str:
    return f'"{argument}"' if not argument or any(c.isspace() for c in argument) else argument


def command_line(
    argv: Sequence[str] | None = None, *, module: str | None = None,
    script: PathLike | None = None, repo_root: Path | None = None,
) -> str:
    """The command as typed: ``python -m <module> …`` or ``python <script> …``,
    the script relative to the repository root, ``argv`` the arguments after it."""
    argv = sys.argv[1:] if argv is None else argv
    if module is not None:
        program = f"-m {module}"
    else:
        program = _repo_relative(
            Path(script if script is not None else sys.argv[0]),
            repo_root or repository_root(), Path.cwd(),
        )
    return " ".join(["python", program, *(_quote(str(arg)) for arg in argv)])


def _repo_relative(path: Path, repo_root: Path, cwd: Path) -> str:
    """``path`` relative to the repository root where it lies inside it, else absolute."""
    resolved = (path if path.is_absolute() else cwd / path).resolve()
    try:
        return resolved.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _summary(error: BaseException) -> str:
    """``Type: first line of the message``, cut to a readable length."""
    message = str(error).strip().splitlines()
    text = f"{type(error).__name__}: {message[0]}" if message else type(error).__name__
    return text[:_SUMMARY_LIMIT]


def _outcome(exit_code: int, error: BaseException | None, message: str | None) -> str:
    if isinstance(error, KeyboardInterrupt):
        return "interrupted"
    if error is not None:
        return f"exit {exit_code}: {_summary(error)}"
    if message:
        return f"exit {exit_code}: {message.strip().splitlines()[0][:_SUMMARY_LIMIT]}"
    if exit_code == INTERRUPTED_EXIT:
        return f"exit {exit_code}: interrupted"
    return f"exit {exit_code}"


class RunManifest:
    """The ``run.json`` of one run directory, opened at a command's start and
    closed at its end. Neither call raises."""

    def __init__(
        self, run_dir: PathLike, *, repo_root: Path | None = None,
        forge_dir: Path | None = None, cwd: Path | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._repo_root = Path(repo_root) if repo_root is not None else repository_root()
        self._cwd = Path(cwd) if cwd is not None else Path.cwd()
        self._forge_dir = forge_dir
        self._clock = clock or (lambda: datetime.now().astimezone())
        run_dir = Path(run_dir)
        self.run_dir = run_dir if run_dir.is_absolute() else self._cwd / run_dir
        self.path = self.run_dir / MANIFEST_NAME
        self._document: dict[str, Any] | None = None
        #: The directories made for the manifest, deepest first.
        self._created: list[Path] = []

    def start(
        self, *, command: str, inputs: Iterable[PathLike | None] = (), create: bool = True,
    ) -> None:
        """Write the manifest, or extend the one already there (a resume).

        ``create`` false leaves a run directory that does not exist yet
        without a manifest: the command's output is a tree it expects to find.
        """
        try:
            self._document = self._start(command, inputs, create)
        except Exception as error:  # noqa: BLE001 — bookkeeping never stops a run
            logger.warning("%s not written: %s", self._label(), error)
            self._document = None

    def finish(
        self, *, exit_code: int, error: BaseException | None = None, message: str | None = None,
    ) -> None:
        """Close the run: ``finished`` on exit 0, else ``failed``, with how it
        ended — the exit code, and the exception's or ``message``'s first line."""
        if self._document is None:
            return
        ok = exit_code == 0 and error is None
        try:
            if not ok and self._wrote_nothing():
                self._remove()
                return
            document = self._read() or self._document
            if document.get("status") in COMMAND_STATUSES or "status" not in document:
                document["status"] = "finished" if ok else "failed"
            document["ended"] = self._now()
            document["outcome"] = _outcome(exit_code, error, message)
            self._write(document)
        except Exception as failure:  # noqa: BLE001 — bookkeeping never stops a run
            logger.warning("%s not completed: %s", self._label(), failure)

    # ── internals ───────────────────────────────────────────────────────

    def _start(
        self, command: str, inputs: Iterable[PathLike | None], create: bool,
    ) -> dict[str, Any] | None:
        if self.run_dir.resolve() == self._repo_root.resolve():
            logger.warning(
                "%s not written: the run directory is the repository root", self._label(),
            )
            return None
        if not self.run_dir.is_dir() and not create:
            return None
        self._created = [
            directory for directory in (self.run_dir, *self.run_dir.parents)
            if not directory.exists()
        ]
        self.run_dir.mkdir(parents=True, exist_ok=True)
        listed = self._relative_all(inputs)
        if self.path.exists():
            document = self._read()
            if document is None:
                logger.warning(
                    "%s is unreadable; left as it is and not updated by this run",
                    self._label(),
                )
                return None
            document.setdefault("commands", []).append(command)
            document["status"] = "running"
            known = document.setdefault("inputs", [])
            known.extend(path for path in listed if path not in known)
        else:
            document = {
                "run": self._relative(self.run_dir),
                "purpose": None,
                "status": "running",
                "superseded_by": None,
                "commands": [command],
                "commits": self._commits(),
                "inputs": listed,
                "continues": [],
                "started": self._now(),
                "ended": None,
                "outcome": None,
            }
        self._write(document)
        return document

    def _wrote_nothing(self) -> bool:
        """Whether this run made the run directory and put nothing but the manifest in it."""
        return bool(self._created) and all(
            entry.name == MANIFEST_NAME for entry in self.run_dir.iterdir()
        )

    def _remove(self) -> None:
        """Take back the manifest and the directories made for it."""
        self.path.unlink(missing_ok=True)
        for directory in self._created:
            try:
                directory.rmdir()
            except OSError:
                break

    def _commits(self) -> dict[str, Any]:
        try:
            return read_commits(self._repo_root, self._forge_dir)
        except Exception as error:  # noqa: BLE001
            logger.warning("%s: commits not read: %s", self._label(), error)
            return {"repo": None, "forge": None, "dirty": False}

    def _relative(self, path: PathLike) -> str:
        return _repo_relative(Path(path), self._repo_root, self._cwd)

    def _relative_all(self, paths: Iterable[PathLike | None]) -> list[str]:
        listed: list[str] = []
        for path in paths:
            if path is None or str(path) == "":
                continue
            relative = self._relative(path)
            if relative not in listed:
                listed.append(relative)
        return listed

    def _read(self) -> dict[str, Any] | None:
        try:
            document = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return document if isinstance(document, dict) else None

    def _write(self, document: dict[str, Any]) -> None:
        temporary = self.path.with_name(f".{MANIFEST_NAME}.{os.getpid()}.tmp")
        try:
            temporary.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
            os.replace(temporary, self.path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    def _now(self) -> str:
        return self._clock().isoformat(timespec="seconds")

    def _label(self) -> str:
        return f"run manifest {self.path}"


def run_recorded(
    run_dir: PathLike, body: Callable[[], int | None], *, command: str,
    inputs: Iterable[PathLike | None] = (), create: bool = True, **manifest_options: Any,
) -> int | None:
    """Run ``body`` with ``run_dir``'s manifest open around it, closed on
    every way out — a return, an exception, ``sys.exit`` and Ctrl-C — and
    hand back whatever ``body`` returned or raised."""
    manifest = RunManifest(run_dir, **manifest_options)
    manifest.start(command=command, inputs=inputs, create=create)
    try:
        code = body()
    except KeyboardInterrupt as interrupt:
        manifest.finish(exit_code=INTERRUPTED_EXIT, error=interrupt)
        raise
    except SystemExit as exit_:
        # Python exits 0 on None, the code on an int, and 1 printing anything else.
        if exit_.code is None or isinstance(exit_.code, int):
            manifest.finish(exit_code=exit_.code or 0)
        else:
            manifest.finish(exit_code=1, message=str(exit_.code))
        raise
    except BaseException as error:
        manifest.finish(exit_code=1, error=error)
        raise
    manifest.finish(exit_code=code or 0)
    return code


def run_script(
    run_dir: PathLike | None, body: Callable[[], int | None], *, script: PathLike,
    argv: Sequence[str] | None = None, inputs: Iterable[PathLike | None] = (),
    create: bool = True,
) -> int | None:
    """:func:`run_recorded` for a script run by path (``python scripts/….py``):
    ``argv`` its arguments (``sys.argv[1:]`` when None); a ``run_dir`` of
    None, an invocation writing nothing, runs ``body`` bare."""
    if run_dir is None:
        return body()
    return run_recorded(
        run_dir, body, command=command_line(argv, script=script), inputs=inputs, create=create,
    )

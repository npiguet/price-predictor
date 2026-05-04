"""Shared helpers for invoking forge-connector and Forge JVM main classes.

Centralises the four classpath/JAR/JVM-launching duplications that previously
lived in `run_convert` and the three sealed connectors, plus the
Windows-vs-POSIX process-tree termination helper.
"""

from __future__ import annotations

import os
import platform
import signal
import subprocess
from pathlib import Path

CONNECTOR_JAR_NAME = "forge-connector-1.0.0-SNAPSHOT-jar-with-dependencies.jar"
FORGE_VERSION = "2.0.13-SNAPSHOT"


def project_root() -> Path:
    """Locate the price-predictor project root from this file."""
    return Path(__file__).resolve().parents[3]


def forge_dir() -> Path:
    """Return the sibling Forge checkout directory (../forge)."""
    return project_root().parent / "forge"


def forge_module_jar(module: str) -> Path:
    """Return the path to a Forge module JAR (e.g. ``forge-game``)."""
    return forge_dir() / module / "target" / f"{module}-{FORGE_VERSION}.jar"


def resolve_connector_jar() -> Path:
    """Return the forge-connector fat JAR path. Raises if not built yet."""
    jar = project_root() / "forge-connector" / "target" / CONNECTOR_JAR_NAME
    if not jar.exists():
        raise FileNotFoundError(
            f"Connector JAR not found at {jar}\n"
            "Build it first: cd forge-connector && mvn package -DskipTests"
        )
    return jar


def build_forge_classpath(
    *,
    include_full_runtime: bool = True,
    include_dependency_glob: bool = False,
) -> str:
    """Build the classpath needed to run a forge-connector main class.

    Args:
        include_full_runtime: include forge-gui and forge-ai in addition to
            forge-game and forge-core. The convert command runs only static
            card-script translation and doesn't need them.
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


def build_jvm_command(
    *,
    main_class: str,
    classpath: str,
    system_properties: dict[str, str] | None = None,
    xmx: str | None = None,
    main_args: list[str] | None = None,
) -> list[str]:
    """Construct a ``java`` subprocess command list."""
    cmd: list[str] = ["java"]
    if system_properties:
        for key, value in system_properties.items():
            cmd.append(f"-D{key}={value}")
    if xmx:
        cmd.append(f"-Xmx{xmx}")
    cmd.extend(["-cp", classpath, main_class])
    if main_args:
        cmd.extend(main_args)
    return cmd


def run_forge_worker(
    main_class: str,
    *,
    main_args: list[str] | None = None,
    xmx: str | None = None,
    input_text: str | None = None,
) -> subprocess.CompletedProcess:
    """Run a one-shot forge-connector main class and return the result.

    Raises FileNotFoundError if ``java`` is not on PATH.
    """
    cmd = build_jvm_command(
        main_class=main_class,
        classpath=build_forge_classpath(),
        main_args=main_args,
        xmx=xmx,
    )
    capture = input_text is not None
    try:
        return subprocess.run(
            cmd,
            input=input_text,
            capture_output=capture,
            text=capture,
            encoding="utf-8" if capture else None,
            check=False,
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "Java not found. Ensure java is on PATH."
        ) from exc


def kill_process_tree(proc: subprocess.Popen) -> None:
    """Kill a subprocess and its entire child tree.

    On Windows, ``proc.terminate()`` only kills the parent — child JVM
    processes survive as orphans. ``taskkill /F /T`` walks the PID tree
    and forces termination. On POSIX, ``os.killpg`` does the same when
    the process was launched with ``start_new_session=True``.
    """
    if proc.poll() is not None:
        return
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
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass

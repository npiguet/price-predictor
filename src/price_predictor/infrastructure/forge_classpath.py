"""Where Forge and the connector are on disk, and the classpath a
forge-connector main class runs on."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

CONNECTOR_JAR_NAME = "forge-connector-1.0.0-SNAPSHOT-jar-with-dependencies.jar"

#: Fallback Forge version, used only when the built JAR cannot be found and so
#: no version can be read off disk. `forge-connector/pom.xml` pins the real one
#: in its `forge.version` property — Maven's `systemPath` cannot glob, so that
#: pin has to be updated by hand on a Forge upgrade.
FORGE_VERSION = "2.0.15-SNAPSHOT"


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

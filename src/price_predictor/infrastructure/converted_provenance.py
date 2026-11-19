"""The conversion record on disk: written by ``convert``, read by every
collecting command (finding 11.3; the rule in
:mod:`price_predictor.domain.converted_provenance`).

``convert`` keeps its record in the ``run.json`` of its run directory, the
``converted/`` directory holding both trees (or the card tree itself when the
two are not siblings). A collector names a converted tree, so the record is
looked for in that tree's ``run.json``, then in its parent's: the first of
the two whose invocations record a conversion is the tree's. There is no
other fallback: a tree converted before the record existed is refused.

The connector JAR's Forge commit is the sibling checkout's HEAD when the
connector was built (``forge-connector/pom.xml``, manifest attribute
``Forge-Commit``): it names the Forge the JAR links against only when Forge
was installed from that same HEAD before the connector (the build order).
"""

from __future__ import annotations

import hashlib
import json
import logging
import zipfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from price_predictor.domain.converted_provenance import (
    CONVERSION_MEMBER,
    REBUILD_CONNECTOR,
    Conversion,
    ConversionRefused,
    latest_conversion,
    manifest_forge_commit,
    require_same_commit,
)
from price_predictor.infrastructure.run_manifest import MANIFEST_NAME, annotate_run, git_output

logger = logging.getLogger(__name__)

#: The suffix of the provenance sidecar beside each converted file.
SIDECAR_SUFFIX = ".provenance.json"
#: The source trees ``convert`` writes, as the sidecars' keys name them.
CARD_TREE = "cardsfolder"
TOKEN_TREE = "tokenscripts"


def sidecar_paths(folder: Path) -> list[Path]:
    """Every sidecar under ``folder``, recursively, in path order so every
    walk over a tree visits it the same way."""
    return sorted(Path(folder).rglob(f"*{SIDECAR_SUFFIX}"))


def sidecar_digest(roots: Mapping[str, Path]) -> str:
    """sha256 over every sidecar under ``roots`` (``tree -> folder``), by
    tree, relative path and size, sorted: cheap enough for tens of thousands
    of files, and moved by a reconversion that changes any sidecar's size."""
    digest = hashlib.sha256()
    for tree, root in sorted(roots.items()):
        for path in sidecar_paths(Path(root)):
            relative = path.relative_to(root).as_posix()
            digest.update(f"{tree}/{relative}:{path.stat().st_size}\n".encode())
    return digest.hexdigest()


def forge_checkout(forge_dir: Path) -> tuple[str | None, bool]:
    """The Forge checkout's full HEAD (None when it is no checkout) and
    whether its working tree has uncommitted changes to tracked files."""
    head = git_output(Path(forge_dir), "rev-parse", "HEAD")
    if head is None:
        return None, False
    status = git_output(Path(forge_dir), "status", "--porcelain", "--untracked-files=no")
    return head, bool(status)


def read_conversion(tree: Path) -> Conversion:
    """The conversion ``tree`` comes from; :class:`ConversionRefused` naming
    the paths looked at when neither records one."""
    candidates = [Path(tree) / MANIFEST_NAME, Path(tree).parent / MANIFEST_NAME]
    for manifest in candidates:
        conversion = latest_conversion(_document(manifest), manifest)
        if conversion is not None:
            return conversion
    raise ConversionRefused(
        f"the converted tree {tree} names no conversion: neither "
        f"{' nor '.join(str(path) for path in candidates)} records one "
        f"(the '{CONVERSION_MEMBER}' member convert writes). Reconvert it with "
        "'python -m price_predictor convert'"
    )


def _document(manifest: Path) -> dict[str, Any]:
    try:
        document = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return document if isinstance(document, dict) else {}


def connector_forge_commit(jar: Path) -> str:
    """The Forge commit the connector JAR's manifest names; refused, naming
    the JAR, when it names none."""
    try:
        with zipfile.ZipFile(jar) as archive:
            text = archive.read("META-INF/MANIFEST.MF").decode("utf-8", errors="replace")
    except (OSError, KeyError, zipfile.BadZipFile) as error:
        raise ConversionRefused(
            f"rebuild the connector: its JAR {jar} cannot be read ({error}); {REBUILD_CONNECTOR}"
        ) from error
    commit = manifest_forge_commit(text)
    if commit is None:
        # "rebuild the connector" first: run.json's outcome keeps 300 characters.
        raise ConversionRefused(
            f"rebuild the connector: its JAR {jar} names no Forge commit in its "
            "manifest, so it was built before the commit was recorded or without "
            f"git; {REBUILD_CONNECTOR}"
        )
    return commit


def collection_conversion(trees: Iterable[Path], *, jar: Path | None = None) -> Conversion:
    """The one conversion ``trees`` come from, checked against the connector
    JAR (by default the one the workers run): refused when a tree names none,
    when they name two, and when the JAR was built at another Forge commit.
    A conversion from a dirty Forge checkout is warned about, not refused."""
    conversions = {conversion.manifest: conversion for conversion in map(read_conversion, trees)}
    if len(conversions) != 1:
        raise ConversionRefused(
            "the converted trees come from "
            + (f"{len(conversions)} conversions ({', '.join(map(str, conversions))}); "
               "name the trees of one" if conversions else "no conversion: name a tree")
        )
    conversion = next(iter(conversions.values()))
    jar = _connector_jar() if jar is None else Path(jar)
    require_same_commit(conversion, connector_forge_commit(jar), jar)
    if conversion.forge_dirty:
        logger.warning(
            "The converted tree %s was converted from a Forge checkout with uncommitted "
            "changes: commit %s does not pin the scripts it converted",
            conversion.run, conversion.forge_commit,
        )
    return conversion


def _connector_jar() -> Path:
    from price_predictor.infrastructure.forge_jvm import resolve_connector_jar

    try:
        return resolve_connector_jar()
    except FileNotFoundError as error:
        raise ConversionRefused(str(error)) from error


def record_conversion(run_dir: Path, conversion: Conversion) -> None:
    """Keep ``conversion`` in ``run_dir``'s ``run.json`` entry as
    ``conversion``: its run, Forge commit, dirty flag and sidecar digest."""
    annotate_run(run_dir, **{CONVERSION_MEMBER: conversion.recorded()})

"""The conversion a converted tree comes from.

An effect record names its abilities by provenance keys, which join against
the sidecars of the converted tree the collection played over, by trait
numbers the Forge scripts and engine decide. ``convert`` records in its
``run.json`` the Forge commit it converted at and a digest of the sidecars it
wrote (:data:`CONVERSION_MEMBER`, one per invocation that changed the tree);
every collecting command reads that record from the tree it is given and
refuses a connector JAR built against another Forge commit, so a shard is
never collected against keys of another conversion.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: The invocation member of the convert's ``run.json`` holding its record:
#: ``{"forge_commit", "forge_dirty", "sidecar_digest"}`` once it finished
#: whole, ``null`` once it may have changed the tree without finishing.
CONVERSION_MEMBER = "conversion"
#: The connector JAR's manifest attribute naming the Forge commit it was built at.
MANIFEST_COMMIT_ATTRIBUTE = "Forge-Commit"
#: The build that writes it.
REBUILD_CONNECTOR = "mvn -f forge-connector/pom.xml install -DskipTests"

_FULL_COMMIT = re.compile(r"[0-9a-f]{40}")


class ConversionRefused(Exception):
    """A collection that cannot name the conversion its keys join against,
    or whose connector was built against another Forge commit."""


def is_full_commit(value: Any) -> bool:
    """Whether ``value`` is a full 40-hex git commit."""
    return isinstance(value, str) and _FULL_COMMIT.fullmatch(value) is not None


@dataclass(frozen=True)
class Conversion:
    """One finished conversion, as its ``run.json`` records it."""

    #: The convert's ``run.json`` it was read from.
    manifest: Path
    forge_commit: str
    #: True when the Forge checkout had uncommitted changes to tracked files,
    #: so the commit does not pin the scripts converted.
    forge_dirty: bool
    sidecar_digest: str
    #: True when the Forge checkout the workers read their scripts from has
    #: uncommitted changes to tracked files (set when a collection starts; a
    #: conversion read from ``run.json`` has not been checked against one).
    checkout_dirty: bool = False

    @property
    def run(self) -> Path:
        """The conversion's run directory."""
        return self.manifest.parent

    def recorded(self) -> dict[str, Any]:
        """The record a collector keeps in its own ``run.json``."""
        return {
            "run": self.run.as_posix(), "forge_commit": self.forge_commit,
            "forge_dirty": self.forge_dirty, "sidecar_digest": self.sidecar_digest,
            "checkout_dirty": self.checkout_dirty,
        }


def latest_conversion(document: Mapping[str, Any], manifest: Path) -> Conversion | None:
    """The conversion the latest invocation that changed the tree recorded,
    or None when no invocation of ``document`` records one.

    Raises :class:`ConversionRefused` when that invocation did not finish
    whole (its record is ``null``) or names no full Forge commit.
    """
    invocations = document.get("invocations")
    for entry in reversed(invocations if isinstance(invocations, list) else []):
        if isinstance(entry, dict) and CONVERSION_MEMBER in entry:
            return _conversion(entry[CONVERSION_MEMBER], manifest)
    return None


def _conversion(record: Any, manifest: Path) -> Conversion:
    if not isinstance(record, dict):
        raise ConversionRefused(
            f"{manifest}: the last convert into this tree did not finish whole, "
            "so the tree is no conversion a collection may join against; "
            "reconvert it with --replace"
        )
    commit = record.get("forge_commit")
    if not is_full_commit(commit):
        raise ConversionRefused(
            f"{manifest}: its conversion names no Forge commit ({commit!r}); "
            "reconvert the tree from a Forge checkout"
        )
    return Conversion(
        manifest=manifest, forge_commit=commit,
        forge_dirty=bool(record.get("forge_dirty")),
        sidecar_digest=str(record.get("sidecar_digest") or ""),
    )


def manifest_forge_commit(manifest_text: str) -> str | None:
    """The full Forge commit a JAR manifest's :data:`MANIFEST_COMMIT_ATTRIBUTE`
    names, or None when it names none (a connector built before it was
    written, or a build that could not ask git)."""
    attributes: dict[str, str] = {}
    name = None
    for line in manifest_text.splitlines():
        if line.startswith(" ") and name is not None:
            attributes[name] += line[1:]
        elif ":" in line:
            name, _, value = line.partition(":")
            attributes[name] = value.strip()
    commit = attributes.get(MANIFEST_COMMIT_ATTRIBUTE)
    return commit if is_full_commit(commit) else None


def require_same_commit(conversion: Conversion, jar_commit: str, jar: Path) -> None:
    """Refuse a connector built against another Forge commit than ``conversion``'s."""
    if jar_commit != conversion.forge_commit:
        # Both commits first: run.json's outcome keeps a message's first 300
        # characters, and the two paths can be long.
        raise ConversionRefused(
            f"the converted tree was converted at Forge {conversion.forge_commit}, but "
            f"the connector was built at Forge {jar_commit}: the records' "
            "provenance keys would join against another conversion's lines (converted tree "
            f"{conversion.run}, connector JAR {jar}). Reconvert the tree at the "
            "connector's commit, or install Forge at the conversion's commit "
            "and rebuild the connector"
        )


def require_checkout_at_jar_commit(
    checkout_commit: str | None, jar_commit: str, forge: Path, jar: Path,
) -> None:
    """Refuse a collection whose workers would read card scripts from a Forge
    checkout at another commit than the connector JAR was built at (or from
    a directory that is no checkout, so no commit names the scripts)."""
    if checkout_commit is None:
        raise ConversionRefused(
            f"the Forge directory {forge} is no git checkout, so the card scripts the "
            f"workers read name no commit (connector JAR {jar}, built at Forge {jar_commit}); "
            "point ../forge at a checkout at the connector's commit"
        )
    if checkout_commit != jar_commit:
        raise ConversionRefused(
            f"the Forge checkout is at {checkout_commit}, but the connector was "
            f"built at Forge {jar_commit}: the workers would play card scripts another "
            f"commit wrote than the ones the keys join against (checkout {forge}, "
            f"connector JAR {jar}). Check out the connector's commit, or reinstall "
            "Forge and rebuild the connector"
        )


def recorded_conversions(document: Mapping[str, Any], manifest: Path) -> list[Conversion]:
    """Every conversion the invocations of a collection's ``run.json`` record,
    in order (each collecting command keeps the one it ran against as
    :data:`CONVERSION_MEMBER`); an invocation that recorded none is passed over."""
    invocations = document.get("invocations")
    return [
        _conversion(entry[CONVERSION_MEMBER], manifest)
        for entry in (invocations if isinstance(invocations, list) else [])
        if isinstance(entry, dict) and isinstance(entry.get(CONVERSION_MEMBER), dict)
    ]


def require_same_conversion(recorded: Conversion, wanted: Conversion) -> None:
    """Refuse records collected against another conversion than ``wanted``'s:
    one that differs in Forge commit or sidecar digest keyed its records to
    other lines than ``wanted``'s tree holds."""
    if (recorded.forge_commit, recorded.sidecar_digest) == (
        wanted.forge_commit, wanted.sidecar_digest,
    ):
        return
    # Both conversions first, and tersely: run.json's outcome keeps a message's
    # first 300 characters, and each conversion is 104 of them.
    raise ConversionRefused(
        f"records: Forge {recorded.forge_commit} sidecars {recorded.sidecar_digest}; "
        f"tree: Forge {wanted.forge_commit} sidecars {wanted.sidecar_digest}: a records "
        f"directory holds one conversion (records {recorded.manifest}, converted tree "
        f"{wanted.run}). Collect into a new directory, or reconvert the tree at the "
        "records' conversion"
    )

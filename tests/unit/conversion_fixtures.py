"""Converted trees with the ``run.json`` ``convert`` keeps, and connector JARs
with the manifest the build writes, for the tests of the conversion a
collection is tied to (``price_predictor.domain.converted_provenance``).

The run.json is hand-built: a real conversion needs the Forge runtime and a
minute of JVM, and the member it carries is pinned by the convert's own
tests (``tests/unit/infrastructure/test_cli_convert_conversion.py``).
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from price_predictor.domain.converted_provenance import Conversion

FORGE_COMMIT = "a" * 40
OTHER_COMMIT = "b" * 40
DIGEST = "d" * 64


def conversion_member(commit: str | None = FORGE_COMMIT, *, dirty: bool = False) -> dict:
    """The ``conversion`` member a whole conversion writes."""
    return {"forge_commit": commit, "forge_dirty": dirty, "sidecar_digest": DIGEST}


def write_conversion(root: Path, *members: dict | None, omit: bool = False) -> Path:
    """``root`` as ``convert`` leaves it: ``cardsfolder/`` and ``tokenscripts/``
    beside ``run.json``, one invocation per member (a whole conversion's by
    default); ``omit`` writes one invocation without the member, as every
    conversion before it existed did."""
    for tree in ("cardsfolder", "tokenscripts"):
        (root / tree).mkdir(parents=True, exist_ok=True)
    members = members or (conversion_member(),)
    invocations = [
        {"command": "python -m price_predictor convert", "outcome": "exit 0"}
        | ({} if omit else {"conversion": member})
        for member in members
    ]
    document = {"run": root.as_posix(), "status": "finished", "invocations": invocations}
    (root / "run.json").write_text(json.dumps(document), encoding="utf-8")
    return root


def write_jar(path: Path, commit: str | None = FORGE_COMMIT) -> Path:
    """A connector JAR whose manifest names ``commit``, or none."""
    lines = ["Manifest-Version: 1.0", "Main-Class: com.pricepredictor.connector.ConvertMain"]
    if commit is not None:
        lines.append(f"Forge-Commit: {commit}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("META-INF/MANIFEST.MF", "\r\n".join(lines) + "\r\n")
    return path


#: What ``SourceTrees.conversion`` returns in a test whose trees are no
#: conversion: the collector tests that are not about it patch it in.
STAND_IN_CONVERSION = Conversion(
    manifest=Path("output/effects/gen3/converted/run.json"),
    forge_commit=FORGE_COMMIT, forge_dirty=False, sidecar_digest=DIGEST,
)
#: The patch target that stands :data:`STAND_IN_CONVERSION` in.
TREES_CONVERSION = "effects.infrastructure.converted_trees.SourceTrees.conversion"

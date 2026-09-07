"""Spawns the Java ``VariantSidecarMain``.

One connector per Java main, mirroring ``keyword_definition_connector``. Runs to
completion rather than being supervised: writing sidecars is a single parse pass
with no game in it.

Why the sidecars come from Java at all: a provenance key's
``index_within_kind`` is the trait's position in Forge's own runtime trait list,
and the collectors read that list directly. Numbering the traits in Python would
be a second implementation of Forge's parser, and the two would drift silently —
the join only fails loudly when a key is missing, not when it points at the
wrong line.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from price_predictor.infrastructure.forge_jvm import (
    build_forge_classpath,
    build_jvm_command,
)

_MAIN_CLASS = "com.pricepredictor.connector.VariantSidecarMain"


class VariantSidecarConnector:
    """Runs ``VariantSidecarMain`` over a variant tree and waits for it."""

    def run(self, variants_path: Path) -> int:
        """Write a sidecar beside every script in ``variants_path``.

        Returns the exit code.

        Raises:
            FileNotFoundError: If the connector JAR is missing (the error names
                the ``mvn package`` command) or java is not on PATH.
        """
        # Reaches GuiBase / FModel in forge-gui through
        # ForgeEnvironmentInitializer, so the full runtime is required — the
        # same classpath the converter needs.
        classpath = build_forge_classpath(
            include_full_runtime=True,
            include_dependency_glob=True,
        )
        cmd = build_jvm_command(
            main_class=_MAIN_CLASS,
            classpath=classpath,
            main_args=["--variants-path", str(variants_path)],
        )
        return subprocess.run(cmd, check=False).returncode

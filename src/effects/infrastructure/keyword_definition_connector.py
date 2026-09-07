"""Spawns the Java ``KeywordDefinitionMain``.

One connector per Java main, mirroring ``sealed``'s ``match_worker_connector``
and ``draft``'s ``draft_worker_connector``. This one runs to completion rather
than being supervised, because reading Forge's keyword table is a single pass
with no game in it.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from price_predictor.infrastructure.forge_jvm import (
    build_forge_classpath,
    build_jvm_command,
)

_MAIN_CLASS = "com.pricepredictor.connector.KeywordDefinitionMain"


class KeywordDefinitionConnector:
    """Runs ``KeywordDefinitionMain`` and waits for it."""

    def run(self, output_path: Path) -> int:
        """Write keyword definitions to ``output_path``; return the exit code.

        Raises:
            FileNotFoundError: If the connector JAR is missing (the error names
                the ``mvn package`` command) or java is not on PATH.
        """
        # KeywordDefinitionMain calls ForgeEnvironmentInitializer.initialize(),
        # which reaches GuiBase / FModel in forge-gui and pulls forge-ai
        # subsystems, so the full runtime is required — same classpath the
        # converter needs.
        classpath = build_forge_classpath(
            include_full_runtime=True,
            include_dependency_glob=True,
        )
        cmd = build_jvm_command(
            main_class=_MAIN_CLASS,
            classpath=classpath,
            main_args=["--output", str(output_path)],
        )
        return subprocess.run(cmd, check=False).returncode

"""DraftWorkerConnector: launches a Java DraftWorkerMain subprocess.

Reuses the shared ``forge_jvm`` helpers exactly as ``MatchWorkerConnector`` /
``PoolConnector`` do. The worker streams one ``<<DRAFT-EVENT-JSON>>`` line per
completed draft on stdout (piped, text-mode, line-buffered); its diagnostics go
to stderr (discarded). Run parameters (set restriction, agent mix, pod size)
are forwarded as JVM system properties.
"""

from __future__ import annotations

import subprocess

from price_predictor.infrastructure.forge_jvm import (
    build_forge_classpath,
    build_jvm_command,
)

_MAIN_CLASS = "com.pricepredictor.connector.DraftWorkerMain"


class DraftWorkerConnector:
    """Spawns a single Java ``DraftWorkerMain`` process with stdout piped as text."""

    def start(
        self,
        *,
        agent_mix: str,
        set_code: str | None = None,
        xmx: str = "1200m",
    ) -> subprocess.Popen[str]:
        """Spawn the worker and return its Popen handle (stdout = text pipe).

        Args:
            agent_mix: canonical ``label:weight,…`` spec; the worker samples each
                seat's agent from it per draft (``-Ddraft.agent.mix``).
            set_code: when given, every draft uses this set (``-Ddraft.set``);
                when None, the worker picks a random sealed-legal set per draft.
            xmx: JVM max heap.

        Raises:
            FileNotFoundError: if the connector JAR is not built.
        """
        system_properties: dict[str, str] = {"draft.agent.mix": agent_mix}
        if set_code is not None:
            system_properties["draft.set"] = set_code

        cmd = build_jvm_command(
            main_class=_MAIN_CLASS,
            classpath=build_forge_classpath(),
            system_properties=system_properties,
            xmx=xmx,
        )
        return subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,  # line-buffered
            start_new_session=True,
        )

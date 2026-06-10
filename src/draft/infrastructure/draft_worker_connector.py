"""DraftWorkerConnector: launches a Java DraftWorkerMain subprocess.

Reuses the shared ``forge_jvm`` helpers exactly as ``MatchWorkerConnector`` /
``PoolConnector`` do. The worker streams one ``<<DRAFT-EVENT-JSON>>`` line per
completed draft on stdout (piped, text-mode, line-buffered), plus — for
model-piloted seats — ``<<DRAFT-PICK-REQUEST>>`` / ``<<DRAFT-ABANDONED>>`` lines
answered over its stdin (the live-play side-channel, contracts/pick-protocol.md).
Its diagnostics go to a per-run stderr log file (FR-015). Run parameters (set
restriction, agent mix, pod size, external-agent set) are forwarded as JVM
system properties.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

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
        external_agents: frozenset[str] | set[str] | None = None,
        run_id: str | None = None,
        log_dir: Path = Path("output/draft"),
        xmx: str = "1200m",
    ) -> subprocess.Popen[str]:
        """Spawn the worker and return its Popen handle (stdout/stdin = text pipes).

        Args:
            agent_mix: canonical ``label:weight,…`` spec; the worker samples each
                seat's agent from it per draft (``-Ddraft.agent.mix``).
            set_code: when given, every draft uses this set (``-Ddraft.set``);
                when None, the worker picks a random sealed-legal set per draft.
            external_agents: mix labels that are model-piloted; the worker routes
                only these seats through the pick request side-channel
                (``-Ddraft.external.agents``). Empty/None ⇒ byte-for-byte gen-1
                behavior (no requests are ever emitted, SC-004).
            run_id: supervisor run id; names the per-run stderr log file.
            log_dir: directory for the per-run worker stderr log (FR-015).
            xmx: JVM max heap.

        Raises:
            FileNotFoundError: if the connector JAR is not built.
        """
        system_properties: dict[str, str] = {"draft.agent.mix": agent_mix}
        if set_code is not None:
            system_properties["draft.set"] = set_code
        if external_agents:
            system_properties["draft.external.agents"] = ",".join(sorted(external_agents))

        cmd = build_jvm_command(
            main_class=_MAIN_CLASS,
            classpath=build_forge_classpath(),
            system_properties=system_properties,
            xmx=xmx,
        )

        # Pipe stdin only when the worker may emit pick requests; redirect stderr
        # to a per-run log so FR-015's diagnostic log exists (gen-1 discarded it).
        stdin = subprocess.PIPE if external_agents else subprocess.DEVNULL
        if external_agents:
            log_dir.mkdir(parents=True, exist_ok=True)
            stderr = open(log_dir / f"worker-{run_id or 'run'}.log", "a", encoding="utf-8")
        else:
            stderr = subprocess.DEVNULL

        return subprocess.Popen(
            cmd,
            stdin=stdin,
            stdout=subprocess.PIPE,
            stderr=stderr,
            text=True,
            encoding="utf-8",   # worker emits UTF-8; don't fall back to the
            errors="replace",   # platform codec (cp1252 on Windows) and choke
            bufsize=1,          # on accented card names
            start_new_session=True,
        )

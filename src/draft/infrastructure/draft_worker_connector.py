"""DraftWorkerConnector: launches a Java DraftWorkerMain subprocess.

Reuses the shared ``forge_classpath`` and ``jvm_processes`` helpers exactly as
``MatchWorkerConnector`` / ``PoolConnector`` do. The worker streams one
``<<DRAFT-EVENT-JSON>>`` line per completed draft on stdout (piped, text-mode,
line-buffered), plus — for model-piloted seats — ``<<DRAFT-PICK-REQUEST>>`` /
``<<DRAFT-ABANDONED>>`` lines answered over its stdin (the live-play
side-channel, contracts/pick-protocol.md). Its diagnostics go to a per-run
stderr log file (FR-015). Run parameters (set restriction, agent mix, pod size,
external-agent set) are forwarded as JVM system properties.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import IO

from price_predictor.infrastructure.forge_classpath import build_forge_classpath
from price_predictor.infrastructure.jvm_processes import JvmOptions, build_jvm_command

_MAIN_CLASS = "com.pricepredictor.connector.DraftWorkerMain"


@dataclass(frozen=True)
class DraftWorkerLaunch:
    """Everything a ``DraftWorkerMain`` worker is started with.

    Attributes:
        agent_mix: canonical ``label:weight,...`` spec; the worker samples each
            seat's agent from it per draft (``-Ddraft.agent.mix``).
        set_code: when given, every draft uses this set (``-Ddraft.set``);
            when None, the worker picks a random sealed-legal set per draft.
        external_agents: mix labels that are model-piloted; the worker routes
            only these seats through the pick request side-channel
            (``-Ddraft.external.agents``). Empty/None => byte-for-byte gen-1
            behavior (no requests are ever emitted, SC-004).
        run_id: supervisor run id; names the per-run stderr log file.
        required_agent: when given, the worker overwrites one uniformly-chosen
            seat with this label if the independent per-seat mix draw produced
            none (``-Ddraft.required.agent``), guaranteeing every played pod
            carries at least one such seat (spec 021 FR-003). None => sampling
            is byte-for-byte unchanged.
        log_dir: directory for the per-run worker stderr log (FR-015); required
            with ``external_agents`` (the only workers that keep a log).
        xmx: JVM max heap.
    """

    agent_mix: str
    set_code: str | None = None
    external_agents: frozenset[str] | set[str] | None = None
    run_id: str | None = None
    required_agent: str | None = None
    log_dir: Path | None = None
    xmx: str = "1200m"

    def system_properties(self) -> dict[str, str]:
        """The worker's run parameters as JVM system properties."""
        properties: dict[str, str] = {"draft.agent.mix": self.agent_mix}
        if self.set_code is not None:
            properties["draft.set"] = self.set_code
        if self.external_agents:
            properties["draft.external.agents"] = ",".join(sorted(self.external_agents))
        if self.required_agent:
            properties["draft.required.agent"] = self.required_agent
        return properties

    def stdin_and_stderr(self) -> tuple[int, int | IO[str]]:
        """The worker's stdin and stderr.

        Stdin is piped only when the worker may emit pick requests; its stderr
        goes to a per-run log so FR-015's diagnostic log exists (gen-1
        discarded it).

        Raises:
            ValueError: ``external_agents`` is given without a ``log_dir``.
        """
        if not self.external_agents:
            return subprocess.DEVNULL, subprocess.DEVNULL
        if self.log_dir is None:
            raise ValueError("log_dir is required when external_agents is given")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        stderr = open(self.log_dir / f"worker-{self.run_id or 'run'}.log", "a", encoding="utf-8")
        return subprocess.PIPE, stderr


class DraftWorkerConnector:
    """Spawns a single Java ``DraftWorkerMain`` process with stdout piped as text."""

    def start(self, launch: DraftWorkerLaunch) -> subprocess.Popen[str]:
        """Spawn the worker as ``launch`` describes it and return its Popen
        handle (stdout/stdin = text pipes).

        Raises:
            ValueError: ``launch`` has ``external_agents`` but no ``log_dir``.
            FileNotFoundError: if the connector JAR is not built.
        """
        cmd = build_jvm_command(
            main_class=_MAIN_CLASS,
            classpath=build_forge_classpath(),
            options=JvmOptions(system_properties=launch.system_properties(), xmx=launch.xmx),
        )
        stdin, stderr = launch.stdin_and_stderr()
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

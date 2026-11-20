"""MatchWorkerConnector: launches a Java MatchWorkerMain subprocess, as a
:class:`WorkerLaunch` describes it."""

from __future__ import annotations

import dataclasses
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import IO

from price_predictor.infrastructure.forge_classpath import build_forge_classpath
from price_predictor.infrastructure.jvm_processes import JvmOptions, build_jvm_command
from price_predictor.infrastructure.worker_stops import (
    RECYCLE_SECONDS_PROPERTY,
    STOP_FILE_PROPERTY,
    graceful_popen_options,
)

# Default weight of the file-sampled side-B method relative to the 4 Forge
# methods (combined weight 10). Single source of truth: the supervisor
# default, the connector default, the CLI fallback, and the CLI help text
# all reference this value.
DEFAULT_SIDE_B_DECKS_WEIGHT = 4

#: The worker's ``-Xmx``. 1200m is containment, not a memory budget. Forge
#: misbehaves late in a JVM's life — runaway allocation, loops that do not
#: terminate — and a small heap turns that into an early, cheap death that
#: ForgeWorkerPool restarts, instead of one worker dragging the machine down
#: hours into a run. The price is a handful of GC crashes per overnight run,
#: which is accepted: a crashed worker costs the game it was playing (cut, no
#: game-end marker) and nothing else. Raising it hides the pathology rather
#: than containing it, and it is paired with the pool's recycle
#: (ForgeWorkerPool._kill_oldest_worker, and with a stop file
#: recycle_seconds) — the two are one measure, not two knobs.
WORKER_XMX = "1200m"


@dataclasses.dataclass(frozen=True)
class WorkerDecks:
    """Where a match worker's decks come from.

    Attributes:
        side_a: When given, the worker samples deck A from this
            generated-decks file instead of using the 4 Forge methods on side
            A (``-Dside.a.decks.file``).
        side_b: When given, the worker rolls deck B between the 4 Forge
            methods (weights 4:3:2:1) and sampling from this file (weight
            ``side_b_weight``); when None, deck B uses Forge methods only
            (``-Dside.b.decks.file``).
        side_b_weight: Weight of the file-sampled side-B method relative to
            the Forge methods; ignored without ``side_b``
            (``-Dside.b.decks.weight``).
        decks_only: Both sides sample from the decks file instead of rolling
            the 4 Forge methods, and no pool is opened
            (``-Dsealed.decks.only=true``): for decks that belong to no set —
            a coverage or variant deck drawn from the whole corpus — where
            resolving the set code would fail.
        exclude_cards: A file of card names no pool may hold
            (``-Dsealed.exclude.cards``). The worker opens its own pool per
            match rather than reading a pools file, so a depleted collection
            run needs the exclusion here; without it it would quietly play
            full-strength pools.
    """

    side_a: Path | None = None
    side_b: Path | None = None
    side_b_weight: int = DEFAULT_SIDE_B_DECKS_WEIGHT
    decks_only: bool = False
    exclude_cards: Path | None = None

    def put_pool_rules(self, properties: dict[str, str]) -> None:
        """Add the exclusion and the decks-only switch."""
        if self.exclude_cards is not None:
            properties["sealed.exclude.cards"] = str(self.exclude_cards)
        if self.decks_only:
            properties["sealed.decks.only"] = "true"

    def put_sides(self, properties: dict[str, str]) -> None:
        """Add the decks files of each side."""
        if self.side_a is not None:
            properties["side.a.decks.file"] = str(self.side_a)
        if self.side_b is not None:
            properties["side.b.decks.file"] = str(self.side_b)
            properties["side.b.decks.weight"] = str(self.side_b_weight)


@dataclasses.dataclass(frozen=True)
class WorkerRecords:
    """What a match worker records of its games beside the match outcomes.

    Attributes:
        directory: When given, the worker also writes effect records to a
            JSONL shard in this directory (``-Deffect.records.dir``). The
            instrumentation rides matches that were going to be played anyway
            and costs no extra simulation.
        worker_index: This worker's slot, which the shard filename and every
            record id carry (``-Deffect.worker.index``). The id's other half —
            the JVM lifetime — is minted by the worker itself, because this
            process is restarted many times per run and a supervisor-supplied
            number would be one more thing a launcher could forget.
        collection_caps: Per-worker collection caps, as produced by
            ``effects.domain.collection_caps.as_system_properties``. Like
            ``worker_index``, passed only alongside ``directory``; the worker
            falls back to its own defaults for anything absent.
        variant_scripts: When given, a directory of perturbed Forge scripts to
            stage into Forge's custom-cards directory before the card database
            loads (``-Deffect.variant.scripts``). Staging also registers every
            name with ``VariantRegistry``, which is what makes a variant's
            provenance key resolve to ``variant-scripts/`` rather than
            ``cardsfolder/``. Read by ``ForgeEnvironmentInitializer.initialize``
            at worker startup, so a script that only appears afterward is
            invisible for the life of the JVM.
    """

    directory: Path | None = None
    worker_index: int = 0
    collection_caps: Mapping[str, object] | None = None
    variant_scripts: Path | None = None

    def put_records(self, properties: dict[str, str]) -> None:
        """Add the records directory, the slot and the caps."""
        if self.directory is None:
            return
        properties["effect.records.dir"] = str(self.directory)
        properties["effect.worker.index"] = str(self.worker_index)
        for key, value in (self.collection_caps or {}).items():
            properties[key] = str(value)

    def put_variants(self, properties: dict[str, str]) -> None:
        """Add the variant scripts directory."""
        if self.variant_scripts is not None:
            properties["effect.variant.scripts"] = str(self.variant_scripts)


@dataclasses.dataclass(frozen=True)
class WorkerStopProtocol:
    """A match worker's side of the stop protocol (``contracts/cli.md`` §
    Match worker stop protocol); without a stop file the worker runs until
    killed.

    Attributes:
        stop_file: The file whose existence asks the worker to stop at its
            game's end (``-Dworker.stop.file``); the supervisor creates it,
            the worker never does.
        recycle_seconds: The JVM age at which the worker stops itself at its
            next game end (``-Dworker.recycle.seconds``).
    """

    stop_file: Path | None = None
    recycle_seconds: int | None = None

    def put(self, properties: dict[str, str]) -> None:
        """Add the stop file and the recycle age."""
        if self.stop_file is not None:
            properties[STOP_FILE_PROPERTY] = str(self.stop_file)
        if self.recycle_seconds is not None:
            properties[RECYCLE_SECONDS_PROPERTY] = str(self.recycle_seconds)


@dataclasses.dataclass(frozen=True)
class WorkerLaunch:
    """Everything a MatchWorkerMain worker is started with.

    Attributes:
        output_file: Where the worker appends match outcomes; None for a
            records-only worker, which builds neither sealed writer.
        run_id: Supervisor-generated UUID stamped onto every match line
            (``-Dmatch.run.id``).
        best_of: Number of games per match (e.g. 3, 7, 17), a positive odd
            integer (``-Dmatch.best.of``).
        decks: Where its decks come from (:class:`WorkerDecks`).
        records: What it records beside the outcomes (:class:`WorkerRecords`).
        stop: Its stop file and recycle age (:class:`WorkerStopProtocol`).
        progress_file: When given, the worker appends one line per completed
            match to this file (``-Dsealed.progress.file``). Written
            unconditionally — independent of ``output_file``/records-only mode
            and of both sealed writers — because a progress tally is not a
            corpus and must not ride the gate that keeps records-only mode out
            of one. This is what lets a records-only round (``output_file`` is
            None) be bounded: without a real destination for ``output_file``,
            nothing else this worker writes grows with each completed match.
    """

    output_file: Path | None
    run_id: str
    best_of: int
    decks: WorkerDecks = dataclasses.field(default_factory=WorkerDecks)
    records: WorkerRecords = dataclasses.field(default_factory=WorkerRecords)
    stop: WorkerStopProtocol = dataclasses.field(default_factory=WorkerStopProtocol)
    progress_file: Path | None = None

    def check(self) -> None:
        """Raise ValueError for a launch the worker cannot run: ``best_of``
        not a positive odd integer, or neither an output file nor effect
        records, with which it would play matches and record nothing."""
        if self.best_of < 1 or self.best_of % 2 == 0:
            raise ValueError(
                f"best_of must be a positive odd integer, got: {self.best_of}"
            )
        if self.output_file is None and self.records.directory is None:
            raise ValueError(
                "the worker needs an output file, effect records, or both; "
                "with neither it would play matches and record nothing"
            )

    def system_properties(self) -> dict[str, str]:
        """The worker's ``-D`` properties, in the order the command passes
        them (a collection cap that names a property set later is replaced
        by it)."""
        properties: dict[str, str] = {
            "match.run.id": self.run_id,
            "match.best.of": str(self.best_of),
        }
        if self.output_file is not None:
            properties["output.file"] = str(self.output_file)
        self.records.put_records(properties)
        self.decks.put_pool_rules(properties)
        if self.progress_file is not None:
            properties["sealed.progress.file"] = str(self.progress_file)
        self.records.put_variants(properties)
        self.stop.put(properties)
        self.decks.put_sides(properties)
        return properties


class MatchWorkerConnector:
    """Spawns a single Java MatchWorkerMain process.

    The returned Popen handle can be waited on and terminated by the supervisor.
    """

    def start(
        self, launch: WorkerLaunch, *, log_file: IO[bytes] | None = None,
    ) -> subprocess.Popen:
        """Spawn a MatchWorkerMain Java subprocess as ``launch`` describes
        it and return its Popen handle. ``log_file``: an open binary file
        for the worker's stdout and stderr; None discards them.

        A worker under the stop protocol (a stop file) runs in a process
        group of its own, so the console's Ctrl-C reaches the supervisor
        alone, which asks it to stop at its game's end; one without keeps
        the console's signal.

        Raises:
            ValueError: For a launch :meth:`WorkerLaunch.check` refuses.
            FileNotFoundError: If the JAR is not found or java is not on PATH.
        """
        launch.check()
        cmd = build_jvm_command(
            main_class="com.pricepredictor.connector.MatchWorkerMain",
            classpath=build_forge_classpath(),
            options=JvmOptions(system_properties=launch.system_properties(), xmx=WORKER_XMX),
        )
        stdio = log_file if log_file is not None else subprocess.DEVNULL
        group = (graceful_popen_options() if launch.stop.stop_file is not None
                 else {"start_new_session": True})
        return subprocess.Popen(cmd, stdout=stdio, stderr=stdio, **group)

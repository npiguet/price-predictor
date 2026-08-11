"""play-draft-games: play Forge matches between decks drafted in the same pod.

The command projects the corpus into a seat table once, starts a pool of Forge
workers over it, and lets them draw their own pairings. Its own job is then only
to count: progress is the number of rows the output file has gained, which is
also the stopping condition and the summary.

Reporting comes from :class:`ForgeWorkerPool`, shared with sealed
``match-outcomes``, so the two commands' status lines cannot drift apart.
"""

from __future__ import annotations

import os
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from draft.domain.seat_table import (
    FORGE_NATIVE_LABEL,
    FORGE_REFERENCE_LABEL,
    write_seat_table,
)
from draft.infrastructure.draft_game_connector import DraftGameConnector
from draft.infrastructure.draft_record_io import read_records
from price_predictor.infrastructure.forge_jvm import ForgeWorkerPool

DEFAULT_DRAFTS_PATH = Path("output/draft/drafts.jsonl")
DEFAULT_OUTPUT_PATH = Path("output/draft/draft-games.txt")
DEFAULT_WORKERS = 12
DEFAULT_BEST_OF = 3


@dataclass(frozen=True, slots=True)
class PlayDraftGamesConfig:
    """Everything the command needs; validated by the CLI before it gets here."""

    drafts_path: Path = DEFAULT_DRAFTS_PATH
    output_path: Path = DEFAULT_OUTPUT_PATH
    run_ids: tuple[str, ...] = ()
    n_pairings: int | None = None
    best_of: int = DEFAULT_BEST_OF
    include_mirrors: bool = False
    forge_native_fraction: float = 0.0
    workers: int = DEFAULT_WORKERS


@dataclass(frozen=True, slots=True)
class RunSummary:
    """What the run recorded, for the exit summary and for tests."""

    matches_played: int
    elapsed_seconds: float
    output_path: Path
    interrupted: bool = False

    def format(self) -> str:
        return "\n".join(
            (
                "",
                "=== play-draft-games ===",
                f"matches played   {self.matches_played}",
                f"elapsed          {format_elapsed(self.elapsed_seconds)}",
                f"output           {self.output_path}",
            )
        )


def format_elapsed(seconds: float) -> str:
    """Render a duration as ``38m 12s``, or ``1h 02m 03s`` past an hour."""
    total = int(seconds)
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    return f"{minutes}m {secs:02d}s"


def selected_records(config: PlayDraftGamesConfig):
    """Stream the corpus records this run is scoped to.

    With no ``--run-id`` the whole corpus is in scope. The filter lives here
    rather than only in the CLI so the seat table can never disagree with what
    startup validated.
    """
    for record in read_records(config.drafts_path):
        if not config.run_ids or record.run_id in config.run_ids:
            yield record


def write_seat_table_file(
    config: PlayDraftGamesConfig, destination: Path,
) -> tuple[int, int, int]:
    """Project the scoped corpus into ``destination``.

    Returns ``(seats, forge_reference_seats, diverted_seats)``.
    """
    with open(destination, "w", encoding="utf-8") as out:
        return write_seat_table(
            selected_records(config),
            out,
            native_fraction=config.forge_native_fraction,
        )


class PlayDraftGamesUseCase:
    """Project, supervise, summarise."""

    def __init__(self, connector: DraftGameConnector | None = None) -> None:
        self._connector = connector or DraftGameConnector()

    def execute(self, config: PlayDraftGamesConfig) -> RunSummary:
        run_id = str(uuid.uuid4())
        # mkstemp hands back an open descriptor; close it or Windows refuses to
        # unlink the file while this process still holds it.
        handle, seats_name = tempfile.mkstemp(prefix="draft-seats-", suffix=".txt")
        os.close(handle)
        seats_file = Path(seats_name)
        started = time.monotonic()
        try:
            seats, reference, diverted = write_seat_table_file(config, seats_file)
            print(f"Seat table: {seats} seats -> {seats_file}")
            if config.forge_native_fraction > 0:
                print(
                    f"Forge-native: {diverted} of {reference} {FORGE_REFERENCE_LABEL} "
                    f"seats diverted to {FORGE_NATIVE_LABEL}"
                )

            pool = self._build_pool(config, seats_file, run_id)
            baseline = pool.output_line_count()
            pool.run()
            played = max(0, pool.output_line_count() - baseline)
            interrupted = pool.interrupted
        finally:
            # The seat table is scratch, not an output; it goes even on interrupt.
            seats_file.unlink(missing_ok=True)

        summary = RunSummary(
            matches_played=played,
            elapsed_seconds=time.monotonic() - started,
            output_path=config.output_path,
            interrupted=interrupted,
        )
        print(summary.format())
        return summary

    def _build_pool(
        self,
        config: PlayDraftGamesConfig,
        seats_file: Path,
        run_id: str,
    ) -> ForgeWorkerPool:
        baseline = _line_count(config.output_path)

        def spawn(worker_id: int):
            return self._connector.start(
                seats_file=seats_file,
                output_file=config.output_path,
                run_id=run_id,
                best_of=config.best_of,
                include_mirrors=config.include_mirrors,
            )

        should_stop = None
        if config.n_pairings is not None:
            target = config.n_pairings

            def should_stop(count: int) -> bool:
                # Rows this run added, so a pre-existing corpus is not counted.
                return (count - baseline) >= target

        return ForgeWorkerPool(
            worker_count=config.workers,
            spawn_worker=spawn,
            output_path=config.output_path,
            should_stop=should_stop,
        )


def _line_count(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        with open(path, "rb") as f:
            return sum(1 for _ in f)
    except OSError:
        return 0

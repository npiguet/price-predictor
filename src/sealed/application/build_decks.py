"""BuildDecksUseCase: build scorer-guided 40-card decks from a sealed pools file."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch

from sealed.application.deck_assembly import (
    assemble_full_deck,
    load_pool_embeddings,
)
from sealed.application.evaluate_scorer import format_decks_for_display, score_decks
from sealed.domain.deck import Deck
from sealed.domain.greedy_deck_builder import NONLAND_DECK_SIZE, GreedyDeckBuilder
from sealed.domain.scorer_model import SetTransformerScorer
from sealed.infrastructure.converted_card_locator import ConvertedCardLocator
from sealed.infrastructure.pool_file_reader import (
    count_complete_lines_and_truncate_partial,
    format_generated_deck,
    parse_pools,
)
from sealed.infrastructure.scorer_store import ScorerStore


def _log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


@dataclass
class BuildDecksConfig:
    pools_path: Path
    label: str
    """Generation-method tag written as the first column of every output line.

    Identifies which build-decks invocation (e.g. ``"gen-3"``) produced each
    deck. Consumed by ``match-outcomes`` self-play as the ``method_A`` /
    ``method_B`` value when a deck sampled from this file is played.
    """
    checkpoint: Path
    cards_path: Path
    output: Path
    #: The ``.npz`` embeddings' tree, laid out like ``cards_path``; None
    #: reads them beside the cards.
    embeddings_path: Path | None = None
    sa_temperature: float = 0.8
    sa_cooling: float = 0.85
    sa_max_iterations: int = 200
    restarts: int | str = 1
    print_decks: bool = False
    resume: bool = False
    """When True, count complete lines already in ``output``, skip that many
    pools from the front of the input pools file, and append remaining decks
    to the existing file instead of truncating it. Lets a long run recover
    from an interruption without redoing pools that already produced a
    deck. With this flag off (the default), ``output`` is overwritten."""


@dataclass(frozen=True)
class _ResumePoint:
    skip: int
    open_mode: str
    pools: list


def _resume_point(config: BuildDecksConfig, pools: list) -> _ResumePoint:
    """Where the run starts: after the decks already written when resuming."""
    if not config.resume:
        return _ResumePoint(0, "w", pools)
    skip = count_complete_lines_and_truncate_partial(config.output)
    if skip > 0:
        _log(
            f"Resume: {skip} decks already in {config.output}, "
            f"skipping pools 1..{skip}."
        )
        pools = pools[skip:]
    return _ResumePoint(skip, "a", pools)


def _log_start(config: BuildDecksConfig, total: int) -> None:
    if config.sa_temperature > 0:
        _log(
            f"Building decks for {total} pools with simulated annealing "
            f"(label={config.label}, T0={config.sa_temperature}, "
            f"cooling={config.sa_cooling}, "
            f"max_iter={config.sa_max_iterations}, "
            f"restarts={config.restarts})..."
        )
    else:
        _log(
            f"Building decks for {total} pools (label={config.label}, "
            f"pure greedy, restarts={config.restarts})..."
        )


@dataclass(frozen=True)
class _BuildRun:
    """What one ``execute`` call builds with and where it writes."""

    model: SetTransformerScorer
    locator: ConvertedCardLocator
    config: BuildDecksConfig
    total: int
    skip: int
    open_mode: str


class _DeckSink:
    """The decks written so far, and the progress log over them."""

    def __init__(self) -> None:
        self.written = 0
        self.built_decks: list[list[str]] = []

    def log_progress(self, i: int, total: int, interval: int) -> None:
        if i % interval == 0 or i == total:
            _log(f"  {i}/{total} pools processed ({self.written} decks written)")


class BuildDecksUseCase:
    """Build one scorer-guided 40-card deck per pool and write the result file.

    Mirrors the build pattern in
    ``sealed.application.evaluate_scorer._build_a_decks``: each pool's
    embeddable card subset is fed to ``GreedyDeckBuilder``; basic lands are
    added by ``compute_basic_lands`` over the chosen nonland texts. Pools
    with fewer than 23 embeddable cards are skipped.
    """

    def execute(self, config: BuildDecksConfig) -> int:
        pools = parse_pools(config.pools_path)
        model = self._load_model(config.checkpoint)
        locator = ConvertedCardLocator(
            config.cards_path, embeddings_path=config.embeddings_path,
        )

        config.output.parent.mkdir(parents=True, exist_ok=True)

        total = len(pools)
        resume = _resume_point(config, pools)
        _log_start(config, total)
        run = _BuildRun(model, locator, config, total, resume.skip, resume.open_mode)
        sink = self._write_decks(run, resume.pools)
        _log(f"Done: {sink.written} decks written to {config.output}")

        if config.print_decks and sink.built_decks:
            scores = score_decks(model, sink.built_decks, locator)
            print(format_decks_for_display(sink.built_decks, locator, scores))
        return sink.written

    def _write_decks(self, run: _BuildRun, pools: list) -> _DeckSink:
        sink = _DeckSink()
        # Log progress at each ~1% of total (so a 1500-pool run prints ~100
        # lines of progress, ~15 pools apart). Floor to 1 so smaller runs
        # still log every pool rather than only at the end. ``total`` is the
        # original full pool count so the progress milestones (135, 150, ...)
        # match what a fresh run would have logged.
        progress_interval = max(1, run.total // 100)
        # buffering=1 is line buffering: every "\n" forces a flush to disk so
        # an interrupted run keeps the decks built so far, and `tail -f` on
        # the output file stays current with the loop.
        with open(
            run.config.output, run.open_mode, buffering=1, encoding="utf-8",
        ) as out:
            for i, pool in enumerate(pools, start=run.skip + 1):
                self._write_pool_deck(run, out, pool, sink)
                sink.log_progress(i, run.total, progress_interval)
        return sink

    def _write_pool_deck(self, run: _BuildRun, out, pool, sink: _DeckSink) -> None:
        deck = self._build_one_deck(run.model, pool.cards, run.locator, run.config)
        if deck is None:
            return
        out.write(
            format_generated_deck(run.config.label, pool.set_code, deck.cards),
        )
        out.write("\n")
        sink.written += 1
        if run.config.print_decks:
            sink.built_decks.append(list(deck.cards))

    def _load_model(self, checkpoint_path: Path) -> SetTransformerScorer:
        store = ScorerStore()
        checkpoint = store.load_checkpoint(checkpoint_path)
        model = SetTransformerScorer(checkpoint.config)
        model.load_state_dict(checkpoint.model_state_dict)
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        return model

    def _build_one_deck(
        self,
        model: SetTransformerScorer,
        pool_names: list[str],
        locator: ConvertedCardLocator,
        config: BuildDecksConfig,
    ) -> Deck | None:
        pool_embeddings, valid_names = load_pool_embeddings(pool_names, locator)
        if len(valid_names) < NONLAND_DECK_SIZE:
            return None

        nonland_deck = GreedyDeckBuilder(
            model, pool_embeddings,
            temperature=config.sa_temperature,
            cooling=config.sa_cooling,
            max_iterations=config.sa_max_iterations,
            restarts=config.restarts,
        ).build(valid_names)
        return Deck.of(assemble_full_deck(nonland_deck, locator))

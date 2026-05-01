"""BuildDecksUseCase: build scorer-guided 40-card decks from a sealed pools file."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from sealed.application.evaluate_scorer import format_decks_for_display, score_decks
from sealed.domain.greedy_deck_builder import NONLAND_DECK_SIZE, GreedyDeckBuilder
from sealed.domain.manabase import compute_basic_lands
from sealed.domain.scorer_model import SetTransformerScorer
from sealed.infrastructure.converted_card_locator import ConvertedCardLocator
from sealed.infrastructure.pool_file_reader import parse_pools
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
    checkpoint: Path = field(
        default_factory=lambda: Path("models/sealed/scorer/latest.pt"),
    )
    cards_path: Path = field(default_factory=lambda: Path("output/cardsfolder/"))
    output: Path = field(
        default_factory=lambda: Path("output/sealed/generated-decks.txt"),
    )
    sa_temperature: float = 0.8
    sa_cooling: float = 0.85
    sa_max_iterations: int = 200
    restarts: int | str = 1
    print_decks: bool = False


class BuildDecksUseCase:
    """Build one scorer-guided 40-card deck per pool and write the result file.

    Mirrors the build pattern in
    ``sealed.application.evaluate_scorer._build_a_decks``: each pool's
    embeddable card subset is fed to ``GreedyDeckBuilder``; basic lands are
    added by ``compute_basic_lands`` over the chosen nonland texts. Pools
    with fewer than 23 embeddable cards are skipped.
    """

    PROGRESS_INTERVAL = 100

    def execute(self, config: BuildDecksConfig) -> int:
        pools = parse_pools(config.pools_path)
        model = self._load_model(config.checkpoint)
        locator = ConvertedCardLocator(config.cards_path)

        config.output.parent.mkdir(parents=True, exist_ok=True)

        total = len(pools)
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
        written = 0
        built_decks: list[list[str]] = []
        with open(config.output, "w", encoding="utf-8") as out:
            for i, (set_code, pool_names) in enumerate(pools, start=1):
                deck = self._build_one_deck(model, pool_names, locator, config)
                if deck is not None:
                    out.write(config.label)
                    out.write(";")
                    out.write(set_code)
                    out.write(";")
                    out.write("|".join(deck))
                    out.write("\n")
                    written += 1
                    if config.print_decks:
                        built_decks.append(deck)
                if i % self.PROGRESS_INTERVAL == 0 or i == total:
                    _log(f"  {i}/{total} pools processed ({written} decks written)")
        _log(f"Done: {written} decks written to {config.output}")

        if config.print_decks and built_decks:
            scores = score_decks(model, built_decks, locator)
            print(format_decks_for_display(built_decks, locator, scores))
        return written

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
    ) -> list[str] | None:
        pool_embeddings: dict[str, np.ndarray] = {}
        valid_names: list[str] = []
        for name in pool_names:
            emb = locator.load_embedding(name)
            if emb is not None:
                pool_embeddings[name] = emb
                valid_names.append(name)

        if len(valid_names) < NONLAND_DECK_SIZE:
            return None

        nonland_deck = GreedyDeckBuilder(
            model, pool_embeddings,
            temperature=config.sa_temperature,
            cooling=config.sa_cooling,
            max_iterations=config.sa_max_iterations,
            restarts=config.restarts,
        ).build(valid_names)
        nonland_texts = [
            text for n in nonland_deck if (text := locator.load_text(n)) is not None
        ]
        lands = compute_basic_lands(nonland_texts)

        full_deck: list[str] = list(nonland_deck)
        for land_name, count in lands.items():
            full_deck.extend([land_name] * count)
        return full_deck

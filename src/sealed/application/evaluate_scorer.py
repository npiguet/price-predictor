"""Evaluation use case: round-robin cross-group deck comparison vs Forge baseline."""

from __future__ import annotations

import random
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from sealed.application.deck_assembly import (
    assemble_full_deck,
    load_pool_embeddings,
)
from sealed.domain.greedy_deck_builder import NONLAND_DECK_SIZE, GreedyDeckBuilder
from sealed.domain.round_robin_results import RoundRobinResults, aggregate_results
from sealed.domain.round_robin_scheduling import row_major_pairings, split_evenly
from sealed.domain.scorer_model import SetTransformerScorer
from sealed.infrastructure.converted_card_locator import (
    BASIC_LAND_NAMES,
    ConvertedCardLocator,
)
from sealed.infrastructure.eligible_sets import eligible_sealed_sets
from sealed.infrastructure.evaluation_connector import EvaluationConnector
from sealed.infrastructure.pool_connector import PoolConnector
from sealed.infrastructure.pool_file_reader import parse_pools
from sealed.infrastructure.scorer_store import ScorerStore


@dataclass
class EvaluateScorerConfig:
    checkpoint: Path
    cards_path: Path
    pools: int = 12
    best_of: int = 3
    workers: int = 4
    work_dir: Path | None = None
    set_code: str | None = None
    resume: bool = False


def score_decks(
    model: SetTransformerScorer,
    decks: list[list[str]],
    locator: ConvertedCardLocator,
) -> list[float]:
    """Score each deck (nonland portion) using the scorer model in one batched forward pass."""
    model.eval()
    if not decks:
        return []

    device = next(model.parameters()).device
    deck_embeddings = [_load_nonland_embeddings(deck, locator) for deck in decks]
    lengths = [len(e) for e in deck_embeddings]
    if max(lengths, default=0) == 0:
        return [0.0] * len(decks)

    d_model = model.d_model
    deck_tensors = [
        torch.from_numpy(np.stack(embeds)) if embeds else torch.zeros(0, d_model)
        for embeds in deck_embeddings
    ]
    cards_t = torch.nn.utils.rnn.pad_sequence(
        deck_tensors, batch_first=True,
    ).to(device)
    lengths_t = torch.tensor(lengths, device=device)
    mask_t = torch.arange(cards_t.size(1), device=device)[None, :] < lengths_t[:, None]

    with torch.no_grad():
        out = model(cards_t, mask_t).squeeze(-1)
    return out.cpu().tolist()


def _load_nonland_embeddings(
    deck: list[str], locator: ConvertedCardLocator,
) -> list[np.ndarray]:
    embeds: list[np.ndarray] = []
    for name in deck:
        if name.lower() in BASIC_LAND_NAMES:
            continue
        emb = locator.load_embedding(name)
        if emb is not None:
            embeds.append(emb)
    return embeds


class EvaluateScorerUseCase:
    """Round-robin evaluation pipeline: build N A-decks and N B-decks, play N² matches."""

    def execute(self, config: EvaluateScorerConfig) -> RoundRobinResults:
        work_dir = self._prepare_work_dir(config.work_dir)
        eval_connector = EvaluationConnector()

        if config.resume:
            worker_files = _discover_worker_files(work_dir)
            if not worker_files:
                raise RuntimeError(
                    f"--resume: no validation-matches-*.txt files in {work_dir}"
                )
            n = _infer_n_pools(worker_files)
            print(
                f"Resuming from {work_dir}: {len(worker_files)} worker shard(s),"
                f" {n}x{n} round-robin ({n * n} matches total)"
            )
        else:
            model = self._load_model(config.checkpoint)
            locator = ConvertedCardLocator(config.cards_path)

            set_code = self._resolve_set_code(config.set_code)
            print(f"Generating pools from set: {set_code}")
            pools = self._generate_pools(config.pools, work_dir, set_code)

            print(f"Building {len(pools)} scorer decks (A)...")
            a_decks = _build_a_decks(model, pools, locator)

            print(f"Building {len(pools)} Forge decks (B)...")
            b_decks = eval_connector.build_forge_decks(pools)

            self._dump_decks(work_dir, model, a_decks, b_decks, locator)

            n = len(a_decks)
            print(f"Writing {n}² = {n * n} round-robin match pairings...")
            worker_files = write_round_robin_matches(
                a_decks, b_decks, config.workers, work_dir,
            )

        print(f"Launching {len(worker_files)} workers (best-of-{config.best_of})...")
        outcome_files = eval_connector.launch_workers(
            worker_files, best_of=config.best_of,
        )

        result = aggregate_results(outcome_files, n_pools=n)
        print(result.format_report())
        return result

    def _load_model(self, checkpoint_path: Path) -> SetTransformerScorer:
        store = ScorerStore()
        checkpoint = store.load_checkpoint(checkpoint_path)
        model = SetTransformerScorer(checkpoint.config)
        model.load_state_dict(checkpoint.model_state_dict)
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        print(f"Scorer model on device: {device}")
        return model

    def _prepare_work_dir(self, work_dir: Path | None) -> Path:
        if work_dir:
            path = Path(work_dir)
        else:
            path = Path(tempfile.mkdtemp(prefix="eval_"))
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _generate_pools(
        self, n_pools: int, work_dir: Path, set_code: str,
    ) -> list[list[str]]:
        pools_path = work_dir / "pools"
        pools_path.mkdir(exist_ok=True)
        PoolConnector().generate(set_code, n_pools, pools_path)
        return [pool.cards for pool in parse_pools(pools_path / "pools.txt")]

    def _resolve_set_code(self, set_code: str | None) -> str:
        """Return the given set code, or a random sealed-legal set when None."""
        if set_code is not None:
            return set_code
        eligible = eligible_sealed_sets()
        if not eligible:
            raise RuntimeError(
                "No eligible sealed-legal sets found in AllPrintings.json"
            )
        return random.choice(eligible)

    def _dump_decks(
        self,
        work_dir: Path,
        model: SetTransformerScorer,
        a_decks: list[list[str]],
        b_decks: list[list[str]],
        locator: ConvertedCardLocator,
    ) -> None:
        a_scores = score_decks(model, a_decks, locator)
        b_scores = score_decks(model, b_decks, locator)
        write_decks_file(work_dir / "decks-scorer.txt", a_decks, locator, a_scores)
        write_decks_file(work_dir / "decks-forge.txt", b_decks, locator, b_scores)


_MANA_BRACE_RE = re.compile(r"\{([^}]+)\}")
_VARIABLE_MANA_TOKENS = frozenset({"X", "Y", "Z"})


def _mana_value(cost: str) -> int:
    """Compute mana value (a.k.a. CMC) from a cost string like ``{2}{R}{R}``.

    Lands and other zero-cost cards return 0. Variable costs ({X}/{Y}/{Z})
    contribute 0. Hybrid/split tokens with digits use the larger digit;
    pure-color hybrids contribute 1.
    """
    if not cost:
        return 0
    total = 0
    for match in _MANA_BRACE_RE.finditer(cost):
        token = match.group(1)
        if token.isdigit():
            total += int(token)
        elif token in _VARIABLE_MANA_TOKENS:
            continue
        elif "/" in token:
            digits = [int(p) for p in token.split("/") if p.isdigit()]
            total += max(digits) if digits else 1
        else:
            total += 1
    return total


def format_decks_for_display(
    decks: list[list[str]],
    locator: ConvertedCardLocator,
    scores: list[float],
) -> str:
    """Format decks for human inspection: one card per line with mana cost,
    blank-line-separated, with a `=== Deck N  score=X.XXXX ===` header.

    Cards within each deck are sorted by mana value ascending, with lands
    (no mana cost) grouped at the bottom and ties broken by name.
    """
    cost_cache: dict[str, str] = {}

    def cost_for(name: str) -> str:
        if name not in cost_cache:
            converted = locator.load_text(name)
            cost_cache[name] = (converted.mana_cost_line() if converted else None) or ""
        return cost_cache[name]

    lines: list[str] = []
    for i, deck in enumerate(decks):
        score = scores[i] if i < len(scores) else 0.0
        lines.append(f"=== Deck {i + 1}  score={score:+.4f} ===")
        deck_with_costs = [(card, cost_for(card)) for card in deck]
        deck_with_costs.sort(key=lambda pair: (pair[1] == "", _mana_value(pair[1]), pair[0]))
        for card, cost in deck_with_costs:
            lines.append(f"{card:<32}{cost}".rstrip())
        lines.append("")

    return "\n".join(lines)


def write_decks_file(
    path: Path,
    decks: list[list[str]],
    locator: ConvertedCardLocator,
    scores: list[float],
) -> None:
    path.write_text(format_decks_for_display(decks, locator, scores), encoding="utf-8")


def _build_a_decks(
    model: SetTransformerScorer,
    pools: list[list[str]],
    locator: ConvertedCardLocator,
) -> list[list[str]]:
    """Build one scorer deck per pool via greedy search. Returns list of 40-card decks."""
    a_decks = []
    for pool_names in pools:
        pool_embeddings, valid_names = load_pool_embeddings(pool_names, locator)
        if len(valid_names) < NONLAND_DECK_SIZE:
            continue

        nonland_deck = GreedyDeckBuilder(model, pool_embeddings).build(valid_names)
        a_decks.append(assemble_full_deck(nonland_deck, locator))

    return a_decks


def write_round_robin_matches(
    a_decks: list[list[str]],
    b_decks: list[list[str]],
    n_workers: int,
    work_dir: Path,
) -> list[Path]:
    """Write N² match lines split into per-worker files (row-major A-vs-B order)."""
    lines = [
        "|".join(a) + ";" + "|".join(b)
        for a, b in row_major_pairings(a_decks, b_decks)
    ]
    worker_files: list[Path] = []
    for i, chunk in enumerate(split_evenly(lines, n_workers)):
        worker_file = work_dir / f"validation-matches-{i}.txt"
        worker_file.write_text("\n".join(chunk) + "\n", encoding="utf-8")
        worker_files.append(worker_file)
    return worker_files


_WORKER_FILE_RE = re.compile(r"^validation-matches-(\d+)\.txt$")


def _discover_worker_files(work_dir: Path) -> list[Path]:
    """Return `validation-matches-*.txt` shards in numeric order."""
    matched: list[tuple[int, Path]] = []
    for path in work_dir.iterdir():
        m = _WORKER_FILE_RE.match(path.name)
        if m and path.is_file():
            matched.append((int(m.group(1)), path))
    matched.sort()
    return [p for _, p in matched]


def _infer_n_pools(worker_files: list[Path]) -> int:
    """Infer the round-robin side N from the total match-line count (N² rows)."""
    total = 0
    for f in worker_files:
        with f.open("r", encoding="utf-8") as fh:
            total += sum(1 for line in fh if line.strip())
    n = int(round(total ** 0.5))
    if n * n != total:
        raise RuntimeError(
            f"--resume: match files total {total} rows, which is not a perfect"
            f" square (cannot infer round-robin N)"
        )
    return n

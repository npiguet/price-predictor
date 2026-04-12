"""Evaluation use case: round-robin cross-group deck comparison vs Forge baseline."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from sealed.domain.scorer_model import SetTransformerScorer
from sealed.infrastructure.evaluation_connector import EvaluationConnector
from sealed.infrastructure.scorer_store import ScorerStore


@dataclass
class EvaluateScorerConfig:
    checkpoint: Path
    cards_path: Path = Path("output/cardsfolder/")
    pools: int = 12
    best_of: int = 3
    workers: int = 4
    work_dir: Path | None = None


_MANA_SYMBOL_RE = re.compile(r"\{([WUBRGC])\}")
_COLOR_TO_LAND = {"W": "Plains", "U": "Island", "B": "Swamp", "R": "Mountain", "G": "Forest"}


def greedy_deck_search(
    model: SetTransformerScorer,
    pool_names: list[str],
    pool_embeddings: dict[str, np.ndarray],
) -> list[str]:
    """Build the best 23-card non-land deck from a pool using greedy single-card swaps.

    Starting from 23 random cards, iteratively try all single-card swaps.
    Apply the swap that improves the score the most. Stop when no swap improves.
    """
    model.eval()
    n_nonland = 23

    if len(pool_names) < n_nonland:
        return pool_names[:len(pool_names)]

    # Start with 23 random cards
    import random
    deck = list(random.sample(pool_names, n_nonland))
    remaining = [c for c in pool_names if c not in deck]

    def score_deck(card_list: list[str]) -> float:
        vecs = np.stack([pool_embeddings[c] for c in card_list])
        cards_t = torch.from_numpy(vecs).unsqueeze(0)
        mask_t = torch.ones(1, len(card_list), dtype=torch.bool)
        with torch.no_grad():
            return model(cards_t, mask_t).item()

    current_score = score_deck(deck)

    while True:
        best_swap = None
        best_score = current_score

        for i, deck_card in enumerate(deck):
            for pool_card in remaining:
                candidate = deck[:i] + [pool_card] + deck[i + 1:]
                candidate_score = score_deck(candidate)
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_swap = (i, deck_card, pool_card)

        if best_swap is None:
            break

        i, old_card, new_card = best_swap
        deck[i] = new_card
        remaining.remove(new_card)
        remaining.append(old_card)
        current_score = best_score

    return deck


def compute_basic_lands(nonland_texts: dict[str, str]) -> dict[str, int]:
    """Compute basic land distribution for a 40-card deck.

    Args:
        nonland_texts: mapping of card name → card text for the 23 non-land cards.

    Returns:
        Mapping of basic land name → count, summing to 17 (40 - 23).
    """
    n_basics = 40 - len(nonland_texts)
    pips = {"W": 0, "U": 0, "B": 0, "R": 0, "G": 0}

    for text in nonland_texts.values():
        for line in text.splitlines():
            if line.startswith("mana cost:"):
                for match in _MANA_SYMBOL_RE.finditer(line):
                    sym = match.group(1)
                    if sym in pips:
                        pips[sym] += 1

    total_pips = sum(pips.values())
    if total_pips == 0:
        # Colorless deck: distribute equally
        pips = {"W": 1, "U": 1, "B": 1, "R": 1, "G": 1}
        total_pips = 5

    lands: dict[str, int] = {}
    remaining = n_basics
    remaining_pips = total_pips

    for color in ["W", "U", "B", "R", "G"]:
        if pips[color] == 0 or remaining == 0:
            continue
        if remaining_pips == pips[color]:
            count = remaining
        else:
            count = round(remaining * pips[color] / remaining_pips)
        count = max(0, min(count, remaining))
        if count > 0:
            lands[_COLOR_TO_LAND[color]] = count
        remaining -= count
        remaining_pips -= pips[color]

    return lands


def aggregate_results(outcome_files: list[Path], n_pools: int) -> dict:
    """Aggregate round-robin evaluation outcomes.

    Outcomes are written in row-major order: match index k corresponds to
    A deck k // n_pools vs B deck k % n_pools.

    Returns a dict with:
        n_pools, n_matches, total_games,
        a_win_rates (list[float]), b_win_rates (list[float]),
        pool_deltas (list[float]),
        a_aggregate_win_rate, b_aggregate_win_rate
    """
    # Read all outcomes in order across worker files
    outcomes: list[tuple[int, int]] = []
    for f in outcome_files:
        if not f.exists():
            continue
        for line in f.read_text(encoding="utf-8").strip().splitlines():
            if not line.strip():
                continue
            parts = line.strip().split(";")
            outcomes.append((int(parts[0]), int(parts[1])))

    n_matches = len(outcomes)
    total_games = sum(wa + wb for wa, wb in outcomes)

    if n_pools == 0 or n_matches == 0:
        return {
            "n_pools": n_pools,
            "n_matches": n_matches,
            "total_games": total_games,
            "a_win_rates": [],
            "b_win_rates": [],
            "pool_deltas": [],
            "a_aggregate_win_rate": 0.0,
            "b_aggregate_win_rate": 0.0,
        }

    # Per-A-deck win rate: A_i plays N matches (one per B deck)
    a_wins_total = [0] * n_pools
    a_games_total = [0] * n_pools
    b_wins_total = [0] * n_pools
    b_games_total = [0] * n_pools

    for k, (wa, wb) in enumerate(outcomes):
        i = k // n_pools   # A deck index
        j = k % n_pools    # B deck index
        games = wa + wb
        a_wins_total[i] += wa
        a_games_total[i] += games
        b_wins_total[j] += wb
        b_games_total[j] += games

    a_win_rates = [
        a_wins_total[i] / max(a_games_total[i], 1) for i in range(n_pools)
    ]
    b_win_rates = [
        b_wins_total[j] / max(b_games_total[j], 1) for j in range(n_pools)
    ]
    pool_deltas = [a_win_rates[i] - b_win_rates[i] for i in range(n_pools)]

    total_a_wins = sum(wa for wa, _ in outcomes)
    total_b_wins = sum(wb for _, wb in outcomes)
    a_aggregate = total_a_wins / max(total_games, 1)
    b_aggregate = total_b_wins / max(total_games, 1)

    return {
        "n_pools": n_pools,
        "n_matches": n_matches,
        "total_games": total_games,
        "a_win_rates": a_win_rates,
        "b_win_rates": b_win_rates,
        "pool_deltas": pool_deltas,
        "a_aggregate_win_rate": a_aggregate,
        "b_aggregate_win_rate": b_aggregate,
    }


class EvaluateScorerUseCase:
    """Round-robin evaluation pipeline: build N A-decks and N B-decks, play N² matches."""

    def execute(self, config: EvaluateScorerConfig) -> dict:
        import tempfile

        store = ScorerStore()
        checkpoint = store.load_checkpoint(config.checkpoint)
        model = SetTransformerScorer(**checkpoint["config"])
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        if config.work_dir:
            work_dir = Path(config.work_dir)
        else:
            work_dir = Path(tempfile.mkdtemp(prefix="eval_"))
        work_dir.mkdir(parents=True, exist_ok=True)

        from sealed.infrastructure.pool_connector import PoolConnector
        pools_path = work_dir / "pools"
        pools_path.mkdir(exist_ok=True)

        connector = PoolConnector()
        connector.generate("RVR", config.pools, pools_path)

        pools_file = pools_path / "pools.txt"
        pools = _parse_pools(pools_file)

        print(f"Building {len(pools)} scorer decks (A)...")
        a_decks = _build_a_decks(model, pools, config.cards_path)

        print(f"Building {len(pools)} Forge decks (B)...")
        eval_connector = EvaluationConnector()
        b_decks = eval_connector.build_forge_decks(pools)

        n = len(a_decks)
        print(f"Writing {n}² = {n * n} round-robin match pairings...")
        worker_files = _write_round_robin_matches(
            a_decks, b_decks, config.workers, work_dir,
        )

        print(f"Launching {config.workers} workers (best-of-{config.best_of})...")
        outcome_files = eval_connector.launch_workers(worker_files, best_of=config.best_of)

        result = aggregate_results(outcome_files, n_pools=n)

        _print_results(result)
        return result


def _print_results(result: dict) -> None:
    n = result["n_pools"]
    n_matches = result["n_matches"]
    total_games = result["total_games"]
    a_agg = result["a_aggregate_win_rate"]
    b_agg = result["b_aggregate_win_rate"]
    deltas = result["pool_deltas"]
    a_rates = result["a_win_rates"]
    b_rates = result["b_win_rates"]

    print("\n=== Evaluation Results ===")
    print(f"Pools: {n}  |  Matches: {n_matches}  |  Games: {total_games}")
    print(f"Scorer aggregate win rate: {a_agg:.1%}")
    print(f"Forge  aggregate win rate: {b_agg:.1%}")

    if a_rates:
        print("\nPer-pool comparison (scorer vs Forge from same pool):")
        for i, (ar, br, delta) in enumerate(zip(a_rates, b_rates, deltas)):
            sign = "+" if delta >= 0 else ""
            print(f"  Pool {i + 1:2d}: scorer {ar:.1%}  forge {br:.1%}  delta {sign}{delta:.1%}")
        mean_delta = sum(deltas) / len(deltas)
        sign = "+" if mean_delta >= 0 else ""
        print(f"Mean delta: {sign}{mean_delta:.1%}")


def _parse_pools(pools_file: Path) -> list[list[str]]:
    """Parse a pools.txt file into a list of pools (each pool = list of card names)."""
    pools = []
    for line in pools_file.read_text(encoding="utf-8").strip().splitlines():
        if line.strip():
            pools.append(line.strip().split("|"))
    return pools


def _build_a_decks(
    model: SetTransformerScorer,
    pools: list[list[str]],
    cards_path: Path,
) -> list[list[str]]:
    """Build one scorer deck per pool via greedy search. Returns list of 40-card decks."""
    a_decks = []
    for pool_names in pools:
        pool_embeddings: dict[str, np.ndarray] = {}
        valid_names = []
        for name in pool_names:
            emb = _load_card_embedding(cards_path, name)
            if emb is not None:
                pool_embeddings[name] = emb
                valid_names.append(name)

        if len(valid_names) < 23:
            continue

        nonland_deck = greedy_deck_search(model, valid_names, pool_embeddings)
        nonland_texts = {n: _load_card_text(cards_path, n) for n in nonland_deck}
        lands = compute_basic_lands(nonland_texts)
        full_deck: list[str] = list(nonland_deck)
        for land_name, count in lands.items():
            full_deck.extend([land_name] * count)

        a_decks.append(full_deck)

    return a_decks


def _write_round_robin_matches(
    a_decks: list[list[str]],
    b_decks: list[list[str]],
    n_workers: int,
    work_dir: Path,
) -> list[Path]:
    """Write N² match lines split into per-worker files.

    Match lines are in row-major order: A_0 vs B_0, A_0 vs B_1, ...,
    A_{N-1} vs B_0, ..., A_{N-1} vs B_{N-1}.

    Returns list of per-worker match file paths.
    """
    lines = []
    for a_deck in a_decks:
        for b_deck in b_decks:
            lines.append("|".join(a_deck) + ";" + "|".join(b_deck))

    total = len(lines)
    chunk_size = total // n_workers
    remainder = total % n_workers
    worker_files = []

    offset = 0
    for i in range(n_workers):
        size = chunk_size + (1 if i < remainder else 0)
        worker_file = work_dir / f"validation-matches-{i}.txt"
        worker_file.write_text(
            "\n".join(lines[offset:offset + size]) + "\n",
            encoding="utf-8",
        )
        worker_files.append(worker_file)
        offset += size

    return worker_files


def _sanitize_name(card_name: str) -> str:
    """Convert a card name to its filesystem filename.

    Unicode accents are stripped (e.g. â → a) to match on-disk filenames.
    Known filename typos are corrected via the corrections table.
    """
    import unicodedata

    from sealed.infrastructure.card_name_corrections import FILENAME_CORRECTIONS

    nfkd = unicodedata.normalize("NFKD", card_name)
    ascii_name = "".join(c for c in nfkd if not unicodedata.combining(c))
    sanitized = (
        ascii_name.lower()
        .replace(" // ", "_")
        .replace(" ", "_")
        .replace("'", "")
        .replace(",", "")
        .replace(":", "")
        .replace("!", "")
        .replace('"', "")
        .replace("&", "")
        .replace("+", "")
        .replace(".", "")
        .replace("-", "_")
        .replace("/", "")
    )
    return FILENAME_CORRECTIONS.get(sanitized, sanitized)


def _find_file(cards_path: Path, card_name: str, ext: str) -> Path | None:
    """Resolve card file, falling back to prefix search for DFCs."""
    resolved = _sanitize_name(card_name)
    if "/" in resolved:
        first_letter, filename = resolved.split("/", 1)
    else:
        filename = resolved
        first_letter = filename[0] if filename else "_"
    letter_dir = cards_path / first_letter

    exact = letter_dir / f"{filename}{ext}"
    if exact.exists():
        return exact

    # Prefix search for double-faced / split / adventure cards
    if letter_dir.is_dir():
        prefix = filename + "_"
        for candidate in letter_dir.iterdir():
            if candidate.suffix == ext and candidate.stem.startswith(prefix):
                return candidate
    return None


def _load_card_text(cards_path: Path, card_name: str) -> str:
    """Load a card's text file content."""
    path = _find_file(cards_path, card_name, ".txt")
    if path:
        return path.read_text(encoding="utf-8")
    return ""


def _load_card_embedding(
    cards_path: Path, card_name: str,
) -> np.ndarray | None:
    """Load a card's .npz embedding."""
    path = _find_file(cards_path, card_name, ".npz")
    if path:
        return np.load(path)["embedding"]
    return None

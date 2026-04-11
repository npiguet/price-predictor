"""Evaluation use case: greedy deck search + Forge baseline evaluation."""

from __future__ import annotations

import random
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
    pools: int = 20
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


def aggregate_results(outcome_files: list[Path]) -> dict:
    """Aggregate evaluation outcomes from worker output files."""
    pools_evaluated = 0
    wins_scorer = 0
    wins_forge = 0
    total_games = 0

    for f in outcome_files:
        if not f.exists():
            continue
        for line in f.read_text(encoding="utf-8").strip().splitlines():
            if not line.strip():
                continue
            parts = line.strip().split(";")
            wa, wb = int(parts[0]), int(parts[1])
            pools_evaluated += 1
            wins_scorer += wa
            wins_forge += wb
            total_games += wa + wb

    win_rate = wins_scorer / max(total_games, 1)
    return {
        "pools_evaluated": pools_evaluated,
        "total_games": total_games,
        "wins_scorer": wins_scorer,
        "wins_forge": wins_forge,
        "win_rate": win_rate,
    }


class EvaluateScorerUseCase:
    """Full evaluation pipeline: generate pools, build decks, play matches, report."""

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

        # Load pool data (reuse generate-pools output or generate fresh)
        from sealed.infrastructure.pool_connector import PoolConnector
        pools_path = work_dir / "pools"
        pools_path.mkdir(exist_ok=True)

        connector = PoolConnector()
        connector.generate("RVR", config.pools, pools_path)

        pools_file = pools_path / "pools.txt"
        pools = _parse_pools(pools_file)

        # Build deck A per pool using greedy search
        matches_file = work_dir / "validation-matches.txt"
        _build_validation_matches(model, pools, config.cards_path, matches_file)

        # Launch workers
        eval_connector = EvaluationConnector()
        split_files = eval_connector.split_matches_file(
            matches_file, config.workers, work_dir,
        )
        outcome_files = eval_connector.launch_workers(split_files)

        # Aggregate results
        result = aggregate_results(outcome_files)

        print("\n=== Evaluation Results ===")
        print(f"Pools evaluated: {result['pools_evaluated']}")
        print(f"Total games:     {result['total_games']}")
        print(f"Scorer wins:     {result['wins_scorer']}")
        print(f"Forge wins:      {result['wins_forge']}")
        print(f"Win rate:        {result['win_rate']:.1%}")

        return result


def _parse_pools(pools_file: Path) -> list[list[str]]:
    """Parse a pools.txt file into a list of pools (each pool = list of card names)."""
    pools = []
    for line in pools_file.read_text(encoding="utf-8").strip().splitlines():
        if line.strip():
            pools.append(line.strip().split("|"))
    return pools


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


def _build_validation_matches(
    model: SetTransformerScorer,
    pools: list[list[str]],
    cards_path: Path,
    matches_file: Path,
) -> None:
    """For each pool, build deck A via greedy search and write validation matches."""
    lines = []

    for pool_names in pools:
        # Load embeddings for pool cards
        pool_embeddings: dict[str, np.ndarray] = {}
        valid_names = []
        for name in pool_names:
            emb = _load_card_embedding(cards_path, name)
            if emb is not None:
                pool_embeddings[name] = emb
                valid_names.append(name)

        if len(valid_names) < 23:
            continue

        # Build deck A (23 non-land cards)
        nonland_deck = greedy_deck_search(model, valid_names, pool_embeddings)

        # Compute basic lands
        nonland_texts = {n: _load_card_text(cards_path, n) for n in nonland_deck}
        lands = compute_basic_lands(nonland_texts)
        full_deck: list[str] = list(nonland_deck)
        for land_name, count in lands.items():
            full_deck.extend([land_name] * count)

        # Write match line: deck_A (40 cards) ; pool_B (all pool cards)
        deck_a_str = "|".join(full_deck)
        pool_b_str = "|".join(pool_names)
        lines.append(f"{deck_a_str};{pool_b_str}")

    matches_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

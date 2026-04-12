"""Evaluation use case: round-robin cross-group deck comparison vs Forge baseline."""

from __future__ import annotations

import random
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from sealed.domain.manabase import compute_basic_lands
from sealed.domain.round_robin_results import RoundRobinResults, aggregate_results
from sealed.domain.scorer_model import SetTransformerScorer
from sealed.infrastructure.converted_card_locator import (
    BASIC_LAND_TITLE_NAMES,
    ConvertedCardLocator,
)
from sealed.infrastructure.evaluation_connector import EvaluationConnector
from sealed.infrastructure.pool_connector import PoolConnector
from sealed.infrastructure.scorer_store import ScorerStore

NONLAND_DECK_SIZE = 23


@dataclass
class EvaluateScorerConfig:
    checkpoint: Path
    cards_path: Path
    pools: int = 12
    best_of: int = 3
    workers: int = 4
    work_dir: Path | None = None


def greedy_deck_search(
    model: SetTransformerScorer,
    pool_names: list[str],
    pool_embeddings: dict[str, np.ndarray],
) -> list[str]:
    """Build the best 23-card non-land deck from a pool using greedy single-card swaps.

    Starts from a random 23-card subset, then iteratively scores all
    (position, candidate) swaps in a single batched forward pass and applies
    the best swap if it improves the score. Stops when no swap improves.
    """
    model.eval()
    if len(pool_names) < NONLAND_DECK_SIZE:
        return list(pool_names)

    device = next(model.parameters()).device
    pool_arr = torch.from_numpy(
        np.stack([pool_embeddings[c] for c in pool_names])
    ).to(device)

    perm = list(range(len(pool_names)))
    random.shuffle(perm)
    deck_idx = perm[:NONLAND_DECK_SIZE]
    rem_idx = perm[NONLAND_DECK_SIZE:]

    current_score = _score_indices(model, pool_arr, deck_idx, device)

    while rem_idx:
        cards_batch, mask_batch = _build_swap_batch(
            pool_arr, deck_idx, rem_idx, device,
        )
        with torch.no_grad():
            scores = model(cards_batch, mask_batch).squeeze(-1)

        best_local = int(scores.argmax().item())
        best_score = float(scores[best_local].item())
        if best_score <= current_score:
            break

        r = len(rem_idx)
        i_pos, j_rem = best_local // r, best_local % r
        deck_idx[i_pos], rem_idx[j_rem] = rem_idx[j_rem], deck_idx[i_pos]
        current_score = best_score

    return [pool_names[i] for i in deck_idx]


def _score_indices(
    model: SetTransformerScorer,
    pool_arr: torch.Tensor,
    idx_list: list[int],
    device: torch.device,
) -> float:
    cards_t = pool_arr[torch.tensor(idx_list, device=device)].unsqueeze(0)
    mask_t = torch.ones(1, len(idx_list), dtype=torch.bool, device=device)
    with torch.no_grad():
        return model(cards_t, mask_t).item()


def _build_swap_batch(
    pool_arr: torch.Tensor,
    deck_idx: list[int],
    rem_idx: list[int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the (n_nonland*R, n_nonland) candidate matrix for one swap step.

    Row ``k`` corresponds to position ``k // R`` of the current deck being
    replaced by remaining card ``k % R``.
    """
    n = NONLAND_DECK_SIZE
    r = len(rem_idx)
    deck_t = torch.tensor(deck_idx, device=device)
    rem_t = torch.tensor(rem_idx, device=device)

    batch = deck_t.unsqueeze(0).expand(n * r, -1).clone()
    positions = torch.arange(n, device=device).repeat_interleave(r)
    replacements = rem_t.repeat(n)
    rows = torch.arange(n * r, device=device)
    batch[rows, positions] = replacements

    cards_batch = pool_arr[batch]
    mask_batch = torch.ones(n * r, n, dtype=torch.bool, device=device)
    return cards_batch, mask_batch


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
    max_len = max((len(e) for e in deck_embeddings), default=0)
    if max_len == 0:
        return [0.0] * len(decks)

    d_model = next(e[0].shape[0] for e in deck_embeddings if e)
    n_decks = len(decks)
    cards_arr = np.zeros((n_decks, max_len, d_model), dtype=np.float32)
    mask_arr = np.zeros((n_decks, max_len), dtype=bool)
    for i, embeds in enumerate(deck_embeddings):
        for j, emb in enumerate(embeds):
            cards_arr[i, j] = emb
        if embeds:
            mask_arr[i, :len(embeds)] = True

    cards_t = torch.from_numpy(cards_arr).to(device)
    mask_t = torch.from_numpy(mask_arr).to(device)
    with torch.no_grad():
        out = model(cards_t, mask_t).squeeze(-1)
    return out.cpu().tolist()


def _load_nonland_embeddings(
    deck: list[str], locator: ConvertedCardLocator,
) -> list[np.ndarray]:
    embeds: list[np.ndarray] = []
    for name in deck:
        if name in BASIC_LAND_TITLE_NAMES:
            continue
        emb = locator.load_embedding(name)
        if emb is not None:
            embeds.append(emb)
    return embeds


class EvaluateScorerUseCase:
    """Round-robin evaluation pipeline: build N A-decks and N B-decks, play N² matches."""

    def execute(self, config: EvaluateScorerConfig) -> RoundRobinResults:
        model = self._load_model(config.checkpoint)
        locator = ConvertedCardLocator(config.cards_path)
        work_dir = self._prepare_work_dir(config.work_dir)

        pools = self._generate_pools(config.pools, work_dir)

        print(f"Building {len(pools)} scorer decks (A)...")
        a_decks = _build_a_decks(model, pools, locator)

        print(f"Building {len(pools)} Forge decks (B)...")
        eval_connector = EvaluationConnector()
        b_decks = eval_connector.build_forge_decks(pools)

        self._dump_decks(work_dir, model, a_decks, b_decks, locator)

        n = len(a_decks)
        print(f"Writing {n}² = {n * n} round-robin match pairings...")
        worker_files = _write_round_robin_matches(
            a_decks, b_decks, config.workers, work_dir,
        )

        print(f"Launching {config.workers} workers (best-of-{config.best_of})...")
        outcome_files = eval_connector.launch_workers(
            worker_files, best_of=config.best_of,
        )

        result = aggregate_results(outcome_files, n_pools=n)
        print(result.format_report())
        return result

    def _load_model(self, checkpoint_path: Path) -> SetTransformerScorer:
        store = ScorerStore()
        checkpoint = store.load_checkpoint(checkpoint_path)
        model = SetTransformerScorer(checkpoint["config"])
        model.load_state_dict(checkpoint["model_state_dict"])
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

    def _generate_pools(self, n_pools: int, work_dir: Path) -> list[list[str]]:
        pools_path = work_dir / "pools"
        pools_path.mkdir(exist_ok=True)
        PoolConnector().generate("RVR", n_pools, pools_path)
        return _parse_pools(pools_path / "pools.txt")

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
        _write_decks_file(work_dir / "decks-scorer.txt", a_decks, locator, a_scores)
        _write_decks_file(work_dir / "decks-forge.txt", b_decks, locator, b_scores)


def _write_decks_file(
    path: Path,
    decks: list[list[str]],
    locator: ConvertedCardLocator,
    scores: list[float],
) -> None:
    """Write decks for human inspection: one card per line with mana cost,
    blank-line-separated, with a `=== Deck N (score=X.XXXX) ===` header."""
    cost_cache: dict[str, str] = {}

    def cost_for(name: str) -> str:
        if name not in cost_cache:
            cost_cache[name] = _extract_mana_cost(locator.load_text(name))
        return cost_cache[name]

    lines: list[str] = []
    for i, deck in enumerate(decks):
        score = scores[i] if i < len(scores) else 0.0
        lines.append(f"=== Deck {i + 1}  score={score:+.4f} ===")
        for card in deck:
            cost = cost_for(card)
            lines.append(f"{card:<32}{cost}".rstrip())
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def _extract_mana_cost(card_text: str) -> str:
    """Return the mana cost string (e.g. `{1}{R}`) from a card text file, or ''."""
    for line in card_text.splitlines():
        if line.startswith("mana cost:"):
            return line[len("mana cost:"):].strip()
    return ""


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
    locator: ConvertedCardLocator,
) -> list[list[str]]:
    """Build one scorer deck per pool via greedy search. Returns list of 40-card decks."""
    a_decks = []
    for pool_names in pools:
        pool_embeddings: dict[str, np.ndarray] = {}
        valid_names = []
        for name in pool_names:
            emb = locator.load_embedding(name)
            if emb is not None:
                pool_embeddings[name] = emb
                valid_names.append(name)

        if len(valid_names) < NONLAND_DECK_SIZE:
            continue

        nonland_deck = greedy_deck_search(model, valid_names, pool_embeddings)
        nonland_texts = [locator.load_text(n) for n in nonland_deck]
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
    """Write N² match lines split into per-worker files (row-major A-vs-B order)."""
    lines = [
        "|".join(a_deck) + ";" + "|".join(b_deck)
        for a_deck in a_decks
        for b_deck in b_decks
    ]
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

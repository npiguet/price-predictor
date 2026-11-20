"""The ``train-picker`` reward path: the frozen scorer, loaded once, scoring
every sampled deck and every deterministic chosen deck.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from sealed.application.picker_pools import PreparedPool
from sealed.application.picker_sampling import SampledDecks
from sealed.domain.scorer_model import SetTransformerScorer
from sealed.infrastructure.scorer_store import ScorerStore

_SCORE_CHUNK = 256  # sub-batch size for frozen-scorer forwards over many decks.


def _select_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_scorer(path: Path, device: torch.device) -> SetTransformerScorer:
    checkpoint = ScorerStore().load_checkpoint(path)
    model = SetTransformerScorer(checkpoint.config)
    model.load_state_dict(checkpoint.model_state_dict)
    model.eval()
    model.to(device)
    return model


def _score_sampled(
    scorer: SetTransformerScorer,
    norm_pool: torch.Tensor,
    decks: SampledDecks,
    device: torch.device,
) -> torch.Tensor:
    """Score every sampled deck with the frozen scorer (FR-012). Returns (B, S)."""
    pick_indices, picked_mask, n_samples = decks.pick_indices, decks.picked_mask, decks.n_samples
    b, max_n, dim = norm_pool.shape
    bs = b * n_samples
    k = pick_indices.shape[1]
    pool_bs = norm_pool.unsqueeze(1).expand(b, n_samples, max_n, dim).reshape(bs, max_n, dim)
    gather_idx = pick_indices.unsqueeze(-1).expand(bs, k, dim)
    deck_cards = torch.gather(pool_bs, 1, gather_idx)
    scores = torch.empty(bs, device=device)
    with torch.no_grad(), torch.autocast(
        device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda"),
    ):
        for start in range(0, bs, _SCORE_CHUNK):
            stop = min(start + _SCORE_CHUNK, bs)
            out = scorer.forward_prenormalized(
                deck_cards[start:stop], picked_mask[start:stop],
            ).squeeze(-1)
            scores[start:stop] = out.float()
    return scores.view(b, n_samples)


def _score_chosen_decks(
    scorer: SetTransformerScorer,
    pools: list[PreparedPool],
    chosen: list[list[int]],
    device: torch.device,
) -> np.ndarray:
    """Score deterministic chosen-card decks. Returns (n_decks,) numpy scores."""
    if not pools:
        return np.zeros(0, dtype=np.float32)
    dim = pools[0].dim
    n = len(pools)
    scores = np.zeros(n, dtype=np.float32)
    for start in range(0, n, _SCORE_CHUNK):
        stop = min(start + _SCORE_CHUNK, n)
        group = list(range(start, stop))
        max_k = max(len(chosen[i]) for i in group)
        cards = np.zeros((len(group), max_k, dim), dtype=np.float32)
        mask = np.zeros((len(group), max_k), dtype=bool)
        for row, i in enumerate(group):
            idxs = chosen[i]
            cards[row, :len(idxs)] = pools[i].embeddings[idxs]
            mask[row, :len(idxs)] = True
        cards_t = torch.from_numpy(cards).to(device)
        mask_t = torch.from_numpy(mask).to(device)
        with torch.no_grad(), torch.autocast(
            device_type=device.type, dtype=torch.float16,
            enabled=(device.type == "cuda"),
        ):
            norm = scorer.normalize_features(cards_t)
            out = scorer.forward_prenormalized(norm, mask_t).squeeze(-1)
        scores[start:stop] = out.float().cpu().numpy()
    return scores

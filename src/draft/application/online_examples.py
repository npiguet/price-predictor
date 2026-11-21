"""One gen-3 learner pick as a training example, and its padded, device-resident
batch and forward pass.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from draft.domain.draft_agent_model import DraftAgentModel
from draft.domain.draft_state import TYPE_PACK

# --------------------------------------------------------------------------- #
# Training examples (data-model §3) — a trimmed sibling of RLExample
# --------------------------------------------------------------------------- #

@dataclass(slots=True)
class OnlineExample:
    """One learner ``(draft, seat, pack, pick)`` state.

    Trimmed against ``rl_examples.RLExample``: no critic target, no GAE
    value cache, and no ``learner_active`` flag — non-learner picks are never
    materialised at all, since nothing but the policy gradient consumes them.
    """

    card_idx: np.ndarray      # (N,) int32, rows into RoundBatch.table
    type_idx: np.ndarray      # (N,) int8, TYPE_*
    packs_ago: np.ndarray     # (N,) int8
    pick_ago: np.ndarray      # (N,) int8
    pack_number: int          # 1-based
    pick_number: int          # 1-based
    action_token: int         # absolute index of the taken PACK token
    advantage: float = 0.0    # the seat's shared, detached A (§4)

    @property
    def n_tokens(self) -> int:
        return int(self.card_idx.shape[0])


@dataclass
class _Batch:
    card_emb: torch.Tensor
    type_idx: torch.Tensor
    packs_ago: torch.Tensor
    pick_ago: torch.Tensor
    card_mask: torch.Tensor
    pack_mask: torch.Tensor
    pack_number: torch.Tensor
    pick_number: torch.Tensor
    action_token: torch.Tensor
    advantage: torch.Tensor


def _collate(
    batch: list[OnlineExample], table: np.ndarray, device: torch.device,
) -> _Batch:
    """Pad to the batch's longest example and place every tensor on ``device``.

    Padding is per batch (length bucketing keeps it small) and the whole batch
    moves in a handful of transfers — never one per item (Principle VIII).
    """
    b = len(batch)
    max_n = max(ex.n_tokens for ex in batch)
    dim = table.shape[1]
    card_emb = np.zeros((b, max_n, dim), dtype=np.float32)
    type_idx = np.zeros((b, max_n), dtype=np.int64)
    packs_ago = np.zeros((b, max_n), dtype=np.int64)
    pick_ago = np.zeros((b, max_n), dtype=np.int64)
    card_mask = np.zeros((b, max_n), dtype=bool)
    pack_mask = np.zeros((b, max_n), dtype=bool)
    for i, ex in enumerate(batch):
        n = ex.n_tokens
        card_emb[i, :n] = table[ex.card_idx]
        type_idx[i, :n] = ex.type_idx
        packs_ago[i, :n] = ex.packs_ago
        pick_ago[i, :n] = ex.pick_ago
        card_mask[i, :n] = True
        pack_mask[i, :n] = ex.type_idx == TYPE_PACK
    return _Batch(
        card_emb=torch.from_numpy(card_emb).to(device),
        type_idx=torch.from_numpy(type_idx).to(device),
        packs_ago=torch.from_numpy(packs_ago).to(device),
        pick_ago=torch.from_numpy(pick_ago).to(device),
        card_mask=torch.from_numpy(card_mask).to(device),
        pack_mask=torch.from_numpy(pack_mask).to(device),
        pack_number=torch.tensor([ex.pack_number for ex in batch], device=device),
        pick_number=torch.tensor([ex.pick_number for ex in batch], device=device),
        action_token=torch.tensor([ex.action_token for ex in batch], device=device),
        advantage=torch.tensor(
            [ex.advantage for ex in batch], device=device, dtype=torch.float32,
        ),
    )


def _forward(model: DraftAgentModel, batch: _Batch) -> tuple[torch.Tensor, torch.Tensor]:
    return model(
        batch.card_emb, batch.type_idx, batch.packs_ago, batch.pick_ago,
        batch.card_mask, batch.pack_number, batch.pick_number,
    )

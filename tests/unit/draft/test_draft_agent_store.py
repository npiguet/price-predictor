"""Draft-disjoint split + checkpoint round-trip + arch-flag guard (FR-035, FR-039, SC-006)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from draft.application.train_draft_agent import (
    DraftExample,
    split_draft_disjoint,
)
from draft.domain.draft_agent_model import DraftAgentConfig, DraftAgentModel
from draft.domain.draft_state import NUM_TYPES
from draft.infrastructure.cli import build_parser, run_train_draft_agent
from draft.infrastructure.draft_agent_store import DraftAgentStore


def _example(draft_index: int) -> DraftExample:
    return DraftExample(
        draft_index=draft_index,
        card_idx=np.zeros(2, dtype=np.int32),
        type_idx=np.array([1, 1], dtype=np.int8),
        packs_ago=np.zeros(2, dtype=np.int8),
        pick_ago=np.zeros(2, dtype=np.int8),
        pack_number=1,
        pick_number=1,
        imitation_active=True,
        target_token=0,
        critic_active=True,
        critic_target=1.0,
    )


def test_split_is_draft_disjoint() -> None:
    # 10 drafts, 3 examples each.
    examples = [_example(i) for i in range(10) for _ in range(3)]
    train, val = split_draft_disjoint(examples, val_fraction=0.2, seed=42)
    train_ids = {ex.draft_index for ex in train}
    val_ids = {ex.draft_index for ex in val}
    assert train_ids.isdisjoint(val_ids)         # no draft straddles the split
    assert len(val_ids) == 2                      # 0.2 * 10 distinct ids
    assert len(train) + len(val) == 30


def test_split_is_deterministic_with_seed() -> None:
    examples = [_example(i) for i in range(10) for _ in range(2)]
    a = split_draft_disjoint(examples, 0.2, seed=42)
    b = split_draft_disjoint(examples, 0.2, seed=42)
    assert {e.draft_index for e in a[1]} == {e.draft_index for e in b[1]}


def test_checkpoint_round_trip_reloads_to_predict(tmp_path: Path) -> None:
    cfg = DraftAgentConfig(embedding_dim=8, packs=3, P=15)
    model = DraftAgentModel(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    store = DraftAgentStore()
    path = tmp_path / "agent.pt"
    store.save_checkpoint(
        model, optimizer, epoch=4, best_val_loss=0.5, config=cfg, path=path,
        critic_mean=2.5, critic_std=1.5, train_config={"lr": 1e-3},
    )

    loaded = store.load_checkpoint(path)
    assert loaded.epoch == 4
    assert loaded.best_val_loss == 0.5
    assert loaded.critic_mean == 2.5
    assert loaded.critic_std == 1.5
    assert loaded.config.embedding_dim == 8

    reloaded = DraftAgentModel(loaded.config)
    reloaded.load_state_dict(loaded.model_state_dict)
    reloaded.eval()
    n = 5
    with torch.no_grad():
        logits, critic = reloaded(
            torch.randn(1, n, 8),
            torch.randint(0, NUM_TYPES, (1, n)),
            torch.zeros(1, n, dtype=torch.long),
            torch.zeros(1, n, dtype=torch.long),
            torch.ones(1, n, dtype=torch.bool),
            torch.tensor([1]),
            torch.tensor([1]),
        )
    assert logits.shape == (1, n)
    assert critic.shape == (1,)


def test_arch_flags_rejected_with_resume() -> None:
    parser = build_parser()
    args = parser.parse_args(
        ["train-draft-agent", "--resume", "x.pt", "--n-layers", "6"],
    )
    assert run_train_draft_agent(args) == 2


def test_resume_and_checkpoint_mutually_exclusive() -> None:
    parser = build_parser()
    args = parser.parse_args(
        ["train-draft-agent", "--resume", "a.pt", "--checkpoint", "b.pt"],
    )
    assert run_train_draft_agent(args) == 2

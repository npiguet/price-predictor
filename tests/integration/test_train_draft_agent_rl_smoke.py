"""End-to-end smoke test for gen-2 RL fine-tuning (spec 020, US1 independent test).

Trains one tiny on-policy corpus for a single epoch from a tiny reference
checkpoint (CPU, faked card embeddings) and asserts the candidate checkpoint is
written with RL metadata and a finite held-out RL objective.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from draft.domain.draft_agent_model import DraftAgentConfig, DraftAgentModel
from draft.domain.draft_geometry import Booster, DraftRecord, Seat
from draft.infrastructure.draft_agent_store import DraftAgentStore
from draft.infrastructure.draft_record_io import format_record_line

pytestmark = pytest.mark.integration

DIM = 8  # concat_width 24 is divisible by the default 8 heads


class _FakeLocator:
    def __init__(self, _path) -> None:
        self._tbl: dict[str, np.ndarray] = {}

    def load_embedding(self, name: str) -> np.ndarray:
        if name not in self._tbl:
            self._tbl[name] = np.random.default_rng(
                abs(hash(name)) % 2**32,
            ).standard_normal(DIM).astype(np.float32)
        return self._tbl[name]


def _write_corpus(path: Path, n_drafts: int) -> None:
    lines = []
    for d in range(n_drafts):
        seats = [Seat("gen-k", [], 2.0 + d), Seat("forge-full", [], 1.0)]
        boosters = [
            Booster("TST", [f"d{d}_k{k}_{j}" for j in range(4)]) for k in range(2)
        ]
        rec = DraftRecord(f"draft{d}", "run", "2026-06-14T00:00:00Z", seats, boosters)
        lines.append(format_record_line(rec))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_rl_training_writes_gen2_checkpoint(tmp_path, monkeypatch) -> None:
    from draft.application import train_draft_agent_rl as rl

    # Tiny gen-1 reference checkpoint.
    cfg = DraftAgentConfig(embedding_dim=DIM, packs=1, P=4)
    model = DraftAgentModel(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ref_path = tmp_path / "gen1.pt"
    DraftAgentStore().save_checkpoint(
        model, optimizer, epoch=0, best_val_loss=0.0, config=cfg, path=ref_path,
        critic_mean=1.5, critic_std=1.0,
    )

    corpus = tmp_path / "drafts-gen1.jsonl"
    _write_corpus(corpus, n_drafts=6)

    # Fake the card-embedding cache and write checkpoints under tmp.
    monkeypatch.setattr(rl, "ConvertedCardLocator", _FakeLocator)
    monkeypatch.chdir(tmp_path)

    config = rl.TrainDraftAgentRLConfig(
        checkpoint=ref_path,
        drafts_path=corpus,
        cards_path=tmp_path,
        learner_agents=("gen-k",),
        rollout_temperature=1.0,
        epochs=1,
        val_fraction=0.34,        # 6 drafts → 2 val, 4 train
        batch_size=4,
        evals_per_epoch=1,
    )
    result = rl.TrainDraftAgentRLUseCase().execute(config)

    assert result.generation == 2
    assert result.best_path.exists()
    assert np.isfinite(result.best_val_loss)

    loaded = DraftAgentStore().load_checkpoint(result.best_path)
    assert loaded.rl_metadata is not None
    assert loaded.rl_metadata["generation"] == 2
    assert loaded.rl_metadata["algorithm"] == "reinforce+gae"
    assert loaded.rl_metadata["rollout_temperature"] == 1.0
    # Critic standardization is inherited from the reference (research D7).
    assert loaded.critic_mean == 1.5
    # latest.pt is also written.
    assert (tmp_path / "models" / "draft" / "agent" / "latest.pt").exists()

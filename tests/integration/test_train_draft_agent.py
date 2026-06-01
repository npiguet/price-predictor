"""Integration smoke: train-draft-agent to completion on a tiny fixture corpus."""

from __future__ import annotations

import string
from pathlib import Path

import numpy as np
import pytest
import torch

from draft.application.train_draft_agent import (
    TrainDraftAgentConfig,
    TrainDraftAgentUseCase,
)
from draft.domain.draft_agent_model import DraftAgentModel
from draft.domain.draft_geometry import Booster, DraftRecord, Seat
from draft.domain.draft_state import NUM_TYPES
from draft.infrastructure.draft_agent_store import DraftAgentStore
from draft.infrastructure.draft_record_io import append_record

pytestmark = pytest.mark.integration

_EMB_DIM = 8
_POD = 4
_PACK_SIZE = 3
_CARD_NAMES = [f"card{a}{b}" for a in string.ascii_lowercase[:4]
               for b in string.ascii_lowercase[:6]]  # 24 distinct names


def _write_embeddings(cards_path: Path) -> None:
    rng = np.random.default_rng(0)
    for name in _CARD_NAMES:
        letter = name[0]
        d = cards_path / letter
        d.mkdir(parents=True, exist_ok=True)
        vec = rng.standard_normal(_EMB_DIM).astype(np.float32)
        np.savez(d / f"{name}.npz", embedding=vec)


def _make_record(draft_id: str, rng: np.random.Generator) -> DraftRecord:
    n_boosters = _POD * 1  # one pack
    picks = rng.choice(_CARD_NAMES, size=(n_boosters, _PACK_SIZE), replace=True)
    boosters = [Booster("TST", list(row)) for row in picks]
    seats = [
        Seat("forge-full", [_CARD_NAMES[0]] * 40, float(rng.uniform(-5, 5)))
        for _ in range(_POD)
    ]
    return DraftRecord(draft_id, "run-1", "2026-06-01T00:00:00Z", seats, boosters)


def test_train_draft_agent_smoke(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    cards_path = tmp_path / "cards"
    _write_embeddings(cards_path)

    drafts_path = tmp_path / "drafts.jsonl"
    rng = np.random.default_rng(1)
    with open(drafts_path, "a", encoding="utf-8") as out:
        for i in range(8):
            append_record(out, _make_record(f"d{i}", rng))

    config = TrainDraftAgentConfig(
        drafts_path=drafts_path,
        cards_path=cards_path,
        epochs=3,
        batch_size=16,
        val_fraction=0.25,
        patience=10,
    )
    result = TrainDraftAgentUseCase().execute(config)

    latest = tmp_path / "models" / "draft" / "agent" / "latest.pt"
    assert latest.exists()
    assert result.best_path.exists()

    # The best checkpoint reloads and produces picks + critic on a held-out state.
    loaded = DraftAgentStore().load_checkpoint(result.best_path)
    model = DraftAgentModel(loaded.config)
    model.load_state_dict(loaded.model_state_dict)
    model.eval()
    n = 4
    with torch.no_grad():
        logits, critic = model(
            torch.randn(1, n, _EMB_DIM),
            torch.randint(0, NUM_TYPES, (1, n)),
            torch.zeros(1, n, dtype=torch.long),
            torch.zeros(1, n, dtype=torch.long),
            torch.ones(1, n, dtype=torch.bool),
            torch.tensor([1]),
            torch.tensor([1]),
        )
    assert logits.shape == (1, n)
    assert critic.shape == (1,)

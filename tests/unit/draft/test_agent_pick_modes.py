"""Pick-mode determinism + rival-checkpoint binding (US3, SC-007, T017)."""

from __future__ import annotations

import numpy as np
import torch

from draft.application.agent_pick_service import AgentPickService
from draft.application.agent_registry import AgentRegistry
from draft.application.generate_draft_data import PickRequest
from draft.domain.draft_agent_model import DraftAgentConfig, DraftAgentModel
from draft.infrastructure.cli import build_parser, run_generate_draft_data
from draft.infrastructure.draft_agent_store import DraftAgentStore

EMB_DIM = 8
P = 6


class _FakeLocator:
    def __init__(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        self._embs: dict[str, np.ndarray] = {}
        self._rng = rng

    def load_embedding(self, name: str):
        if name not in self._embs:
            self._embs[name] = self._rng.standard_normal(EMB_DIM).astype(np.float32)
        return self._embs[name].copy()


def _checkpoint(tmp_path, name, *, init_seed=0):
    torch.manual_seed(init_seed)
    config = DraftAgentConfig(embedding_dim=EMB_DIM, packs=3, P=P, n_layers=1, n_heads=1)
    model = DraftAgentModel(config)
    optimizer = torch.optim.AdamW(model.parameters())
    path = tmp_path / name
    DraftAgentStore().save_checkpoint(
        model, optimizer, epoch=0, best_val_loss=0.0, config=config, path=path,
        critic_mean=0.0, critic_std=1.0,
    )
    return path


def _requests():
    pack = ["A", "B", "C", "D", "E"]
    return [
        PickRequest("d1", 0, "draft-agent", 8, 1, i, "TST", list(pack[: 5 - (i - 1)]))
        for i in range(1, 4)
    ]


def test_sample_mode_is_reproducible_under_same_seed(tmp_path) -> None:
    ckpt = _checkpoint(tmp_path, "a.pt")
    # A shared locator gives both services identical embeddings; same seed ⇒
    # identical multinomial draws over the same logits (SC-007, FR-005).
    loc1, loc2 = _FakeLocator(7), _FakeLocator(7)
    svc1 = AgentPickService(ckpt, loc1, pick_mode="sample", temperature=1.5, seed=123)
    svc2 = AgentPickService(ckpt, loc2, pick_mode="sample", temperature=1.5, seed=123)
    picks1 = [svc1.pick(r) for r in _requests()]
    picks2 = [svc2.pick(r) for r in _requests()]
    assert picks1 == picks2


def test_temperature_must_be_positive_with_sample_mode() -> None:
    args = build_parser().parse_args(
        ["generate-draft-data", "--n-drafts", "1", "--pick-mode", "sample",
         "--temperature", "0"]
    )
    assert run_generate_draft_data(args) == 2


def test_distinct_labels_build_distinct_services(tmp_path) -> None:
    a = _checkpoint(tmp_path, "a.pt", init_seed=1)
    b = _checkpoint(tmp_path, "b.pt", init_seed=2)
    registry = AgentRegistry.build(
        {"a": a, "b": b}, {"forge-full", "a", "b"}, locator=_FakeLocator(0),
    )
    assert registry.external_labels == frozenset({"a", "b"})
    assert registry._services["a"] is not registry._services["b"]

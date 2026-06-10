"""AgentPickService: masked argmax pick, un-embeddable handling, faults (T006)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from draft.application.agent_pick_service import AgentPickService, PickFault
from draft.application.generate_draft_data import PickRequest
from draft.domain.draft_agent_model import DraftAgentConfig, DraftAgentModel
from draft.infrastructure.draft_agent_store import DraftAgentStore

EMB_DIM = 8
P = 4


class _FakeLocator:
    """Returns a fixed random embedding for known cards, None otherwise."""

    def __init__(self, embeddable: set[str]) -> None:
        rng = np.random.default_rng(0)
        self._embs = {name: rng.standard_normal(EMB_DIM).astype(np.float32) for name in embeddable}

    def load_embedding(self, name: str):
        emb = self._embs.get(name)
        return None if emb is None else emb.copy()


def _checkpoint(tmp_path):
    config = DraftAgentConfig(embedding_dim=EMB_DIM, packs=3, P=P, n_layers=1, n_heads=1)
    model = DraftAgentModel(config)
    optimizer = torch.optim.AdamW(model.parameters())
    path = tmp_path / "agent.pt"
    DraftAgentStore().save_checkpoint(
        model, optimizer, epoch=0, best_val_loss=0.0, config=config, path=path,
        critic_mean=0.0, critic_std=1.0,
    )
    return path


def _request(pack, *, pick_number=1):
    return PickRequest(
        draft_id="d1", seat=0, agent="draft-agent", pod_size=8,
        pack_number=1, pick_number=pick_number, set_code="TST", pack=pack,
    )


def test_argmax_returns_a_held_pack_card(tmp_path) -> None:
    pack = ["A", "B", "C", "D"]
    service = AgentPickService(_checkpoint(tmp_path), _FakeLocator(set(pack)))
    chosen = service.pick(_request(pack))
    assert chosen in pack


def test_individual_unembeddable_card_is_dropped_not_a_fault(tmp_path) -> None:
    pack = ["A", "B", "MISSING", "C"]
    # "MISSING" has no embedding; the pick must still succeed among the rest.
    service = AgentPickService(_checkpoint(tmp_path), _FakeLocator({"A", "B", "C"}))
    chosen = service.pick(_request(pack))
    assert chosen in {"A", "B", "C"}
    assert chosen != "MISSING"


def test_entirely_unembeddable_pack_is_a_fault(tmp_path) -> None:
    pack = ["X", "Y", "Z"]
    service = AgentPickService(_checkpoint(tmp_path), _FakeLocator(set()))
    with pytest.raises(PickFault):
        service.pick(_request(pack))


def test_pick_number_beyond_checkpoint_capacity_is_a_fault(tmp_path) -> None:
    pack = ["A", "B"]
    service = AgentPickService(_checkpoint(tmp_path), _FakeLocator(set(pack)))
    with pytest.raises(PickFault):
        service.pick(_request(pack, pick_number=P + 1))

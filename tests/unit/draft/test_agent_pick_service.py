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


# ── from_model: the live in-training policy pilots the learner seats ──────────
# (contracts/rollout-stream.md § 2, research D3, spec 021 FR-012)

def _live_model():
    config = DraftAgentConfig(embedding_dim=EMB_DIM, packs=3, P=P, n_layers=1, n_heads=1)
    torch.manual_seed(0)
    model = DraftAgentModel(config)
    model.eval()
    return model, config


def _from_model(model, config, locator, **overrides):
    kwargs = {"device": torch.device("cpu"), "pick_mode": "argmax"}
    kwargs.update(overrides)
    return AgentPickService.from_model(model, config, locator, **kwargs)


def test_from_model_picks_like_the_path_loading_constructor(tmp_path) -> None:
    """Same weights in, same pick out — from_model only changes where they come from."""
    pack = ["A", "B", "C", "D"]
    locator = _FakeLocator(set(pack))
    path = _checkpoint(tmp_path)

    loaded = AgentPickService(path, locator, device=torch.device("cpu"))
    ckpt = DraftAgentStore().load_checkpoint(path)
    model = DraftAgentModel(ckpt.config)
    model.load_state_dict(ckpt.model_state_dict)
    model.eval()
    wrapped = _from_model(model, ckpt.config, locator)

    assert wrapped.pick(_request(pack)) == loaded.pick(_request(pack))
    assert wrapped.config.embedding_dim == ckpt.config.embedding_dim


def test_from_model_sees_a_weight_change_on_the_very_next_pick() -> None:
    """The service holds the model by reference — FR-012 needs no explicit push."""
    pack = ["A", "B", "C", "D"]
    model, config = _live_model()
    service = _from_model(model, config, _FakeLocator(set(pack)))

    before = service.pick(_request(pack))
    # Drive the policy head hard toward a different card, with no push of any kind.
    changed = None
    with torch.no_grad():
        for _ in range(50):
            for p in model.policy_head.parameters():
                p.add_(torch.randn_like(p))
            after = service.pick(_request(pack))
            if after != before:
                changed = after
                break
    assert changed is not None, "a weight change must be visible without a push"


def test_from_model_does_not_touch_mode_or_device() -> None:
    """The caller owns the model: the online trainer flips train()/eval() itself."""
    model, config = _live_model()
    model.train()  # the trainer's update phase

    calls: list[str] = []
    original_eval, original_train, original_to = model.eval, model.train, model.to
    model.eval = lambda *a, **k: (calls.append("eval"), original_eval(*a, **k))[1]
    model.train = lambda *a, **k: (calls.append("train"), original_train(*a, **k))[1]
    model.to = lambda *a, **k: (calls.append("to"), original_to(*a, **k))[1]

    _from_model(model, config, _FakeLocator({"A"}))

    assert calls == []
    assert model.training, "from_model must leave the caller's mode alone"


def test_from_model_honours_pick_mode_and_seed() -> None:
    pack = ["A", "B", "C", "D"]
    model, config = _live_model()
    locator = _FakeLocator(set(pack))

    a = _from_model(model, config, locator, pick_mode="sample", temperature=2.0, seed=7)
    b = _from_model(model, config, locator, pick_mode="sample", temperature=2.0, seed=7)

    picks_a = [a.pick(_request(pack)) for _ in range(6)]
    picks_b = [b.pick(_request(pack)) for _ in range(6)]
    assert picks_a == picks_b, "identically seeded services reproduce picks"
    assert all(pick in pack for pick in picks_a)


def test_from_model_rejects_an_unknown_pick_mode() -> None:
    model, config = _live_model()
    with pytest.raises(ValueError):
        _from_model(model, config, _FakeLocator({"A"}), pick_mode="beam")


def test_from_model_faults_on_an_entirely_unembeddable_pack() -> None:
    model, config = _live_model()
    service = _from_model(model, config, _FakeLocator(set()))
    with pytest.raises(PickFault):
        service.pick(_request(["X", "Y"]))

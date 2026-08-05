"""Per-category pick modes: the learner samples at -T, frozen agents play argmax.

Covers spec FR-004 / FR-021 and research D5. Sampling a frozen agent would not
merely lower its own score — a draft allocates one fixed pool of cards, so a card
it wrongly declines flows downstream to its podmates, the learner among them. A
sampled field hands the learner cards a properly-playing agent would have kept.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from draft.application import train_draft_agent_online as online
from draft.application.train_draft_agent_online import (
    TrainDraftAgentOnlineConfig,
    TrainDraftAgentOnlineUseCase,
)
from draft.domain.draft_agent_model import DraftAgentConfig, DraftAgentModel
from draft.infrastructure.draft_agent_store import DraftAgentStore

EMB_DIM = 8
LEARNER = "gen-3"
ANCHOR = "gen-1"
TEMPERATURE = 3.0


class _Captured(Exception):
    """Abort ``execute`` once the registry call has been recorded."""


@pytest.fixture
def cards_path(tmp_path: Path) -> Path:
    folder = tmp_path / "cardsfolder"
    folder.mkdir()
    rng = np.random.default_rng(0)
    for name in ("c0", "c1", "c2"):
        np.savez(
            folder / f"{name}.npz",
            embedding=rng.standard_normal(EMB_DIM).astype(np.float32),
        )
    return folder


@pytest.fixture
def base_checkpoint(tmp_path: Path) -> Path:
    config = DraftAgentConfig(
        embedding_dim=EMB_DIM, packs=3, P=4, n_layers=1, n_heads=1,
    )
    model = DraftAgentModel(config)
    optimizer = torch.optim.AdamW(model.parameters())
    path = tmp_path / "gen1.pt"
    DraftAgentStore().save_checkpoint(
        model, optimizer, 0, 0.0, config, path, critic_mean=0.0, critic_std=1.0,
    )
    return path


@pytest.fixture
def registry_call(
    tmp_path: Path, cards_path: Path, base_checkpoint: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict:
    """Run ``execute`` up to ``AgentRegistry.build`` and capture its arguments."""
    monkeypatch.setattr(online, "CHECKPOINT_DIR", tmp_path / "models")
    monkeypatch.setattr(
        "draft.application.generate_draft_data.build_labeler",
        lambda config, *, locator=None: object(),
    )

    captured: dict = {}

    def capture(agent_checkpoints, mix_labels, **kwargs):
        captured["agent_checkpoints"] = agent_checkpoints
        captured["mix_labels"] = mix_labels
        captured.update(kwargs)
        raise _Captured

    monkeypatch.setattr(
        "draft.application.agent_registry.AgentRegistry.build", capture,
    )

    config = TrainDraftAgentOnlineConfig(
        learner_label=LEARNER,
        learner_checkpoint=base_checkpoint,
        frozen={ANCHOR: base_checkpoint, "gen-2": base_checkpoint},
        anchor=ANCHOR,
        mix=[(LEARNER, 3), (ANCHOR, 5), ("gen-2", 1), ("forge-r100", 1)],
        scorer_checkpoint=base_checkpoint,
        cards_path=cards_path,
        rollout_temperature=TEMPERATURE,
        output_path=tmp_path / "drafts.jsonl",
    )
    with pytest.raises(_Captured):
        TrainDraftAgentOnlineUseCase().execute(config)
    return captured


def test_every_service_is_supplied_preloaded(registry_call: dict) -> None:
    """Nothing is loaded by path, so ``build``'s single pick-mode never applies."""
    assert registry_call["agent_checkpoints"] == {}
    assert set(registry_call["preloaded"]) == {LEARNER, ANCHOR, "gen-2"}


def test_the_learner_samples_at_the_rollout_temperature(registry_call: dict) -> None:
    learner = registry_call["preloaded"][LEARNER]
    assert learner._pick_mode == "sample"
    assert learner._temperature == TEMPERATURE


def test_every_frozen_agent_plays_argmax(registry_call: dict) -> None:
    for label in (ANCHOR, "gen-2"):
        assert registry_call["preloaded"][label]._pick_mode == "argmax", label


def test_the_learner_service_wraps_the_live_model(registry_call: dict) -> None:
    """It must be the in-training instance, not a reloaded copy (FR-012)."""
    learner = registry_call["preloaded"][LEARNER]
    frozen = registry_call["preloaded"][ANCHOR]
    assert learner._model is not frozen._model


def test_mix_labels_still_cover_every_category(registry_call: dict) -> None:
    assert registry_call["mix_labels"] == {LEARNER, ANCHOR, "gen-2", "forge-r100"}

"""AgentRegistry: LABEL=PATH parsing, label + geometry validation (T007)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from draft.application.agent_registry import (
    AgentRegistry,
    AgentRegistryError,
    parse_agent_checkpoints,
)
from draft.domain.draft_agent_model import DraftAgentConfig, DraftAgentModel
from draft.infrastructure.draft_agent_store import DraftAgentStore

EMB_DIM = 8


class _FakeLocator:
    def load_embedding(self, name: str):
        return None


def _save(tmp_path, name, *, packs=3, P=15):
    config = DraftAgentConfig(embedding_dim=EMB_DIM, packs=packs, P=P, n_layers=1, n_heads=1)
    model = DraftAgentModel(config)
    optimizer = torch.optim.AdamW(model.parameters())
    path = tmp_path / name
    DraftAgentStore().save_checkpoint(
        model, optimizer, epoch=0, best_val_loss=0.0, config=config, path=path,
        critic_mean=0.0, critic_std=1.0,
    )
    return path


# ── LABEL=PATH parsing ────────────────────────────────────────────────────────

def test_parse_label_path_and_bare_path() -> None:
    parsed = parse_agent_checkpoints(["a=models/a.pt", "models/latest.pt"])
    assert parsed == {"a": Path("models/a.pt"), "draft-agent": Path("models/latest.pt")}


def test_parse_rejects_malformed_and_duplicate() -> None:
    with pytest.raises(AgentRegistryError):
        parse_agent_checkpoints(["=nopath.pt"])      # empty label
    with pytest.raises(AgentRegistryError):
        parse_agent_checkpoints(["a=", ])            # empty path
    with pytest.raises(AgentRegistryError):
        parse_agent_checkpoints(["a=x.pt", "a=y.pt"])  # duplicate label


# ── label validation (FR-011) ────────────────────────────────────────────────

def test_unknown_mix_label_fails_fast(tmp_path) -> None:
    ckpt = _save(tmp_path, "a.pt")
    with pytest.raises(AgentRegistryError):
        # "rival" is neither a Forge built-in nor bound.
        AgentRegistry.build(
            {"a": ckpt}, {"forge-full", "a", "rival"}, locator=_FakeLocator(),
        )


def test_forge_builtins_and_bound_labels_pass(tmp_path) -> None:
    ckpt = _save(tmp_path, "a.pt")
    registry = AgentRegistry.build(
        {"a": ckpt}, {"forge-full", "forge-r30", "a"}, locator=_FakeLocator(),
    )
    assert registry.external_labels == frozenset({"a"})


# ── geometry validation (FR-012) ─────────────────────────────────────────────

def test_packs_mismatch_fails_fast(tmp_path) -> None:
    ckpt = _save(tmp_path, "a.pt", packs=2)  # live PACKS is 3
    with pytest.raises(AgentRegistryError):
        AgentRegistry.build({"a": ckpt}, {"a"}, locator=_FakeLocator())


def test_capacity_below_pack_size_fails_fast(tmp_path) -> None:
    ckpt = _save(tmp_path, "a.pt", P=10)
    with pytest.raises(AgentRegistryError):
        AgentRegistry.build({"a": ckpt}, {"a"}, locator=_FakeLocator(), pack_size=15)


def test_missing_checkpoint_file_fails_fast(tmp_path) -> None:
    with pytest.raises(AgentRegistryError):
        AgentRegistry.build(
            {"a": tmp_path / "nope.pt"}, {"a"}, locator=_FakeLocator(),
        )

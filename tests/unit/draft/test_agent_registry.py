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


# ── preloaded services (research D4, spec 021 FR-003) ────────────────────────

def _preloaded_service(*, packs=3, P=15):
    """A live-model service, as the online trainer's learner supplies."""
    from draft.application.agent_pick_service import AgentPickService

    config = DraftAgentConfig(embedding_dim=EMB_DIM, packs=packs, P=P, n_layers=1, n_heads=1)
    model = DraftAgentModel(config)
    model.eval()
    return AgentPickService.from_model(
        model, config, _FakeLocator(), device=torch.device("cpu"),
    )


def test_preloaded_label_counts_as_bound(tmp_path) -> None:
    """The learner's label is not a checkpoint path, but it is still bound."""
    ckpt = _save(tmp_path, "gen1.pt")
    registry = AgentRegistry.build(
        {"gen-1": ckpt}, {"gen-3", "gen-1", "forge-r30"},
        locator=_FakeLocator(),
        preloaded={"gen-3": _preloaded_service()},
    )
    assert registry.external_labels == frozenset({"gen-1", "gen-3"})


def test_preloaded_does_not_excuse_an_unknown_mix_label(tmp_path) -> None:
    with pytest.raises(AgentRegistryError):
        AgentRegistry.build(
            {}, {"gen-3", "rival"}, locator=_FakeLocator(),
            preloaded={"gen-3": _preloaded_service()},
        )


def test_preloaded_services_get_the_same_geometry_checks() -> None:
    with pytest.raises(AgentRegistryError, match="packs"):
        AgentRegistry.build(
            {}, {"gen-3"}, locator=_FakeLocator(),
            preloaded={"gen-3": _preloaded_service(packs=2)},
        )


def test_preloaded_service_pack_size_capacity_is_checked() -> None:
    with pytest.raises(AgentRegistryError, match="capacity"):
        AgentRegistry.build(
            {}, {"gen-3"}, locator=_FakeLocator(), pack_size=15,
            preloaded={"gen-3": _preloaded_service(P=10)},
        )


def test_a_label_bound_both_ways_is_rejected(tmp_path) -> None:
    ckpt = _save(tmp_path, "gen3.pt")
    with pytest.raises(AgentRegistryError):
        AgentRegistry.build(
            {"gen-3": ckpt}, {"gen-3"}, locator=_FakeLocator(),
            preloaded={"gen-3": _preloaded_service()},
        )


def test_preloaded_service_is_the_instance_used_for_picks() -> None:
    service = _preloaded_service()
    registry = AgentRegistry.build(
        {}, {"gen-3"}, locator=_FakeLocator(), preloaded={"gen-3": service},
    )
    assert registry._services["gen-3"] is service


def test_omitting_preloaded_is_unchanged(tmp_path) -> None:
    ckpt = _save(tmp_path, "a.pt")
    registry = AgentRegistry.build(
        {"a": ckpt}, {"forge-full", "a"}, locator=_FakeLocator(),
    )
    assert registry.external_labels == frozenset({"a"})

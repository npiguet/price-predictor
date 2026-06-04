"""Plateau LR annealing for train-draft-agent: controller, decay timing, resume."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from draft.application.train_draft_agent import (
    TrainDraftAgentConfig,
    _make_scheduler,
    _maybe_decay,
    _PlateauLR,
    _resume_or_build,
    _validate_lr_decay,
)
from draft.domain.draft_agent_model import DraftAgentConfig, DraftAgentModel
from draft.infrastructure.draft_agent_store import DraftAgentStore

# --------------------------------------------------------------------------- #
# Controller
# --------------------------------------------------------------------------- #

def test_multiplier_tracks_decay_count() -> None:
    ctrl = _PlateauLR(base_lr=3e-5, factor=0.1, min_lr=3e-8, decay_count=2)
    assert ctrl.multiplier() == pytest.approx(0.01)
    assert ctrl.current_lr() == pytest.approx(3e-7)


def test_floor_caps_the_number_of_decays() -> None:
    # min_lr = base * 1e-3 with factor 0.1 => exactly 3 decays reach the floor.
    ctrl = _PlateauLR(base_lr=3e-5, factor=0.1, min_lr=3e-8)
    assert ctrl.can_decay()              # -> 3e-6
    assert ctrl.decay() == pytest.approx(3e-6)
    assert ctrl.can_decay()              # -> 3e-7
    ctrl.decay()
    assert ctrl.can_decay()              # -> 3e-8 == floor (>=)
    assert ctrl.decay() == pytest.approx(3e-8)
    assert not ctrl.can_decay()          # next (3e-9) is below the floor
    assert ctrl.decay_count == 3


# --------------------------------------------------------------------------- #
# Decay decision
# --------------------------------------------------------------------------- #

def test_maybe_decay_fires_at_patience_and_is_disabled_by_none() -> None:
    ctrl = _PlateauLR(base_lr=1e-4, factor=0.1, min_lr=1e-9)
    assert _maybe_decay(2, patience=3, controller=ctrl) is None   # not stuck yet
    assert _maybe_decay(3, patience=3, controller=ctrl) == pytest.approx(1e-5)
    assert ctrl.decay_count == 1
    # patience=None is the feature switch: never decays.
    assert _maybe_decay(99, patience=None, controller=ctrl) is None
    assert ctrl.decay_count == 1


def _simulate_never_improving(decay_patience: int | None, stop_patience: int):
    """Mirror run_eval's counter logic for a val sequence that never improves.

    Returns (decay_evals, first_stop_eval, final_decay_count).
    """
    ctrl = _PlateauLR(base_lr=3e-5, factor=0.1, min_lr=3e-8)  # 3 decays available
    evals_since_best = 0
    decay_evals: list[int] = []
    first_stop_eval: int | None = None
    for i in range(1, 200):
        evals_since_best += 1  # never a new best
        if _maybe_decay(evals_since_best, decay_patience, ctrl) is not None:
            decay_evals.append(i)
            evals_since_best = 0
        if first_stop_eval is None and evals_since_best >= stop_patience:
            first_stop_eval = i
            break
    return decay_evals, first_stop_eval, ctrl.decay_count


def test_decay_resets_counter_and_early_stop_only_fires_at_floor() -> None:
    decay_evals, first_stop_eval, decays = _simulate_never_improving(
        decay_patience=3, stop_patience=30,
    )
    # Decays at 3, 6, 9 (each resets the window); then floor reached.
    assert decay_evals == [3, 6, 9]
    assert decays == 3
    # After the last decay (eval 9) the floor is hit, so the counter runs the
    # full stop-patience: early stop at 9 + 30 = 39 (i.e. only at min_lr).
    assert first_stop_eval == 39


def test_disabled_decay_matches_plain_early_stop() -> None:
    decay_evals, first_stop_eval, decays = _simulate_never_improving(
        decay_patience=None, stop_patience=30,
    )
    assert decay_evals == []          # never decays
    assert decays == 0
    assert first_stop_eval == 30      # identical to today's constant-LR behavior


# --------------------------------------------------------------------------- #
# Scheduler folding
# --------------------------------------------------------------------------- #

def test_scheduler_folds_live_decay_multiplier() -> None:
    lin = torch.nn.Linear(2, 2)
    opt = torch.optim.AdamW([{"params": list(lin.parameters()), "lr": 1e-3}])
    ctrl = _PlateauLR(base_lr=1e-3, factor=0.1, min_lr=1e-9)
    sched = _make_scheduler(opt, total_steps=10, warmup_frac=0.05, controller=ctrl)

    opt.step()
    sched.step()  # past the 1-step warmup -> full base LR
    assert opt.param_groups[0]["lr"] == pytest.approx(1e-3)

    ctrl.decay()  # a mid-run decay must take effect on the next step
    opt.step()
    sched.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(1e-4)


# --------------------------------------------------------------------------- #
# Config validation
# --------------------------------------------------------------------------- #

def test_validate_lr_decay_accepts_disabled_and_valid() -> None:
    _validate_lr_decay(TrainDraftAgentConfig())  # disabled: no-op
    _validate_lr_decay(
        TrainDraftAgentConfig(lr_decay_patience=10, patience=30, lr_decay_factor=0.1)
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lr_decay_patience": 0},                       # must be >= 1
        {"lr_decay_patience": 10, "lr_decay_factor": 1.0},   # factor in (0,1)
        {"lr_decay_patience": 10, "lr_decay_factor": 0.0},
        {"lr_decay_patience": 30, "patience": 30},      # must be < patience
        {"lr_decay_patience": 40, "patience": 30},
    ],
)
def test_validate_lr_decay_rejects_bad_combos(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        _validate_lr_decay(TrainDraftAgentConfig(**kwargs))


# --------------------------------------------------------------------------- #
# Resume: inherit decay_count, reset on an explicit --lr override
# --------------------------------------------------------------------------- #

def _save_agent(tmp_path: Path, *, lr: float, decay_count: int) -> Path:
    cfg = DraftAgentConfig(embedding_dim=8, packs=3, P=15)
    model = DraftAgentModel(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    path = tmp_path / "agent.pt"
    DraftAgentStore().save_checkpoint(
        model, opt, epoch=2, best_val_loss=0.5, config=cfg, path=path,
        critic_mean=0.0, critic_std=1.0, train_config={"lr": lr},
        lr_decay_count=decay_count,
    )
    return path


def test_resume_inherits_decay_count_when_lr_unchanged(tmp_path: Path) -> None:
    path = _save_agent(tmp_path, lr=3e-5, decay_count=2)
    cfg = TrainDraftAgentConfig(resume=path, lr=3e-5)  # same base LR
    resume, _, _ = _resume_or_build(cfg, DraftAgentStore(), 8, 3, 15)
    assert resume.lr_decay_count == 2


def test_resume_resets_decay_count_on_explicit_lr_override(tmp_path: Path) -> None:
    path = _save_agent(tmp_path, lr=3e-5, decay_count=2)
    cfg = TrainDraftAgentConfig(resume=path, lr=1e-6)  # different base LR
    resume, _, _ = _resume_or_build(cfg, DraftAgentStore(), 8, 3, 15)
    assert resume.lr_decay_count == 0

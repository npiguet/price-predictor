"""Resume precedence for train-draft-agent: CLI > resumed checkpoint > default."""

from __future__ import annotations

from pathlib import Path

import torch

from draft.application.train_draft_agent import _make_scheduler, _reapply_resume_lr
from draft.infrastructure.cli import _resolve_train_agent_config, build_parser


def _args(argv: list[str]):
    return build_parser().parse_args(["train-draft-agent", *argv])


def test_fresh_run_uses_defaults_and_cli_overrides() -> None:
    args = _args(["--lr", "1e-3"])
    cfg = _resolve_train_agent_config(args, resumed_train_config=None)
    assert cfg.lr == 1e-3                 # explicit CLI
    assert cfg.batch_size == 32           # dataclass default
    assert cfg.val_fraction == 0.0025     # dataclass default
    assert cfg.evals_per_epoch == 100     # dataclass default
    assert cfg.patience == 30             # dataclass default
    assert cfg.imitation_agents == ("forge-full",)
    assert cfg.drafts_path == Path("output/draft/drafts.jsonl")


def test_resume_inherits_then_cli_overrides() -> None:
    resumed = {
        "lr": 1e-4,
        "batch_size": 128,
        "val_fraction": 0.01,
        "evals_per_epoch": 50,
        "patience": 20,
        "imitation_weight": 2.0,
        "imitation_agents": ["forge-full", "forge-r30"],
        "drafts_path": "old/drafts.jsonl",
        "cards_path": "old/cards/",
    }
    # Override only the LR (the annealing case) + batch_size on the CLI.
    args = _args(["--resume", "ckpt.pt", "--lr", "3e-5", "--batch-size", "64"])
    cfg = _resolve_train_agent_config(args, resumed)

    assert cfg.lr == 3e-5                       # CLI override
    assert cfg.batch_size == 64                 # CLI override
    assert cfg.val_fraction == 0.01             # inherited from checkpoint
    assert cfg.evals_per_epoch == 50            # inherited
    assert cfg.patience == 20                   # inherited
    assert cfg.imitation_weight == 2.0          # inherited
    assert cfg.imitation_agents == ("forge-full", "forge-r30")  # inherited, as tuple
    assert cfg.drafts_path == Path("old/drafts.jsonl")  # inherited path
    assert cfg.cards_path == Path("old/cards/")
    assert cfg.resume == Path("ckpt.pt")


def test_resume_falls_back_to_default_when_neither_cli_nor_checkpoint() -> None:
    # Checkpoint predates a flag (e.g. no evals_per_epoch stored) -> dataclass default.
    resumed = {"lr": 1e-4}  # sparse train_config
    args = _args(["--resume", "ckpt.pt"])
    cfg = _resolve_train_agent_config(args, resumed)
    assert cfg.lr == 1e-4                # from checkpoint
    assert cfg.evals_per_epoch == 100    # default (absent in checkpoint, no CLI)
    assert cfg.patience == 30            # default
    assert cfg.imitation_agents == ("forge-full",)  # default


def test_resume_lr_override_survives_scheduler_rebuild() -> None:
    """A --lr override must reach the scheduler's base_lr, not be clobbered.

    Regression: a previous run's LambdaLR bakes ``initial_lr`` into the optimizer
    param group, which is saved in the optimizer state dict. On resume that stale
    ``initial_lr`` would override the new --lr unless explicitly dropped, so the
    warmup scheduler would silently ramp back to the *old* LR on its first step().
    """
    model = torch.nn.Linear(4, 4)

    # Original run at the 3e-4 default; its scheduler injects initial_lr=3e-4,
    # which then lives in the saved optimizer state dict.
    old_opt = torch.optim.AdamW([{"params": list(model.parameters()), "lr": 3e-4}])
    _make_scheduler(old_opt, total_steps=1000, warmup_frac=0.05)
    saved_state = old_opt.state_dict()
    assert saved_state["param_groups"][0]["initial_lr"] == 3e-4

    # Resume at an overridden 3e-6.
    new_lr = 3e-6
    new_opt = torch.optim.AdamW([{"params": list(model.parameters()), "lr": new_lr}])
    new_opt.load_state_dict(saved_state)
    _reapply_resume_lr(new_opt, new_lr)
    scheduler = _make_scheduler(new_opt, total_steps=1000, warmup_frac=0.05)

    assert scheduler.base_lrs == [new_lr]          # base recaptured from --lr
    new_opt.step()
    scheduler.step()
    # First warmup step is base_lr × (1/warmup_steps), bounded by the new base —
    # the bug let this jump back to the inherited 3e-4.
    assert new_opt.param_groups[0]["lr"] <= new_lr

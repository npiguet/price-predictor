"""train-draft-agent-rl CLI: startup validation, exit codes, resume precedence."""

from __future__ import annotations

import pytest

from draft.infrastructure.cli import (
    _resolve_train_agent_rl_config,
    build_parser,
    run_train_draft_agent_rl,
)


def _args(argv: list[str]):
    return build_parser().parse_args(["train-draft-agent-rl", *argv])


def test_requires_exactly_one_of_checkpoint_or_resume() -> None:
    assert run_train_draft_agent_rl(_args([])) == 2  # neither
    assert run_train_draft_agent_rl(
        _args(["--checkpoint", "a.pt", "--resume", "b.pt"]),
    ) == 2  # both


def test_empty_learner_agents_rejected() -> None:
    args = _args(["--checkpoint", "ref.pt", "--rollout-temperature", "1.0"])
    assert run_train_draft_agent_rl(args) == 2


def test_missing_rollout_temperature_rejected() -> None:
    args = _args(["--checkpoint", "ref.pt", "--learner-agents", "gen-k"])
    assert run_train_draft_agent_rl(args) == 2


def test_nonpositive_rollout_temperature_rejected() -> None:
    args = _args([
        "--checkpoint", "ref.pt", "--learner-agents", "gen-k",
        "--rollout-temperature", "0",
    ])
    assert run_train_draft_agent_rl(args) == 2


def test_unknown_architecture_flag_is_a_parse_error() -> None:
    # The RL parser defines no architecture flags; passing one is a usage error.
    with pytest.raises(SystemExit):
        _args(["--checkpoint", "ref.pt", "--d-model", "256"])


def test_resume_precedence_cli_over_resumed_over_default() -> None:
    # Resumed config supplies learner/temperature/lr; CLI overrides lr only.
    args = _args(["--resume", "run.pt", "--lr", "2e-4"])
    resumed = {
        "learner_agents": ["gen-k"],
        "rollout_temperature": 1.5,
        "lr": 1e-4,           # overridden by CLI 2e-4
        "gae_lambda": 0.9,    # inherited
    }
    config = _resolve_train_agent_rl_config(args, resumed)
    assert config.learner_agents == ("gen-k",)   # inherited (CLI absent)
    assert config.rollout_temperature == 1.5      # inherited
    assert config.lr == 2e-4                       # CLI override
    assert config.gae_lambda == 0.9               # inherited
    assert config.value_weight == 1.0             # dataclass default (neither set)

"""Every path a draft command reads or writes is a required flag: no default names a
location under ``output/`` or ``models/`` (finding 16.6, task L06d)."""

from __future__ import annotations

import pytest

from draft.infrastructure.cli import build_parser

_GENERATE = [
    "generate-draft-data", "--n-drafts", "1", "--scorer-checkpoint", "s.pt",
    "--picker-checkpoint", "p.pt", "--cards-path", "cards", "--output-path", "d.jsonl",
    "--logs-dir", "logs",
]
_ONLINE = [
    "train-draft-agent-online", "--learner", "gen-3=a.pt", "--agent-temp", "gen-3=2.0",
    "--scorer-checkpoint", "s.pt", "--cards-path", "cards", "--output-path", "d.jsonl",
    "--checkpoint-dir", "run",
]
_VALIDATE = [
    "validate-builder", "--pools-from", "d.jsonl", "--scorer-checkpoint", "s.pt",
    "--picker-checkpoint", "p.pt", "--cards-path", "cards",
]
_ANALYZE = [
    "analyze-generated-decks", "--agent", "forge-full", "--drafts-path", "d.jsonl",
    "--cards-path", "cards",
]
_PLAY = ["play-draft-games", "--drafts-path", "d.jsonl", "--output-path", "g.txt"]
_OFFLINE = [
    "train-draft-agent", "--drafts-path", "d.jsonl", "--cards-path", "cards",
    "--checkpoint-dir", "run",
]
_RL = [
    "train-draft-agent-rl", "--checkpoint", "ref.pt", "--learner-agents", "gen-2",
    "--rollout-temperature", "1.0", "--drafts-path", "d.jsonl", "--cards-path", "cards",
    "--checkpoint-dir", "run",
]

#: (full command line, flags that parse-time ``required=True`` refuses without)
_PARSE_REQUIRED = [
    (_GENERATE, ["--scorer-checkpoint", "--output-path", "--logs-dir", "--cards-path"]),
    (_ONLINE, ["--scorer-checkpoint", "--output-path", "--checkpoint-dir", "--cards-path"]),
    (_VALIDATE, ["--scorer-checkpoint", "--picker-checkpoint", "--cards-path"]),
    (_ANALYZE, ["--drafts-path", "--cards-path"]),
    (_PLAY, ["--drafts-path", "--output-path"]),
]
_PARSE_CASES = [
    pytest.param(argv, flag, id=f"{argv[0]}{flag}")
    for argv, flags in _PARSE_REQUIRED for flag in flags
]


def _without(argv: list[str], flag: str) -> list[str]:
    at = argv.index(flag)
    return argv[:at] + argv[at + 2:]


@pytest.mark.parametrize(("argv", "flag"), _PARSE_CASES)
def test_the_command_refuses_to_start_without_the_flag(argv, flag, capsys):
    with pytest.raises(SystemExit) as stop:
        build_parser().parse_args(_without(argv, flag))
    assert stop.value.code == 2
    assert flag in capsys.readouterr().err


@pytest.mark.parametrize("argv", [_GENERATE, _ONLINE, _VALIDATE, _ANALYZE, _PLAY, _OFFLINE, _RL],
                         ids=lambda argv: argv[0])
def test_the_help_names_no_output_or_models_default(argv, capsys):
    with pytest.raises(SystemExit):
        build_parser().parse_args([argv[0], "--help"])
    help_text = capsys.readouterr().out
    assert "output/" not in help_text
    assert "models/" not in help_text


@pytest.mark.parametrize(("argv", "flag"), [
    pytest.param(argv, flag, id=f"{argv[0]}{flag}")
    for argv in (_OFFLINE, _RL)
    for flag in ("--drafts-path", "--cards-path", "--checkpoint-dir")
])
def test_a_fresh_trainer_refuses_to_start_without_the_flag(argv, flag, capsys):
    args = build_parser().parse_args(_without(argv, flag))
    with pytest.raises(SystemExit) as stop:
        args.func(args)
    assert stop.value.code == 2
    assert flag in capsys.readouterr().err


@pytest.mark.parametrize("argv", [_GENERATE, _ONLINE], ids=lambda argv: argv[0])
def test_the_picker_checkpoint_is_required_only_by_the_picker_build_method(argv, capsys):
    """The default build method of generate-draft-data is the picker; the online one's is greedy."""
    without = _without(argv, "--picker-checkpoint") if "--picker-checkpoint" in argv else argv
    picker = [*without, "--build-method", "picker"]
    args = build_parser().parse_args(picker)
    assert args.picker_checkpoint is None
    assert args.func(args) == 2
    assert "--picker-checkpoint" in capsys.readouterr().err


def test_the_online_trainer_s_run_manifest_is_its_checkpoint_directory():
    from pathlib import Path

    from draft.infrastructure.cli import RUN_MANIFESTS

    args = build_parser().parse_args(_ONLINE)
    assert RUN_MANIFESTS["train-draft-agent-online"].run_dir(args) == Path("run")

"""Every path a sealed command reads or writes is a required flag: no default names a
location under ``output/`` or ``models/``."""

from __future__ import annotations

import pytest

from sealed.infrastructure.cli import build_parser

_ANALYZE = ["analyze-generated-decks", "d.txt", "--cards-path", "cards"]
_BUILD_VOCAB = ["build-vocab", "--cards-folder", "cards", "--vocab-path", "vocab.txt"]
_TRAIN_ENCODER = [
    "train-encoder", "--cards-played-path", "played.txt", "--cards-folder", "cards",
    "--vocab-path", "vocab.txt", "--model-output", "run",
]
_ENCODE = [
    "encode-cards", "--encoder-checkpoint", "e.pt", "--vocab-path", "vocab.txt",
    "--cards-path", "cards",
]
_POOLS = ["generate-pools", "--pools-path", "pools"]
_BUILD_DECKS = [
    "build-decks", "--pools-path", "pools.txt", "--label", "x", "--checkpoint", "s.pt",
    "--cards-path", "cards", "--output", "decks.txt",
]
_PICK = [
    "pick-decks", "--pools-path", "pools.txt", "--picker-checkpoint", "p.pt",
    "--cards-path", "cards", "--label", "x", "--output", "decks.txt",
]
_EVALUATE = ["evaluate-scorer", "--checkpoint", "s.pt", "--cards-path", "cards"]
_SCORER = [
    "train-scorer", "--outcomes-path", "o.txt", "--cards-path", "cards",
    "--checkpoint-dir", "run",
]
_PICKER = [
    "train-picker", "--pools-path", "pools.txt", "--scorer-checkpoint", "s.pt",
    "--cards-path", "cards", "--checkpoint-dir", "run",
]
_MATCHES = ["match-outcomes", "--outcomes-dir", "out"]

_ALL = [_ANALYZE, _BUILD_VOCAB, _TRAIN_ENCODER, _ENCODE, _POOLS, _BUILD_DECKS, _PICK,
        _EVALUATE, _SCORER, _PICKER, _MATCHES]

#: (command line, flags the parser itself refuses without)
_PARSE_REQUIRED = [
    (_ANALYZE, ["--cards-path"]),
    (_BUILD_VOCAB, ["--cards-folder", "--vocab-path"]),
    (_TRAIN_ENCODER, ["--cards-played-path", "--cards-folder", "--vocab-path", "--model-output"]),
    (_ENCODE, ["--vocab-path", "--cards-path"]),
    (_POOLS, ["--pools-path"]),
    (_BUILD_DECKS, ["--checkpoint", "--cards-path", "--output"]),
    (_PICK, ["--picker-checkpoint", "--cards-path", "--output"]),
    (_EVALUATE, ["--cards-path"]),
]
#: (command line, flags a fresh run is refused without, at run time, exit 2)
_RUN_REQUIRED = [
    (_SCORER, ["--outcomes-path", "--cards-path", "--checkpoint-dir"]),
    (_PICKER, ["--scorer-checkpoint", "--cards-path", "--checkpoint-dir"]),
]


def _without(argv: list[str], flag: str) -> list[str]:
    at = argv.index(flag)
    return argv[:at] + argv[at + 2:]


def _cases(table):
    return [pytest.param(argv, flag, id=f"{argv[0]}{flag}") for argv, flags in table
            for flag in flags]


@pytest.mark.parametrize(("argv", "flag"), _cases(_PARSE_REQUIRED))
def test_the_command_refuses_to_start_without_the_flag(argv, flag, capsys):
    with pytest.raises(SystemExit) as stop:
        build_parser().parse_args(_without(argv, flag))
    assert stop.value.code == 2
    assert flag in capsys.readouterr().err


def test_analyze_generated_decks_refuses_to_start_without_a_decks_file(capsys):
    with pytest.raises(SystemExit) as stop:
        build_parser().parse_args(["analyze-generated-decks", "--cards-path", "cards"])
    assert stop.value.code == 2
    assert "decks_files" in capsys.readouterr().err


@pytest.mark.parametrize(("argv", "flag"), _cases(_RUN_REQUIRED))
def test_a_fresh_trainer_refuses_to_start_without_the_flag(argv, flag, capsys):
    args = build_parser().parse_args(_without(argv, flag))
    assert args.func(args) == 2
    assert flag in capsys.readouterr().err


#: match-outcomes' help gives the layout of a converted tree as an example path, so it is
#: checked on its own flags below.
@pytest.mark.parametrize("argv", [a for a in _ALL if a is not _MATCHES],
                         ids=lambda argv: argv[0])
def test_the_help_names_no_output_or_models_default(argv, capsys):
    with pytest.raises(SystemExit):
        build_parser().parse_args([argv[0], "--help"])
    help_text = capsys.readouterr().out
    assert "output/" not in help_text
    assert "models/" not in help_text


def test_encode_cards_refuses_to_start_without_an_encoder_source(capsys):
    args = build_parser().parse_args(_without(_ENCODE, "--encoder-checkpoint"))
    assert args.func(args) == 2
    assert "--encoder-checkpoint" in capsys.readouterr().err


def test_match_outcomes_has_no_default_corpora_directory(capsys):
    assert build_parser().parse_args(["match-outcomes"]).outcomes_dir is None


def test_match_outcomes_refuses_to_start_without_a_place_for_its_corpora(capsys):
    args = build_parser().parse_args(["match-outcomes"])
    assert args.func(args) == 2
    err = capsys.readouterr().err
    assert "--outcomes-dir" in err
    assert "--effect-records" in err

"""``python -m sealed``'s run manifests: every subcommand declared, one written end to end."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from price_predictor.infrastructure import run_manifest
from price_predictor.infrastructure.cli_run_manifest import READ_ONLY, subcommand_names
from sealed.infrastructure import cli


@pytest.fixture(autouse=True)
def fixed_commits(monkeypatch):
    monkeypatch.setattr(
        run_manifest, "read_commits",
        lambda *_args, **_kw: {"repo": "abc1234", "forge": None, "dirty": True},
    )


def test_every_subcommand_is_declared():
    assert sorted(cli.RUN_MANIFESTS) == sorted(subcommand_names(cli.build_parser()))


def test_build_decks_writes_its_manifest_beside_the_decks(tmp_path: Path, monkeypatch):
    out = tmp_path / "decks-run" / "generated-decks.txt"

    def build_decks(args):
        Path(args.output).write_text("gen-2;RVR;A|B\n", encoding="utf-8")
        return 0

    monkeypatch.setattr(cli, "run_build_decks", build_decks)
    argv = ["build-decks", "--pools-path", str(tmp_path / "pools.txt"), "--label", "gen-2",
            "--checkpoint", str(tmp_path / "scorer.pt"), "--cards-path", str(tmp_path / "cards"),
            "--output", str(out)]
    monkeypatch.setattr(sys, "argv", ["sealed", *argv])

    with pytest.raises(SystemExit) as exit_:
        cli.main()

    assert exit_.value.code == 0
    document = json.loads((out.parent / "run.json").read_text(encoding="utf-8"))
    assert document["run"] == out.parent.resolve().as_posix()
    assert document["status"] == "finished"
    assert document["outcome"] == "exit 0"
    assert document["commands"] == ["python -m sealed " + " ".join(argv)]
    assert document["commits"] == {"repo": "abc1234", "forge": None, "dirty": True}
    assert document["inputs"] == [
        (tmp_path / name).resolve().as_posix() for name in ("pools.txt", "scorer.pt", "cards")
    ]
    assert document["purpose"] is None and document["continues"] == []


def test_a_failing_command_is_recorded_failed(tmp_path: Path, monkeypatch):
    out = tmp_path / "decks-run" / "generated-decks.txt"
    monkeypatch.setattr(cli, "run_build_decks", lambda _args: 2)
    monkeypatch.setattr(sys, "argv", ["sealed", "build-decks", "--pools-path", "p",
                                      "--label", "x", "--output", str(out)])

    with pytest.raises(SystemExit) as exit_:
        cli.main()

    assert exit_.value.code == 2
    document = json.loads((out.parent / "run.json").read_text(encoding="utf-8"))
    assert (document["status"], document["outcome"]) == ("failed", "exit 2")


def test_match_outcomes_records_in_its_effect_records_directory():
    args = cli.build_parser().parse_args(
        ["match-outcomes", "--effect-records", "output/effects/gen3/records/2026-11-13-x"],
    )
    assert cli.RUN_MANIFESTS["match-outcomes"].run_dir(args) == Path(
        "output/effects/gen3/records/2026-11-13-x",
    )


def test_generate_pools_resolves_the_set_placeholder():
    args = cli.build_parser().parse_args(
        ["generate-pools", "--set", "RVR", "--pools-path", "output/sealed/gen1/pools/{set}"],
    )
    assert cli.RUN_MANIFESTS["generate-pools"].run_dir(args) == Path(
        "output/sealed/gen1/pools/RVR",
    )


def test_a_resumed_trainer_records_where_its_checkpoint_lives():
    args = cli.build_parser().parse_args(
        ["train-scorer", "--resume", "models/sealed/gen1/scorer/2026-11-13-l6/latest.pt"],
    )
    assert cli.RUN_MANIFESTS["train-scorer"].run_dir(args) == Path(
        "models/sealed/gen1/scorer/2026-11-13-l6",
    )


def test_analysis_writes_none():
    assert cli.RUN_MANIFESTS["analyze-generated-decks"] is READ_ONLY

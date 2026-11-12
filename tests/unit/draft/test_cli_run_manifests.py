"""``python -m draft``'s run manifests: every subcommand declared, one written end to end."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from draft.infrastructure import cli
from price_predictor.infrastructure import run_manifest
from price_predictor.infrastructure.cli_run_manifest import READ_ONLY, subcommand_names


@pytest.fixture(autouse=True)
def fixed_commits(monkeypatch):
    monkeypatch.setattr(
        run_manifest, "read_commits",
        lambda *_args, **_kw: {"repo": "abc1234", "forge": "d44cf28", "dirty": False},
    )


def test_every_subcommand_is_declared():
    assert sorted(cli.RUN_MANIFESTS) == sorted(subcommand_names(cli.build_parser()))


def test_play_draft_games_records_an_interrupt(tmp_path: Path, monkeypatch):
    out = tmp_path / "games-run" / "draft-games.txt"
    drafts = tmp_path / "drafts.jsonl"

    def play(args):
        Path(args.output_path).write_text("", encoding="utf-8")
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "run_play_draft_games", play)
    argv = ["play-draft-games", "--drafts-path", str(drafts), "--output-path", str(out)]
    monkeypatch.setattr(sys, "argv", ["draft", *argv])

    with pytest.raises(KeyboardInterrupt):
        cli.main()

    document = json.loads((out.parent / "run.json").read_text(encoding="utf-8"))
    assert document["run"] == out.parent.resolve().as_posix()
    assert document["status"] == "failed"
    assert document["outcome"] == "interrupted"
    assert document["commands"] == ["python -m draft " + " ".join(argv)]
    assert document["inputs"] == [drafts.resolve().as_posix()]
    assert document["ended"] is not None


def test_generate_draft_data_lists_its_agent_checkpoints():
    args = cli.build_parser().parse_args([
        "generate-draft-data", "--n-drafts", "1", "--output-path", "output/draft/x/drafts.jsonl",
        "--agent-checkpoint", "gen-2=models/draft/agent/a.pt",
    ])
    spec = cli.RUN_MANIFESTS["generate-draft-data"]
    assert spec.run_dir(args) == Path("output/draft/x")
    assert Path("models/draft/agent/a.pt") in spec.input_paths(args)


def test_validation_writes_none():
    assert cli.RUN_MANIFESTS["validate-builder"] is READ_ONLY

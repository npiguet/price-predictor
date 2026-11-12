"""Wiring CLI subcommands to the run manifest, and ``python -m price_predictor``'s table."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

from price_predictor.infrastructure import cli, run_manifest
from price_predictor.infrastructure.cli_run_manifest import (
    READ_ONLY,
    SUBCOMMAND_DEST,
    Writes,
    dispatch,
    labelled_paths,
    subcommand_names,
    tag_subcommands,
)

COMMITS = {"repo": "abc1234", "forge": "d44cf28", "dirty": False}


@pytest.fixture(autouse=True)
def fixed_commits(monkeypatch):
    monkeypatch.setattr(run_manifest, "read_commits", lambda *_args, **_kw: dict(COMMITS))


def assert_well_formed(run_dir: Path, *, command: str, status: str = "finished") -> dict:
    """``run_dir``'s manifest, checked field by field against the skill's shape."""
    document = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert document["run"] == run_dir.resolve().as_posix()
    assert document["purpose"] is None
    assert document["status"] == status
    assert document["superseded_by"] is None
    assert document["commands"] == [command]
    assert document["commits"] == COMMITS
    assert document["continues"] == []
    assert document["started"] and document["ended"]
    assert document["outcome"].startswith("exit ")
    return document


def toy_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="toy")
    commands = parser.add_subparsers()
    write = commands.add_parser("write")
    write.add_argument("--out", type=Path)
    write.add_argument("--source", type=Path)
    write.set_defaults(func=lambda args: 0)
    nested = commands.add_parser("train").add_subparsers()
    nested.add_parser("fast").set_defaults(func=lambda args: 0)
    return parser


class TestSubcommands:
    def test_names_every_leaf_nested_ones_space_joined(self):
        assert subcommand_names(toy_parser()) == ["write", "train fast"]

    def test_tagging_records_the_name_in_the_parsed_arguments(self):
        parser = toy_parser()
        tag_subcommands(parser)
        assert getattr(parser.parse_args(["train", "fast"]), SUBCOMMAND_DEST) == "train fast"


class TestWrites:
    def test_an_output_directory_is_the_run_directory(self):
        args = argparse.Namespace(out=Path("output/x"), source=None)
        assert Writes("out").run_dir(args) == Path("output/x")

    def test_an_output_file_s_directory_is_the_run_directory(self):
        args = argparse.Namespace(out="output/x/decks.txt")
        assert Writes("out", file=True).run_dir(args) == Path("output/x")

    def test_an_unset_output_writes_no_manifest(self):
        assert Writes("out").run_dir(argparse.Namespace(out=None)) is None

    def test_when_false_writes_no_manifest(self):
        args = argparse.Namespace(out="x", verify=True)
        assert Writes("out", when=lambda a: not a.verify).run_dir(args) is None

    def test_inputs_flatten_and_skip_unset_values(self):
        args = argparse.Namespace(a="one.txt", b=None, c=[Path("two"), None, "three"])
        spec = Writes("a", inputs=("a", "b", "c", lambda _args: Path("four")))
        assert spec.input_paths(args) == [Path("one.txt"), Path("two"), Path("three"),
                                          Path("four")]

    def test_labelled_paths(self):
        assert labelled_paths(["gen-1=models/a.pt", "models/b.pt"]) == [
            Path("models/a.pt"), Path("models/b.pt"),
        ]
        assert labelled_paths(None) == []


class TestDispatch:
    def test_a_writing_subcommand_runs_inside_its_manifest(self, tmp_path: Path):
        parser = toy_parser()
        tag_subcommands(parser)
        out = tmp_path / "run"
        argv = ["write", "--out", str(out), "--source", str(tmp_path / "in.txt")]
        args = parser.parse_args(argv)
        table = {"write": Writes("out", inputs=("source",)), "train fast": READ_ONLY}

        assert dispatch(args, table, module="toy", argv=argv) == 0

        document = assert_well_formed(out, command="python -m toy " + " ".join(argv))
        assert document["inputs"] == [(tmp_path / "in.txt").resolve().as_posix()]

    def test_a_read_only_subcommand_writes_none(self, tmp_path: Path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        parser = toy_parser()
        tag_subcommands(parser)
        args = parser.parse_args(["train", "fast"])
        assert dispatch(args, {"train fast": READ_ONLY}, module="toy", argv=[]) == 0
        assert list(tmp_path.iterdir()) == []

    def test_a_manifest_that_cannot_be_resolved_never_stops_the_command(self, tmp_path):
        parser = toy_parser()
        tag_subcommands(parser)
        args = parser.parse_args(["write"])

        def broken(_args):
            raise RuntimeError("cannot resolve")

        assert dispatch(args, {"write": Writes(broken)}, module="toy", argv=[]) == 0


class TestPricePredictorCli:
    def test_every_subcommand_is_declared(self):
        assert sorted(cli.RUN_MANIFESTS) == sorted(subcommand_names(cli.build_parser()))

    def test_vocabulary_writes_its_manifest(self, tmp_path: Path, monkeypatch):
        from price_predictor import __main__ as entry

        out, cards = tmp_path / "vocab-run", tmp_path / "cardsfolder"

        def build_vocabulary(args):
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
            (Path(args.output_dir) / "vocab.txt").write_text("[PAD]\n", encoding="utf-8")
            return 0

        monkeypatch.setattr(cli, "run_vocabulary", build_vocabulary)
        argv = ["vocabulary", "--output-dir", str(out), "--cards-path", str(cards),
                "--printings-path", str(tmp_path / "AllPrintings.json")]
        monkeypatch.setattr(sys, "argv", ["price_predictor", *argv])

        assert entry.main() == 0

        document = assert_well_formed(out, command="python -m price_predictor " + " ".join(argv))
        assert document["outcome"] == "exit 0"
        assert document["inputs"] == [
            cards.resolve().as_posix(), (tmp_path / "AllPrintings.json").resolve().as_posix(),
        ]
        assert sorted(p.name for p in out.iterdir()) == ["run.json", "vocab.txt"]

    def test_convert_records_in_the_directory_holding_both_trees(self):
        args = cli.build_parser().parse_args([
            "convert", "--output-path", "output/price-predictor/gen1/converted/cardsfolder",
            "--tokens-output-path", "output/price-predictor/gen1/converted/tokenscripts",
        ])
        assert cli.RUN_MANIFESTS["convert"].run_dir(args) == Path(
            "output/price-predictor/gen1/converted",
        )

    def test_predict_writes_none(self):
        assert cli.RUN_MANIFESTS["predict sklearn"] is READ_ONLY

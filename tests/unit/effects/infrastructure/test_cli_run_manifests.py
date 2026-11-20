"""``python -m effects``'s run manifests: every subcommand declared, one written
end to end, and no reader of a run directory taking ``run.json`` for data."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from effects.infrastructure import cli
from effects.infrastructure.shard_io import iter_shards
from price_predictor.infrastructure import run_manifest
from price_predictor.infrastructure.cli_run_manifest import subcommand_names
from tests.unit.effects.cli_args import cli_args, parse_cli


@pytest.fixture(autouse=True)
def fixed_commits(monkeypatch):
    monkeypatch.setattr(
        run_manifest, "read_commits",
        lambda *_args, **_kw: {"repo": "abc1234", "forge": "d44cf28", "dirty": False},
    )


def run_main(monkeypatch, argv: list[str]) -> int:
    monkeypatch.setattr(sys, "argv", ["effects", *argv])
    with pytest.raises(SystemExit) as exit_:
        cli.main()
    return exit_.value.code


def test_every_subcommand_is_declared():
    assert sorted(cli.RUN_MANIFESTS) == sorted(subcommand_names(cli.build_parser()))


def test_holdout_cards_writes_its_manifest(tmp_path: Path, monkeypatch):
    out = tmp_path / "holdout-run" / "holdout-cards.txt"
    cards = tmp_path / "cardsfolder"

    def holdout_cards(args):
        args.out.write_text("Lightning Bolt\n", encoding="utf-8")
        return 0

    monkeypatch.setattr(cli, "run_holdout_cards", holdout_cards)
    argv = ["holdout-cards", "--out", str(out), "--cards-folder", str(cards),
            "--printings-path", str(tmp_path / "AllPrintings.json")]

    assert run_main(monkeypatch, argv) == 0

    document = json.loads((out.parent / "run.json").read_text(encoding="utf-8"))
    assert document["run"] == out.parent.resolve().as_posix()
    assert (document["status"], document["outcome"]) == ("finished", "exit 0")
    assert document["commands"] == ["python -m effects " + " ".join(argv)]
    assert document["inputs"] == [
        cards.resolve().as_posix(), (tmp_path / "AllPrintings.json").resolve().as_posix(),
    ]


def test_an_exception_is_recorded_and_still_raised(tmp_path: Path, monkeypatch):
    out = tmp_path / "corpus-run"

    def build_corpus(args):
        (Path(args.output) / "manifest.previous.json").write_text("{}", encoding="utf-8")
        raise RuntimeError("a shard holds a mixed generation")

    monkeypatch.setattr(cli, "run_build_corpus", build_corpus)
    monkeypatch.setattr(sys, "argv", ["effects", *cli_args(["build-corpus", "--output", str(out)])])

    with pytest.raises(RuntimeError):
        cli.main()

    document = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert document["status"] == "failed"
    assert document["outcome"] == "exit 1: RuntimeError: a shard holds a mixed generation"


def test_verify_writes_none(tmp_path: Path, monkeypatch):
    out = tmp_path / "corpus-run"
    monkeypatch.setattr(cli, "run_build_corpus", lambda _args: 0)
    assert run_main(monkeypatch, cli_args(["build-corpus", "--output", str(out), "--verify"])) == 0
    assert not out.exists()


def test_a_collector_records_in_the_corpus_it_extends():
    corpus = "output/effects/gen3/records/2026-11-13-depleted"
    args = parse_cli(["collect-coverage", "--training-corpus", corpus])
    spec = cli.RUN_MANIFESTS["collect-coverage"]
    assert spec.run_dir(args) == Path(corpus)
    assert Path(corpus) / "holdout-cards.txt" in spec.input_paths(args)


def test_the_trainer_manifests_the_directory_it_is_given():
    args = parse_cli([
        "train-effect-model", "--corpus", "c", "--variant", "identity",
        "--model-output", "models/effects/gen3/effect-model/run",
    ])
    assert cli.RUN_MANIFESTS["train-effect-model"].run_dir(args) == Path(
        "models/effects/gen3/effect-model/run",
    )


def test_build_vocab_manifests_the_directory_of_its_vocabulary():
    args = parse_cli([
        "build-vocab", "--vocab-path", "output/effects/gen3/inputs/run/vocab.txt",
        "--keyword-definitions", "k.json",
    ])
    assert cli.RUN_MANIFESTS["build-vocab"].run_dir(args) == Path(
        "output/effects/gen3/inputs/run",
    )


def test_shard_discovery_never_takes_the_manifest_for_a_shard(tmp_path: Path):
    (tmp_path / "nested").mkdir()
    (tmp_path / "run.json").write_text("{}", encoding="utf-8")
    (tmp_path / "nested" / "run.json").write_text("{}", encoding="utf-8")
    assert iter_shards(tmp_path) == []


class TestTheCollectorsListWhatTheyRead:
    """P1 #10: ``collect-variants`` listed neither the variant tree it reads
    and extends nor the rules catalogue its steered round reads; V4 #5:
    ``collect-coverage --count-from`` reads earlier runs."""

    def test_collect_variants_lists_its_variant_tree(self):
        args = parse_cli([
            "collect-variants", "--variant-scripts", "output/effects/gen3/inputs/variants",
        ])
        inputs = cli.RUN_MANIFESTS["collect-variants"].input_paths(args)
        assert Path("output/effects/gen3/inputs/variants") in inputs
        assert Path("rules") not in inputs, "the untargeted round reads no catalogue"

    def test_the_steered_round_lists_the_rules_catalogue(self):
        args = parse_cli([
            "collect-variants", "--round", "steered", "--rule-counts", "counts.json",
        ])
        inputs = cli.RUN_MANIFESTS["collect-variants"].input_paths(args)
        assert Path("rules") in inputs
        assert Path("counts.json") in inputs

    def test_collect_coverage_lists_every_count_from_directory(self):
        args = parse_cli([
            "collect-coverage", "--count-from", "runs/a", "--count-from", "runs/b",
        ])
        inputs = cli.RUN_MANIFESTS["collect-coverage"].input_paths(args)
        assert {Path("runs/a"), Path("runs/b")} <= set(inputs)


def test_shard_discovery_never_takes_the_sealed_corpora_for_a_shard(tmp_path: Path):
    """A match-outcomes collection keeps match-outcomes.txt and cards-played.txt
    in its run directory, beside the shards every reader walks."""
    (tmp_path / "match-outcomes.txt").write_text("row\n", encoding="utf-8")
    (tmp_path / "cards-played.txt").write_text("row\n", encoding="utf-8")
    assert iter_shards(tmp_path) == []

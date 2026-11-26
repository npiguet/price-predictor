"""The commands name every path they read and write (artifact-layout rules 2 and 3).

No command runs on a default path that predates the layout: each flag that
once defaulted to one is required, a refusal's reason reaches ``run.json``,
and ``run.json`` lists the paths a command takes from a corpus manifest or a
checkpoint as well as those a flag names.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from effects.infrastructure import cli
from tests.unit.effects.cli_args import REQUIRED_FLAGS, cli_args, parse_cli

#: The flags a command takes beyond the ones in ``REQUIRED_FLAGS``.
EXTRA_ARGUMENTS = {"train-effect-model": ["--corpus", "corpus"]}

REQUIRED_CASES = [
    (command, flag) for command, flags in REQUIRED_FLAGS.items() for flag in flags
]


@pytest.mark.parametrize(("command", "flag"), REQUIRED_CASES)
def test_a_path_flag_that_once_had_a_default_is_refused_when_absent(command, flag, capsys):
    arguments = [command, *EXTRA_ARGUMENTS.get(command, [])]
    complete = cli_args(arguments)
    at = complete.index(flag)
    del complete[at:at + 2]

    with pytest.raises(SystemExit) as refused:
        cli.build_parser().parse_args(complete)

    assert refused.value.code == 2, "argparse's usage error keeps its own exit code"
    assert flag in capsys.readouterr().err


def test_no_flag_help_prints_a_pre_layout_default(capsys):
    """``--help`` names no ``output/effects/records/``-style default path."""
    for command in REQUIRED_FLAGS:
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args([command, "--help"])
        help_text = capsys.readouterr().out
        for pre_layout in (
            "output/effects/records", "output/effects/reports", "output/effects/abilities",
            "output/effects/corpus", "output/effects/variant-scripts", "models/effects/vocab",
            "output/effects/keyword-definitions", "output/effects/rule-counts",
            "output/sealed",
        ):
            assert pre_layout not in help_text, f"{command} --help names {pre_layout}"


def _main(monkeypatch, argv: list[str]) -> int | None:
    monkeypatch.setattr(sys, "argv", ["effects", *argv])
    with pytest.raises(SystemExit) as finished:
        cli.main()
    return finished.value.code


def test_a_refusal_s_reason_is_written_to_run_json(tmp_path: Path, monkeypatch):
    """A refusal mapped by ``exits_with`` reaches the manifest with the message
    the console shows, not a bare ``exit 1``."""
    cache = tmp_path / "cache-run"

    def refuse(config):
        (config.output_root / "partial.npz").write_bytes(b"")
        raise FileNotFoundError("no source tree at output/effects/gen3/converted/cardsfolder")

    cache.mkdir()
    monkeypatch.setattr("effects.application.encode_abilities.run", refuse)

    code = _main(monkeypatch, [
        "encode-abilities", "--output-root", str(cache), "--checkpoint", str(tmp_path / "x.pt"),
    ])

    assert code == 1
    document = json.loads((cache / "run.json").read_text(encoding="utf-8"))
    assert document["status"] == "failed"
    assert document["outcome"] == (
        "exit 1: no source tree at output/effects/gen3/converted/cardsfolder"
    )


def _corpus_recording_variants(root: Path, variants: str) -> Path:
    from effects.infrastructure.corpus_store import CorpusStore
    from tests.unit.effects.domain.test_corpus_manifest import manifest

    corpus = root / "corpus"
    corpus.mkdir()
    CorpusStore(corpus).save(manifest(variant_scripts=variants))
    return corpus


def test_the_trainer_lists_the_variant_tree_it_takes_from_the_corpus_manifest(tmp_path: Path):
    corpus = _corpus_recording_variants(tmp_path, "output/effects/gen3/inputs/2026-11-01-variants")
    args = parse_cli(["train-effect-model", "--corpus", str(corpus)])

    inputs = cli.RUN_MANIFESTS["train-effect-model"].input_paths(args)

    assert Path("output/effects/gen3/inputs/2026-11-01-variants") in inputs
    assert corpus in inputs


def test_the_flag_s_variant_tree_replaces_the_manifest_s(tmp_path: Path):
    corpus = _corpus_recording_variants(tmp_path, "output/effects/gen3/inputs/recorded")
    args = parse_cli([
        "train-effect-model", "--corpus", str(corpus), "--variant-scripts", "inputs/chosen",
    ])

    inputs = cli.RUN_MANIFESTS["train-effect-model"].input_paths(args)

    assert Path("inputs/chosen") in inputs
    assert Path("output/effects/gen3/inputs/recorded") not in inputs


def _checkpoint(root: Path, corpus: Path) -> tuple[Path, Path, Path]:
    """A saved checkpoint recording a vocabulary, a keyword table and ``corpus``."""
    import torch

    from effects.domain.ability_encoder import AbilityEncoder, AbilityEncoderConfig
    from effects.domain.checkpoint_layout import layout_settings
    from effects.domain.effect_model import EffectModel, EffectModelConfig
    from effects.domain.settings_shape import SETTINGS_SHAPE, SETTINGS_SHAPE_KEY
    from effects.domain.split_provenance import SplitProvenance
    from effects.infrastructure.effect_model_store import (
        EffectCheckpoint,
        EffectModelStore,
        content_hash,
    )

    vocab, keywords = root / "recorded-vocab.txt", root / "recorded-keywords.json"
    vocab.write_text("[PAD]\n[UNK]\n", encoding="utf-8")
    keywords.write_text("{}", encoding="utf-8")
    model = EffectModelConfig(
        global_features=3, act_features=3, player_features=3, card_features=3,
        e_dim=4, d_model=8, n_layers=1, n_heads=2, ff_dim=16,
    )
    torch.manual_seed(0)
    encoder = AbilityEncoderConfig(vocab_size=10, e_dim=4, d_model=8, n_layers=1)
    store = EffectModelStore(root / "models")
    store.save(EffectCheckpoint(
        encoder_config=encoder,
        model_config=model, encoder_state=AbilityEncoder(encoder).state_dict(),
        model_state=EffectModel(model).state_dict(),
        provenance=SplitProvenance(
            vocab_path=str(vocab), keyword_definitions_path=str(keywords),
            vocab_hash=content_hash(vocab), keyword_definitions_hash=content_hash(keywords),
            corpus_path=str(corpus), corpus_digest="digest",
        ),
        training_settings={**layout_settings(), SETTINGS_SHAPE_KEY: SETTINGS_SHAPE},
    ))
    return store.latest_path(), vocab, keywords


def test_the_evaluator_lists_what_it_takes_from_the_checkpoint(tmp_path: Path):
    corpus = _corpus_recording_variants(tmp_path, "output/effects/gen3/inputs/recorded")
    checkpoint, vocab, keywords = _checkpoint(tmp_path, corpus)
    args = parse_cli(["evaluate-effect-model", "--checkpoint", str(checkpoint)])

    inputs = cli.RUN_MANIFESTS["evaluate-effect-model"].input_paths(args)

    assert {vocab, keywords, corpus, Path("output/effects/gen3/inputs/recorded")} <= set(inputs)


def test_encode_abilities_lists_the_vocabulary_and_keywords_the_checkpoint_recorded(
    tmp_path: Path,
):
    checkpoint, vocab, keywords = _checkpoint(tmp_path, tmp_path / "corpus")
    args = parse_cli(["encode-abilities", "--checkpoint", str(checkpoint)])

    inputs = cli.RUN_MANIFESTS["encode-abilities"].input_paths(args)

    assert {checkpoint, vocab, keywords} <= set(inputs)
    assert tmp_path / "corpus" not in inputs, "the cache is built without the corpus"


def test_an_unreadable_checkpoint_lists_the_flags_alone(tmp_path: Path):
    args = parse_cli([
        "evaluate-effect-model", "--checkpoint", str(tmp_path / "absent.pt"),
        "--vocab-path", "given-vocab.txt",
    ])

    inputs = cli.RUN_MANIFESTS["evaluate-effect-model"].input_paths(args)

    assert Path("given-vocab.txt") in inputs


def test_the_keyword_extraction_s_refusal_reaches_run_json(tmp_path: Path, monkeypatch):
    """The worker's failure is a refusal with a reason, not a bare ``exit 1``."""
    output = tmp_path / "inputs" / "keywords.json"
    output.parent.mkdir()

    def no_worker(path):
        raise FileNotFoundError("the connector jar is not built: run mvn install")

    monkeypatch.setattr(
        "effects.infrastructure.one_shot_workers.write_keyword_definitions", no_worker,
    )

    code = _main(monkeypatch, ["extract-keyword-definitions", "--output", str(output)])

    assert code == 1
    document = json.loads((output.parent / "run.json").read_text(encoding="utf-8"))
    assert document["outcome"] == "exit 1: the connector jar is not built: run mvn install"


def test_the_trainer_records_its_seed_and_device_in_run_json(tmp_path: Path):
    """A seed drawn from the OS is in ``run.json``, not in ``console.log`` alone."""
    from effects.application.train_effect_model import TrainEffectModelConfig
    from price_predictor.infrastructure.run_manifest import RunCommand, run_recorded
    from tests.unit.effects.builders import make_training_loop

    model_output = tmp_path / "model"
    config = TrainEffectModelConfig.unpathed(corpus=str(tmp_path), model_output=model_output)

    run_recorded(
        model_output, lambda: make_training_loop(config) and 0,
        RunCommand("train-effect-model"),
    )

    entry = json.loads((model_output / "run.json").read_text(encoding="utf-8"))["invocations"][-1]
    assert isinstance(entry["seed"], int)
    assert entry["device"] in {"cpu", "cuda", "cuda:0", "mps"}

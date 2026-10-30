"""What ``train-effect-model`` says before it starts stepping (FR-088b, FR-146).

The startup line sizes the holdout by ability texts, which the split is keyed
on, as well as by the cards that carry them.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from effects.application.train_effect_model import (
    CorpusFacts,
    HoldoutMismatchError,
    MissingTrainingInputError,
    TrainEffectModelConfig,
    run,
)
from effects.domain.corpus_manifest import CorpusManifest
from effects.domain.shard_generation import MixedGenerationsError
from effects.domain.text_holdout import HoldoutUnit
from effects.infrastructure.corpus_store import CorpusStore
from effects.infrastructure.shard_io import write_shard

#: Three texts ``text_is_held_out`` selects at the default permille, found by
#: search: the trainer recomputes the holdout from its sidecars at startup and
#: refuses a corpus whose recorded texts it cannot reproduce (FR-041).
_HELD_OUT_TEXTS = (
    "SP$ Bespoke | Weird$ 249", "SP$ Bespoke | Weird$ 281", "SP$ Bespoke | Weird$ 82",
)
_CARD_TEXTS = {
    "Lightning Bolt": [_HELD_OUT_TEXTS[0], _HELD_OUT_TEXTS[1]],
    "Shock": [_HELD_OUT_TEXTS[2]],
    "Grizzly Bears": ["SP$ Bespoke | Weird$ 0"],
}


def _card_tree(root):
    """A converted tree whose holdout recomputes to ``_HELD_OUT_TEXTS``."""
    from effects.domain.provenance import ProvenanceKey, ProvenanceSidecar, SidecarLine
    from effects.infrastructure.sidecar_io import sidecar_path_for
    from tests.unit.effects.builders import write_sidecar

    for name, texts in _CARD_TEXTS.items():
        stem = name.lower().replace(" ", "_")
        script = f"cardsfolder/{stem[0]}/{stem}.txt"
        path = root / stem[0] / f"{stem}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"name: {name}\n", encoding="utf-8")
        write_sidecar(ProvenanceSidecar(card=name, script_file=script, lines=tuple(
            SidecarLine(line_index=i, line_kind="spell",
                        provenance=(ProvenanceKey(script, 0, "spell", i),), script_text=text)
            for i, text in enumerate(texts)
        )), sidecar_path_for(path))
    return root


def _manifest(**overrides) -> CorpusManifest:
    fields = dict(
        seed=1, surface="script", vocab_path="models/effects/vocab-script.txt",
        variant_scripts="", holdout_permille=20, holdout_max_carriers=8,
        text_cap=0, card_disjoint_text_cap=0, game_disjoint_target=0,
        training_records=1, class_mix={}, delivered_mix={},
        held_out_cards=("Lightning Bolt", "Shock"),
        held_out_texts=_HELD_OUT_TEXTS,
        card_disjoint_games=("cg",), game_disjoint_games=("gg",),
        rarity={}, sources=(), per_class={},
        per_stratum={"gate-one": 4096}, unique_texts={}, shortfall={},
    )
    fields.update(overrides)
    return CorpusManifest(**fields)


@pytest.fixture
def corpus(tmp_path, make_record):
    """A curated dataset with just enough on disk for ``run`` to reach the loop."""
    store = CorpusStore(tmp_path / "corpus")
    store.training_dir.mkdir(parents=True)
    store.samples_dir.mkdir(parents=True)
    write_shard(store.training_dir / "t.jsonl.gz",
                [make_record(record_id="t0", game_id="tg")])
    for stratum in ("card-disjoint", "game-disjoint"):
        write_shard(store.sample_path(stratum),
                    [make_record(record_id=stratum, game_id=stratum)])
    store.save(_manifest())
    return store


@pytest.fixture
def config(corpus, tmp_path):
    return TrainEffectModelConfig(
        corpus=str(corpus.directory),
        cards_folders=(_card_tree(tmp_path / "cardsfolder"),),
        vocab_path="models/effects/vocab-script.txt",
    )


@pytest.fixture(autouse=True)
def started(monkeypatch):
    """Stop at the loop's door: these tests are about what ``run`` logs and
    hands the loop. Yields every loop that reached it."""
    from effects.application import training_loop

    loops = []

    def execute(self):
        loops.append(self)
        return 0

    monkeypatch.setattr(training_loop.TrainingLoop, "execute", execute)
    return loops


def test_the_startup_line_counts_held_out_texts_not_only_cards(config, caplog):
    """The split is keyed on ability texts; cards are what carries them.

    A line reporting only the card count says nothing about the population
    gate 1 is scored over.
    """
    with caplog.at_level(logging.INFO):
        assert run(config) == 0

    assert "3 held-out ability text(s) on 2 card(s)" in caplog.text


def test_a_corpus_without_training_shards_is_refused(corpus, config, started):
    (corpus.training_dir / "t.jsonl.gz").unlink()
    with pytest.raises(MissingTrainingInputError, match="No training shards under"):
        run(config)
    assert not started


def test_a_corpus_without_validation_samples_is_refused(corpus, config, started):
    corpus.sample_path("game-disjoint").unlink()
    with pytest.raises(MissingTrainingInputError, match="has no validation samples"):
        run(config)
    assert not started


def test_the_loop_reads_the_corpus_through_the_manifests_facts(corpus, config, started):
    assert run(config) == 0
    facts = started[-1].spec.facts
    manifest = corpus.load()
    assert facts == CorpusFacts.of(manifest)
    assert facts.split.card_disjoint_games == frozenset({"cg"})
    assert facts.split.game_disjoint_games == frozenset({"gg"})
    assert facts.gate_one_records == 4096
    assert facts.digest == manifest.digest()


# ── the variant-script tree ──────────────────────────────────────────────


def test_an_unset_variant_tree_is_the_one_the_corpus_was_built_against(
    corpus, config, started, caplog,
):
    """Evaluation takes it from the manifest; a training run that did not
    would read every variant ability the build kept as no text."""
    corpus.save(_manifest(variant_scripts="output/effects/variant-scripts"))
    with caplog.at_level(logging.INFO):
        assert run(config) == 0
    assert started[-1].config.variant_scripts == Path("output/effects/variant-scripts")
    assert "the corpus manifest's" in caplog.text


def test_a_named_variant_tree_wins_over_the_manifests(corpus, config, started, caplog):
    corpus.save(_manifest(variant_scripts="output/effects/variant-scripts"))
    config.variant_scripts = Path("elsewhere")
    with caplog.at_level(logging.INFO):
        assert run(config) == 0
    assert started[-1].config.variant_scripts == Path("elsewhere")
    assert "elsewhere (--variant-scripts)" in caplog.text


def test_no_variant_tree_is_read_when_neither_names_one(config, started, caplog):
    with caplog.at_level(logging.INFO):
        assert run(config) == 0
    assert started[-1].config.variant_scripts is None
    assert "No variant-script tree" in caplog.text


# ── encoder size and sidecar roots (FR-061, FR-062, FR-063b) ─────────────


def _train_args(tmp_path, *extra: str):
    from effects.infrastructure.cli import build_parser

    return build_parser().parse_args(["train-effect-model", "--corpus", str(tmp_path), *extra])


def test_the_encoder_size_flags_default_to_four_and_256(tmp_path):
    args = _train_args(tmp_path)
    assert (args.encoder_layers, args.encoder_d_model) == (4, 256)
    assert (args.e_noise, args.value_weight) == (0.1, 0.05)


def test_a_width_not_divisible_by_the_head_count_is_refused_before_training(
    tmp_path, monkeypatch, caplog,
):
    """Spec Story 5 scenario 7."""
    from effects.application import train_effect_model
    from effects.infrastructure import cli

    started = []
    monkeypatch.setattr(train_effect_model, "run", lambda config: started.append(1))
    args = _train_args(tmp_path, "--encoder-d-model", "250")
    with caplog.at_level(logging.ERROR):
        assert cli.run_train_effect_model(args) == 2
    assert not started
    assert "not divisible" in caplog.text


@pytest.mark.parametrize("failure, code", [
    ("MissingTrainingInputError", 1),
    ("HoldoutMismatchError", 1),
    ("MixedGenerationsError", 1),
    ("ValidationGameInTrainingError", 1),
    ("ResumeMismatchError", 2),
    ("FileNotFoundError", 2),
])
def test_an_operator_error_out_of_training_exits_with_its_code(
    tmp_path, monkeypatch, caplog, failure, code,
):
    """A refused corpus, a resume under other settings, and a missing manifest
    or resume file end in one logged line, not a traceback."""
    from effects.application import train_effect_model
    from effects.infrastructure import cli, effect_model_store

    kind = {
        "MissingTrainingInputError": train_effect_model.MissingTrainingInputError,
        "HoldoutMismatchError": train_effect_model.HoldoutMismatchError,
        "MixedGenerationsError": MixedGenerationsError,
        "ValidationGameInTrainingError": train_effect_model.ValidationGameInTrainingError,
        "ResumeMismatchError": effect_model_store.ResumeMismatchError,
        "FileNotFoundError": FileNotFoundError,
    }[failure]

    def refuse(config):
        raise kind("refused for the test")

    monkeypatch.setattr(train_effect_model, "run", refuse)
    with caplog.at_level(logging.ERROR):
        assert cli.run_train_effect_model(_train_args(tmp_path)) == code
    assert "refused for the test" in caplog.text


@pytest.mark.parametrize(
    "shape", [{"encoder_d_model": 250}, {"encoder_d_model": 0}, {"encoder_layers": 0}],
)
def test_the_config_refuses_an_encoder_shape_it_cannot_build(shape):
    with pytest.raises(ValueError, match="--encoder"):
        TrainEffectModelConfig(**shape)


def test_the_size_flags_reach_the_trainers_config(tmp_path):
    from effects.infrastructure.cli import config_from_args

    config = config_from_args(TrainEffectModelConfig, _train_args(
        tmp_path, "--encoder-layers", "6", "--encoder-d-model", "384",
        "--value-weight", "0.2", "--e-noise", "0.05",
    ))
    assert (config.encoder_layers, config.encoder_d_model) == (6, 384)
    assert (config.value_weight, config.e_noise) == (0.2, 0.05)


def test_a_kept_aside_tree_resolves_the_keys_of_its_original(tmp_path):
    """Stage 0 reads gen-1's sidecars from a copy while stage 1 reconverts."""
    import shutil
    from pathlib import Path

    from effects.domain.provenance import ProvenanceKey
    from tests.unit.effects.builders import make_preparer

    fixture = Path(__file__).parents[3] / "fixtures" / "effects" / "gen1-sidecars"
    kept = tmp_path / "gen1-cardsfolder"
    shutil.copytree(fixture / "cardsfolder", kept)
    key = ProvenanceKey("cardsfolder/q/quintessential_katana.txt", 0, "static", 0)
    config = TrainEffectModelConfig(corpus=str(tmp_path), cards_folders=(kept,))

    trainer = make_preparer(tmp_path, cards_folders=(kept,)).sidecars
    for sidecars in (trainer, config.source_trees.sidecar_cache()):
        assert sidecars.path_for(key.script_file).parent.parent == kept
        sidecars.resolution_of(key)  # neither raises nor reports no tree


# ── the holdout unit and its recompute (FR-041, FR-042; T043) ────────────


def test_the_trainer_takes_the_manifests_unit_when_the_flag_is_absent(config, started):
    """A gen-1 manifest records no unit and reads as ``text``."""
    assert run(config) == 0
    assert started[-1].spec.facts.holdout.unit is HoldoutUnit.TEXT
    assert config.holdout_unit is None


def test_a_unit_differing_from_the_manifests_is_refused(config):
    config.holdout_unit = HoldoutUnit.TEMPLATE
    with pytest.raises(HoldoutMismatchError, match="--holdout-unit template differs"):
        run(config)


def test_a_recomputed_holdout_that_disagrees_is_refused(corpus, config):
    corpus.save(_manifest(held_out_texts=("SP$ Bespoke | Weird$ 82",)))
    with pytest.raises(HoldoutMismatchError, match="recomputed"):
        run(config)


def test_a_real_gen1_manifest_is_accepted_under_its_text_unit(
    tmp_path, make_record, started,
):
    """The trimmed real gen-1 manifest records no unit; the trainer reads text."""
    import json
    from pathlib import Path

    from effects.domain.corpus_manifest import CorpusManifest as Manifest

    raw = json.loads((Path(__file__).parents[3] / "fixtures" / "effects"
                      / "gen1-manifest-trimmed.json").read_text(encoding="utf-8"))
    raw["held_out_texts"] = list(_HELD_OUT_TEXTS)
    store = CorpusStore(tmp_path / "corpus")
    store.training_dir.mkdir(parents=True)
    store.samples_dir.mkdir(parents=True)
    write_shard(store.training_dir / "t.jsonl.gz", [make_record(record_id="t0", game_id="tg")])
    for stratum in ("card-disjoint", "game-disjoint"):
        write_shard(store.sample_path(stratum), [make_record(record_id=stratum, game_id=stratum)])
    store.save(Manifest.from_dict(raw))
    config = TrainEffectModelConfig(
        corpus=str(store.directory), cards_folders=(_card_tree(tmp_path / "cardsfolder"),),
        vocab_path="models/effects/vocab-script.txt",
    )

    assert run(config) == 0
    assert started[-1].spec.facts.holdout.unit is HoldoutUnit.TEXT


def test_a_records_set_mixing_generations_is_refused(corpus, config, make_record):
    """FR-033: refused before any record is read."""
    from effects.infrastructure.record_codec import record_to_dict

    gen1 = record_to_dict(make_record(record_id="old", game_id="og"))
    del gen1["random_seat"]
    import gzip
    import json

    with gzip.open(corpus.training_dir / "gen1.jsonl.gz", "wt", encoding="utf-8") as out:
        out.write(json.dumps(gen1) + "\n")
    with pytest.raises(MixedGenerationsError, match="mixes gen-1 shards"):
        run(config)

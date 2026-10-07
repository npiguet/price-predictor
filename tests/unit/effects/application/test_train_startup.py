"""What ``train-effect-model`` says before it starts stepping (FR-088b, FR-146).

Three of these cover flags whose help text promised something the trainer does
not do: ``--split-from`` inherits nothing now that ``--corpus`` is required, and
``--context-cache`` is constructed but never read. The fourth covers the
startup line, which sized the holdout by cards while the split is keyed on
ability texts.
"""

from __future__ import annotations

import logging

import pytest

from effects.application.train_effect_model import (
    TrainEffectModelConfig,
    run,
)
from effects.domain.corpus_manifest import CorpusManifest
from effects.infrastructure.corpus_store import CorpusStore
from effects.infrastructure.record_io import write_shard
from tests.unit.effects.domain.conftest import (  # noqa: F401
    ability_key,
    make_record,
    snapshot,
)

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
    from effects.infrastructure.sidecar_io import sidecar_path_for, write_sidecar

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
def corpus(tmp_path, make_record):  # noqa: F811
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
def _no_training(monkeypatch):
    """Stop at the loop's door: these tests are about what ``run`` logs."""
    from effects.application import training_loop

    monkeypatch.setattr(
        training_loop.TrainingLoop, "execute", lambda self: 0,
    )


def test_split_from_is_reported_as_read_by_nothing(config, caplog):
    config.split_from = "models/effects/effect-model/latest.pt"

    with caplog.at_level(logging.WARNING):
        assert run(config) == 0

    assert "--split-from" in caplog.text
    assert "nothing is read" in caplog.text


def test_no_split_from_warning_when_the_flag_is_absent(config, caplog):
    with caplog.at_level(logging.WARNING):
        assert run(config) == 0

    assert "--split-from" not in caplog.text


def test_context_cache_is_reported_as_not_wired_in(config, caplog):
    config.context_cache = True

    with caplog.at_level(logging.WARNING):
        assert run(config) == 0

    assert "--context-cache" in caplog.text
    assert "re-encoded live" in caplog.text


def test_the_startup_line_counts_held_out_texts_not_only_cards(config, caplog):
    """The split is keyed on ability texts; cards are what carries them.

    A line reporting only the card count says nothing about the population
    gate 1 is scored over.
    """
    with caplog.at_level(logging.INFO):
        assert run(config) == 0

    assert "3 held-out ability text(s) on 2 card(s)" in caplog.text


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


def test_the_size_flags_reach_the_trainers_config(tmp_path):
    from effects.infrastructure.cli import train_config_from

    config = train_config_from(_train_args(
        tmp_path, "--encoder-layers", "6", "--encoder-d-model", "384",
        "--value-weight", "0.2", "--e-noise", "0.05",
    ))
    assert (config.encoder_layers, config.encoder_d_model) == (6, 384)
    assert (config.value_weight, config.e_noise) == (0.2, 0.05)


def test_a_kept_aside_tree_resolves_the_keys_of_its_original(tmp_path):
    """Stage 0 reads gen-1's sidecars from a copy while stage 1 reconverts."""
    import shutil
    from pathlib import Path

    from effects.application.training_loop import TrainingLoop
    from effects.domain.provenance import ProvenanceKey
    from effects.infrastructure.model_runner import build_sidecars

    fixture = Path(__file__).parents[3] / "fixtures" / "effects" / "gen1-sidecars"
    kept = tmp_path / "gen1-cardsfolder"
    shutil.copytree(fixture / "cardsfolder", kept)
    key = ProvenanceKey("cardsfolder/q/quintessential_katana.txt", 0, "static", 0)
    config = TrainEffectModelConfig(corpus=str(tmp_path), cards_folders=(kept,))

    loop = TrainingLoop.__new__(TrainingLoop)
    loop.config = config
    for sidecars in (loop._build_sidecars(), build_sidecars(config)):
        assert sidecars.path_for(key.script_file).parent.parent == kept
        sidecars.resolution_of(key)  # neither raises nor reports no tree


# ── the holdout unit and its recompute (FR-041, FR-042; T043) ────────────


def test_the_trainer_takes_the_manifests_unit_when_the_flag_is_absent(config):
    """A gen-1 manifest records no unit and reads as ``text``."""
    assert run(config) == 0
    assert config.holdout_unit == "text"


def test_a_unit_differing_from_the_manifests_is_refused(config, caplog):
    config.holdout_unit = "template"
    with caplog.at_level(logging.ERROR):
        assert run(config) == 1
    assert "--holdout-unit template differs" in caplog.text


def test_a_recomputed_holdout_that_disagrees_is_refused(corpus, config, caplog):
    corpus.save(_manifest(held_out_texts=("SP$ Bespoke | Weird$ 82",)))
    with caplog.at_level(logging.ERROR):
        assert run(config) == 1
    assert "recomputed" in caplog.text


def test_a_real_gen1_manifest_is_accepted_under_its_text_unit(tmp_path, make_record):  # noqa: F811
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
    assert config.holdout_unit == "text"


def test_a_records_set_mixing_generations_is_refused(corpus, config, caplog, make_record):  # noqa: F811
    """FR-033: refused before any record is read."""
    from effects.infrastructure.record_io import record_to_dict

    gen1 = record_to_dict(make_record(record_id="old", game_id="og"))
    del gen1["random_seat"]
    import gzip
    import json

    with gzip.open(corpus.training_dir / "gen1.jsonl.gz", "wt", encoding="utf-8") as out:
        out.write(json.dumps(gen1) + "\n")
    with caplog.at_level(logging.ERROR):
        assert run(config) == 1
    assert "mixes gen-1 shards" in caplog.text

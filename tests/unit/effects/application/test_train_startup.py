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


def _manifest(**overrides) -> CorpusManifest:
    fields = dict(
        seed=1, surface="script", vocab_path="models/effects/vocab-script.txt",
        variant_scripts="", holdout_permille=20, holdout_max_carriers=8,
        text_cap=0, card_disjoint_text_cap=0, game_disjoint_target=0,
        training_records=1, class_mix={}, delivered_mix={},
        held_out_cards=("Lightning Bolt", "Shock"),
        held_out_texts=("a", "b", "c"),
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
def config(corpus):
    return TrainEffectModelConfig(
        corpus=str(corpus.directory),
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

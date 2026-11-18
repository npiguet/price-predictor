"""Fixtures the application tests share: the gen-3 records, and a small
corpus over them that a real training run can read."""

from __future__ import annotations

import pytest
import torch

from effects.domain.ability_tokenizer import SPECIAL_TOKENS, AbilityTokenizer
from effects.domain.baselines import baseline_masks
from effects.infrastructure.shard_io import write_shard
from effects.infrastructure.sidecar_io import SidecarCache
from effects.infrastructure.tokenizer_loading import load_keyword_definitions
from price_predictor.infrastructure.tokenizer_store import save_vocabulary
from tests.unit.effects.application.geometry_checks_support import SourceRows, _write_source
from tests.unit.effects.builders import (
    FIXTURE_KEYWORD_DEFINITIONS,
    FIXTURE_TRAINING_SHARDS,
    GEN3_SIDECAR_FOLDERS,
    curated_fixture_records,
    make_batcher,
)
from tests.unit.effects.domain.conftest import ability_key, make_record, snapshot  # noqa: F401


@pytest.fixture(scope="module")
def records():
    return curated_fixture_records()


def _write_training_shards(root, records, held):
    shards = []
    for index in range(FIXTURE_TRAINING_SHARDS):
        path = root / f"train{index}.jsonl.gz"
        write_shard(path, [
            r for i, r in enumerate(records)
            if i % FIXTURE_TRAINING_SHARDS == index and r.game_id != held
        ])
        shards.append(path)
    return shards


def _write_vocabulary(root, records):
    cards, tokens = GEN3_SIDECAR_FOLDERS
    sidecars = SidecarCache({"cardsfolder": cards, "tokenscripts": tokens})
    definitions = load_keyword_definitions(FIXTURE_KEYWORD_DEFINITIONS)
    base = {token: index for index, token in enumerate(SPECIAL_TOKENS)}
    bare = AbilityTokenizer(base, definitions, surface="script")
    batcher = make_batcher(
        tokenizer=bare, sidecars=sidecars, masks=baseline_masks("full"),
        surface="script", e_dim=16, widths={}, device=torch.device("cpu"),
    )
    texts = [*batcher.batch_texts(records)]
    texts += [d.reminder_template for d in definitions.values() if d.reminder_template]
    vocab = dict(base)
    for text in texts:
        for token in bare.tokenize(text):
            vocab.setdefault(token.text, len(vocab))
    vocab_path = root / "vocab-script.txt"
    save_vocabulary(vocab, vocab_path)
    return vocab_path


@pytest.fixture(scope="module")
def corpus(tmp_path_factory, records):
    """Three training shards, the validation sample, and a vocabulary file.

    ``held`` is a validation game: its records are in the sample and in no
    training shard, as build-corpus routes them.
    """
    root = tmp_path_factory.mktemp("corpus")
    held = sorted({record.game_id for record in records})[0]
    shards = _write_training_shards(root, records, held)
    sample = root / "validation.jsonl.gz"
    write_shard(sample, records[:24])
    return type("Corpus", (), {
        "root": root, "shards": shards, "sample": sample, "held": held,
        "vocab": _write_vocabulary(root, records),
    })


@pytest.fixture
def fake_supervisor(monkeypatch):
    """The records-only supervisor a collector builds, stood in for.

    Auto-specced, so a method the real one lacks, or a call its signature
    refuses, fails here as it would there; never interrupted. The stand-in
    constructor is ``.constructor``, for the keywords a collector built it with.
    """
    from unittest.mock import create_autospec

    from effects.infrastructure import collector_connector

    supervisor = create_autospec(collector_connector.CollectorSupervisor, instance=True)
    supervisor.interrupted = False
    supervisor.constructor = create_autospec(
        collector_connector.CollectorSupervisor, return_value=supervisor)
    monkeypatch.setattr(collector_connector, "CollectorSupervisor", supervisor.constructor)
    return supervisor


@pytest.fixture
def cache(tmp_path):
    cards = tmp_path / "cardsfolder"
    (cards / "s").mkdir(parents=True)
    abilities = tmp_path / "abilities"
    _write_source(
        cards, abilities,
        SourceRows(
            "Serra Angel",
            "s/serra_angel",
            ["Flying", "Vigilance"],
            [[1.0, 0.0], [0.0, 1.0]],
            prose=["flying", "vigilance"],
        ),
    )
    _write_source(
        cards, abilities,
        SourceRows(
            "Shock",
            "s/shock",
            ["DealDamage | NumDmg$ 2"],
            [[0.7, 0.7]],
            prose=["CARDNAME deals 2 damage to any target."],
        ),
    )
    return cards, abilities


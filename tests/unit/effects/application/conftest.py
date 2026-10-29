"""Fixtures the application tests share: the gen-1 records, and a small
corpus over them that a real training run can read."""

from __future__ import annotations

import pytest
import torch

from effects.domain.ability_tokenizer import SPECIAL_TOKENS, AbilityTokenizer
from effects.domain.baselines import baseline_masks
from effects.infrastructure.record_io import read_shard, write_shard
from effects.infrastructure.sidecar_io import SidecarCache
from effects.infrastructure.tokenizer_loading import load_keyword_definitions
from price_predictor.infrastructure.tokenizer_store import save_vocabulary
from tests.unit.effects.builders import (
    EFFECTS_FIXTURES,
    GEN1_KEYWORD_DEFINITIONS,
    GEN1_SIDECAR_FOLDERS,
    GEN1_TRAINING_SHARDS,
    make_batcher,
)
from tests.unit.effects.domain.conftest import ability_key, make_record, snapshot  # noqa: F401


@pytest.fixture(scope="module")
def records():
    return list(read_shard(EFFECTS_FIXTURES / "gen1-records.jsonl.gz"))


@pytest.fixture(scope="module")
def corpus(tmp_path_factory, records):
    """Three training shards, the validation sample, and a vocabulary file."""
    root = tmp_path_factory.mktemp("corpus")
    games = sorted({record.game_id for record in records})
    held = games[0]
    shards = []
    for index in range(GEN1_TRAINING_SHARDS):
        path = root / f"train{index}.jsonl.gz"
        # Every shard keeps a few of the held game's records, which the
        # inherited split must keep out of training wherever it lands.
        write_shard(path, [
            r for i, r in enumerate(records)
            if i % GEN1_TRAINING_SHARDS == index or r.game_id == held
        ])
        shards.append(path)
    sample = root / "validation.jsonl.gz"
    write_shard(sample, records[:24])

    cards, tokens = GEN1_SIDECAR_FOLDERS
    sidecars = SidecarCache({"cardsfolder": cards, "tokenscripts": tokens})
    definitions = load_keyword_definitions(GEN1_KEYWORD_DEFINITIONS)
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
    return type("Corpus", (), {
        "root": root, "shards": shards, "sample": sample, "held": held,
        "vocab": vocab_path,
    })


@pytest.fixture
def fake_supervisor(monkeypatch):
    """The records-only supervisor a collector builds, stood in for.

    ``spec``-ed, so a method the real one lacks fails here as it would there,
    and never interrupted; the stand-in constructor is ``.constructor``, for
    the keywords a collector built it with.
    """
    from unittest.mock import MagicMock

    from effects.infrastructure import collector_connector

    supervisor = MagicMock(spec=collector_connector.CollectorSupervisor)
    supervisor.interrupted = False
    supervisor.constructor = MagicMock(return_value=supervisor)
    monkeypatch.setattr(collector_connector, "CollectorSupervisor", supervisor.constructor)
    return supervisor

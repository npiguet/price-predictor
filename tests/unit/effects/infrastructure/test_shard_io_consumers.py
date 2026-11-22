"""Shard IO: every consumer reads gen-3 records only (FR-020).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from effects.infrastructure.shard_io import (
    iter_shards,
)
from tests.unit.effects.builders import GEN3_RECORDS
from tests.unit.effects.infrastructure.shard_io_support import (
    _EARLIER,
    _curated,
    _older_shard,
)


def _build_corpus(root: Path, corpus: Path) -> None:
    from effects.application.build_corpus import _require_generation

    _require_generation(corpus)


def _train(root: Path, corpus: Path) -> None:
    from effects.application.train_effect_model import TrainEffectModelConfig, run
    from tests.unit.effects.application.test_train_startup import _card_tree
    from tests.unit.effects.builders import write_vocabulary

    keywords = root / "keyword-definitions.json"
    keywords.write_text("{}", encoding="utf-8")
    run(TrainEffectModelConfig.unpathed(
        corpus=str(corpus), cards_folders=(_card_tree(root / "cardsfolder"),),
        vocab_path=write_vocabulary(root / "vocab-script.txt", "script"),
        keyword_definitions=keywords,
    ))


def _evaluate(root: Path, corpus: Path) -> None:
    import torch

    from effects.application.evaluation_context import load_context
    from effects.domain.ability_encoder import AbilityEncoderConfig
    from effects.domain.checkpoint_layout import layout_settings
    from effects.domain.effect_model import EffectModel, EffectModelConfig
    from effects.domain.split_provenance import SplitProvenance
    from effects.infrastructure.corpus_store import CorpusStore
    from effects.infrastructure.effect_model_store import (
        EffectCheckpoint,
        EffectModelStore,
        content_hash,
    )
    from tests.unit.effects.configs import evaluation_config

    vocab, keywords = root / "vocab-script.txt", root / "kw.json"
    vocab.write_text("[PAD]\n[UNK]\n", encoding="utf-8")
    keywords.write_text("{}", encoding="utf-8")
    model = EffectModelConfig(
        global_features=3, act_features=3, player_features=3, card_features=3,
        e_dim=4, d_model=8, n_layers=1, n_heads=2, ff_dim=16,
    )
    torch.manual_seed(0)
    store = EffectModelStore(root / "models")
    store.save(EffectCheckpoint(
        encoder_config=AbilityEncoderConfig(vocab_size=10, e_dim=4, d_model=8, n_layers=1),
        model_config=model, encoder_state={},
        model_state=EffectModel(model).state_dict(),
        provenance=SplitProvenance(
            vocab_path=str(vocab), keyword_definitions_path=str(keywords),
            vocab_hash=content_hash(vocab), keyword_definitions_hash=content_hash(keywords),
            corpus_path=str(corpus), corpus_digest=CorpusStore(corpus).load().digest(),
        ),
        training_settings=layout_settings(),
    ))
    load_context(evaluation_config(checkpoint=store.latest_path()))


def _probe(root: Path, corpus: Path) -> None:
    from effect_knowledge_probes import common

    list(common.read_games(iter_shards(corpus), {"any game"}))


class TestEveryConsumerReadsGen3Only:
    """FR-020: ``build-corpus``, ``train-effect-model``,
    ``evaluate-effect-model`` and the knowledge probes refuse a gen-1 and a
    gen-2 records set before reading a record, naming its generation."""

    _CONSUMERS = {"build-corpus": _build_corpus, "train": _train,
                  "evaluate": _evaluate, "probes": _probe}

    @pytest.fixture(autouse=True)
    def _scripts_on_the_path(self):
        import sys

        scripts = str(Path(__file__).parents[4] / "scripts")
        if scripts not in sys.path:
            sys.path.insert(0, scripts)

    @pytest.mark.parametrize("generation", sorted(_EARLIER))
    @pytest.mark.parametrize("consumer", sorted(_CONSUMERS))
    def test_an_earlier_generation_is_refused_naming_its_shard(
        self, tmp_path, consumer, generation,
    ):
        from effects.domain.corpus_plan import BuildCorpusError
        from effects.domain.shard_generation import UnacceptedGenerationError

        corpus = _curated(tmp_path, _older_shard(tmp_path / "old.jsonl.gz", generation))
        with pytest.raises((UnacceptedGenerationError, BuildCorpusError),
                           match=rf"line 1: the record is {generation} \(it carries no "
                                 r"generation member\); every reader reads gen-3 records only"):
            self._CONSUMERS[consumer](tmp_path, corpus)

    def test_a_gen3_set_passes_the_check(self, tmp_path):
        from effects.infrastructure.shard_io import require_generation

        require_generation(iter_shards(_curated(tmp_path, GEN3_RECORDS)))

"""A tiny gen-3 checkpoint on disk, read as a probing run reads it."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from effect_knowledge_probes import (
    common,
    labels,
    ladder,
    record_walk,
    sweeps,
)
from effect_knowledge_probes.method_c_probe import method_c

from tests.unit.effects.builders import (
    GEN3_SIDECAR_FOLDERS,
)
from tests.unit.scripts.knowledge_probe_support import E_DIM, probabilities


@pytest.fixture(scope="module")
def checkpoint_path(tmp_path_factory, records, sidecars):
    """A gen-3 checkpoint at random weights, saved by the store as a training
    run saves one, with a vocabulary over the fixture's texts."""
    import torch

    from effects.application.run_context import training_settings
    from effects.application.train_effect_model import (
        CorpusFacts,
        RunSpec,
        TrainEffectModelConfig,
    )
    from effects.domain.ability_encoder import AbilityEncoder, AbilityEncoderConfig
    from effects.domain.ability_tokenizer import SPECIAL_TOKENS, AbilityTokenizer
    from effects.domain.baselines import baseline_masks
    from effects.domain.effect_model import EffectModel, EffectModelConfig
    from effects.domain.input_layout import SlotKind
    from effects.domain.slot_widths import layout_widths
    from effects.domain.split_provenance import SplitProvenance
    from effects.infrastructure.effect_model_store import EffectCheckpoint, EffectModelStore
    from effects.infrastructure.tokenizer_loading import load_keyword_definitions
    from effects.infrastructure.vocabulary_file import save_vocabulary
    from tests.unit.effects.batch_builders import make_batcher
    from tests.unit.effects.builders import FIXTURE_KEYWORD_DEFINITIONS

    root = tmp_path_factory.mktemp("checkpoint")
    definitions = load_keyword_definitions(FIXTURE_KEYWORD_DEFINITIONS)
    base = {token: index for index, token in enumerate(SPECIAL_TOKENS)}
    bare = AbilityTokenizer(base, definitions, surface="script")
    widths = layout_widths()
    probe = make_batcher(tokenizer=bare, sidecars=sidecars, masks=baseline_masks("full"),
                         surface="script", e_dim=E_DIM, widths=widths,
                         device=torch.device("cpu"))
    vocab = dict(base)
    texts = [*probe.batch_texts(records)]
    texts += [d.reminder_template for d in definitions.values() if d.reminder_template]
    for text in texts:
        for token in bare.tokenize(text):
            vocab.setdefault(token.text, len(vocab))
    vocab_path = root / "vocab-script.txt"
    save_vocabulary(vocab, "script", vocab_path)
    encoder_config = AbilityEncoderConfig(vocab_size=len(vocab), e_dim=E_DIM, d_model=16,
                                          n_layers=1, n_heads=2)
    model_config = EffectModelConfig(
        global_features=widths[SlotKind.GLOBAL], act_features=widths[SlotKind.ACT],
        player_features=widths[SlotKind.PLAYER], card_features=widths[SlotKind.CARD],
        ability_features=widths[SlotKind.ABILITY], e_dim=E_DIM, vocab_size=len(vocab),
        encoder_d_model=16, d_model=16, n_layers=2, n_heads=2, ff_dim=32,
    )
    torch.manual_seed(0)
    store = EffectModelStore(root / "effect-model")
    store.save(EffectCheckpoint(
        encoder_config=encoder_config, model_config=model_config,
        encoder_state=AbilityEncoder(encoder_config).state_dict(),
        model_state=EffectModel(model_config).state_dict(),
        provenance=SplitProvenance(
            vocab_path=str(vocab_path), keyword_definitions_path=str(FIXTURE_KEYWORD_DEFINITIONS),
            vocab_hash=content_hash(vocab_path),
            keyword_definitions_hash=content_hash(FIXTURE_KEYWORD_DEFINITIONS),
        ),
        training_settings=training_settings(
            RunSpec(TrainEffectModelConfig.unpathed(corpus="unused"), CorpusFacts(), seed=1),
            api_types=[], param_keys=[]),
    ))
    return store.latest_path()


def content_hash(path):
    from effects.infrastructure.effect_model_store import content_hash as digest

    return digest(path)


@pytest.fixture(scope="module")
def loaded(checkpoint_path, records):
    """The checkpoint loaded as a probing run loads it, on the CPU."""
    import torch

    from effects.application import model_loading

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(model_loading, "default_device", lambda: torch.device("cpu"))
        return common.load_probe_model(checkpoint_path, list(GEN3_SIDECAR_FOLDERS), records)


@pytest.fixture(scope="module")
def items_by_record(records, sidecars):
    join = labels.TriggerJoin()
    for record in records:
        join.observe(record)
    joined = join.result()
    acting = common.acting_text_function(sidecars, "script")
    return {r.record_id: found for r in records
            if (found := labels.record_items(r, sidecars, joined, acting))}


def _assert_one_width(rows) -> None:
    """One feature width per rung over a target's rows, the rungs' widths related as built."""
    for rung in ladder.PROBED_RUNGS:
        assert len({r.features[rung].shape for r in rows}) == 1
    zero = rows[0].features["0"].shape[0]
    assert rows[0].features["1"].shape[0] == zero + 3 * E_DIM
    assert rows[0].features["1w"].shape == rows[0].features["1"].shape
    assert all(np.isfinite(r.rung3) for r in rows)


class TestExtraction:
    """``ladder.extract_features`` on a gen-3 checkpoint loaded from disk."""

    def test_every_rung_has_one_width_per_target(self, loaded, records, items_by_record):
        chosen = [r for r in records if r.record_id in items_by_record]
        act_hidden: dict = {}
        extracted = ladder.extract_features(
            loaded, chosen, items_by_record,
            ladder.ExtractionOptions(batch_size=8, act_hidden=act_hidden))
        assert extracted and act_hidden
        assert loaded.device.type == "cpu"
        for target in {i.target for i in extracted}:
            _assert_one_width([i for i in extracted if i.target == target])

    def test_the_per_layer_view_reads_every_trunk_layer(self, loaded, records, items_by_record):
        record = next(r for r in records if r.record_id in items_by_record)
        extracted = ladder.extract_features(
            loaded, [record], {record.record_id: items_by_record[record.record_id]},
            ladder.ExtractionOptions(per_layer=True))
        depth = len(loaded.model.trunk.layers)
        assert depth == 2
        assert all(len(item.layers) == depth for item in extracted)
        assert extracted[0].layers[0].shape == extracted[0].features["2"].shape

    def test_a_toughness_sweep_reads_one_prediction_per_value(self, loaded, records, sidecars):
        selector = sweeps.SweepSelector(sidecars)
        for record in records:
            selector.observe(record)
        rows = selector.result()["toughness"]
        assert rows
        record = next(r for r in records if r.record_id == rows[0]["record_id"])
        curve = sweeps.run_sweep("toughness", loaded, record, rows[0])
        assert [p["value"] for p in curve] == list(sweeps.TOUGHNESS_VALUES)
        probabilities([p["prediction"] for p in curve])

    @pytest.mark.parametrize("layers", [0, 1])
    def test_method_c_trains_a_shallow_trunk_beside_the_full_model(
        self, loaded, records, tmp_path, layers,
    ):
        from effects.infrastructure.shard_io import write_shard

        (tmp_path / "training").mkdir()
        write_shard(tmp_path / "training" / "shard-00001.jsonl.gz", records[:24])
        args = SimpleNamespace(method_c_layers=layers, method_c_steps=2)
        result = method_c(args, loaded, [tmp_path / "training" / "shard-00001.jsonl.gz"],
                          records[:16])
        assert result["steps"] == 2 and result["layers"] == layers
        assert set(result["shallow"]) == set(result["full"])
        assert all(np.isfinite(v) for v in result["shallow"].values())


class TestGen3Only:
    def test_a_checkpoint_of_another_layout_is_refused(self, checkpoint_path, tmp_path):
        import torch

        payload = torch.load(checkpoint_path, weights_only=False)
        payload["training_settings"] = {**payload["training_settings"], "input_layout": "gen-2"}
        path = tmp_path / "gen2.pt"
        torch.save(payload, path)
        with pytest.raises(SystemExit, match="gen-2"):
            common.load_checkpoint(path)
        assert common.load_checkpoint(checkpoint_path).training_settings

    def test_gen2_records_are_refused(self, tmp_path):
        """A gen-3 fixture record less ``generation`` is gen-2's."""
        import gzip
        import json

        from effects.domain.shard_generation import OlderGenerationError
        from tests.unit.effects.builders import gen3_fixture_records

        directory = tmp_path / "game-disjoint"
        directory.mkdir()
        with gzip.open(directory / "shard-00000.jsonl.gz", "wt", encoding="utf-8") as out:
            for data in gen3_fixture_records():
                del data["generation"]
                out.write(json.dumps(data) + "\n")
        with pytest.raises(OlderGenerationError, match="the record is gen-2 "):
            next(record_walk.records(sorted(directory.iterdir())))
        with pytest.raises(OlderGenerationError, match="the record is gen-2 "):
            next(common.read_games(sorted(directory.iterdir()), {"any"}))

"""``scoring_targets``: the freeze's scoring step over a curated corpus."""

from __future__ import annotations

import torch

from effects.application.selection_deviance import scoring_targets
from effects.application.train_effect_model import TrainEffectModelConfig
from effects.domain.corpus_manifest import CorpusManifest
from tests.unit.effects.builders import FIXTURE_KEYWORD_DEFINITIONS, GEN3_SIDECAR_FOLDERS

_SCORED = 16


def _manifest(vocab) -> CorpusManifest:
    return CorpusManifest(
        seed=1, surface="script", vocab_path=str(vocab), variant_scripts="",
        holdout_permille=20, holdout_max_carriers=8, text_cap=0,
        card_disjoint_text_cap=0, training_records=1, class_mix={}, delivered_mix={},
        held_out_cards=(), held_out_texts=(), card_disjoint_games=(),
        game_disjoint_games=(), rarity={}, sources=(), per_class={}, per_stratum={},
        unique_texts={}, shortfall={}, rule_counts={"records": 0, "rules": {}},
        rule_map_digest="",
    )


def _targets(corpus, records):
    prepare = scoring_targets(
        corpus.root, _manifest(corpus.vocab), cards_folders=GEN3_SIDECAR_FOLDERS)
    return prepare(list(records[:_SCORED]))


def _same(left, right) -> bool:
    """Equal tensors, or equal tuples of them."""
    if isinstance(left, torch.Tensor):
        return torch.equal(left, right)
    return len(left) == len(right) and all(_same(a, b) for a, b in zip(left, right))


def test_the_scoring_step_returns_targets_over_the_fixture_corpus(corpus, records) -> None:
    targets, heads = _targets(corpus, records)
    assert targets.gate.shape[0] == _SCORED
    assert isinstance(heads, dict)


def test_the_keyword_table_changes_no_target(corpus, records, monkeypatch) -> None:
    """The scoring step reads targets only; the table feeds the encoder's text."""
    without_table, heads_without = _targets(corpus, records)
    unpathed = TrainEffectModelConfig.unpathed
    monkeypatch.setattr(
        TrainEffectModelConfig, "unpathed",
        classmethod(lambda cls, **o: unpathed(
            **{**o, "keyword_definitions": FIXTURE_KEYWORD_DEFINITIONS})))
    with_table, heads_with = _targets(corpus, records)
    assert torch.equal(without_table.gate, with_table.gate)
    assert torch.equal(without_table.mask, with_table.mask)
    assert without_table.fields.values.keys() == with_table.fields.values.keys()
    for name, value in with_table.fields.values.items():
        assert torch.equal(without_table.fields.values[name], value), name
        assert torch.equal(without_table.fields.present[name], with_table.fields.present[name])
    assert heads_without.keys() == heads_with.keys()
    for key in heads_with:
        assert _same(heads_without[key], heads_with[key]), key


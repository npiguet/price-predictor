"""The readers that weigh a validation row multiply by its ``class_weight``
(FR-059a, FR-068a): on a stratum with no rule copy, a row of class weight k
reads as the k repeated rows a validation stratum wrote before the weight
existed, in the freeze's deviance losses and standard errors and in the
control score's deviance component; a reader taking each record once
ignores the weight.

The records are the gen-3 fixture's, each given a natural copy's curation
whose class weight is an integer (1, 2 or 3 by sampling class), so the
repeated rows it stands for can be rebuilt exactly."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace

import pytest

from effects.application.control_score import (
    ControlFlags,
    ControlScorer,
    FrozenSelection,
    ScoredCheckpoint,
    score_half,
)
from effects.application.control_stratum import HalfScope, ScoringSource
from effects.application.freeze_selection_set import (
    FreezePorts,
    FreezeThresholds,
    RuleTestRules,
    SelectionSetRefused,
    freeze_selection_set,
)
from effects.application.record_forward import RecordForward
from effects.application.selection_deviance import TargetDevianceLosses, scoring_targets
from effects.application.step_loss import StepLossComputer
from effects.application.surface_batching import SurfaceBatcher
from effects.domain.corpus_manifest import ClassCounts, CorpusManifest, SourceShard
from effects.domain.records import Curation
from effects.domain.sampling import sampling_class
from effects.domain.score_scales import loss_ratio_standard_error
from effects.domain.selection_set import DEVIANCE, SampleRecord, item_key
from tests.unit.effects.builders import (
    FIXTURE_RULE_OUTPUTS,
    FIXTURE_STRATA,
    GEN3_SIDECAR_FOLDERS,
    fixture_run,
    fixture_selection_set,
)

#: Records the freeze's reader scores.
_SCORED = 48
GD = "game-disjoint"


def _class_weights(records) -> dict[str, int]:
    """An integer weight per sampling class: 1, 2, 3 in turn over the sorted classes."""
    classes = sorted({str(sampling_class(record)) for record in records})
    return {name: 1 + index % 3 for index, name in enumerate(classes)}


def _weighted(records) -> list:
    """Each record once, a natural copy weighing its class's weight."""
    weights = _class_weights(records)
    return [replace(record, curation=Curation(
        1, 1, class_weight=float(weights[str(sampling_class(record))]))) for record in records]


def _repeated(records) -> list:
    """Each record repeated as often as its class weighs, every row at weight one:
    the stratum a class mix reached by repetition wrote."""
    weights = _class_weights(records)
    return [replace(record, curation=Curation(1, 1))
            for record in records for _ in range(weights[str(sampling_class(record))])]


def _by_game(records, losses) -> dict[str, float]:
    sums: dict[str, float] = defaultdict(float)
    for record, loss in zip(records, losses, strict=True):
        sums[record.game_id] += loss
    return dict(sums)


def _scoring_manifest(vocab) -> CorpusManifest:
    return CorpusManifest(
        seed=1, surface="script", vocab_path=str(vocab), variant_scripts="",
        holdout_permille=20, holdout_max_carriers=8, text_cap=0,
        card_disjoint_text_cap=0, training_records=1, class_mix={}, delivered_mix={},
        held_out_cards=(), held_out_texts=(), card_disjoint_games=(),
        game_disjoint_games=(), rarity={}, sources=(), per_class={}, per_stratum={},
        unique_texts={}, shortfall={}, rule_counts={"records": 0, "rules": {}},
        rule_map_digest="",
    )


@pytest.fixture(scope="module")
def chosen(records):
    """At most two records a game, so both halves of the freeze hold games."""
    taken: dict[str, int] = defaultdict(int)
    rows = []
    for record in records:
        if taken[record.game_id] < 2 and len(rows) < _SCORED:
            taken[record.game_id] += 1
            rows.append(record)
    assert len(set(_class_weights(rows)[str(sampling_class(r))] for r in rows)) == 3
    return rows


@pytest.fixture(scope="module")
def deviance_reader(corpus):
    prepare = scoring_targets(corpus.root, _scoring_manifest(corpus.vocab),
                              cards_folders=GEN3_SIDECAR_FOLDERS)
    return TargetDevianceLosses(prepare, FIXTURE_RULE_OUTPUTS)


# ── the freeze ──────────────────────────────────────────────────────────


def test_a_rows_deviance_losses_are_its_repeated_rows(deviance_reader, chosen) -> None:
    weighted, repeated = _weighted(chosen), _repeated(chosen)
    ours = deviance_reader.losses(weighted)
    theirs = deviance_reader.losses(repeated)
    assert ours.keys() == theirs.keys()
    for name, by_mode in theirs.items():
        for mode, losses in by_mode.items():
            mine = _by_game(weighted, ours[name][mode])
            assert mine == pytest.approx(_by_game(repeated, losses), rel=1e-6, abs=1e-6), (
                name, mode)


def _freeze_manifest(records) -> CorpusManifest:
    games = sorted({record.game_id for record in records})
    halves = {GD: {"selection": tuple(games[0::2]), "reporting": tuple(games[1::2])}}
    return CorpusManifest(
        seed=42, surface="script", vocab_path="models/effects/vocab-script.txt",
        variant_scripts="", holdout_permille=20, holdout_max_carriers=8, text_cap=200,
        card_disjoint_text_cap=50, training_records=0, class_mix={"rewrite": 1.0},
        delivered_mix={"rewrite": 1.0}, held_out_cards=(), held_out_texts=(),
        card_disjoint_games=(), game_disjoint_games=tuple(games), rarity={},
        sources=(SourceShard("a.jsonl.gz", 1),), per_class={"rewrite": ClassCounts(1, 1, 0, 1)},
        per_stratum={}, unique_texts={}, shortfall={}, halves=halves,
    )


def _deviance_errors(stratum, reader) -> dict:
    """The deviance items' standard errors a freeze of ``stratum`` records."""
    rules = RuleTestRules(tag=lambda record: [], declared=frozenset(), trained=frozenset(),
                          blocked=frozenset(), unreadable=frozenset())
    try:
        frozen = freeze_selection_set(
            _freeze_manifest(stratum), {GD: lambda: [stratum]},
            ports=FreezePorts(rules=rules, probes=None, grids=[], deviance=reader),
            thresholds=FreezeThresholds(min_games_per_half=1, max_item_error=10.0,
                                        max_component_error=10.0))
    except SelectionSetRefused as refused:  # no probe or grid: those components miss
        frozen = refused.selection
    prefix = item_key(GD, "selection", DEVIANCE, "")
    return {key: error for key, error in frozen.standard_errors.items()
            if key.startswith(prefix) or key.startswith(item_key(GD, "reporting", DEVIANCE, ""))}


def test_the_freezes_deviance_errors_weigh_a_row_by_its_class_weight(
        deviance_reader, chosen) -> None:
    ours = _deviance_errors(_weighted(chosen), deviance_reader)
    theirs = _deviance_errors(_repeated(chosen), deviance_reader)
    assert ours.keys() == theirs.keys() and ours
    for key, by_mode in theirs.items():
        for mode, error in by_mode.items():
            assert ours[key][mode] == pytest.approx(error, rel=1e-6, abs=1e-9), (key, mode)


def test_the_standard_error_is_the_weighted_games_effective_count() -> None:
    """``√(Σ S_g²) / Σ S_g`` is ``1 / √n_eff``, Kish's effective number of
    games under each game's weighted loss ``S_g``."""
    games, losses, weights = ["a", "a", "b", "c"], [1.0, 2.0, 1.0, 4.0], [3.0, 1.0, 2.0, 1.0]
    sums = {"a": 5.0, "b": 2.0, "c": 4.0}
    effective = sum(sums.values()) ** 2 / sum(s * s for s in sums.values())
    assert loss_ratio_standard_error(games, losses, weights) == pytest.approx(
        1 / effective ** 0.5)


# ── the control score ───────────────────────────────────────────────────


@pytest.fixture(scope="module")
def run(corpus, records, tmp_path_factory):
    return fixture_run(corpus, records, tmp_path_factory.mktemp("class-weights"))


def _repeated_samples(selection, records):
    """``selection``'s deviance sample with every copy the repeated stratum holds."""
    weights = _class_weights(records)
    weight_of = {record.record_id: weights[str(sampling_class(record))] for record in records}
    return replace(selection, deviance_samples={
        stratum: {half: tuple(SampleRecord(row.record_id, row.game_id, copy)
                              for row in rows for copy in range(weight_of[row.record_id]))
                  for half, rows in by_half.items()}
        for stratum, by_half in selection.deviance_samples.items()})


def _training_deviance(run, selection, stratum) -> dict:
    scorer = ControlScorer.prepare(
        FrozenSelection(selection, dict.fromkeys(FIXTURE_STRATA, lambda: [stratum])),
        ScoringSource(run.preparer, list(run.fields.values()), FIXTURE_RULE_OUTPUTS,
                      chunk_records=8, batch_size=8),
        ControlFlags("hard", 0.3, 0.0), HalfScope(components=(DEVIANCE,)))
    score = scorer.epoch(RecordForward(run.encoder, run.model, run.scoring),
                         StepLossComputer(run.spec.config, run.device), run.fields[max(run.fields)])
    return {name: score.strata[name].items[DEVIANCE] for name in FIXTURE_STRATA}


def _checkpoint_deviance(run, selection, stratum) -> dict:
    template = run.preparer.scoring_batcher()
    batcher = SurfaceBatcher(stager=template.stager, resolver=template.resolver,
                             spec=template.spec, materializer=run.scoring)
    score = score_half(
        FrozenSelection(selection, dict.fromkeys(FIXTURE_STRATA, lambda: [stratum])),
        ScoredCheckpoint(run.encoder, run.model, batcher, FIXTURE_RULE_OUTPUTS,
                         config=run.spec.config),
        ControlFlags("soft", 0.3, 0.0), HalfScope("reporting", (DEVIANCE,)))
    return {name: score.strata[name].items[DEVIANCE] for name in FIXTURE_STRATA}


@pytest.mark.parametrize("scored", [_training_deviance, _checkpoint_deviance],
                         ids=["training", "checkpoint"])
def test_the_deviance_component_reads_a_row_as_its_repeated_rows(run, records, scored) -> None:
    records = list(records)
    selection = fixture_selection_set(records)
    ours = scored(run, selection, _weighted(records))
    theirs = scored(run, _repeated_samples(selection, records), _repeated(records))
    for stratum, items in theirs.items():
        assert items and ours[stratum].keys() == items.keys()
        for name, value in items.items():
            assert ours[stratum][name] == pytest.approx(value, rel=1e-6, abs=1e-6), (
                stratum, name)


def test_the_validation_loss_takes_each_record_once_whatever_its_weight(run, records) -> None:
    """The per-epoch validation pass reads a row at its rule-mask weight
    alone: a class weight changes no target weight there."""
    import torch

    from effects.application.host_cache import map_tensors

    fields = tuple(run.fields[max(run.fields)])
    plain = [replace(record, curation=Curation(1, 1)) for record in records[:16]]
    heavy = [replace(record, curation=Curation(1, 1, class_weight=3.0))
             for record in records[:16]]
    got, want = [], []
    map_tensors(run.preparer.prepare_scoring_step(heavy, fields).targets,
                lambda t: got.append(t) or t)
    map_tensors(run.preparer.prepare_scoring_step(plain, fields).targets,
                lambda t: want.append(t) or t)
    assert len(got) == len(want) and all(torch.equal(a, b) for a, b in zip(got, want))

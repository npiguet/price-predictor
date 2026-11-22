"""What the survey counts for records no stratum can hold: tags of refused and
turn-action records sit under their own source, ``refused``, and a trigger
with no sidecar is refused and counted."""

from __future__ import annotations

import shutil

import pytest

from effects.application.corpus_survey import (
    REFUSED_SOURCE,
    ReadConfig,
    SurveyConfig,
    WorkerContext,
    run_survey,
)
from effects.domain.record_quality import QualityDefect
from effects.domain.records import RecordKind
from effects.domain.stratum_placement import PlacementRule
from tests.unit.effects.application.test_build_corpus_survey import (
    RecordSpec,
    a_config,
    a_record,
    converted_tree,
)
from tests.unit.effects.builders import (
    GEN3_RECORDS,
    GEN3_SIDECAR_FOLDERS,
    held_out_cards,
    write_ended_shard,
)


@pytest.fixture(scope="module")
def fixture_survey(tmp_path_factory):
    from effects.infrastructure.rules_catalogue_io import load_scenarios

    root = tmp_path_factory.mktemp("refused-tags")
    shutil.copyfile(GEN3_RECORDS, root / GEN3_RECORDS.name)
    config = SurveyConfig(
        read=ReadConfig(
            records_dir=root, max_events=64, held_out=held_out_cards(), seed=42,
            sidecar_roots={"cardsfolder": GEN3_SIDECAR_FOLDERS[0],
                           "tokenscripts": GEN3_SIDECAR_FOLDERS[1]},
            catalogue=load_scenarios(),
        ),
        heap_cap=200,
        placement=PlacementRule(game_disjoint_share=0.1, keyword_share=0.1,
                                card_disjoint_text_cap=0, seed=42),
    )
    return run_survey(root, config=config, workers=1), config


def test_the_tags_of_turn_action_records_are_counted_under_refused(fixture_survey):
    """A turn action (the game-end record among them) is written to no
    stratum: its tags are no stratum's, and the natural sources count only
    what a stratum can hold. The fixture's untaps carry 703.4c eight times."""
    survey, _config = fixture_survey
    counts = survey.totals.rule_counts.counts
    assert counts["703.4c", "fires", REFUSED_SOURCE] == 8
    assert not any(
        n for (rule, _p, source), n in counts.items()
        if rule == "703.4c" and source != REFUSED_SOURCE
    )


def test_every_refused_or_turn_action_record_tag_is_counted_under_refused_alone(
    fixture_survey,
):
    survey, config = fixture_survey
    context = WorkerContext(config.read)
    from effects.infrastructure.shard_io import read_shard

    held = 0
    for record in read_shard(config.read.records_dir / GEN3_RECORDS.name):
        if record.kind is RecordKind.TURN_ACTION or context.refusal(record) is not None:
            held += len(context.tags(record).tags)
    counts = survey.totals.rule_counts.counts
    refused = sum(n for (_r, _p, source), n in counts.items() if source == REFUSED_SOURCE)
    assert refused == held


def test_a_variant_trigger_with_no_sidecar_is_refused_and_counted_by_class(tmp_path):
    from effects.domain.provenance import ProvenanceKey

    variant = (ProvenanceKey("variant-scripts/bear_variant_ab12.txt", 0, "trigger", 0),)
    kept = (ProvenanceKey("cardsfolder/b/bear.txt", 0, "trigger", 0),)
    write_ended_shard(tmp_path / "run.0-a.jsonl.gz", [
        a_record(RecordSpec(record_id="run.0-a.0", kind="trigger", ability=variant)),
        a_record(RecordSpec(record_id="run.0-a.1", kind="trigger", ability=kept)),
    ])
    roots = converted_tree(tmp_path, *kept)
    roots["variant-scripts"] = tmp_path / "converted" / "variant-scripts"

    survey = run_survey(tmp_path, config=a_config(tmp_path, sidecar_roots=roots), workers=1)

    refusals = survey.totals.refusals
    assert refusals.by_reason == {QualityDefect.NO_ACTING_TEXT: 1}
    assert sum(refusals.by_class.values()) == 1
    assert sum(survey.totals.kept.values()) == 1


def test_a_rule_counted_only_under_refused_is_tagged_nowhere_written():
    from effects.application.build_corpus import tagged_nowhere_written
    from effects.domain.rule_budgets import BudgetedRecord, BudgetInputs, plan_rule_budgets
    from effects.domain.rule_catalogue import Polarity
    from effects.domain.rule_detectors import RuleSource
    from effects.domain.strata import Stratum

    records = [BudgetedRecord(
        game="g1", record_hash=1, klass="effects", text="t1", natural=1,
        source=RuleSource.FORGE_AI, tags=(("702.16e", Polarity.FIRES),))]
    budgets = plan_rule_budgets(records, BudgetInputs(
        class_targets={"effects": 2}, budgeted=set(), reuse_cap=1, text_cap=0))
    empty = plan_rule_budgets([], BudgetInputs(
        class_targets={}, budgeted=set(), reuse_cap=1, text_cap=0))
    counts = {"rules": {
        "702.16e": {"fires": {"forge_ai": 1}},
        "104.2a": {"fires": {REFUSED_SOURCE: 2}},
        "703.4c": {"fires": {REFUSED_SOURCE: 8}},
    }}

    named = tagged_nowhere_written(counts, budgets, {Stratum.CARD_DISJOINT: empty})

    assert named == ("104.2a", "703.4c")

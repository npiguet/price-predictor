"""Rule budgets: added copies, the text cap and the manifest's rule rows."""

from __future__ import annotations

from dataclasses import replace

import pytest

from effects.domain.event_schema import Event, EventType
from effects.domain.provenance import ProvenanceKey
from effects.domain.records import (
    EffectRecord,
    RecordKind,
)
from effects.infrastructure.shard_io import (
    read_records,
)
from tests.unit.effects.application.build_corpus_support import (
    _BEAR_FILE,
    _BOLT_KEY,
    _BUDGET_FREE_RULES,
    _BUDGET_RULE,
    _SHOCK_KEY,
    _TIMESTAMP,
    CorpusFixture,
    _budget_rules,
    _resolution,
    _snapshot,
    make_record,
)


def _damaged(record_id: str, game: str, *, hits: int) -> EffectRecord:
    """A bolt resolution dealing ``hits`` damage events to ``E0``, a creature on its board."""
    record = CorpusFixture.resolution(record_id, game=game, events=tuple(
        Event(type=EventType.DAMAGE_DEALT, subjects=("E0",), params={"amount": 1},
              attributed_to="root")
        for _ in range(hits)
    ))
    return replace(record, state=_snapshot(("grizzly bears",)))


@pytest.fixture
def budget_build(tmp_path, a_corpus):
    """``a_corpus`` and one training game of ten bolts, two of them tagged
    with :data:`_BUDGET_RULE`, built under a catalogue budgeting that rule.

    The game's draw keeps it out of the candidate pool, so it trains; the
    precision target is off, since no validation game carries the rule.
    """
    from effects.domain.corpus_curation import game_disjoint_draw

    game = next(f"g-dmg-{i}" for i in range(100) if game_disjoint_draw(f"g-dmg-{i}") >= 0.1)
    a_corpus.with_extra_records([
        *(_damaged(f"{game}.tagged{i}", game, hits=2) for i in range(2)),
        *(_damaged(f"{game}.plain{i}", game, hits=1) for i in range(8)),
    ])
    store = a_corpus.build(
        tmp_path / "out", rules=_BUDGET_FREE_RULES, rules_dir=_budget_rules(tmp_path / "rules"),
        game_disjoint_share=0.01, game_disjoint_keyword_share=0.01,
        class_mix={"resolution-effect": 1.0}, validation_sample=0,
    )
    return store, game


def test_the_text_cap_bounds_a_text_s_natural_and_added_copies_together(tmp_path, a_corpus):
    """The build hands the rule budgets every natural copy of the text in the
    class (``SelectionPlan.text_planned``), untagged records' included, so the
    copies added for a rule never take the text past ``--text-cap``."""
    from effects.domain.corpus_curation import game_disjoint_draw

    game = next(f"g-dmg-{i}" for i in range(100) if game_disjoint_draw(f"g-dmg-{i}") >= 0.1)
    a_corpus.with_extra_records([
        *(_damaged(f"{game}.tagged{i}", game, hits=2) for i in range(2)),
        *(_damaged(f"{game}.plain{i}", game, hits=1) for i in range(8)),
        # Another text's records, so the class asks the rule for many copies.
        *(_resolution(f"{game}.shock{i}", game, ability=(_SHOCK_KEY,)) for i in range(40)),
    ])
    store = a_corpus.build(
        tmp_path / "out", rules=_BUDGET_FREE_RULES, rules_dir=_budget_rules(tmp_path / "rules"),
        game_disjoint_share=0.01, game_disjoint_keyword_share=0.01,
        class_mix={"resolution-effect": 1.0}, validation_sample=0, text_cap=6,
    )
    bolts = [r for r in read_records(store.training_dir)
             if r.game_id == game and r.ability == (_BOLT_KEY,)]
    assert 0 < len(bolts) <= 6


def _assert_copies_say_how_they_were_written(
    training: list[EffectRecord], record_id: str,
) -> None:
    copies = [r for r in training if r.record_id == record_id]
    curation = copies[0].curation
    assert len(copies) == curation.copies <= 4
    assert sum(r.curation.drawn_rule is None for r in copies) == curation.natural_share
    assert all(r.curation.drawn_rules == (_BUDGET_RULE,) for r in copies)


def _assert_added_copies_are_the_budgeted_ones(added: list[EffectRecord], game: str) -> None:
    assert {r.curation.drawn_rule for r in added} == {_BUDGET_RULE}
    assert {r.record_id for r in added} <= {f"{game}.tagged0", f"{game}.tagged1"}
    _assert_added_copies_read_the_damaged_creature(added)


def _assert_added_copies_read_the_damaged_creature(added: list[EffectRecord]) -> None:
    """The damaged creature is the outcome; the bolt's controller is read by its
    condition."""
    assert all(r.curation.outcome_entities == ("E0",) for r in added)
    assert all(r.actor_player in r.curation.condition_entities for r in added)


def _drawn_rules_of(training: list[EffectRecord], id_prefix: str) -> set[tuple[str, ...]]:
    return {r.curation.drawn_rules for r in training if r.record_id.startswith(id_prefix)}


def test_added_copies_carry_their_rule_and_natural_copies_none(budget_build):
    """FR-055, research T1: every copy says how it was written."""
    store, game = budget_build
    training = list(read_records(store.training_dir))
    assert all(record.curation is not None for record in training)
    added = [r for r in training if r.curation.drawn_rule is not None]
    natural = [r for r in training if r.curation.drawn_rule is None]
    assert added and natural
    _assert_added_copies_are_the_budgeted_ones(added, game)
    for record_id in {r.record_id for r in added}:
        _assert_copies_say_how_they_were_written(training, record_id)
    assert _drawn_rules_of(training, f"{game}.plain") <= {()}


def test_the_manifest_reports_the_rule_row_and_its_one_side(budget_build):
    """FR-054, FR-054a, FR-060: a fires-only rule's half is filled from its
    records, the other half stays its shortfall and the rule is one-sided."""
    store, game = budget_build
    manifest = store.load()
    row = manifest.rules["resolution-effect"][_BUDGET_RULE]["fires"]
    tagged = [r for r in read_records(store.training_dir)
              if r.record_id.startswith(f"{game}.tagged")]
    assert row.available == {"forge_ai": 2}
    assert row.written == {"forge_ai": len(tagged)}
    assert row.added["forge_ai"] == sum(r.curation.drawn_rule is not None for r in tagged)
    assert row.shortfall == max(0, row.share - len(tagged))
    assert manifest.rules["resolution-effect"][_BUDGET_RULE]["does_not_fire"].shortfall == (
        row.share)
    assert manifest.one_sided_rules == {"resolution-effect": (_BUDGET_RULE,)}
    assert manifest.budgeted_rules == {_BUDGET_RULE}
    assert manifest.rule_outputs(_BUDGET_RULE) == ("damage_taken",)
    # The damaged creature carries damage_taken, so the output is effective.
    assert manifest.effective_rule_outputs[_BUDGET_RULE] == ("damage_taken",)


def test_a_decision_whose_candidate_key_is_unjoinable_is_refused(tmp_path, a_corpus):
    """Story 6 acceptance 5 (FR-059): a candidate's key inside a range its
    sidecar declares and in neither list is ``unjoinable-key``."""
    from effects.domain.record_quality import QualityDefect
    from effects.domain.records import Candidate, PlayabilityDecisionPayload, PlayabilitySubkind

    def decision(record_id: str, key: ProvenanceKey) -> EffectRecord:
        return make_record(
            RecordKind.PLAYABILITY, subkind=PlayabilitySubkind.DECISION,
            record_id=record_id, game_id="g-clean-1", timestamp=_TIMESTAMP,
            payload=PlayabilityDecisionPayload(candidates=(Candidate(
                ability=(key,), can_play=True, affordable=True, has_legal_target=True,
                timing=True, failed_checks=(), land_play=False),)),
        )

    unjoinable = ProvenanceKey(_BEAR_FILE, 0, "spell", 1, option=0)
    # Two joinable decisions beside it, so the class is not mostly refused.
    store = a_corpus.with_extra_records([
        decision("g-clean-1.decide", unjoinable),
        decision("g-clean-1.bolt0", _BOLT_KEY), decision("g-clean-1.bolt1", _BOLT_KEY),
    ]).build(tmp_path / "out")
    assert store.load().quality_dropped.get(QualityDefect.UNJOINABLE_KEY) == 1
    written = {r.record_id for d in (store.training_dir, store.card_disjoint_dir,
                                     store.game_disjoint_dir) for r in read_records(d)}
    assert "g-clean-1.decide" not in written


# ── the gen-3 fixture under the real catalogue (T140) ────────────────────

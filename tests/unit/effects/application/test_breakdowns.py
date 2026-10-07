"""The evaluation report's breakdowns (T106; spec Story 6 scenarios 1–5).

The rows are real gen-1 records from the fixture, grouped by the families
``rule_family`` gives them over their real sidecars; only the losses are made
up, because what is under test is the grouping, not the model.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from effects.application import breakdowns as bd
from effects.application.breakdowns import RecordResult
from effects.application.evaluate_effect_model import (
    CARD_DISJOINT,
    GAME_DISJOINT,
    CheckStatus,
    breakdown_checks,
)
from effects.domain.rule_families import rule_family
from effects.infrastructure.record_io import read_shard
from effects.infrastructure.sidecar_io import SidecarCache

_FIXTURES = Path(__file__).parents[3] / "fixtures" / "effects"


@pytest.fixture(scope="module")
def rows() -> list[RecordResult]:
    """One row per real fixture record, with a deterministic made-up loss."""
    sidecars = SidecarCache({
        "cardsfolder": _FIXTURES / "gen1-sidecars" / "cardsfolder",
        "tokenscripts": _FIXTURES / "gen1-sidecars" / "tokenscripts",
    })
    out = []
    for position, record in enumerate(read_shard(_FIXTURES / "gen1-records.jsonl.gz")):
        text = None
        for key in record.ability or ():
            line = sidecars.line_for(key)
            if line is not None and line.script_text:
                text = line.script_text
                break
        out.append(RecordResult(
            record_id=record.record_id,
            game_id=record.game_id,
            kind=record.kind.value,
            subkind=record.subkind.value if record.subkind else None,
            text=text,
            family=rule_family(record, sidecars),
            random_seat=record.random_seat,
            what_if=record.what_if,
            losses={"gate": float(position % 3), bd.TOTAL: float(position % 5)},
        ))
    return out


def _row(record_id, *, text="t", game="g1", total=1.0, **fields) -> RecordResult:
    losses = {bd.TOTAL: total}
    losses.update({k: v for k, v in fields.items() if k in ("gate", "zone_outcome")})
    rest = {k: v for k, v in fields.items() if k not in losses}
    return RecordResult(
        record_id=record_id, game_id=game, kind="resolution", text=text,
        losses=losses, **rest,
    )


class TestPerText:
    """Scenario 1: per record and as a per-text mean."""

    def test_a_text_with_many_records_weighs_once_in_the_per_text_mean(self):
        rows = [_row(f"a{i}", text="common", total=1.0) for i in range(9)]
        rows.append(_row("b", text="rare", total=11.0))
        overall = bd.overall(rows)
        assert overall.per_record.means[bd.TOTAL] == pytest.approx(2.0)
        assert overall.per_text.means[bd.TOTAL] == pytest.approx(6.0)
        assert overall.per_text.counts[bd.TOTAL] == 2

    def test_a_field_is_averaged_over_the_records_that_supervise_it(self):
        rows = [_row("a", zone_outcome=2.0), _row("b")]
        assert bd.field_means(rows).means["zone_outcome"] == pytest.approx(2.0)
        assert bd.field_means(rows).counts["zone_outcome"] == 1

    def test_records_with_no_text_are_left_out_of_the_per_text_mean(self):
        rows = [_row("a", text=None, total=9.0), _row("b", text="x", total=1.0)]
        assert bd.overall(rows).per_text.means[bd.TOTAL] == pytest.approx(1.0)
        assert bd.overall(rows).per_record.means[bd.TOTAL] == pytest.approx(5.0)


class TestRarityAndFamily:
    def test_texts_bucket_by_the_corpus_rarity_table(self):
        rows = [_row("a", text="seen-once"), _row("b", text="common")]
        groups = bd.by_rarity_bucket(rows, {"seen-once": 1, "common": 40})
        assert list(groups) == ["1", "20+"]

    def test_a_text_the_table_never_counted_buckets_by_its_own_games(self):
        rows = [_row("a", text="new", game="g1"), _row("b", text="new", game="g2"),
                _row("c", text="new", game="g3")]
        assert list(bd.by_rarity_bucket(rows, {})) == ["2-4"]

    def test_real_records_group_by_their_rule_family(self, rows):
        families = bd.by_family(rows)
        assert len(families) > 3
        assert sum(g.per_record.records for g in families.values()) == len(rows)

    def test_the_mean_over_families_weighs_each_family_once(self):
        rows = [_row(f"a{i}", family="Draw", total=1.0) for i in range(9)]
        rows.append(_row("b", family="Pump", total=11.0))
        mean = bd.mean_over_families(bd.by_family(rows))
        assert mean.per_record.means[bd.TOTAL] == pytest.approx(6.0)


class TestMemorizationGap:
    """Scenario 2: per family, game-disjoint minus card-disjoint."""

    def test_the_gap_is_game_disjoint_minus_card_disjoint_on_shared_fields(self):
        card = bd.by_family([_row("a", family="Draw", total=3.0, gate=1.0)])
        game = bd.by_family([_row("b", family="Draw", total=1.0)])
        gap = bd.memorization_gap(game, card)
        assert gap == {"Draw": {bd.TOTAL: pytest.approx(-2.0)}}

    def test_a_family_in_one_stratum_alone_has_no_gap(self):
        card = bd.by_family([_row("a", family="Draw")])
        game = bd.by_family([_row("b", family="Pump")])
        assert bd.memorization_gap(game, card) == {}


class TestPolicyAndDecisionSlices:
    """Scenarios 3 and 4."""

    def test_random_seat_records_are_reported_apart(self, rows):
        off = replace(rows[0], random_seat=True)
        groups = bd.policy_slice([off, *rows[1:]])
        assert groups[bd.OFF_POLICY].per_record.records == 1
        assert groups[bd.ON_POLICY].per_record.records == len(rows) - 1

    def test_legality_records_split_into_real_what_if_and_unknown(self, rows):
        legality = [r for r in rows if r.subkind in bd.LEGALITY_SUBKINDS]
        assert len(legality) >= 3
        marked = [
            replace(legality[0], what_if=False),
            replace(legality[1], what_if=True),
            *legality[2:],
        ]
        groups = bd.decision_slice([*marked, *rows])
        assert groups[bd.REAL].per_record.records == 1
        assert groups[bd.WHAT_IF].per_record.records == 1
        assert bd.UNKNOWN in groups

    def test_decision_records_never_enter_the_legality_slice(self, rows):
        decisions = [r for r in rows if r.subkind == "decision"]
        assert decisions
        assert bd.decision_slice(decisions) == {}


class TestWithheldKeyword:
    """Scenario 5: withheld keyword beside trained keywords."""

    def test_records_split_by_whether_they_carry_the_withheld_keyword(self):
        rows = [
            _row("a", keywords=frozenset({"first_strike"})),
            _row("b", keywords=frozenset({"flying"})),
            _row("c"),
        ]
        groups = bd.keyword_slice(rows, "First Strike")
        assert groups[bd.WITHHELD].per_record.records == 1
        assert groups[bd.TRAINED].per_record.records == 1

    def test_a_keyword_value_is_not_part_of_its_name(self):
        assert bd.normalize_keyword("Ward:2") == "ward"


class TestTheReport:
    def test_every_section_appears_for_a_gen2_shaped_set(self, rows):
        legality = next(r for r in rows if r.subkind in bd.LEGALITY_SUBKINDS)
        card = [*rows, replace(rows[0], random_seat=True),
                replace(legality, what_if=False)]
        checks = breakdown_checks(
            {CARD_DISJOINT: card, GAME_DISJOINT: rows},
            games_by_text={}, withheld_keyword=None,
        )
        names = [check.name for check in checks]
        assert CARD_DISJOINT in names
        assert any(name.startswith(f"{CARD_DISJOINT} rarity ") for name in names)
        assert any(name.startswith(f"{CARD_DISJOINT} family ") for name in names)
        assert f"{CARD_DISJOINT} mean over families" in names
        assert any(name.startswith("memorization-gap ") for name in names)
        assert f"{CARD_DISJOINT} policy off" in names
        assert f"{CARD_DISJOINT} legality real" in names
        assert all(check.status is not CheckStatus.FAIL for check in checks)

    def test_a_gen1_set_reports_the_policy_and_decision_slices_as_skipped(self, rows):
        checks = {c.name: c for c in breakdown_checks(
            {CARD_DISJOINT: rows, GAME_DISJOINT: rows},
            games_by_text={}, withheld_keyword="flying",
        )}
        assert checks[f"{CARD_DISJOINT} off-policy"].status is CheckStatus.SKIPPED
        assert checks[f"{CARD_DISJOINT} real-decision legality"].status is (
            CheckStatus.SKIPPED
        )

    def test_the_per_text_mean_is_reported_beside_the_per_record_one(self, rows):
        overall = next(
            c for c in breakdown_checks(
                {CARD_DISJOINT: rows}, games_by_text={}, withheld_keyword=None,
            ) if c.name == CARD_DISJOINT
        )
        assert bd.TOTAL in overall.values and f"text:{bd.TOTAL}" in overall.values

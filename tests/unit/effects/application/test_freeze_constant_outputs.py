"""The freeze leaves out a rule-test output whose targets are the same in both
polarities, as the control score does (finding 12.9).

The records are the real gen-3 fixture's, each its own game. In it the
playability records alone supervise ``can_play`` (seven say 1, 29 say 0) and
train no gate, so a rule tagged ``fires`` on the first and ``does_not_fire``
on the second, plus ten records a polarity that train the gate and leave the
actor unaffected, has an informative ``can_play`` and a ``gate`` that is 0 on
every row it reads.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace

import pytest

from effects.application.freeze_selection_set import (
    ProbeItem,
    RuleTestRules,
    SelectionSetRefused,
    freeze_selection_set,
)
from effects.domain.rule_catalogue import Polarity
from effects.domain.rule_detectors import RuleTag
from effects.domain.rule_output_targets import (
    RecordIndicators,
    TargetMeans,
    record_indicators,
)
from effects.domain.score_scales import clustered_auc_standard_error
from effects.domain.selection_set import (
    NO_DIFFERING_OUTPUT,
    RULE_TESTS,
    SAME_TARGETS,
    LeftOut,
    error_of,
    item_key,
)
from tests.unit.effects.application.test_control_score import STRATA, _prepare, _selection
from tests.unit.effects.application.test_freeze_selection_set import (
    GD,
    LENIENT,
    _Grid,
    _Losses,
    _manifest,
    _reader,
)
from tests.unit.effects.builders import fixture_run

FIRES, NOT = Polarity.FIRES, Polarity.DOES_NOT_FIRE
RULE = "121.2"
HALVES = ("selection", "reporting")


def _eligible(record) -> bool:
    return not (record.fork or record.interventional or record.synthetic
                or record.aimed_at is not None or record.directed is not None)


def _playable(eligible) -> list:
    """The playability records, ``fires`` where they say ``can_play``."""
    said = [(r, record_indicators("can_play", r)) for r in eligible]
    return [(r, FIRES if value == [1.0] else NOT) for r, value in said if value is not None]


def _reads_the_gate_only(record) -> bool:
    """A record saying nothing of ``can_play`` that trains the gate and
    leaves its actor unaffected."""
    if record_indicators("can_play", record) is not None:
        return False
    return RecordIndicators(record).of("gate", (record.actor_player,)) == [0.0]


@pytest.fixture(scope="module")
def tagged(records):
    """``[(record, polarity)]``, each record its own game."""
    eligible = [r for r in records if _eligible(r)]
    unaffected = [r for r in eligible if _reads_the_gate_only(r)]
    rows = _playable(eligible) + [(r, FIRES) for r in unaffected[:10]] + [
        (r, NOT) for r in unaffected[10:20]]
    assert Counter(p for _, p in rows) == {FIRES: 17, NOT: 39}
    return [(replace(r, game_id=f"g{i:03d}"), p) for i, (r, p) in enumerate(rows)]


class _Probes:
    """One binary probe type, its label alternating by game."""

    def items(self, records):
        for record in records:
            yield ProbeItem("dies", "binary", record.record_id, "E1",
                            float(int(record.game_id[1:]) // 2 % 2))


def _freeze(tagged, outputs):
    tags = {r.record_id: [RuleTag(RULE, p, observed=p is FIRES, condition=(),
                                  outcome=(r.actor_player,), scenario=f"{RULE}-1")]
            for r, p in tagged}
    rules = RuleTestRules(tag=lambda record: tags.get(record.record_id, []),
                          declared=frozenset({RULE}), trained=frozenset({RULE}),
                          outputs={RULE: outputs})
    games = [r.game_id for r, _ in tagged]
    halves = {GD: {"selection": games[0::2], "reporting": games[1::2]}}
    return freeze_selection_set(
        _manifest(halves), {GD: _reader([r for r, _ in tagged])}, rules=rules,
        probes=_Probes(), grids=[_Grid()], deviance=_Losses(), **LENIENT)


def _left_out(frozen, half):
    return frozen.left_out.get(GD, {}).get(half, {}).get(RULE_TESTS, ())


def test_a_rule_with_one_constant_output_is_frozen_with_it_named(tagged):
    frozen = _freeze(tagged, ("gate", "can_play"))
    for half in HALVES:
        assert RULE in frozen.rule_tests[GD][half]
        assert LeftOut(f"{RULE}/gate", SAME_TARGETS) in _left_out(frozen, half)


def test_its_standard_error_counts_the_rows_its_informative_output_reads(tagged):
    reads = {r.record_id for r, _ in tagged if record_indicators("can_play", r) is not None}
    frozen = _freeze(tagged, ("gate", "can_play"))
    for half in HALVES:
        sides = frozen.rule_tests[GD][half][RULE]
        every = [[row.game_id for row in sides[p]] for p in ("fires", "does_not_fire")]
        read = [[row.game_id for row in sides[p] if row.record_id in reads]
                for p in ("fires", "does_not_fire")]
        expected = clustered_auc_standard_error(*read)
        assert expected != pytest.approx(clustered_auc_standard_error(*every))
        error = error_of(frozen.standard_errors[item_key(GD, half, RULE_TESTS, RULE)])
        assert error == pytest.approx(expected)


def test_a_rule_whose_every_output_is_constant_is_left_out_and_counts_no_error(tagged):
    with pytest.raises(SelectionSetRefused) as refused:
        _freeze(tagged, ("gate",))
    frozen = refused.value.selection
    for half in HALVES:
        assert RULE not in frozen.rule_tests[GD][half]
        assert LeftOut(RULE, NO_DIFFERING_OUTPUT) in _left_out(frozen, half)
        assert frozen.standard_errors[item_key(GD, half, RULE_TESTS, RULE)] is None
    assert {(miss.half, miss.component) for miss in refused.value.misses} == {
        (half, RULE_TESTS) for half in HALVES}


# ── the scorer and the freeze read one target ───────────────────────────


OUTPUTS = ("gate", "zone_outcome", "damage_taken", "pt_change", "can_play", "trigger_fired",
           "created_objects")


@pytest.fixture(scope="module")
def run(corpus, records, tmp_path_factory):
    return fixture_run(corpus, records, tmp_path_factory.mktemp("constant-outputs"))


def test_the_scorer_settles_the_outputs_the_freeze_keeps_in_the_same_direction(run, records):
    selection = _selection(records)
    scorer, _, _ = _prepare(run, records, selection, rule_outputs={"r.board": OUTPUTS},
                            strict=False)
    by_id = {r.record_id: RecordIndicators(r) for r in records}
    for stratum in STRATA:
        sides = selection.rule_tests[stratum]["selection"]["r.board"]
        settled = {}
        for output in OUTPUTS:
            means = TargetMeans()
            for polarity, fires in (("fires", True), ("does_not_fire", False)):
                means.observe_all(fires, (by_id[row.record_id].of(output, row.entities)
                                          for row in sides[polarity]))
            settled[output] = means.settle()
        kept = scorer.strata[stratum].rules.get("r.board", {})
        assert set(kept) == {output for output, found in settled.items() if found}
        assert len(kept) not in (0, len(OUTPUTS))
        for output, reading in kept.items():
            assert (reading.candidate, reading.sign) == (
                settled[output].candidate, settled[output].sign)

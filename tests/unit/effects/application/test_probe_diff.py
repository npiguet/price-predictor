"""The probe diff: the model's move beside Forge's on the probed combats (FR-116).

Each probe fork is joined to the real combat it mirrors by ``mirror_of`` and
the probed keyword and carrier, whichever comes first in the stream; Forge's
move is the real record's target minus the fork's, and each pair scores the
model's move over Forge's. Reported beside gate 2, it changes no verdict.
"""

from __future__ import annotations

import dataclasses
import math

import pytest

from effects.application import gate_two
from effects.application.gate_two_moves import PredictedMove
from effects.application.probe_diff import (
    ProbeJoin,
    ProbeKey,
    _RealSide,
    forge_moves,
    observed_value,
)
from effects.domain.damage_step_keywords import (
    DAMAGE_STEP_KEYWORDS,
    KEYWORDS_BY_NAME,
    combat_participant,
)
from effects.domain.effect_head_layout import ZONE_OUTCOMES, HeadField
from effects.domain.effect_targets import RecordTargets, derive_targets
from effects.domain.evaluation_report import CheckName, CheckStatus, EvaluationReport
from effects.domain.event_schema import Event, EventType
from effects.domain.records import CombatPayload, ForkKind, RecordKind
from effects.domain.score_scales import clustered_mean_standard_error
from tests.unit.effects.domain.test_damage_step_predicates import _unblocked_attacker
from tests.unit.effects.record_builders import make_record

_LIFELINK = KEYWORDS_BY_NAME["lifelink"]
_GAIN = _LIFELINK.effects[0]
_DEATHTOUCH_KILL = KEYWORDS_BY_NAME["deathtouch"].effects[0]


def _hit(amount: int) -> tuple[Event, ...]:
    """The lifelinker A's hit on P1, and P1's loss."""
    return (
        Event(EventType.DAMAGE_DEALT, ("P1",), {"amount": amount, "combat": True, "source": "A"}),
        Event(EventType.LIFE_CHANGE, ("P1",), {"delta": -amount}),
    )


def _lifelink_combat(dealt: int = 3, *, game: str = "g1", record_id: str = "g1.1"):
    """A 3-power lifelinker unblocked at P1 that dealt ``dealt``, its
    controller P0 gaining as much, and the carrier's gate-2 participant."""
    state, carrier = _unblocked_attacker(3, 3, keywords=("lifelink",))
    events = (*_hit(dealt), Event(EventType.LIFE_CHANGE, ("P0",), {"delta": dealt}))
    record = dataclasses.replace(
        make_record(RecordKind.COMBAT, state=state, payload=CombatPayload(events=events)),
        game_id=game, record_id=record_id)
    participant = combat_participant(
        record, carrier, lambda entity: set(entity.granted_temporary.keywords))
    return record, participant


def _probe_of(real, *, dealt: int = 3, mirror_of: str | None = None, keyword="lifelink"):
    """The real combat re-run with ``keyword`` stripped from A: the hit
    lands, and nobody gains."""
    payload = dataclasses.replace(
        real.payload, events=_hit(dealt), probed_keyword=keyword, probed_entity="A")
    return dataclasses.replace(
        real, record_id=f"{real.record_id}.probe", fork=True, fork_kind=ForkKind.PROBE,
        mirror_of=mirror_of or real.record_id, payload=payload)


def _move(predicted: float) -> list[PredictedMove]:
    return [PredictedMove(_GAIN, "P0", predicted)]


def _line(join: ProbeJoin, keyword: str):
    return next(check for check in join.checks(DAMAGE_STEP_KEYWORDS)
                if check.name == f"{CheckName.PROBE_DIFF} {keyword}")


class TestTheJoin:
    def test_a_fork_read_before_its_real_record_pairs(self):
        real, participant = _lifelink_combat()
        join = ProbeJoin()
        join.fork(_probe_of(real))
        join.observe(real, _LIFELINK, participant, _move(1.5))
        assert join.paired["lifelink"] == 1
        assert (join.forks, join.observed) == ({}, {})
        assert join.moves["lifelink"].per_effect[_GAIN.label].shares == [0.5]

    def test_a_fork_read_after_its_real_record_pairs(self):
        real, participant = _lifelink_combat()
        join = ProbeJoin()
        join.observe(real, _LIFELINK, participant, _move(1.5))
        join.fork(_probe_of(real))
        assert join.paired["lifelink"] == 1
        assert (join.forks, join.observed) == ({}, {})
        assert join.moves["lifelink"].per_effect[_GAIN.label].shares == [0.5]

    def test_a_fork_whose_real_combat_was_never_scored_stays_unpaired(self):
        real, participant = _lifelink_combat()
        join = ProbeJoin()
        join.observe(real, _LIFELINK, participant, _move(1.5))
        join.fork(_probe_of(real, mirror_of="g1.other"))
        assert join.paired.total() == 0
        assert list(join.forks) == [ProbeKey("g1.other", "lifelink", "A")]
        assert _line(join, "lifelink").detail.endswith("paired 0, forks unpaired 1")

    def test_a_real_combat_with_no_fork_is_neither_paired_nor_a_fork_unpaired(self):
        real, participant = _lifelink_combat()
        join = ProbeJoin()
        join.observe(real, _LIFELINK, participant, _move(1.5))
        assert join.paired.total() == 0
        assert _line(join, "lifelink").detail == "share none; paired 0, forks unpaired 0"

    def test_first_and_double_strike_forks_are_counted_and_not_joined(self):
        real, _ = _lifelink_combat()
        join = ProbeJoin()
        join.fork(_probe_of(real, keyword="first_strike"))
        assert (join.forks, join.not_measured) == ({}, {"first_strike": 1})
        assert all("first_strike" not in check.name for check in join.checks(DAMAGE_STEP_KEYWORDS))


class TestForgesMove:
    def test_it_is_the_real_records_target_minus_the_forks(self):
        """With lifelink P0 gained 3; without it, nothing: Forge's move is +3."""
        real, _ = _lifelink_combat()
        side = _RealSide("g1", tuple(_move(1.5)),
                         (observed_value(derive_targets(real), _GAIN, "P0"),))
        assert forge_moves(side, derive_targets(_probe_of(real))) == {
            _GAIN.label: [(1.5, 3.0)]}

    def test_an_unaffected_subject_reads_zero(self):
        real, _ = _lifelink_combat()
        assert observed_value(derive_targets(_probe_of(real)), _GAIN, "P0") == 0.0

    def test_a_death_reads_one_at_its_class_and_zero_at_another(self):
        targets = RecordTargets()
        entry = targets.entry("B")
        entry.affected = True
        entry.set(HeadField.ZONE_OUTCOME, ZONE_OUTCOMES.index("died"))
        assert observed_value(targets, _DEATHTOUCH_KILL, "B") == 1.0
        entry.set(HeadField.ZONE_OUTCOME, ZONE_OUTCOMES.index("exiled"))
        assert observed_value(targets, _DEATHTOUCH_KILL, "B") == 0.0

    def test_a_field_the_record_cannot_say_is_left_out(self):
        targets = RecordTargets()
        targets.entry("P0").affected = True
        targets.entry("P0").forget(HeadField.LIFE_DELTA)
        side = _RealSide("g1", tuple(_move(1.0)), (2.0,))
        assert observed_value(targets, _GAIN, "P0") is None
        assert forge_moves(side, targets) == {}


class TestTheMeasure:
    def test_each_pair_scores_the_models_move_over_forges_clustered_by_game(self):
        """Forge moves +3 on each pair; the model 3, 1.5 (game g1) and 0 (g2):
        shares 1, 0.5, 0, mean 0.5, residuals 0.5, 0 | -0.5: √0.5 / 3, times √2 for two games."""
        join = ProbeJoin()
        for index, (game, predicted) in enumerate([("g1", 3.0), ("g1", 1.5), ("g2", 0.0)]):
            real, participant = _lifelink_combat(game=game, record_id=f"{game}.{index}")
            join.observe(real, _LIFELINK, participant, _move(predicted))
            join.fork(_probe_of(real))
        (effect,) = join.moves["lifelink"].summary(_LIFELINK).effects
        assert effect.combats == 3
        assert effect.mean_share == pytest.approx(0.5)
        assert effect.standard_error == pytest.approx(math.sqrt(0.5) * math.sqrt(2) / 3)
        assert effect.standard_error == pytest.approx(
            clustered_mean_standard_error([1.0, 0.5, 0.0], ["g1", "g1", "g2"]))

    def test_a_pair_where_forge_moved_nothing_is_counted_apart(self):
        """The real hit was prevented: neither record gains life."""
        real, participant = _lifelink_combat(0)
        join = ProbeJoin()
        join.observe(real, _LIFELINK, participant, _move(-0.4))
        join.fork(_probe_of(real, dealt=0))
        (effect,) = join.moves["lifelink"].summary(_LIFELINK).effects
        assert (effect.combats, effect.zero_size) == (0, 1)
        assert effect.zero_size_move == pytest.approx(0.4)


class TestTheReportLines:
    def test_a_summary_line_then_one_line_per_measured_keyword(self):
        real, participant = _lifelink_combat()
        join = ProbeJoin()
        join.observe(real, _LIFELINK, participant, _move(1.5))
        join.fork(_probe_of(real))
        join.fork(_probe_of(real, keyword="double_strike"))
        report = EvaluationReport()
        report.extend(join.checks(DAMAGE_STEP_KEYWORDS))
        lines = report.render().splitlines()[:-1]
        assert lines[0] == (
            "[----] probe-diff: the model's move over Forge's (f: the real combat's "
            "targets minus its probe fork's), 1 probed combats paired, 0 forks "
            "unpaired; routes nothing; first and double strike not measured (forks: "
            "double_strike 1): a fork deals both damage steps where the real record is one")
        assert [line.split(":")[0] for line in lines[1:]] == [
            f"[----] probe-diff {row.keyword}" for row in DAMAGE_STEP_KEYWORDS[2:]]
        assert lines[2] == (
            "[----] probe-diff lifelink: share 0.50; controller.life_delta 0.50 ± n/a "
            "(q 0.50/0.50/0.50) over 1; paired 1, forks unpaired 0")
        # scripts/effect_gen2_report/build_data.gate2 reads only lines holding n=.
        assert not any("n=" in line for line in lines)

    def test_a_join_that_raised_is_one_skip_line_naming_the_exception(self, caplog):
        real, participant = _lifelink_combat()
        join = ProbeJoin()
        join.observe(real, _LIFELINK, participant, [PredictedMove(_GAIN, "P0", 1.0), None])
        join.fork(_probe_of(real))
        (check,) = join.checks(DAMAGE_STEP_KEYWORDS)
        assert (check.name, check.status) == (CheckName.PROBE_DIFF, CheckStatus.SKIPPED)
        assert check.detail.startswith("AttributeError")
        assert "Traceback" in caplog.text


# ── on the fixture's probed combats, through gate 2's one pass ───────────


@pytest.fixture(scope="module")
def fixture_pass(corpus, records, tmp_path_factory):
    from effects.application.model_loading import Runnable
    from effects.infrastructure.record_codec import record_from_dict
    from tests.unit.effects.builders import gen3_fixture_records
    from tests.unit.effects.training_builders import fixture_run

    run = fixture_run(corpus, records, tmp_path_factory.mktemp("probe-diff"))
    batcher = run.preparer.scoring_batcher()
    batcher.materializer = run.scoring
    runnable = Runnable(run.encoder, run.model, batcher, tuple(run.fields[1]))

    def verdicts(probes):
        return gate_two.keyword_verdicts(
            map(record_from_dict, gen3_fixture_records()), lambda _head: runnable,
            DAMAGE_STEP_KEYWORDS, probes)

    join = ProbeJoin()
    return verdicts(join), verdicts(None), join


def test_the_fixtures_trample_pair_scores_against_forges_move(fixture_pass):
    _, _, join = fixture_pass
    checks = {check.name: check for check in join.checks(DAMAGE_STEP_KEYWORDS)}
    assert "1 probed combats paired, 4 forks unpaired" in checks[CheckName.PROBE_DIFF].detail
    assert "(forks: first_strike 1)" in checks[CheckName.PROBE_DIFF].detail
    assert (join.paired, join.not_measured) == ({"trample": 1}, {"first_strike": 1})
    (trample,) = join.moves["trample"].summary(KEYWORDS_BY_NAME["trample"]).effects
    assert (trample.label, trample.combats, trample.zero_size) == (
        "defending_player.life_delta", 1, 0)
    assert checks["probe-diff trample"].detail.endswith("paired 1, forks unpaired 2")
    assert checks["probe-diff deathtouch"].detail == "share none; paired 0, forks unpaired 2"


def test_the_probe_diff_changes_no_verdict(fixture_pass):
    """Every verdict, reason and moves line is the same with the join as
    without it (the moves compared rendered: an unmeasured figure is NaN)."""
    joined, alone, _ = fixture_pass
    assert [dataclasses.replace(v, moves=None) for v in joined] == [
        dataclasses.replace(v, moves=None) for v in alone]
    assert [v.moves.render() for v in joined] == [v.moves.render() for v in alone]
    assert any(v.qualifying_records for v in joined)

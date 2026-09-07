"""The coverage collector's three rules (T102).

The consult ranks but never drops; satisfaction counts acting hosts, event
subjects and referenced refs but **not** mere snapshot presence; and a card with
no progress retires, which is the only thing that makes the run terminate.
"""

from __future__ import annotations

import pytest

from effects.application.collect_coverage import (
    DECK_SIZE,
    DEFAULT_DECKS_PER_ROUND,
    DEFAULT_NO_PROGRESS_ROUNDS,
    DEFAULT_TARGET_RECORDS,
    NONLANDS_PER_DECK,
    CardCoverage,
    CollectCoverageConfig,
    ConsultVerdict,
    count_coverage,
    deck_weights,
    is_complete,
    qualifying_cards,
    rank_by_consult,
    residues,
    retire_stalled,
)
from effects.domain.event_schema import Event, EventType
from effects.domain.records import (
    EffectRecord,
    Moment,
    RecordKind,
    ResolutionPayload,
)
from effects.domain.state_snapshot import EntityState, GlobalState, Refs, StateSnapshot


def _entity(entity_id: str, name: str) -> EntityState:
    return EntityState(
        id=entity_id, name=name, zone="battlefield", controller="P0",
    )


def _record(entities, *, source=None, targets=(), events=()) -> EffectRecord:
    return EffectRecord(
        record_id="r.0.1", run_id="r", timestamp="t", game_id="r.0.1",
        kind=RecordKind.RESOLUTION, moment=Moment.RESOLUTION,
        actor_player="P0",
        state=StateSnapshot(
            global_=GlobalState(
                turn=1, phase="main1", active="P0", priority="P0", stack_size=0,
            ),
            players=(),
            entities=tuple(entities),
            refs=Refs(source=source, targets=tuple(targets)),
        ),
        payload=ResolutionPayload(events=tuple(events)),
    )


class TestSatisfactionCounting:
    def test_the_acting_lines_host_counts(self):
        record = _record([_entity("E1", "Shock")], source="E1")
        assert qualifying_cards(record) == {"Shock"}

    def test_a_referenced_ref_counts(self):
        record = _record(
            [_entity("E1", "Shock"), _entity("E2", "Grizzly Bears")],
            source="E1", targets=("E2",),
        )
        assert qualifying_cards(record) == {"Shock", "Grizzly Bears"}

    def test_an_event_subject_counts(self):
        record = _record(
            [_entity("E1", "Shock"), _entity("E2", "Grizzly Bears")],
            source="E1",
            events=(Event(
                type=EventType.DAMAGE_DEALT, subjects=("E2",),
                params={"amount": 2},
            ),),
        )
        assert qualifying_cards(record) == {"Shock", "Grizzly Bears"}

    def test_mere_snapshot_presence_does_not_count(self):
        """A card on the battlefield while something else resolves teaches
        the model nothing about that card."""
        record = _record(
            [_entity("E1", "Shock"), _entity("E9", "Bystander")], source="E1",
        )
        assert "Bystander" not in qualifying_cards(record)

    def test_a_vanilla_creature_can_still_be_satisfied(self):
        """The unit is not resolution records: a vanilla creature has no
        acting line and is still coverable through combat."""
        record = _record(
            [_entity("E1", "Grizzly Bears")],
            events=(Event(
                type=EventType.DAMAGE_DEALT, subjects=("E1",),
                params={"amount": 3, "combat": True},
            ),),
        )
        assert qualifying_cards(record) == {"Grizzly Bears"}

    def test_counting_tallies_across_the_whole_corpus(self):
        records = [
            _record([_entity("E1", "Shock")], source="E1") for _ in range(3)
        ]
        assert count_coverage(records)["Shock"] == 3


class TestConsultRanksButNeverDrops:
    def test_an_uncastable_card_keeps_a_weight(self):
        """Being in a game is the precondition an intervention forks from."""
        weights = rank_by_consult(
            {"Weird Card": 50.0},
            {"Weird Card": ConsultVerdict.UNCASTABLE},
        )
        assert "Weird Card" in weights
        assert weights["Weird Card"] > 0

    def test_a_castable_card_outranks_an_uncastable_one(self):
        weights = rank_by_consult(
            {"a": 50.0, "b": 50.0},
            {"a": ConsultVerdict.CASTABLE, "b": ConsultVerdict.UNCASTABLE},
        )
        assert weights["a"] > weights["b"]

    def test_an_unconsulted_card_ranks_as_castable(self):
        """The safe direction: under-ranking merely slows the run down."""
        weights = rank_by_consult({"a": 50.0, "b": 50.0}, {})
        assert weights["a"] == weights["b"] == 50.0

    def test_no_card_is_ever_removed_by_the_consult(self):
        weights = {"a": 1.0, "b": 2.0, "c": 3.0}
        verdicts = dict.fromkeys(weights, ConsultVerdict.UNCASTABLE)
        assert set(rank_by_consult(weights, verdicts)) == set(weights)


class TestDeckWeighting:
    def test_a_card_with_no_records_weighs_most(self):
        coverage = {
            "empty": CardCoverage("empty", records=0),
            "half": CardCoverage("half", records=25),
        }
        weights = deck_weights(coverage, 50)
        assert weights["empty"] > weights["half"]

    def test_a_satisfied_card_drops_out_of_the_draw(self):
        coverage = {"done": CardCoverage("done", records=50)}
        assert deck_weights(coverage, 50) == {}

    def test_a_retired_card_drops_out_too(self):
        coverage = {"gone": CardCoverage("gone", records=0, retired=True)}
        assert deck_weights(coverage, 50) == {}

    def test_a_card_one_short_still_gets_drawn(self):
        """Linear in the shortfall, so the last records do not take forever."""
        coverage = {"nearly": CardCoverage("nearly", records=49)}
        assert deck_weights(coverage, 50)["nearly"] == 1.0


class TestRetirement:
    def test_a_card_that_gained_a_record_resets_its_counter(self):
        coverage = {"a": CardCoverage("a", records=5, rounds_without_progress=2)}
        retire_stalled(coverage, {"a": 3}, no_progress_rounds=3)
        assert coverage["a"].rounds_without_progress == 0
        assert not coverage["a"].retired

    def test_a_stalled_card_retires_after_the_configured_rounds(self):
        coverage = {"a": CardCoverage("a", records=0)}
        for _ in range(DEFAULT_NO_PROGRESS_ROUNDS - 1):
            retire_stalled(coverage, {"a": 0}, no_progress_rounds=3)
            assert not coverage["a"].retired
        retired = retire_stalled(coverage, {"a": 0}, no_progress_rounds=3)
        assert coverage["a"].retired
        assert retired == ["a"]

    def test_an_already_retired_card_is_not_retired_twice(self):
        coverage = {"a": CardCoverage("a", retired=True)}
        assert retire_stalled(coverage, {"a": 0}, no_progress_rounds=1) == []

    def test_the_run_terminates_once_every_card_is_done(self):
        coverage = {
            "satisfied": CardCoverage("satisfied", records=50),
            "retired": CardCoverage("retired", records=0, retired=True),
        }
        assert is_complete(coverage, 50)

    def test_the_run_continues_while_any_card_is_live(self):
        coverage = {"live": CardCoverage("live", records=1)}
        assert not is_complete(coverage, 50)


class TestResidues:
    def test_each_unsatisfied_card_is_counted_under_its_verdict(self):
        coverage = {
            "cannot": CardCoverage(
                "cannot", records=0, retired=True,
                verdict=ConsultVerdict.UNCASTABLE,
            ),
            "short": CardCoverage(
                "short", records=10, retired=True,
                verdict=ConsultVerdict.CASTABLE,
            ),
            "done": CardCoverage("done", records=50),
        }
        report = residues(coverage, 50)
        assert report.uncastable == ("cannot",)
        assert report.castable_but_short == ("short",)

    def test_a_satisfied_card_is_in_neither_residue(self):
        coverage = {"done": CardCoverage("done", records=50)}
        report = residues(coverage, 50)
        assert not report.uncastable
        assert not report.castable_but_short

    def test_the_report_says_where_the_residues_go(self):
        """Both fall to stage-three interventions; SC-006 turns on saying so."""
        rendered = residues({}, 50).render(50)
        assert "stage-three interventions" in rendered

    def test_an_unconsulted_short_card_counts_as_castable(self):
        coverage = {
            "unknown": CardCoverage("unknown", records=1, retired=True),
        }
        assert residues(coverage, 50).castable_but_short == ("unknown",)


class TestConfig:
    def test_the_defaults_are_the_contracts(self):
        config = CollectCoverageConfig()
        assert config.effect_records.as_posix() == "output/effects/records"
        assert config.target_records == DEFAULT_TARGET_RECORDS == 50
        assert config.decks_per_round == DEFAULT_DECKS_PER_ROUND == 500
        assert config.no_progress_rounds == DEFAULT_NO_PROGRESS_ROUNDS == 3
        assert config.split_from is None

    def test_the_coverage_unit_comes_from_the_card_tree_alone(self):
        """A token script is not a deckable card and could never be retired."""
        config = CollectCoverageConfig()
        assert config.coverage_folder().name == "cardsfolder"

    def test_a_reordered_cards_folder_still_finds_the_card_tree(self):
        from pathlib import Path

        config = CollectCoverageConfig(cards_folders=(
            Path("output/tokenscripts/"), Path("output/cardsfolder/"),
        ))
        assert config.coverage_folder().name == "cardsfolder"

    def test_a_deck_is_forty_cards_with_twenty_three_nonlands(self):
        assert DECK_SIZE == 40
        assert NONLANDS_PER_DECK == 23


class TestHeldOutExclusion:
    def test_no_split_holds_nothing_out(self):
        from effects.application.collect_coverage import load_held_out

        assert load_held_out(None) == frozenset()

    def test_held_out_cards_are_absent_from_the_coverage_unit(self, tmp_path):
        from effects.application.collect_coverage import build_coverage_units

        (tmp_path / "s").mkdir()
        for stem, name in (("serra_angel", "Serra Angel"), ("shock", "Shock")):
            (tmp_path / "s" / f"{stem}.txt").write_text(
                f"name: {name}\ntypes: creature\n", encoding="utf-8",
            )
        units = build_coverage_units(tmp_path, frozenset({"Shock"}))
        assert set(units) == {"Serra Angel"}


@pytest.mark.parametrize("records,target,expected", [(0, 50, False), (50, 50, True)])
def test_satisfied_is_records_at_or_above_target(records, target, expected):
    assert CardCoverage("x", records=records).satisfied(target) is expected

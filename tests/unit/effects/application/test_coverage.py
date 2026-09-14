"""The coverage collector's three rules (T102).

The consult ranks but never drops; satisfaction counts acting hosts, event
subjects and referenced refs but **not** mere snapshot presence; and a card with
no progress retires, which is the only thing that makes the run terminate.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from effects.application.collect_coverage import (
    DEFAULT_DECKS_PER_ROUND,
    DEFAULT_NO_PROGRESS_ROUNDS,
    DEFAULT_TARGET_RECORDS,
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
        retire_stalled(
            coverage, {"a": 3}, no_progress_rounds=3, decked={"a"},
        )
        assert coverage["a"].rounds_without_progress == 0
        assert not coverage["a"].retired

    def test_a_stalled_card_retires_after_the_configured_rounds(self):
        coverage = {"a": CardCoverage("a", records=0)}
        for _ in range(DEFAULT_NO_PROGRESS_ROUNDS - 1):
            retire_stalled(
                coverage, {"a": 0}, no_progress_rounds=3, decked={"a"},
            )
            assert not coverage["a"].retired
        retired = retire_stalled(
            coverage, {"a": 0}, no_progress_rounds=3, decked={"a"},
        )
        assert coverage["a"].retired
        assert retired == ["a"]

    def test_an_already_retired_card_is_not_retired_twice(self):
        coverage = {"a": CardCoverage("a", retired=True)}
        assert retire_stalled(
            coverage, {"a": 0}, no_progress_rounds=1, decked={"a"},
        ) == []

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


class TestCoverageDeckBuilding:
    """A coverage deck is 40 cards and favours the cards that need records."""

    def _texts(self, names):
        return {n: f"name: {n}\nmana cost: {{1}}{{G}}\ntypes: creature\n" for n in names}

    def test_a_deck_is_forty_cards_with_twenty_three_nonlands(self):
        import random

        from effects.application.collect_coverage import build_coverage_decks

        names = [f"card {i}" for i in range(40)]
        decks = build_coverage_decks(
            {n: 1.0 for n in names}, self._texts(names), count=3,
            rng=random.Random(1),
        )

        basics = {
            "Plains", "Island", "Swamp", "Mountain", "Forest", "Wastes",
        }
        assert len(decks) == 3
        for deck in decks:
            assert len(deck) == 40
            assert sum(1 for card in deck if card not in basics) == 23, (
                "a coverage deck is 23 nonlands plus basics; asserting only "
                "the total lets a 40-basic deck through"
            )

    def test_a_heavier_card_appears_more_often(self):
        import random

        from effects.application.collect_coverage import build_coverage_decks

        names = [f"card {i}" for i in range(40)]
        weights = {n: 1.0 for n in names}
        weights["card 0"] = 500.0
        decks = build_coverage_decks(
            weights, self._texts(names), count=40, rng=random.Random(1),
        )

        flat = [n for deck in decks for n in deck]
        assert flat.count("card 0") > flat.count("card 1")

    def test_a_card_with_no_converted_text_is_skipped(self):
        """A weight can name a card the tree has no text for; basics need text."""
        import random

        from effects.application.collect_coverage import build_coverage_decks

        names = [f"card {i}" for i in range(40)]
        decks = build_coverage_decks(
            {**{n: 1.0 for n in names}, "ghost": 5.0},
            self._texts(names), count=2, rng=random.Random(1),
        )

        assert all("ghost" not in deck for deck in decks)


class TestHeldOutExclusion:
    def test_no_split_holds_nothing_out(self):
        from effects.application.collect_coverage import load_held_out

        assert load_held_out(None) == frozenset()

    def test_held_out_cards_are_absent_from_the_coverage_unit(self, tmp_path):
        from effects.application.collect_coverage import build_coverage_units
        from effects.application.train_effect_model import load_card_files

        (tmp_path / "s").mkdir()
        for stem, name in (("serra_angel", "Serra Angel"), ("shock", "Shock")):
            (tmp_path / "s" / f"{stem}.txt").write_text(
                f"name: {name}\ntypes: creature\n", encoding="utf-8",
            )
        units = build_coverage_units(
            load_card_files(tmp_path), frozenset({"Shock"}),
        )
        assert set(units) == {"Serra Angel"}


@pytest.mark.parametrize("records,target,expected", [(0, 50, False), (50, 50, True)])
def test_satisfied_is_records_at_or_above_target(records, target, expected):
    assert CardCoverage("x", records=records).satisfied(target) is expected


class TestRunClosesTheSupervisor:
    """final-fix-3.md item 5: ``CollectorSupervisor.stop()`` -- the only
    caller of ``WorkerLogFiles.close_all()`` -- was never invoked by
    ``run()``, so a coverage run's worker log handles (opened for exactly the
    latched effect-record failure reporters F4 exists to make visible to an
    operator) stayed open until the interpreter exited on its own, not when
    the run itself finished.
    """

    def _config(self, tmp_path, **overrides):
        cardsfolder = tmp_path / "cardsfolder"
        cardsfolder.mkdir()
        (cardsfolder / "bears.txt").write_text(
            "name: Grizzly Bears\ntypes: creature\n", encoding="utf-8",
        )
        overrides.setdefault("target_records", 0)
        return CollectCoverageConfig(
            cards_folders=(cardsfolder,),
            effect_records=tmp_path / "records",
            **overrides,
        )

    def test_stop_runs_even_though_no_round_was_ever_needed(
        self, tmp_path, monkeypatch,
    ):
        """``target_records=0`` satisfies every card before the ``while``
        loop's first check, so ``play_round`` is never called -- proving
        ``stop()`` is not merely tacked on after the last round, but wraps
        the supervisor's whole lifetime regardless of how many rounds ran."""
        from unittest.mock import MagicMock

        import effects.application.collect_coverage as collect_coverage
        import effects.infrastructure.collector_connector as collector_connector

        monkeypatch.setattr(
            collect_coverage, "consult_castability", lambda names, folder: {},
        )
        supervisor = MagicMock()
        monkeypatch.setattr(
            collector_connector, "CollectorSupervisor",
            MagicMock(return_value=supervisor),
        )

        code = collect_coverage.run(self._config(tmp_path))

        assert code == 0
        supervisor.play_round.assert_not_called()
        supervisor.stop.assert_called_once()

    def test_stop_runs_even_if_a_round_raises(self, tmp_path, monkeypatch):
        """The property a bare call after the loop cannot give: a round that
        raises must still close the log handles and shut the pool down
        before the exception is let through, not leave the handles open on
        the way out."""
        from unittest.mock import MagicMock

        import effects.application.collect_coverage as collect_coverage
        import effects.infrastructure.collector_connector as collector_connector

        monkeypatch.setattr(
            collect_coverage, "consult_castability", lambda names, folder: {},
        )
        supervisor = MagicMock()
        supervisor.play_round.side_effect = RuntimeError("boom")
        monkeypatch.setattr(
            collector_connector, "CollectorSupervisor",
            MagicMock(return_value=supervisor),
        )

        with pytest.raises(RuntimeError, match="boom"):
            collect_coverage.run(self._config(tmp_path, target_records=50))

        supervisor.play_round.assert_called_once()
        supervisor.stop.assert_called_once()


class TestExclusionsWithoutACheckpoint:
    """The holdout no longer needs a trained model to exist (T171).

    `--split-from` was the only way to learn the holdout when it was derived at
    training time. It is now derived from the converted tree and two flags, so
    requiring a checkpoint makes coverage collection wait for a training run
    that is itself supposed to come after it — which the quickstart papered over
    by telling an operator to run step 6 once and come back.
    """

    def test_the_depletion_list_is_read_and_folded(self, tmp_path) -> None:
        from effects.application.collect_coverage import load_exclusions

        listed = tmp_path / "holdout-cards.txt"
        listed.write_text("Soul Echo\nLim-Dûl's Vault\n", encoding="utf-8")

        excluded = load_exclusions(split_from=None, exclude_cards=listed)

        assert excluded == frozenset({"soul echo", "lim-dûl's vault"}), (
            "the file carries Forge's printed case; coverage decks are built "
            "from converted names, which are lowercase"
        )

    def test_neither_source_means_no_exclusions(self) -> None:
        from effects.application.collect_coverage import load_exclusions

        assert load_exclusions(split_from=None, exclude_cards=None) == frozenset()

    def test_both_sources_at_once_is_refused(self, tmp_path) -> None:
        """Two spellings of the same holdout is the bug this feature just had."""
        from effects.application.collect_coverage import load_exclusions

        listed = tmp_path / "holdout-cards.txt"
        listed.write_text("Soul Echo\n", encoding="utf-8")

        with pytest.raises(ValueError, match="one source"):
            load_exclusions(split_from=tmp_path / "latest.pt", exclude_cards=listed)


class TestTheRoundPlaysTheComputedDecks:
    """The weights have to reach the table, which is what never happened.

    `play_round` received correctly-computed weights and discarded them, so the
    workers played ordinary sealed self-play: a slower duplicate of
    `match-outcomes` that reaches none of the cards coverage exists for.
    """

    def test_the_decks_played_are_built_from_the_weights(self, tmp_path):
        from effects.application import collect_coverage
        from effects.infrastructure import collector_connector, record_io

        cards = tmp_path / "cardsfolder"
        (cards / "a").mkdir(parents=True)
        (cards / "a" / "aa.txt").write_text(
            "name: aa\nmana cost: {G}\ntypes: creature\n", encoding="utf-8",
        )
        (cards / "a" / "bb.txt").write_text(
            "name: bb\nmana cost: {G}\ntypes: creature\n", encoding="utf-8",
        )
        supervisor = MagicMock()
        supervisor.interrupted = False

        # Two candidates, and a recount that plays out over three rounds, so
        # the test can tell "decks track the weights `run` recomputes each
        # round" apart from "decks track a fixture that happens to have one
        # candidate" -- a single-card pool gives `rng.choices` no real choice
        # to make, and a single round never exercises the recompute at all.
        # Round 1 sees no qualifying records yet, so "aa" and "bb" still
        # weigh the same -- uninteresting, and that round's decks are
        # overwritten before the test ever reads them. Round 2's recount
        # reports "aa" at 9 of a target of 10 (nearly satisfied) while "bb"
        # stays at 0, so round 3 computes sharply different weights (aa
        # shortfall 1, bb shortfall 10). Round 3 is also the *last* round:
        # its own recount (still "aa": 10, "bb" untouched) satisfies "aa"
        # and is "bb"'s third consecutive stalled round, so both are done
        # and `run` returns without overwriting round 3's decks -- which are
        # exactly the ones this test reads.
        aa_record = _record([_entity("E1", "aa")], source="E1")
        recounts = iter([[], [aa_record] * 9, [aa_record] * 10])

        with patch.object(collector_connector, "CollectorSupervisor",
                          return_value=supervisor), \
             patch.object(collect_coverage, "consult_castability", return_value={}), \
             patch.object(record_io, "read_records",
                          side_effect=lambda directory: next(recounts)):
            collect_coverage.run(
                collect_coverage.CollectCoverageConfig(
                    effect_records=tmp_path / "records",
                    cards_folders=(cards,),
                    target_records=10,
                    decks_per_round=150,
                )
            )

        decks_file = supervisor.play_round.call_args.args[0]
        rows = decks_file.read_text(encoding="utf-8").splitlines()
        assert rows, "no decks were written"
        assert all(row.split(";")[2].count("|") == 39 for row in rows)
        assert supervisor.play_round.call_args.kwargs["matches"] == 150

        flat = [name for row in rows for name in row.split(";")[2].split("|")]
        assert flat.count("bb") > flat.count("aa") * 3, (
            "bb is 10 records short of target and aa is only 1 short after "
            "the round-2 recount, so the final round's decks must favour bb "
            "heavily -- a 'deck whatever is in texts, ignore the weights "
            "argument' implementation would split them roughly evenly "
            "instead and fail this assertion"
        )


class TestARoundEndsInItsReport:
    """Final review, CRITICAL 1, at the level the operator sees.

    ``run``'s ``finally`` calls ``CollectorSupervisor.stop()``, which called a
    pool method that does not exist, so every round ended in
    ``AttributeError``: ``residues(...).render(...)`` (FR-051) never printed
    and ``run`` never returned its code. Every other test in this file mocks
    the supervisor, and a ``MagicMock`` answers to any method name at all --
    so none of them could see it. This one mocks neither the supervisor nor
    the pool, stubbing only ``ForgeWorkerPool.run``, the blocking loop that
    would otherwise spawn real JVMs.
    """

    def _one_card_config(self, tmp_path, **overrides):
        cards = tmp_path / "cardsfolder"
        cards.mkdir()
        (cards / "aa.txt").write_text(
            "name: aa\nmana cost: {G}\ntypes: creature\n", encoding="utf-8",
        )
        (tmp_path / "records").mkdir()
        return CollectCoverageConfig(
            effect_records=tmp_path / "records",
            cards_folders=(cards,),
            target_records=1,
            decks_per_round=2,
            no_progress_rounds=1,
            workers=1,
            **overrides,
        )

    def test_the_run_finishes_reports_and_returns_zero(self, tmp_path, caplog):
        import logging

        from effects.application import collect_coverage
        from price_predictor.infrastructure.forge_jvm import ForgeWorkerPool

        config = self._one_card_config(tmp_path)

        with patch.object(ForgeWorkerPool, "run", autospec=True) as pool_run, \
             patch.object(collect_coverage, "consult_castability", return_value={}), \
             caplog.at_level(logging.INFO):
            code = collect_coverage.run(config)

        assert pool_run.called, "no round was ever played"
        assert code == 0
        assert any("Residues:" in record.getMessage() for record in caplog.records), (
            "the run ended without printing the two residues FR-051 requires"
        )


class TestCtrlCEndsTheRun:
    """Final review, IMPORTANT 3: nothing in ``effects`` read
    ``ForgeWorkerPool.interrupted``, so ``run`` built a fresh pool for the
    next round and Ctrl-C degraded to "skip this round, start another".

    ``spec=CollectorSupervisor`` rather than a bare ``MagicMock`` on purpose:
    a bare one answers to ``interrupted`` whether or not the supervisor
    exposes it, which is the same blindness that let CRITICAL 1 ship.
    """

    def _two_round_config(self, tmp_path):
        cards = tmp_path / "cardsfolder"
        cards.mkdir()
        (cards / "aa.txt").write_text(
            "name: aa\nmana cost: {G}\ntypes: creature\n", encoding="utf-8",
        )
        return CollectCoverageConfig(
            effect_records=tmp_path / "records",
            cards_folders=(cards,),
            target_records=50,
            decks_per_round=2,
            no_progress_rounds=5,
        )

    def _run_with(self, tmp_path, *, interrupted: bool):
        from effects.application import collect_coverage
        from effects.infrastructure import collector_connector

        supervisor = MagicMock(spec=collector_connector.CollectorSupervisor)
        supervisor.interrupted = interrupted

        with patch.object(collector_connector, "CollectorSupervisor",
                          return_value=supervisor), \
             patch.object(collect_coverage, "consult_castability", return_value={}):
            code = collect_coverage.run(self._two_round_config(tmp_path))
        return code, supervisor

    def test_an_interrupted_round_is_the_last_round(self, tmp_path):
        code, supervisor = self._run_with(tmp_path, interrupted=True)

        assert supervisor.play_round.call_count == 1, (
            "a second round was started after the operator interrupted the "
            "first one"
        )
        assert code == 130, "an interrupted run must not report success"

    def test_an_uninterrupted_run_plays_every_round_it_needs(self, tmp_path):
        """The control: five stalled rounds, then retirement ends the run."""
        code, supervisor = self._run_with(tmp_path, interrupted=False)

        assert supervisor.play_round.call_count == 5
        assert code == 0


class TestOnlyDeckedCardsCountAStalledRound:
    """Final review, IMPORTANT 4: ``retire_stalled`` counted a stalled round
    against every card that gained no qualifying record, including cards no
    deck that round ever held.

    At the defaults -- 33,680 cards against 500 decks of 23 nonland slots --
    a given card is left out of a whole round with probability about 0.71, so
    roughly 36% of the corpus retired after three rounds without a single
    attempt. FR-050's letter was met (the run terminates) but it terminated
    having collected almost nothing while reporting a huge "castable but
    short" residue.
    """

    def test_a_card_no_deck_held_does_not_count_a_stalled_round(self):
        coverage = {"undecked": CardCoverage("undecked", records=0)}

        for _ in range(5):
            retire_stalled(
                coverage, {"undecked": 0}, no_progress_rounds=1,
                decked=frozenset(),
            )

        assert coverage["undecked"].rounds_without_progress == 0
        assert not coverage["undecked"].retired, (
            "a card that was never put in a deck retired without one attempt"
        )

    def test_the_round_reports_exactly_the_names_it_decked(self, tmp_path):
        """``run`` must hand ``retire_stalled`` the round's own decks, not the
        whole coverage unit -- otherwise the guard above never fires."""
        from effects.application import collect_coverage
        from effects.infrastructure import collector_connector

        # Sixty candidates against a single deck of 23 nonland slots: most of
        # the unit cannot be in the round at all, which is the real shape of
        # the finding (33,680 cards against 500 x 23 slots).
        names = [f"c{index:02d}" for index in range(60)]
        cards = tmp_path / "cardsfolder"
        cards.mkdir()
        for name in names:
            (cards / f"{name}.txt").write_text(
                f"name: {name}\nmana cost: {{G}}\ntypes: creature\n",
                encoding="utf-8",
            )
        supervisor = MagicMock(spec=collector_connector.CollectorSupervisor)
        supervisor.interrupted = False
        seen = []
        real = collect_coverage.retire_stalled

        def _spy(coverage, previous, *, no_progress_rounds, decked):
            seen.append(frozenset(decked))
            return real(
                coverage, previous, no_progress_rounds=no_progress_rounds,
                decked=decked,
            )

        with patch.object(collector_connector, "CollectorSupervisor",
                          return_value=supervisor), \
             patch.object(collect_coverage, "consult_castability", return_value={}), \
             patch.object(collect_coverage, "retire_stalled", _spy):
            collect_coverage.run(CollectCoverageConfig(
                effect_records=tmp_path / "records",
                cards_folders=(cards,),
                target_records=1,
                decks_per_round=1,
                no_progress_rounds=1,
            ))

        decks_file = supervisor.play_round.call_args.args[0]
        decked_names = {
            name
            for row in decks_file.read_text(encoding="utf-8").splitlines()
            for name in row.split(";")[2].split("|")
        }
        assert seen, "retire_stalled was never called"
        assert len(seen[0] & set(names)) < len(names), (
            "the whole coverage unit was reported as decked; one deck of 23 "
            "slots cannot have held all 60 candidates"
        )
        # The last round is the one whose decks file survived the loop.
        assert seen[-1] == decked_names

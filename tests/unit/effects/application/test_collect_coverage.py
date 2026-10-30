"""The coverage collector (T102): its rounds, its counting and its refusals.

A card with no progress retires, which is the only thing that makes the run
terminate; the unit's own rules are in ``tests/unit/effects/domain/test_coverage.py``.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from effects.application.collect_coverage import (
    DEFAULT_NO_PROGRESS_ROUNDS,
    CollectCoverageConfig,
    residues,
)
from effects.application.collector_common import DeckHoldout
from effects.domain.coverage import (
    CardCoverage,
    ConsultVerdict,
    is_complete,
    retire_stalled,
)
from effects.domain.event_schema import Event, EventType
from effects.domain.records import (
    RecordKind,
    ResolutionPayload,
)
from tests.unit.effects.builders import (
    fresh_interpreter_env,
    held_out_cards,
    make_entity,
)
from tests.unit.effects.domain.test_coverage import _record


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


class TestCoverageDeckBuilding:
    """A coverage deck is 40 cards and favours the cards that need records."""

    def _candidates(self, names):
        from effects.domain.coverage import DeckCandidate

        return {n: DeckCandidate(n, "{1}{G}") for n in names}

    def test_a_deck_is_forty_cards_with_twenty_three_nonlands(self):
        import random

        from effects.application.collector_common import build_coverage_decks

        names = [f"card {i}" for i in range(40)]
        decks = build_coverage_decks(
            {n: 1.0 for n in names}, self._candidates(names), count=3,
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

        from effects.application.collector_common import build_coverage_decks

        names = [f"card {i}" for i in range(40)]
        weights = {n: 1.0 for n in names}
        weights["card 0"] = 500.0
        decks = build_coverage_decks(
            weights, self._candidates(names), count=40, rng=random.Random(1),
        )

        flat = [n for deck in decks for n in deck]
        assert flat.count("card 0") > flat.count("card 1")

    def test_a_card_with_no_converted_text_is_skipped(self):
        """A weight can name a card the tree has no text for; basics need its cost."""
        import random

        from effects.application.collector_common import build_coverage_decks

        names = [f"card {i}" for i in range(40)]
        decks = build_coverage_decks(
            {**{n: 1.0 for n in names}, "ghost": 5.0},
            self._candidates(names), count=2, rng=random.Random(1),
        )

        assert all("ghost" not in deck for deck in decks)


class TestHeldOutExclusion:
    def test_no_split_holds_nothing_out(self):
        from effects.application.collector_common import load_held_out

        assert load_held_out(None) == frozenset()

    def test_held_out_cards_are_absent_from_the_coverage_unit(self, tmp_path):
        from effects.application.collect_coverage import build_coverage_units
        from effects.infrastructure.converted_trees import ConvertedTrees

        (tmp_path / "s").mkdir()
        for stem, name in (("serra_angel", "Serra Angel"), ("shock", "Shock")):
            (tmp_path / "s" / f"{stem}.txt").write_text(
                f"name: {name}\ntypes: creature\n", encoding="utf-8",
            )
        units = build_coverage_units(
            ConvertedTrees({"cardsfolder": tmp_path}).card_files, frozenset({"Shock"}),
        )
        assert set(units) == {"Serra Angel"}


class TestExclusionsWithoutACheckpoint:
    """The holdout no longer needs a trained model to exist (T171).

    `--split-from` was the only way to learn the holdout when it was derived at
    training time. It is now derived from the converted tree and two flags, so
    requiring a checkpoint makes coverage collection wait for a training run
    that is itself supposed to come after it — which the quickstart papered over
    by telling an operator to run step 6 once and come back.
    """

    def test_the_depletion_list_is_read_and_folded(self, tmp_path) -> None:
        listed = tmp_path / "holdout-cards.txt"
        listed.write_text("Soul Echo\nLim-Dûl's Vault\n", encoding="utf-8")

        excluded = DeckHoldout(exclude_cards=listed).exclusions()

        assert excluded == frozenset({"soul echo", "lim-dûl's vault"}), (
            "the file carries Forge's printed case; coverage decks are built "
            "from converted names, which are lowercase"
        )

    def test_neither_source_means_no_exclusions(self) -> None:
        assert DeckHoldout().exclusions() == frozenset()

    def test_both_sources_at_once_is_refused(self, tmp_path) -> None:
        """Two spellings of the same holdout is the bug this feature just had."""
        listed = tmp_path / "holdout-cards.txt"
        listed.write_text("Soul Echo\n", encoding="utf-8")

        with pytest.raises(ValueError, match="one source"):
            DeckHoldout(split_from=tmp_path / "latest.pt", exclude_cards=listed)

    def test_both_sources_on_a_config_are_refused_when_it_runs(self, tmp_path) -> None:
        """A config built in code, not from flags, still names one holdout."""
        from effects.application import collect_coverage

        listed = tmp_path / "holdout-cards.txt"
        listed.write_text("Soul Echo\n", encoding="utf-8")
        cards = tmp_path / "cardsfolder"
        cards.mkdir()
        (cards / "bears.txt").write_text(
            "name: Grizzly Bears\ntypes: creature\n", encoding="utf-8",
        )

        with pytest.raises(ValueError, match="one source"):
            collect_coverage.run(collect_coverage.CollectCoverageConfig(
                cards_folders=(cards,), effect_records=tmp_path / "records",
                deck_holdout=DeckHoldout(
                    split_from=tmp_path / "latest.pt", exclude_cards=listed,
                ),
            ))


class TestTheRoundPlaysTheComputedDecks:
    """The weights have to reach the table, which is what never happened.

    `play_round` received correctly-computed weights and discarded them, so the
    workers played ordinary sealed self-play: a slower duplicate of
    `match-outcomes` that reaches none of the cards coverage exists for.
    """

    def test_the_decks_played_are_built_from_the_weights(self, tmp_path, fake_supervisor):
        from effects.application import collect_coverage

        cards = tmp_path / "cardsfolder"
        (cards / "a").mkdir(parents=True)
        (cards / "a" / "aa.txt").write_text(
            "name: aa\nmana cost: {G}\ntypes: creature\n", encoding="utf-8",
        )
        (cards / "a" / "bb.txt").write_text(
            "name: bb\nmana cost: {G}\ntypes: creature\n", encoding="utf-8",
        )
        supervisor = fake_supervisor

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
        # Each round's play writes the shard that round produced, which is how
        # the real counter learns anything: it reads shards it has not seen and
        # adds to a running total. Round 1 produces nothing, round 2 nine "aa"
        # records, round 3 the tenth.
        from effects.infrastructure.record_codec import format_record_line

        records_dir = tmp_path / "records"
        records_dir.mkdir()
        produced = iter([0, 9, 1])

        def _play(_decks, _decks_file, *, label, matches):
            count = next(produced, 0)
            if not count:
                return
            line = format_record_line(
                _record([make_entity("E1", "aa")], source="E1")
            )
            shard = records_dir / f"run.0-{matches}{count}.jsonl"
            shard.write_text("\n".join([line] * count) + "\n", encoding="utf-8")

        supervisor.play_round.side_effect = _play

        with patch.object(collect_coverage, "consult_castability", return_value={}):
            collect_coverage.run(
                collect_coverage.CollectCoverageConfig(
                    effect_records=tmp_path / "records",
                    cards_folders=(cards,),
                    target_records=10,
                    decks_per_round=150,
                )
            )

        decks = supervisor.play_round.call_args.args[0]
        assert decks, "no decks were played"
        assert all(len(deck) == 40 for deck in decks)
        assert supervisor.play_round.call_args.kwargs["matches"] == 150

        flat = [name for deck in decks for name in deck]
        assert flat.count("bb") > flat.count("aa") * 3, (
            "bb is 10 records short of target and aa is only 1 short after "
            "the round-2 recount, so the final round's decks must favour bb "
            "heavily -- a 'deck whatever is in texts, ignore the weights "
            "argument' implementation would split them roughly evenly "
            "instead and fail this assertion"
        )


class TestARoundEndsInItsReport:
    """A round ends in its report, at the level the operator sees.

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
    """Nothing in ``effects`` used to read
    ``ForgeWorkerPool.interrupted``, so ``run`` built a fresh pool for the
    next round and Ctrl-C degraded to "skip this round, start another".

    The ``fake_supervisor`` is ``spec``-ed on purpose: a bare ``MagicMock``
    answers to ``interrupted`` whether or not the supervisor exposes it, which
    is the same blindness ``TestARoundEndsInItsReport`` exists for.
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

    def _run_with(self, tmp_path, supervisor, *, interrupted: bool):
        from effects.application import collect_coverage

        supervisor.interrupted = interrupted
        with patch.object(collect_coverage, "consult_castability", return_value={}):
            code = collect_coverage.run(self._two_round_config(tmp_path))
        return code, supervisor

    def test_an_interrupted_round_is_the_last_round(self, tmp_path, fake_supervisor):
        code, supervisor = self._run_with(tmp_path, fake_supervisor, interrupted=True)

        assert supervisor.play_round.call_count == 1, (
            "a second round was started after the operator interrupted the "
            "first one"
        )
        assert code == 130, "an interrupted run must not report success"

    def test_an_uninterrupted_run_plays_every_round_it_needs(self, tmp_path, fake_supervisor):
        """The control: five stalled rounds, then retirement ends the run."""
        code, supervisor = self._run_with(tmp_path, fake_supervisor, interrupted=False)

        assert supervisor.play_round.call_count == 5
        assert code == 0


class TestOnlyDeckedCardsCountAStalledRound:
    """``retire_stalled`` used to count a stalled round
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

    def test_the_round_reports_exactly_the_names_it_decked(self, tmp_path, fake_supervisor):
        """``run`` must hand ``retire_stalled`` the round's own decks, not the
        whole coverage unit -- otherwise the guard above never fires."""
        from effects.application import collect_coverage

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
        supervisor = fake_supervisor
        seen = []
        real = collect_coverage.retire_stalled

        def _spy(coverage, previous, *, no_progress_rounds, decked):
            seen.append(frozenset(decked))
            return real(
                coverage, previous, no_progress_rounds=no_progress_rounds,
                decked=decked,
            )

        with patch.object(collect_coverage, "consult_castability", return_value={}), \
             patch.object(collect_coverage, "retire_stalled", _spy):
            collect_coverage.run(CollectCoverageConfig(
                effect_records=tmp_path / "records",
                cards_folders=(cards,),
                target_records=1,
                decks_per_round=1,
                no_progress_rounds=1,
            ))

        decked_names = {
            name for deck in supervisor.play_round.call_args.args[0] for name in deck
        }
        assert seen, "retire_stalled was never called"
        assert len(seen[0] & set(names)) < len(names), (
            "the whole coverage unit was reported as decked; one deck of 23 "
            "slots cannot have held all 60 candidates"
        )
        # The last round is the one whose decks file survived the loop.
        assert seen[-1] == decked_names


class TestIncrementalCounting:
    """Counting each shard once instead of the whole corpus every round.

    The recount re-read every shard under `--effect-records` after each round.
    Against a corpus of 1,502 shards that measured ~66 minutes per round,
    against ~35 minutes of actual play — so a coverage run spent most of its
    wall clock re-deriving numbers that had not changed.

    Safe because a round's workers are all terminated before `play_round`
    returns (`ForgeWorkerPool._supervisor_loop` calls `_terminate_all` when the
    stop condition fires), and because a shard file is never reopened: its name
    carries the worker's JVM lifetime, so each round's workers write new files.
    A counted shard is therefore final.
    """

    def _shard(self, directory: Path, name: str, cards: list[str]) -> Path:
        """One shard whose every record qualifies for the named card."""
        from effects.infrastructure.record_codec import format_record_line

        directory.mkdir(parents=True, exist_ok=True)
        lines = [
            format_record_line(
                _record([make_entity(f"E-{card}", card)], source=f"E-{card}")
            )
            for card in cards
        ]
        path = directory / f"{name}.jsonl"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def test_the_first_pass_counts_every_shard(self, tmp_path) -> None:
        from effects.application.collect_coverage import CoverageCounter

        self._shard(tmp_path, "a.0-x", ["alpha", "beta"])
        self._shard(tmp_path, "a.0-y", ["alpha"])
        coverage = {n: CardCoverage(n) for n in ("alpha", "beta")}

        CoverageCounter().update(coverage, tmp_path)

        assert coverage["alpha"].records == 2
        assert coverage["beta"].records == 1

    def test_a_second_pass_over_the_same_shards_adds_nothing(self, tmp_path) -> None:
        """The bug this replaces: every round re-counted the whole corpus."""
        from effects.application.collect_coverage import CoverageCounter

        self._shard(tmp_path, "a.0-x", ["alpha"])
        coverage = {"alpha": CardCoverage("alpha")}
        counter = CoverageCounter()

        counter.update(coverage, tmp_path)
        counter.update(coverage, tmp_path)

        assert coverage["alpha"].records == 1

    def test_a_new_shard_adds_to_the_running_total(self, tmp_path) -> None:
        from effects.application.collect_coverage import CoverageCounter

        self._shard(tmp_path, "a.0-x", ["alpha"])
        coverage = {"alpha": CardCoverage("alpha")}
        counter = CoverageCounter()
        counter.update(coverage, tmp_path)

        self._shard(tmp_path, "a.0-y", ["alpha", "alpha"])
        counter.update(coverage, tmp_path)

        assert coverage["alpha"].records == 3

    def test_it_reports_how_many_shards_it_read(self, tmp_path) -> None:
        """The operator needs to see the round cost fall, not infer it."""
        from effects.application.collect_coverage import CoverageCounter

        self._shard(tmp_path, "a.0-x", ["alpha"])
        counter = CoverageCounter()
        coverage = {"alpha": CardCoverage("alpha")}

        assert counter.update(coverage, tmp_path) == 1
        assert counter.update(coverage, tmp_path) == 0


class TestCountingCrossesTheNameBoundary:
    """Coverage units are converted names; record entities are Forge's (T174).

    `build_coverage_units` keys off `ConvertedTrees.card_files`, which reads names out of
    converted card text — lowercased, `soul echo`. A record's `EntityState.name`
    is Forge's own, `Soul Echo`. Looking one up in the other matches nothing, so
    every card counts zero records forever: nothing is ever satisfied, the
    weights never fall, and cards leave the run only by retiring.

    The same converted-tree/Forge boundary that broke depletion, the split, the
    variant skip and the holdout file. Fifth instance.
    """

    def test_a_printed_case_record_counts_toward_a_converted_case_card(
        self, tmp_path,
    ) -> None:
        from effects.application.collect_coverage import CoverageCounter
        from effects.infrastructure.record_codec import format_record_line

        shard = tmp_path / "run.0-x.jsonl"
        shard.write_text(
            format_record_line(
                _record([make_entity("E1", "Soul Echo")], source="E1")
            ) + "\n",
            encoding="utf-8",
        )
        coverage = {"soul echo": CardCoverage("soul echo")}

        CoverageCounter().update(coverage, tmp_path)

        assert coverage["soul echo"].records == 1

    def test_a_card_outside_the_coverage_unit_is_still_ignored(
        self, tmp_path,
    ) -> None:
        from effects.application.collect_coverage import CoverageCounter
        from effects.infrastructure.record_codec import format_record_line

        shard = tmp_path / "run.0-x.jsonl"
        shard.write_text(
            format_record_line(
                _record([make_entity("E1", "Some Token")], source="E1")
            ) + "\n",
            encoding="utf-8",
        )
        coverage = {"soul echo": CardCoverage("soul echo")}

        CoverageCounter().update(coverage, tmp_path)

        assert coverage["soul echo"].records == 0


class TestCountingReportsProgress:
    """A silent 35-minute startup is indistinguishable from a hang.

    Moving the corpus scan from per-round to once made the round loop fast and
    the startup long. Without output it looks exactly like the hang this whole
    area kept producing, which is the one shape an operator must never have to
    guess at.
    """

    def _shards(self, directory: Path, count: int) -> None:
        from effects.infrastructure.record_codec import format_record_line

        directory.mkdir(parents=True, exist_ok=True)
        line = format_record_line(_record([make_entity("E1", "aa")], source="E1"))
        for n in range(count):
            (directory / f"run.0-{n:03d}.jsonl").write_text(
                line + "\n", encoding="utf-8",
            )

    def test_it_reports_progress_while_reading(self, tmp_path, caplog) -> None:
        from effects.application.collect_coverage import CoverageCounter

        self._shards(tmp_path, 120)
        coverage = {"aa": CardCoverage("aa")}

        with caplog.at_level("INFO"):
            CoverageCounter(progress_every=50).update(coverage, tmp_path)

        progress = [r for r in caplog.messages if "of 120 shard" in r]
        assert len(progress) >= 2, f"expected periodic progress, got {progress}"

    def test_a_short_read_says_nothing(self, tmp_path, caplog) -> None:
        """Counting three shards should not narrate itself."""
        from effects.application.collect_coverage import CoverageCounter

        self._shards(tmp_path, 3)
        coverage = {"aa": CardCoverage("aa")}

        with caplog.at_level("INFO"):
            CoverageCounter(progress_every=50).update(coverage, tmp_path)

        assert not [r for r in caplog.messages if "shard" in r and "of 3" in r]

    def test_the_held_out_round_reports_progress_too(self, tmp_path, caplog) -> None:
        """Its first pass reads the same whole corpus, and was silent."""
        from effects.application.collect_coverage import TextCoverageCounter
        from effects.domain.coverage import TextCoverage

        self._shards(tmp_path, 120)
        counter = TextCoverageCounter(
            {"cardsfolder": str(tmp_path / "none")}, _holding({"t"}), progress_every=50,
        )

        with caplog.at_level("INFO"):
            read = counter.update({"t": TextCoverage(text="t")}, tmp_path, workers=1)

        assert read == 120
        progress = [r for r in caplog.messages if "of 120 shard" in r]
        assert len(progress) >= 2, f"expected periodic progress, got {progress}"

    def test_a_counting_worker_never_loads_torch(self, tmp_path) -> None:
        """One pool worker per core, each loading torch's CUDA DLLs, exhausts
        the Windows paging file beside a training job — and counting a shard
        needs nothing from torch."""
        import subprocess
        import sys

        self._shards(tmp_path, 1)
        shard = next(tmp_path.iterdir())
        probe = (
            "import sys\n"
            "from pathlib import Path\n"
            "from effects.application.collect_coverage import count_shard\n"
            f"count_shard(Path({str(shard)!r}))\n"
            "print('torch' in sys.modules)\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True, text=True, check=True, env=fresh_interpreter_env(),
        )
        assert result.stdout.strip() == "False"


class TestParallelCounting:
    """Counting is CPU-bound per shard and the shards are independent.

    Each shard contributes a count per card and nothing else, so the work maps
    cleanly: read shards in parallel, sum the per-shard counters, apply once.
    Summation is associative and commutative, so the result cannot depend on
    how the shards were divided or in what order they finished — which is the
    property worth pinning, since a parallel count that merely *usually* agrees
    with the serial one is worse than no parallelism at all.
    """

    def _shards(self, directory: Path, spec: dict[str, int]) -> None:
        from effects.infrastructure.record_codec import format_record_line

        directory.mkdir(parents=True, exist_ok=True)
        for n, (card, count) in enumerate(spec.items()):
            line = format_record_line(
                _record([make_entity("E1", card)], source="E1")
            )
            (directory / f"run.0-{n:03d}.jsonl").write_text(
                "\n".join([line] * count) + "\n", encoding="utf-8",
            )

    def test_parallel_and_serial_agree(self, tmp_path) -> None:
        from effects.application.collect_coverage import CoverageCounter

        spec = {f"Card {i}": i + 1 for i in range(12)}
        self._shards(tmp_path, spec)
        expected = {f"card {i}": i + 1 for i in range(12)}

        serial = {n: CardCoverage(n) for n in expected}
        CoverageCounter().update(serial, tmp_path, workers=1)
        parallel = {n: CardCoverage(n) for n in expected}
        CoverageCounter().update(parallel, tmp_path, workers=4)

        assert {n: c.records for n, c in serial.items()} == expected
        assert {n: c.records for n, c in parallel.items()} == expected

    def test_a_card_split_across_shards_sums(self, tmp_path) -> None:
        """The reduce step, which is where a parallel count goes wrong."""
        from effects.application.collect_coverage import CoverageCounter
        from effects.infrastructure.record_codec import format_record_line

        tmp_path.mkdir(parents=True, exist_ok=True)
        line = format_record_line(_record([make_entity("E1", "Soul Echo")], source="E1"))
        for n in range(6):
            (tmp_path / f"run.0-{n}.jsonl").write_text(
                "\n".join([line] * 5) + "\n", encoding="utf-8",
            )

        coverage = {"soul echo": CardCoverage("soul echo")}
        CoverageCounter().update(coverage, tmp_path, workers=3)

        assert coverage["soul echo"].records == 30

    def test_already_counted_shards_stay_counted_once(self, tmp_path) -> None:
        from effects.application.collect_coverage import CoverageCounter

        self._shards(tmp_path, {"Alpha": 4})
        coverage = {"alpha": CardCoverage("alpha")}
        counter = CoverageCounter()

        counter.update(coverage, tmp_path, workers=2)
        counter.update(coverage, tmp_path, workers=2)

        assert coverage["alpha"].records == 4


# ── the held-out coverage round (FR-035–FR-038; T044, T045) ──────────────


_FIXTURES = Path(__file__).parents[3] / "fixtures" / "effects"


class TestHeldOutTextUnit:
    def test_a_held_out_text_is_one_only_listed_cards_carry(self):
        from effects.domain.coverage import held_out_texts_of

        texts = held_out_texts_of(frozenset({"alpha", "beta"}), {
            "alpha": ["SP$ Rare | Thing$ 1", "SP$ Common"],
            "beta": ["SP$ Rare  | Thing$ 1"],
            "gamma": ["SP$ Common"],
        })
        assert texts == {"SP$ Rare | Thing$ 1": frozenset({"alpha", "beta"})}

    def test_a_shard_counts_distinct_games_per_text(self):
        """Counted as build-corpus counts a held-out text's games (FR-053):
        from resolution records only, so a text whose triggers fire but never
        resolve is not satisfied."""
        from effects.application.collect_coverage import count_text_shard

        resolutions, everything = _texts_by_game(_GEN1_SHARD)
        assert resolutions and everything != resolutions
        counted = count_text_shard(
            _GEN1_SHARD, roots=_GEN1_ROOTS, held_out=_holding(everything),
        )
        assert counted == resolutions

    def test_a_game_goes_to_the_first_acting_text_held_out_by_its_unit(
        self, tmp_path,
    ):
        """The text build-corpus credits, not the first text the round lists.

        Under ``template`` the first acting key prints ``NumDmg$ 2``, whose
        template is held out though no listed card prints that exact text;
        build-corpus credits the game to it, so the round's listed ``Draw``
        text, acting second, must not count the game either.
        """
        from effects.application.collect_coverage import (
            TextCoverageCounter,
            count_text_shard,
        )
        from effects.domain.coverage import TextCoverage
        from effects.domain.provenance import ProvenanceKey, ProvenanceSidecar, SidecarLine
        from effects.domain.text_holdout import HoldoutUnit
        from effects.infrastructure.shard_io import write_shard
        from tests.unit.effects.builders import make_record, write_sidecar

        damage, draw = "SP$ DealDamage | NumDmg$ 2", "SP$ Draw | NumCards$ 1"
        keys = []
        for stem, text in (("variant", damage), ("listed", draw)):
            key = ProvenanceKey(f"cardsfolder/a/{stem}.txt", 0, "spell", 0)
            write_sidecar(
                ProvenanceSidecar(card=stem, script_file=key.script_file, lines=(
                    SidecarLine(0, "spell", (key,), script_text=text),
                )),
                tmp_path / "cardsfolder" / "a" / f"{stem}.provenance.json",
            )
            keys.append(key)
        shard = tmp_path / "records" / "run.0-a.jsonl.gz"
        write_shard(shard, [make_record(ability=tuple(keys), payload=ResolutionPayload(
            events=(Event(type=EventType.CARD_DRAWN, subjects=("P0",)),),
        ))])
        roots = {"cardsfolder": str(tmp_path / "cardsfolder")}
        held_out = _holding({"SP$ DealDamage | NumDmg$ 3", draw}, HoldoutUnit.TEMPLATE)
        coverage = {draw: TextCoverage(text=draw)}

        assert count_text_shard(shard, roots=roots, held_out=held_out) == {damage: {"g1"}}
        TextCoverageCounter(roots, held_out).update(coverage, tmp_path / "records", workers=1)
        assert coverage[draw].games == set()

    def test_a_refused_resolution_does_not_count_toward_its_text(self, tmp_path):
        """A record the quality rules refuse never reaches the manifest."""
        from dataclasses import replace

        from effects.application.collect_coverage import count_text_shard
        from effects.domain.record_quality import MAX_EVENTS_PER_RECORD
        from effects.infrastructure.shard_io import read_shard, write_shard

        effects = [
            record for record in read_shard(_GEN1_SHARD)
            if isinstance(record.payload, ResolutionPayload)
        ]
        flood = (Event(type=EventType.DAMAGE_DEALT, subjects=("P0",)),) * (
            MAX_EVENTS_PER_RECORD + 1
        )
        clean, flooded = tmp_path / "clean.jsonl.gz", tmp_path / "flooded.jsonl.gz"
        write_shard(clean, effects)
        write_shard(flooded, [
            replace(record, payload=replace(record.payload, events=flood))
            for record in effects
        ])
        _, everything = _texts_by_game(_GEN1_SHARD)
        held_out = _holding(everything)

        assert count_text_shard(clean, roots=_GEN1_ROOTS, held_out=held_out)
        assert count_text_shard(flooded, roots=_GEN1_ROOTS, held_out=held_out) == {}

    def test_a_process_reads_each_sidecar_once_across_shards(self, monkeypatch):
        """A sidecar cache per shard re-read every sidecar the shard named."""
        from effects.application import collect_coverage
        from effects.infrastructure import converted_trees, sidecar_io

        reads: list[Path] = []
        real_read = sidecar_io.read_sidecar

        def counting_read(path):
            reads.append(Path(path))
            return real_read(path)

        monkeypatch.setattr(sidecar_io, "read_sidecar", counting_read)
        converted_trees._process_sidecars.cache_clear()
        roots = {
            "cardsfolder": str(_FIXTURES / "gen1-sidecars" / "cardsfolder"),
            "tokenscripts": str(_FIXTURES / "gen1-sidecars" / "tokenscripts"),
        }
        shard = _FIXTURES / "gen1-records.jsonl.gz"
        first = collect_coverage.count_text_shard(shard, roots=roots, held_out=_holding(()))
        read_by_one = len(reads)
        second = collect_coverage.count_text_shard(shard, roots=roots, held_out=_holding(()))
        converted_trees._process_sidecars.cache_clear()

        assert read_by_one > 0
        assert len(reads) == read_by_one
        assert first == second == {}

    def test_satisfaction_is_distinct_games_at_the_floor(self):
        from effects.domain.coverage import TextCoverage

        unit = TextCoverage(text="t", games={"g1", "g2", "g3", "g4"})
        assert not unit.satisfied(5)
        unit.games.add("g5")
        assert unit.satisfied(5) and unit.progress == 5

    def test_the_round_ends_without_waiting_on_uncastable_texts(self):
        """FR-037 / spec Story 2 scenario 6."""
        from effects.application.collect_coverage import text_residues
        from effects.domain.coverage import TextCoverage, awaited_texts, is_complete

        coverage = {
            "done": TextCoverage(text="done", games={f"g{i}" for i in range(5)}),
            "retired": TextCoverage(text="retired", games={"g1"}, retired=True),
            "stuck": TextCoverage(text="stuck", castable=False),
        }
        assert set(awaited_texts(coverage)) == {"done", "retired"}
        assert is_complete(awaited_texts(coverage), 5)
        residues = text_residues(coverage, 5)
        assert residues.under_floor == {"retired": 1}
        assert residues.uncastable == ("stuck",)
        assert "stuck" in residues.render(5)

    def test_decks_hold_only_listed_cards_plus_basics(self):
        """Spec Story 2 scenario 5."""
        import random

        from effects.application.collector_common import build_coverage_decks
        from effects.domain.coverage import (
            DeckCandidate,
            TextCoverage,
            text_deck_weights,
        )

        carriers = {"t1": frozenset({"alpha"}), "t2": frozenset({"beta"})}
        coverage = {"t1": TextCoverage(text="t1"), "t2": TextCoverage(
            text="t2", games={f"g{i}" for i in range(5)},
        )}
        weights = text_deck_weights(coverage, carriers, 5)
        assert weights == {"alpha": 5.0}
        candidates = {
            "alpha": DeckCandidate("alpha", "{R}"), "beta": DeckCandidate("beta", None),
        }
        decks = build_coverage_decks(weights, candidates, 3, rng=random.Random(0))
        basics = {"Plains", "Island", "Swamp", "Mountain", "Forest"}
        assert all(set(deck) <= {"alpha"} | basics for deck in decks)


_GEN1_SHARD = _FIXTURES / "gen1-records.jsonl.gz"
_GEN1_ROOTS = {
    "cardsfolder": str(_FIXTURES / "gen1-sidecars" / "cardsfolder"),
    "tokenscripts": str(_FIXTURES / "gen1-sidecars" / "tokenscripts"),
}


def _holding(texts, unit=None):
    """A holdout of exactly ``texts``, by ``unit`` (``text`` by default)."""
    from effects.domain.text_holdout import HoldoutUnit

    return held_out_cards(texts=texts, unit=unit or HoldoutUnit.TEXT)


def _texts_by_game(shard: Path) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """``text -> games`` over the shard's resolution records, and over every
    record: each record counted through its first key with a script text."""
    from effects.domain.text_holdout import normalize_script_text
    from effects.infrastructure.shard_io import read_shard
    from effects.infrastructure.sidecar_io import SidecarCache

    cache = SidecarCache({tree: Path(root) for tree, root in _GEN1_ROOTS.items()})
    resolutions: dict[str, set[str]] = {}
    everything: dict[str, set[str]] = {}
    for record in read_shard(shard):
        for key in record.ability or ():
            line = cache.line_for(key)
            if line is None or not line.script_text:
                continue
            text = normalize_script_text(line.script_text)
            everything.setdefault(text, set()).add(record.game_id)
            if record.kind is RecordKind.RESOLUTION:
                resolutions.setdefault(text, set()).add(record.game_id)
            break
    return resolutions, everything


class TestOnlyCardsRefusals:
    """FR-038 / spec Story 2 scenario 7: refused before any worker starts."""

    @pytest.mark.parametrize("extra", [
        ("--exclude-cards", "holdout.txt"),
        ("--split-from", "models/x/latest.pt"),
    ])
    def test_only_cards_beside_another_holdout_source_is_refused(self, extra):
        from effects.infrastructure.cli import build_parser, collector_config

        args = build_parser().parse_args(
            ["collect-coverage", "--only-cards", "holdout.txt", *extra],
        )
        with pytest.raises(ValueError, match="--only-cards"):
            collector_config(CollectCoverageConfig, args)

    def test_only_cards_beside_a_training_corpus_is_refused(self, tmp_path):
        from effects.infrastructure.cli import build_parser, run_collect_coverage

        corpus = tmp_path / "depleted"
        corpus.mkdir()
        (corpus / "holdout-cards.txt").write_text("Soul Echo\n", encoding="utf-8")
        args = build_parser().parse_args([
            "collect-coverage", "--only-cards", "holdout.txt",
            "--training-corpus", str(corpus),
        ])
        with patch("effects.application.collect_coverage.run") as collect:
            assert run_collect_coverage(args) == 1
        collect.assert_not_called()

    def test_alone_it_selects_the_text_unit(self):
        from effects.infrastructure.cli import build_parser, collector_config

        config = collector_config(CollectCoverageConfig, build_parser().parse_args(
            ["collect-coverage", "--only-cards", "holdout.txt", "--min-text-games", "7"],
        ))
        assert config.deck_holdout.only_cards == Path("holdout.txt")
        assert config.min_text_games == 7


class TestNonDeckTypesLeaveTheUnit:
    """Planes, schemes, vanguards and the like never go in a deck.

    Forge's card database does not load them as deck cards, so a coverage deck
    holding one fails to materialize. With no records they weigh the most and
    land in most decks: on the gen-2 corpus 71% of a round's decks held one.
    """

    @pytest.mark.parametrize("types,deckable", [
        ("creature", True),
        ("legendary planeswalker", True),
        ("artifact creature", True),
        ("scheme", False),
        ("ongoing scheme", False),
        ("vanguard", False),
        ("plane dominaria", False),
        ("phenomenon", False),
        ("conspiracy", False),
        ("artifact contraption", False),
        ("artifact attraction", False),
    ])
    def test_the_types_line_decides(self, types, deckable):
        from effects.domain.coverage import is_deckable

        assert is_deckable(f"name: x\ntypes: {types}\n") is deckable

    def test_a_card_without_a_types_line_is_kept(self):
        from effects.domain.coverage import is_deckable

        assert is_deckable("name: x\n")

    def test_the_training_unit_holds_no_non_deck_card(self, tmp_path, fake_supervisor):
        from effects.application import collect_coverage

        cards = tmp_path / "cardsfolder"
        (cards / "a").mkdir(parents=True)
        for stem, types in (("aa", "creature"), ("my_plan", "scheme"),
                            ("squee", "vanguard")):
            (cards / "a" / f"{stem}.txt").write_text(
                f"name: {stem}\nmana cost: {{G}}\ntypes: {types}\n",
                encoding="utf-8",
            )
        (tmp_path / "records").mkdir()
        consulted: list[list[str]] = []

        def _consult(names):
            consulted.append(list(names))
            return {}

        with patch.object(collect_coverage, "consult_castability", _consult):
            collect_coverage.run(collect_coverage.CollectCoverageConfig(
                effect_records=tmp_path / "records", cards_folders=(cards,),
                target_records=1, decks_per_round=5,
            ))

        assert consulted == [["aa"]]
        decked = {name for deck in fake_supervisor.play_round.call_args.args[0] for name in deck}
        assert "my_plan" not in decked and "squee" not in decked

    def test_the_held_out_round_drops_non_deck_carriers_only(
        self, tmp_path, caplog, fake_supervisor,
    ):
        """A text a non-deck held-out card shares stays held out.

        Which texts are held out is decided over every listed card: dropping
        the scheme from the listed set first would make it an unlisted
        carrier of the shared text and take that text out of the round.
        """
        import logging

        from effects.application import collect_coverage

        cards = tmp_path / "cardsfolder"
        (cards / "a").mkdir(parents=True)
        for stem, types in (("alpha", "creature"), ("plan", "scheme"),
                            ("gamma", "creature")):
            (cards / "a" / f"{stem}.txt").write_text(
                f"name: {stem}\nmana cost: {{G}}\ntypes: {types}\n",
                encoding="utf-8",
            )
        listed = tmp_path / "holdout-cards.txt"
        listed.write_text("alpha\nplan\n", encoding="utf-8")
        (tmp_path / "records").mkdir()
        texts_by_card = {
            "alpha": ["SP$ Shared", "SP$ Alpha"],
            "plan": ["SP$ Shared", "SP$ Plan"],
            "gamma": ["SP$ Common"],
        }
        consulted: list[list[str]] = []

        def _consult(names):
            consulted.append(list(names))
            return {}

        with patch.object(collect_coverage, "consult_castability", _consult), \
             patch("effects.infrastructure.converted_trees.load_card_texts",
                   return_value=texts_by_card), \
             caplog.at_level(logging.INFO, logger=collect_coverage.__name__):
            collect_coverage.run(collect_coverage.CollectCoverageConfig(
                effect_records=tmp_path / "records", cards_folders=(cards,),
                deck_holdout=DeckHoldout(only_cards=listed), min_text_games=0,
            ))

        assert consulted == [["alpha"]]
        assert "Held-out round: 2 text(s) on 1 card(s)" in caplog.text


class TestHeldOutRoundCaps:
    """FR-035: full strength for the card's own records, not for the AI's planning."""

    def test_every_playability_record_is_kept(self):
        from effects.application.collect_coverage import held_out_round_caps
        from effects.domain.collection_caps import CollectionCaps

        assert held_out_round_caps(CollectionCaps()).playability_rate == 1.0

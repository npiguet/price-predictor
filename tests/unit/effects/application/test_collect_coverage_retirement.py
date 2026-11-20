"""The coverage collector (T102): retirement, the residues and the decks.

A card with no progress retires, which is the only thing that makes the run
terminate; the unit's own rules are in ``tests/unit/effects/domain/test_coverage.py``.
"""

from __future__ import annotations

from unittest.mock import patch

from effects.application.collect_coverage import (
    DEFAULT_NO_PROGRESS_ROUNDS,
    CollectCoverageConfig,
    residues,
)
from effects.domain.coverage import (
    CardCoverage,
    ConsultVerdict,
    is_complete,
    retire_stalled,
)
from tests.unit.effects.application.collect_coverage_support import _turn_action
from tests.unit.effects.builders import (
    make_entity,
    write_plain_shard,
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
        from effects.application import collect_coverage, collection_rounds

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
        real = collection_rounds.retire_stalled

        def _spy(coverage, previous, *, no_progress_rounds, decked, written):
            seen.append(frozenset(written))
            return real(
                coverage, previous, no_progress_rounds=no_progress_rounds,
                decked=decked, written=written,
            )

        with patch.object(collect_coverage, "consult_castability", return_value={}), \
             patch.object(collection_rounds, "retire_stalled", _spy):
            collect_coverage.run(CollectCoverageConfig(
                effect_records=tmp_path / "records",
                cards_folders=(cards,),
                target_records=1,
                decks_per_round=1,
                no_progress_rounds=1,
                workers=1,
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


class TestARoundChargesTheCardsItsGamesShowed:
    """python D8: retirement is charged to the cards the round's games showed,
    not to every card of every deck written."""

    def test_a_written_card_no_game_showed_is_not_charged(self, tmp_path, fake_supervisor):
        from effects.application import collect_coverage, collection_rounds, coverage_rounds

        cards = tmp_path / "cardsfolder"
        cards.mkdir()
        for name in ("shown", "unseen"):
            (cards / f"{name}.txt").write_text(
                f"name: {name}\nmana cost: {{G}}\ntypes: creature\n", encoding="utf-8",
            )
        records_dir = tmp_path / "records"
        records_dir.mkdir()
        played = iter(range(100))

        def _play(_decks, _decks_file, spec):
            # The round's games show "shown" in hand, crediting nothing.
            record = _record([make_entity("E9", "other")], source="E9")
            seen_record = _turn_action([make_entity("E1", "shown", zone="hand")], [])
            write_plain_shard(records_dir, f"r{next(played)}", [record, seen_record])

        fake_supervisor.play_round.side_effect = _play
        calls = []
        real = collection_rounds.retire_stalled

        def _spy(coverage, previous, **kwargs):
            calls.append(kwargs)
            return real(coverage, previous, **kwargs)

        with patch.object(collect_coverage, "consult_castability", return_value={}), \
                patch.object(collection_rounds, "retire_stalled", _spy), \
                patch.object(coverage_rounds, "build_coverage_decks",
                             return_value=[["shown", "unseen"]]):
            collect_coverage.run(CollectCoverageConfig(
                effect_records=records_dir, cards_folders=(cards,),
                target_records=1, decks_per_round=1, no_progress_rounds=1, workers=1,
            ))

        first = calls[0]
        assert first["written"] == {"shown", "unseen"}
        assert first["decked"] == {"shown"}

"""The coverage collector (T102): which cards the unit and the decks hold:
non-deck types, the wording round, a hung consult and a card Forge cannot
resolve."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from effects.application.collect_coverage import CollectCoverageConfig
from effects.application.collector_common import DeckHoldout
from effects.domain.coverage import ConsultVerdict
from tests.unit.effects.application.collect_coverage_support import (
    _holding,
    _two_cards,
)


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
                target_records=1, decks_per_round=5, workers=1,
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

        from effects.application import collect_coverage, held_out_round

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

        held_out = _holding({"SP$ Shared", "SP$ Alpha", "SP$ Plan"})
        with patch.object(held_out_round, "consult_castability", _consult), \
             patch("effects.infrastructure.converted_trees.load_card_texts",
                   return_value=texts_by_card), \
             patch("effects.infrastructure.converted_trees.ConvertedTrees.holdout",
                   return_value=held_out), \
             caplog.at_level(logging.INFO, logger=held_out_round.__name__):
            collect_coverage.run(collect_coverage.CollectCoverageConfig(
                effect_records=tmp_path / "records", cards_folders=(cards,),
                deck_holdout=DeckHoldout(only_cards=listed), min_text_games=0,
                workers=1,
            ))

        assert consulted == [["alpha"]]
        assert "Held-out round: 2 text(s) on 1 card(s)" in caplog.text


class TestTheWordingRound:
    """FR-038: ``--wording-cards`` runs the card round over the listed cards
    alone, the coverage unit and the decks both, with the holdout still kept
    out."""

    @staticmethod
    def _cards(tmp_path: Path) -> Path:
        cards = tmp_path / "cardsfolder"
        (cards / "a").mkdir(parents=True)
        # Converted text spells names in lower case, as convert writes them.
        for stem, name in (("prodigal", "prodigal pyromancer"), ("bears", "grizzly bears"),
                           ("pinger", "cunning sparkmage")):
            (cards / "a" / f"{stem}.txt").write_text(
                f"name: {name}\nmana cost: {{R}}\ntypes: creature\n", encoding="utf-8",
            )
        (tmp_path / "records").mkdir()
        return cards

    def _run(self, tmp_path, listed: str, **config):
        from effects.application import collect_coverage

        cards = self._cards(tmp_path)
        wording = tmp_path / "wording.txt"
        wording.write_text(listed, encoding="utf-8")
        consulted: list[list[str]] = []

        def _consult(names):
            consulted.append(list(names))
            return {}

        with patch.object(collect_coverage, "consult_castability", _consult):
            code = collect_coverage.run(collect_coverage.CollectCoverageConfig(
                effect_records=tmp_path / "records", cards_folders=(cards,),
                target_records=1, decks_per_round=5, no_progress_rounds=1, workers=1,
                wording_cards=wording, **config,
            ))
        return code, consulted

    def test_the_unit_and_the_decks_hold_only_listed_cards(self, tmp_path, fake_supervisor):
        code, consulted = self._run(tmp_path, "prodigal pyromancer\nCunning Sparkmage\n")

        assert code == 1, "the stand-in rounds write no record, which fails the run"
        assert consulted == [["cunning sparkmage", "prodigal pyromancer"]]
        decked = {name for deck in fake_supervisor.play_round.call_args.args[0]
                  for name in deck}
        assert "grizzly bears" not in decked
        assert decked & {"cunning sparkmage", "prodigal pyromancer"}

    def test_a_held_out_listed_card_stays_out(self, tmp_path, fake_supervisor):
        holdout = tmp_path / "holdout-cards.txt"
        holdout.write_text("Cunning Sparkmage\n", encoding="utf-8")

        _, consulted = self._run(
            tmp_path, "Prodigal Pyromancer\nCunning Sparkmage\n",
            deck_holdout=DeckHoldout(exclude_cards=holdout),
        )

        assert consulted == [["prodigal pyromancer"]]

    def test_a_listed_name_with_no_card_is_named(self, tmp_path, fake_supervisor, caplog):
        self._run(tmp_path, "Prodigal Pyromancer\nTim the Enchanter\n")

        assert "tim the enchanter" in caplog.text

    def test_a_list_matching_no_card_starts_no_round(self, tmp_path, fake_supervisor):
        code, consulted = self._run(tmp_path, "Tim the Enchanter\n")

        assert code == 1
        assert consulted == []
        fake_supervisor.play_round.assert_not_called()

    def test_it_is_refused_beside_the_held_out_round(self, tmp_path):
        with pytest.raises(ValueError, match="--wording-cards or --only-cards"):
            CollectCoverageConfig(
                effect_records=tmp_path / "records",
                wording_cards=tmp_path / "w.txt",
                deck_holdout=DeckHoldout(only_cards=tmp_path / "h.txt"),
            )


class TestAHungConsultOnlyRanks:
    """P1 #12: the castability consult runs under a timeout; a consult killed
    at it degrades like one that could not run, every card ranked castable."""

    def test_a_timed_out_consult_yields_no_verdicts(self, caplog):
        from effects.application import collect_coverage
        from effects.infrastructure import one_shot_workers
        from price_predictor.infrastructure.jvm_processes import ForgeWorkerTimeout

        with patch.object(one_shot_workers, "castability_verdicts",
                          side_effect=ForgeWorkerTimeout("CastabilityMain ran past its timeout")):
            assert collect_coverage.consult_castability(["aa"]) == {}
        assert "ran past its timeout" in caplog.text


class TestACardForgeCannotResolve:
    """P1 #11: a decks-only round refuses a whole deck naming a card Forge's
    card database does not hold, so one such card, unrecorded and weighted
    the most, failed every match its decks reached until it retired unseen.
    The consult answers every name the database holds: one it leaves
    unanswered leaves the unit and the decks."""

    def _run(self, tmp_path, fake_supervisor, verdicts):
        from effects.application import collect_coverage

        cards = _two_cards(tmp_path, "aa", "zz")
        written = []
        fake_supervisor.play_round.side_effect = (
            lambda decks, *_a, **_k: written.extend(name for deck in decks for name in deck)
        )
        with patch.object(collect_coverage, "consult_castability", return_value=verdicts):
            collect_coverage.run(CollectCoverageConfig(
                effect_records=tmp_path / "records", cards_folders=(cards,),
                target_records=1, decks_per_round=4, no_progress_rounds=1, workers=1,
            ))
        return written

    def test_an_unanswered_card_is_never_decked(self, tmp_path, fake_supervisor, caplog):
        written = self._run(tmp_path, fake_supervisor, {"aa": ConsultVerdict.CASTABLE})

        assert "aa" in written
        assert "zz" not in written
        assert "not in Forge's card database" in caplog.text

    def test_the_dropped_count_and_share_are_loud_and_recorded(
        self, tmp_path, fake_supervisor, caplog,
    ):
        """No floor stops the run, but the report cannot be missed: the count,
        the share of the pool, what a large share means, and run.json."""
        self._run(tmp_path, fake_supervisor, {"aa": ConsultVerdict.CASTABLE})

        assert "1 of 2 card(s) (50.0%)" in caplog.text
        assert "../forge is older than the converted cards" in caplog.text
        fake_supervisor.record.assert_any_call(
            unresolved_cards=1, unresolved_share=0.5, unresolved_examples=["zz"],
        )

    def test_a_consult_that_answered_nothing_drops_nothing(self, tmp_path, fake_supervisor):
        """A consult that could not run proves nothing about any card."""
        written = self._run(tmp_path, fake_supervisor, {})

        assert {"aa", "zz"} <= set(written)

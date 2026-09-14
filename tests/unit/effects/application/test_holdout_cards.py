"""``holdout-cards``: the depletion list `generate-pools` reads (T162, T163).

The list and the trainer's split must agree exactly. They are two programs
reading two different things — one the converted tree, the other its own
config — and a disagreement is silent: pools would be depleted of one set of
cards while the split held out another, so held-out cards would reach training
games and the card-disjoint stratum would hold games that are not disjoint.
"""

from __future__ import annotations

import pytest

from effects.application.holdout_cards import depletion_list


def test_the_list_is_the_cards_the_trainer_holds_out() -> None:
    """One rule, exercised through both entry points (T163)."""
    from effects.application.train_effect_model import text_keyed_holdout

    card_files = {
        "Searing Spear": "cardsfolder/s/searing_spear.txt",
        "Lightning Strike": "cardsfolder/l/lightning_strike.txt",
        "Grizzly Bears": "cardsfolder/g/grizzly_bears.txt",
    }
    texts_by_card = {
        "Searing Spear": ["SP$ DealDamage | NumDmg$ 3 | ValidTgts$ Any"],
        "Lightning Strike": ["SP$ DealDamage | NumDmg$ 3 | ValidTgts$ Any"],
        "Grizzly Bears": [],
    }

    for permille in (0, 250, 500, 750, 1000):
        listed = depletion_list(
            card_files, texts_by_card, permille=permille, max_carriers=8,
        )
        held = text_keyed_holdout(
            card_files, texts_by_card, permille=permille, max_carriers=8,
        )
        assert set(listed) == held.names, f"disagreed at permille={permille}"


def test_the_list_is_sorted_so_two_runs_produce_the_same_file() -> None:
    """A file that reordered between runs would look like a changed holdout."""
    card_files = {
        "Zed": "cardsfolder/z/zed.txt",
        "Alice": "cardsfolder/a/alice.txt",
        "Mallory": "cardsfolder/m/mallory.txt",
    }
    texts_by_card = {
        name: [f"SP$ Bespoke | Who$ {name}"] for name in card_files
    }

    listed = depletion_list(
        card_files, texts_by_card, permille=1000, max_carriers=8,
    )

    assert listed == sorted(listed)
    assert listed == ["alice", "mallory", "zed"], (
        "with no printings file to resolve against, the list keeps the "
        "converted tree's spelling"
    )


def test_the_cli_defaults_match_the_trainer_constants() -> None:
    """Two copies of the same number, kept honest.

    The CLI restates them so `--help` stays free of torch, and a drift would
    deplete pools against one holdout while the split used another.
    """
    from effects.application.train_effect_model import (
        HOLDOUT_MAX_CARRIERS,
        HOLDOUT_PERMILLE,
    )
    from effects.infrastructure import cli

    assert cli.HOLDOUT_PERMILLE == HOLDOUT_PERMILLE
    assert cli.HOLDOUT_MAX_CARRIERS == HOLDOUT_MAX_CARRIERS


def test_a_checkpoint_records_the_holdout_flags_it_trained_under() -> None:
    """A depleted corpus was composed against these values (T169, FR-134).

    Without them, a later run inheriting the split has no way to tell whether
    the corpus it is reading was depleted against the same holdout, and the
    disagreement is silent.
    """
    from effects.infrastructure.effect_model_store import SplitProvenance

    provenance = SplitProvenance(
        held_out_cards=("Alpha",), holdout_permille=20, holdout_max_carriers=8,
    )

    restored = SplitProvenance.from_dict(provenance.as_dict())

    assert restored.holdout_permille == 20
    assert restored.holdout_max_carriers == 8


class TestTheTrainerTakesTheHoldoutFlags:
    """The quickstart tells an operator to pass these at training (FR-088).

    They have to be flags, not just config fields: a depleted corpus was
    composed against particular values, and a run that cannot be told about
    them would silently train against a different holdout than the one its
    pools were depleted for.
    """

    def _config(self, *argv):
        from effects.infrastructure.cli import build_parser, train_config_from

        return train_config_from(build_parser().parse_args(
            ["train-effect-model", *argv]
        ))

    def test_the_defaults_are_the_trainer_constants(self):
        from effects.application.train_effect_model import (
            HOLDOUT_MAX_CARRIERS,
            HOLDOUT_PERMILLE,
            MIN_HOLDOUT_RECORDS,
        )

        config = self._config()

        assert config.holdout_permille == HOLDOUT_PERMILLE
        assert config.holdout_max_carriers == HOLDOUT_MAX_CARRIERS
        assert config.min_holdout_records == MIN_HOLDOUT_RECORDS

    def test_each_one_is_overridable(self):
        config = self._config(
            "--holdout-permille", "50",
            "--holdout-max-carriers", "3",
            "--min-holdout-records", "500",
        )

        assert config.holdout_permille == 50
        assert config.holdout_max_carriers == 3
        assert config.min_holdout_records == 500


def test_the_min_holdout_default_matches_the_trainer_constant() -> None:
    from effects.application.train_effect_model import MIN_HOLDOUT_RECORDS
    from effects.infrastructure import cli

    assert cli.MIN_HOLDOUT_RECORDS == MIN_HOLDOUT_RECORDS


class TestTheFileSpeaksForgesSpelling:
    """`holdout-cards.txt` crosses into Forge, so it should carry Forge's names.

    Folding at every comparison makes the case irrelevant to correctness. The
    file is still written in printed case, because it is an interface file: an
    operator reads it, Forge's own `getName()` is what it will be matched
    against, and a future consumer that forgets to fold should not be the one
    that discovers the convention.
    """

    def test_names_are_resolved_to_their_printed_spelling(self) -> None:
        from effects.application.holdout_cards import depletion_list

        listed = depletion_list(
            {"soul echo": "cardsfolder/s/soul_echo.txt"},
            {"soul echo": ["SP$ Bespoke | Weird$ True"]},
            permille=1000, max_carriers=8,
            canonical_names={"Soul Echo": "2004-10-01"},
        )

        assert listed == ["Soul Echo"]

    def test_an_unresolvable_name_keeps_the_converted_spelling(self) -> None:
        """A token or a card MTGJSON has never heard of still gets depleted."""
        from effects.application.holdout_cards import depletion_list

        listed = depletion_list(
            {"nonesuch": "cardsfolder/n/nonesuch.txt"},
            {"nonesuch": ["SP$ Bespoke | Weird$ True"]},
            permille=1000, max_carriers=8,
            canonical_names={"Soul Echo": "2004-10-01"},
        )

        assert listed == ["nonesuch"]


class TestTheTrainingCorpusFlag:
    """One flag for a corpus and the holdout list that depleted it.

    They belong together: the list is what the corpus was built against, and
    naming them separately invites the mismatch that silently produces an
    undepleted corpus. `--training-corpus DIR` says both live in DIR.
    """

    def _config(self, *argv):
        from effects.infrastructure.cli import build_parser, coverage_config_from

        return coverage_config_from(
            build_parser().parse_args(["collect-coverage", *argv])
        )

    def test_it_sets_both_the_records_dir_and_the_holdout_list(self, tmp_path):
        corpus = tmp_path / "depleted"
        corpus.mkdir()
        (corpus / "holdout-cards.txt").write_text("Soul Echo\n", encoding="utf-8")

        config = self._config("--training-corpus", str(corpus))

        assert config.effect_records == corpus
        assert config.exclude_cards == corpus / "holdout-cards.txt"

    def test_a_corpus_without_a_holdout_list_is_refused(self, tmp_path):
        """Silently running undepleted is the failure this whole area had."""
        from effects.infrastructure.cli import build_parser, coverage_config_from

        corpus = tmp_path / "depleted"
        corpus.mkdir()

        with pytest.raises(ValueError, match="holdout-cards.txt"):
            coverage_config_from(build_parser().parse_args(
                ["collect-coverage", "--training-corpus", str(corpus)]
            ))

    def test_it_refuses_to_share_the_stage_with_the_flags_it_implies(self, tmp_path):
        from effects.infrastructure.cli import build_parser, coverage_config_from

        corpus = tmp_path / "depleted"
        corpus.mkdir()
        (corpus / "holdout-cards.txt").write_text("Soul Echo\n", encoding="utf-8")

        with pytest.raises(ValueError, match="one source"):
            coverage_config_from(build_parser().parse_args([
                "collect-coverage", "--training-corpus", str(corpus),
                "--effect-records", str(tmp_path / "elsewhere"),
            ]))

    def test_without_it_the_ordinary_flags_still_work(self, tmp_path):
        config = self._config("--effect-records", str(tmp_path / "records"))

        assert config.effect_records == tmp_path / "records"
        assert config.exclude_cards is None

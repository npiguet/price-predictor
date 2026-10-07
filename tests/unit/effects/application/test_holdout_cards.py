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


# ── the template unit (FR-040, FR-043; T040, T042) ──────────────────────


def test_the_report_counts_templates_texts_cards_and_the_depleted_share():
    from effects.application.holdout_cards import holdout_report

    card_files = {f"card {i}": f"cardsfolder/c/card_{i}.txt" for i in range(6)}
    texts_by_card = {
        "card 0": ["SP$ Bespoke | Weird$ 1"],
        "card 1": ["SP$ Bespoke | Weird$ 2"],          # same template as card 0
        "card 2": ["SP$ Other | Thing$ X"],
        "card 3": ["SP$ Other | Thing$ X"],            # same text as card 2
        "card 4": [], "card 5": [],
    }
    report = holdout_report(
        card_files, texts_by_card, permille=1000, max_carriers=8, unit="template",
    )

    assert report.keys == 2
    assert report.texts == 3
    assert len(report.names) == 4
    assert report.depleted_share == 4 / 6


def test_number_only_variants_land_on_the_same_side_of_the_list():
    from effects.domain.text_holdout import masked_template

    card_files = {f"card {i}": f"cardsfolder/c/card_{i}.txt" for i in range(40)}
    texts_by_card = {
        f"card {i}": [f"SP$ DealDamage | NumDmg$ {i % 4} | ValidTgts$ Kind{i // 4}"]
        for i in range(40)
    }
    listed = set(depletion_list(
        card_files, texts_by_card, permille=500, max_carriers=8, unit="template",
    ))
    by_template: dict[str, set[bool]] = {}
    for card, texts in texts_by_card.items():
        by_template.setdefault(masked_template(texts[0]), set()).add(card in listed)
    assert all(len(sides) == 1 for sides in by_template.values())


def test_the_cli_defaults_to_the_template_unit():
    from effects.infrastructure.cli import build_parser

    args = build_parser().parse_args(["holdout-cards", "--out", "x.txt"])
    assert args.holdout_unit == "template"
    args = build_parser().parse_args(["build-corpus"])
    assert args.holdout_unit == "template"
    args = build_parser().parse_args(["train-effect-model", "--corpus", "c"])
    assert args.holdout_unit is None


def test_a_checkpoint_records_the_holdout_unit_and_reads_none_as_text():
    from effects.infrastructure.effect_model_store import SplitProvenance

    template = SplitProvenance(held_out_cards=("Alpha",), holdout_unit="template")
    assert SplitProvenance.from_dict(template.as_dict()).holdout_unit == "template"

    text = SplitProvenance(held_out_cards=("Alpha",))
    assert "holdout_unit" not in text.as_dict()
    assert SplitProvenance.from_dict(text.as_dict()).holdout_unit == "text"


def test_a_real_gen1_checkpoint_split_reads_as_the_text_unit():
    """The 09.17f checkpoint's recorded split, as written, carries no unit."""
    from effects.infrastructure.effect_model_store import SplitProvenance

    recorded = {
        "held_out_cards": ["a-moss-pit skeleton"], "card_disjoint_games": [],
        "game_disjoint_games": [], "vocab_path": "models/effects/vocab-script.txt",
        "keyword_definitions_path": "output/effects/keyword-definitions.json",
        "vocab_hash": "47c816e1", "keyword_definitions_hash": "93b710ba",
        "withheld_keyword": None, "holdout_permille": 20, "holdout_max_carriers": 8,
        "corpus_path": "output/effects/corpus/",
        "corpus_digest": "e3c0c9bb913b9aeda0d07eb587b5445c",
    }
    provenance = SplitProvenance.from_dict(recorded)
    assert provenance.holdout_unit == "text"
    assert provenance.as_dict() == {
        **recorded,
        **{k: tuple(v) for k, v in recorded.items() if isinstance(v, list)},
    }

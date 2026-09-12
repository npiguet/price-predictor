"""``holdout-cards``: the depletion list `generate-pools` reads (T162, T163).

The list and the trainer's split must agree exactly. They are two programs
reading two different things — one the converted tree, the other its own
config — and a disagreement is silent: pools would be depleted of one set of
cards while the split held out another, so held-out cards would reach training
games and the card-disjoint stratum would hold games that are not disjoint.
"""

from __future__ import annotations

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
    assert listed == ["Alice", "Mallory", "Zed"]


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

"""The build's flags: counts, caps, help text and keyword lists."""

from __future__ import annotations

import pytest

from effects.domain.corpus_plan import BuildCorpusError
from tests.unit.effects.application.build_corpus_support import (
    _BUDGET_FREE_RULES,
)
from tests.unit.effects.file_builders import write_vocabulary


def test_the_trainer_refuses_a_real_dataset_built_on_the_other_surface(
    default_build, shared_corpus, tmp_path,
):
    """The manifest's ``surface`` reaches the trainer (FR-141).

    A real ``build-corpus`` output, read back by ``train_effect_model.run``
    under a vocabulary on the other surface. The guard fires before the
    training loop is imported, so this needs no torch and no GPU — and it is
    the only thing in the suite that round-trips a written dataset into the
    trainer at all.
    """
    from effects.application.train_effect_model import (
        SurfaceMismatchError,
        TrainEffectModelConfig,
    )
    from effects.application.train_effect_model import run as train

    assert default_build.load().surface == "prose"

    with pytest.raises(SurfaceMismatchError, match="prose"):
        train(TrainEffectModelConfig.unpathed(
            corpus=str(default_build.directory), cards_folders=shared_corpus.cards,
            vocab_path=write_vocabulary(tmp_path / "vocab.txt", "script"),
        ))


# ── what a count may be, and what zero means for each ─────────────────


@pytest.mark.parametrize("field,flag", [
    ("text_cap", "--text-cap"),
    ("card_disjoint_text_cap", "--card-disjoint-text-cap"),
    ("reuse_cap", "--reuse-cap"),
    ("training_records", "--training-records"),
])
def test_a_negative_count_is_refused(tmp_path, shared_corpus, field, flag):
    with pytest.raises(ValueError, match=flag):
        shared_corpus.config(tmp_path / "curated", **{field: -1})


def test_a_text_cap_of_zero_means_no_cap_rather_than_keep_nothing(
    tmp_path, shared_corpus,
):
    """Zero admits everything, the way ``CapHeap`` reads it. Documented here
    because the other reading — keep nothing — is the one the word "cap"
    suggests, and it would silently empty the training corpus."""
    manifest = shared_corpus.build(
        tmp_path / "curated", text_cap=0,
        # One class, so the class mixture asks for every record the class
        # holds: whatever is dropped then is the text cap's doing.
        class_mix={"resolution-effect": 1.0},
    ).load()

    assert sum(c.dropped_by_cap for c in manifest.per_class.values()) == 0
    assert manifest.per_stratum["training"] > 0


def test_a_card_disjoint_text_cap_of_zero_means_no_cap_either(tmp_path, shared_corpus):
    store = shared_corpus.build(tmp_path / "curated", card_disjoint_text_cap=0)

    assert store.load().card_disjoint_games == ("g-tainted",)


# ── stage-four variants (FR-141: they are ability texts like any other) ──


def test_the_max_events_help_says_zero_disables_the_rule():
    """The zero reading must survive in --help, not just in the docstring:
    this is what would have caught it silently vanishing."""
    from effects.infrastructure.cli import build_parser

    subparsers_action = next(
        action for action in build_parser()._actions
        if getattr(action, "choices", None) and "build-corpus" in action.choices
    )
    max_events_action = next(
        action for action in subparsers_action.choices["build-corpus"]._actions
        if "--max-events-per-record" in action.option_strings
    )
    assert "0 disables" in max_events_action.help


# ── C1: combat is exempt from the unattributed rule ───────────────────


def test_the_game_disjoint_keywords_parse_as_a_comma_separated_list():
    from tests.unit.effects.cli_args import parse_cli

    args = parse_cli(["build-corpus", "--game-disjoint-keywords", "trample,infect"])
    assert args.game_disjoint_keywords == ("trample", "infect")


def test_an_unscored_game_disjoint_keyword_is_refused(tmp_path, shared_corpus):
    with pytest.raises(BuildCorpusError, match="flying"):
        shared_corpus.config(tmp_path, rules=_BUDGET_FREE_RULES, game_disjoint_keywords=("flying",))

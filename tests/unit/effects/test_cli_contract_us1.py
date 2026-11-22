"""Every flag and default of the US1 CLI surface (T039).

Parsed values only, no execution, with one deliberate exception
(``TestVariantAcceptsCorpusAtThePreflightCheck``): a defaulted flag can drift
silently, but a dispatcher's own pre-flight guard rejecting an argument the
library function underneath it now accepts is a defect a parse-only test
cannot see at all, because it never runs the code that guards.

The stage-four rows (``--variant-scripts`` on the trainer's own default,
``--surface script``'s vocabulary path) are US4's, and are covered by its own
contract test.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from effects.infrastructure.cli import build_parser
from tests.unit.effects.cli_args import parse_cli


def parse(*argv: str):
    return parse_cli(argv)


#: ``train-effect-model`` takes no defaults without a corpus any more.
TRAIN = ("train-effect-model", "--corpus", "output/effects/corpus")

#: What ``--cards-folder`` parses to when it is not passed.
BOTH_TREES = (
    Path("output/effects/gen3/converted/cardsfolder"),
    Path("output/effects/gen3/converted/tokenscripts"),
)


class TestBuildVocab:
    def test_the_defaults_are_the_contracts(self):
        args = parse("build-vocab")
        assert args.surface == "prose"
        assert args.cards_folders == BOTH_TREES
        assert args.target_size == 5000

    def test_cards_folder_is_repeatable(self):
        """Every command takes ``--cards-folder`` through one helper, so one
        command's parse stands for all of them."""
        args = parse(
            "build-vocab", "--cards-folder", "a/", "--cards-folder", "b/",
        )
        assert args.cards_folders == (Path("a/"), Path("b/"))

    def test_one_cards_folder_replaces_both_defaults(self):
        """argparse's own ``append`` would add to the default, not replace it."""
        args = parse("build-vocab", "--cards-folder", "a/")
        assert args.cards_folders == (Path("a/"),)

    def test_the_surface_choices_are_prose_and_script(self):
        assert parse("build-vocab", "--surface", "script").surface == "script"
        with pytest.raises(SystemExit):
            parse("build-vocab", "--surface", "oracle")


class TestExtractKeywordDefinitions:
    def test_the_output_is_the_flags_value(self):
        args = parse("extract-keyword-definitions", "--output", "inputs/run/keywords.json")
        assert args.output == Path("inputs/run/keywords.json")


class TestTrainEffectModel:
    def test_the_paths_are_the_flags_values(self):
        args = parse(
            *TRAIN, "--vocab-path", "v.txt", "--keyword-definitions", "k.json",
            "--model-output", "models/run",
        )
        assert args.corpus == "output/effects/corpus"
        assert args.vocab_path == Path("v.txt")
        assert args.keyword_definitions == Path("k.json")
        assert args.model_output == Path("models/run")
        assert args.variant_scripts is None

    def test_the_architecture_defaults_are_the_contracts(self):
        args = parse(*TRAIN)
        assert args.e_dim == 256
        assert args.e_noise == 0.1
        assert args.keyword_expand_p == 0.25
        assert args.context_dropout == 0.15

    def test_the_auxiliary_weights_are_the_contracts(self):
        args = parse(*TRAIN)
        assert args.mlm_weight == 0.1
        assert args.mlm_mask_prob == 0.15
        assert args.api_weight == 0.003

    def test_the_schedule_defaults_are_the_contracts(self):
        args = parse(*TRAIN)
        assert args.curriculum_epoch == 3
        assert args.batch_size == 32
        assert args.grad_accum == 1
        assert args.steps_per_epoch == 5000
        assert args.epochs == 40
        assert args.patience == 5

    def test_the_corpus_is_required(self):
        """FR-125 withdrawn: raw shards are no longer a training input."""
        with pytest.raises(SystemExit):
            parse("train-effect-model")

    def test_the_variant_choices_are_the_five(self):
        for variant in ("full", "identity", "state-only", "no-state", "taxonomy"):
            assert parse(*TRAIN, "--variant", variant).variant == variant
        with pytest.raises(SystemExit):
            parse(*TRAIN, "--variant", "oracle")

    def test_full_is_the_default_variant(self):
        assert parse(*TRAIN).variant == "full"

    def test_withhold_keyword_defaults_to_none(self):
        assert parse(*TRAIN).withhold_keyword is None
        assert parse(
            *TRAIN, "--withhold-keyword", "cascade",
        ).withhold_keyword == "cascade"


class TestVariantAcceptsCorpusAtThePreflightCheck:
    """A baseline variant with ``--corpus`` reaches the
    trainer: the curated dataset's manifest is the split it inherits.

    The real trainer entry point (``train_effect_model.run``) is monkeypatched
    out, so getting past the CLI is what this test proves, not a full training
    run, and it neither imports torch nor reads a real corpus off disk.
    """

    def test_a_variant_with_corpus_gets_past_the_preflight_check(self, monkeypatch):
        from effects.infrastructure.cli import run_train_effect_model

        monkeypatch.setattr(
            "effects.application.train_effect_model.run", lambda config: 0,
        )
        args = parse(
            "train-effect-model", "--variant", "identity",
            "--corpus", "output/effects/corpus",
        )

        assert run_train_effect_model(args) == 0


class TestBuildCorpusDispatch:
    """What ``run_build_corpus`` turns into an exit code, and what it does not.

    ``except ValueError`` was catching three unrelated things at once: the
    operator conditions it was written for, and two internal-invariant
    violations (``decide``'s cap mismatch, ``CapHeap.merge``'s) where a
    traceback is the useful output. A distinct ``BuildCorpusError`` separates
    them, so a bug in the builder can no longer arrive as a one-line log entry
    that reads like a misconfigured run.
    """

    def test_a_verify_against_a_directory_with_no_manifest_reports_cleanly(
        self, tmp_path,
    ):
        from effects.infrastructure.cli import run_build_corpus

        args = parse(
            "build-corpus", "--output", str(tmp_path / "nothing-here"), "--verify",
        )

        assert run_build_corpus(args) == 1

    def test_an_operator_condition_becomes_an_exit_code(self, tmp_path, monkeypatch):
        from effects.domain.corpus_plan import BuildCorpusError
        from effects.infrastructure.cli import run_build_corpus

        def explode(config):
            raise BuildCorpusError("nothing to build from")

        monkeypatch.setattr("effects.application.build_corpus.build", explode)

        assert run_build_corpus(parse("build-corpus")) == 1

    def test_an_internal_invariant_violation_keeps_its_traceback(self, monkeypatch):
        """A plain ``ValueError`` out of the builder is a bug, not a bad flag."""
        from effects.infrastructure.cli import run_build_corpus

        def explode(config):
            raise ValueError("cannot merge heaps with mismatched caps")

        monkeypatch.setattr("effects.application.build_corpus.build", explode)

        with pytest.raises(ValueError, match="mismatched caps"):
            run_build_corpus(parse("build-corpus"))


class TestEncodeAbilities:
    def test_the_defaults_are_the_contracts(self):
        args = parse("encode-abilities", "--output-root", "cache/run", "--checkpoint", "c.pt")
        assert args.variant == "full"
        assert args.checkpoint == Path("c.pt")
        assert args.output_root == Path("cache/run")
        assert args.clean is False

    def test_the_vocabulary_paths_default_to_the_checkpoints(self):
        """FR-097: resolved from the loaded checkpoint, not from a constant."""
        args = parse("encode-abilities")
        assert args.vocab_path is None
        assert args.keyword_definitions is None


class TestEvaluateEffectModel:
    def test_the_defaults_are_the_contracts(self):
        args = parse("evaluate-effect-model", "--checkpoint", "models/run/latest.pt")
        assert args.checkpoint == Path("models/run/latest.pt")
        assert not hasattr(args, "records_dir")
        assert args.baseline_checkpoints is None
        assert not hasattr(args, "sealed_encoder_checkpoint")

    def test_variant_checkpoint_is_repeatable(self):
        args = parse(
            "evaluate-effect-model",
            "--variant-checkpoint", "identity=a.pt",
            "--variant-checkpoint", "no-state=b.pt",
        )
        assert args.baseline_checkpoints == [
            ("identity", Path("a.pt")), ("no-state", Path("b.pt")),
        ]

    def test_an_unknown_variant_checkpoint_name_is_a_usage_error(self):
        with pytest.raises(SystemExit):
            parse("evaluate-effect-model", "--variant-checkpoint", "identiy=a.pt")

    def test_the_vocabulary_paths_default_to_the_checkpoints(self):
        args = parse("evaluate-effect-model")
        assert args.vocab_path is None
        assert args.keyword_definitions is None

    def test_the_corpus_defaults_to_the_checkpoints_recorded_one(self):
        """FR-147: an explicit --corpus overrides; absent, the checkpoint's
        own recorded path is what gets verified."""
        args = parse("evaluate-effect-model")
        assert args.corpus is None


class TestSubcommandTable:
    def test_every_subcommand_is_registered(self):
        parser = build_parser()
        action = next(a for a in parser._actions if isinstance(a.choices, dict))
        assert set(action.choices) == {
            "build-vocab", "extract-keyword-definitions", "collect-coverage",
            "build-corpus", "field-coverage", "validate-corpus",
            "collect-variants", "train-effect-model", "encode-abilities",
            "evaluate-effect-model", "holdout-cards", "rule-counts",
        }

    def test_bare_invocation_prints_help_rather_than_failing(self):
        assert getattr(build_parser().parse_args([]), "func", None) is None


class TestTrainEffectModelSurfaceMismatch:
    """A surface disagreement is only knowable once the manifest is read, so
    it surfaces out of ``run`` rather than at the parser. It still has to
    arrive as an exit code, the way every sibling guard does."""

    def test_it_becomes_an_exit_code_rather_than_a_traceback(self, monkeypatch):
        from effects.application.train_effect_model import SurfaceMismatchError
        from effects.infrastructure.cli import run_train_effect_model

        def explode(config):
            raise SurfaceMismatchError("built on 'script', --vocab-path is 'prose'")

        monkeypatch.setattr("effects.application.train_effect_model.run", explode)
        args = parse("train-effect-model", "--corpus", "output/effects/corpus")

        assert run_train_effect_model(args) == 1


class TestBuildCorpus:
    """Every ``build-corpus`` default, pinned.

    The defaults are contract here more sharply than anywhere else on this
    surface: each one is written into the manifest and each one changes the
    dataset's digest, so a default that drifts changes what a corpus — and
    every checkpoint pinned to it — means, silently and after the fact.
    """

    def test_the_path_defaults_are_the_contracts(self):
        args = parse(
            "build-corpus", "--records-dir", "records/run", "--output", "corpus/run",
            "--vocab-path", "v.txt",
        )
        assert args.records_dir == Path("records/run")
        assert args.output == Path("corpus/run")
        assert args.cards_folders == BOTH_TREES
        assert args.vocab_path == "v.txt"  # a string: the manifest records it
        assert args.variant_scripts is None

    def test_the_holdout_defaults_match_the_trainers(self):
        """The same two values must be passed here and at `holdout-cards`, or
        the depleted corpus and the split disagree. The trainer no longer
        spells them at all: it reads the manifest this build writes."""
        args = parse("build-corpus")
        assert (args.holdout_permille, args.holdout_max_carriers) == (20, 8)
        holdout = parse("holdout-cards", "--out", "output/effects/holdout.txt")
        assert (holdout.holdout_permille, holdout.holdout_max_carriers) == (20, 8)

    def test_the_selection_defaults_are_the_contracts(self):
        args = parse("build-corpus")
        assert args.text_cap == 200
        assert args.card_disjoint_text_cap == 50
        assert args.game_disjoint_share == 0.01
        assert args.game_disjoint_keyword_share == 0.15
        assert args.game_disjoint_keywords == (
            "first strike", "deathtouch", "trample", "indestructible", "wither", "infect",
        )
        assert args.reuse_cap == 4
        assert args.training_records == 0  # no ceiling
        assert args.class_mix is None  # the training mixture
        assert args.seed == 42

    def test_the_run_defaults_are_the_contracts(self):
        args = parse("build-corpus")
        assert args.workers == 12  # explicit and bounded, never the CPU count
        for out_of_range in ("0", "13"):
            with pytest.raises(SystemExit):
                parse("build-corpus", "--workers", out_of_range)
        assert args.verify is False
        assert parse("build-corpus", "--verify").verify is True

    def test_the_class_mix_is_parsed_through_the_trainers_own_syntax(self):
        args = parse("build-corpus", "--class-mix", "rewrite=0.5,combat=0.5")
        assert args.class_mix == {"rewrite": 0.5, "combat": 0.5}

    def test_a_class_the_mixture_has_no_name_for_is_refused_at_the_parser(self):
        with pytest.raises(SystemExit):
            parse("build-corpus", "--class-mix", "oracle=1.0")

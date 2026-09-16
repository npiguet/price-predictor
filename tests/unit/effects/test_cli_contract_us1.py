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

import pytest

from effects.infrastructure.cli import build_parser


def parse(*argv: str):
    return build_parser().parse_args(argv)


class TestBuildVocab:
    def test_the_defaults_are_the_contracts(self):
        args = parse("build-vocab")
        assert args.surface == "prose"
        assert args.cards_folders is None  # resolved to both converted trees
        assert args.vocab_path is None  # resolved per surface
        assert args.keyword_definitions == "output/effects/keyword-definitions.json"
        assert args.target_size == 5000

    def test_cards_folder_is_repeatable(self):
        args = parse(
            "build-vocab", "--cards-folder", "a/", "--cards-folder", "b/",
        )
        assert args.cards_folders == ["a/", "b/"]

    def test_the_default_resolves_to_both_converted_trees(self):
        from effects.infrastructure.cli import resolve_cards_folders

        folders = resolve_cards_folders(parse("build-vocab").cards_folders)
        assert [f.as_posix() for f in folders] == [
            "output/cardsfolder", "output/tokenscripts",
        ]

    def test_the_surface_choices_are_prose_and_script(self):
        assert parse("build-vocab", "--surface", "script").surface == "script"
        with pytest.raises(SystemExit):
            parse("build-vocab", "--surface", "oracle")

    def test_an_explicit_vocab_path_is_carried(self):
        args = parse("build-vocab", "--vocab-path", "models/x.txt")
        assert args.vocab_path == "models/x.txt"


class TestExtractKeywordDefinitions:
    def test_the_output_default_is_the_contracts(self):
        args = parse("extract-keyword-definitions")
        assert args.output == "output/effects/keyword-definitions.json"


class TestTrainEffectModel:
    def test_the_path_defaults_are_the_contracts(self):
        args = parse("train-effect-model")
        assert args.records_dir == "output/effects/records/"
        assert args.vocab_path == "models/effects/vocab.txt"
        assert args.printings_path == "resources/AllPrintings.json"
        assert args.keyword_definitions == "output/effects/keyword-definitions.json"
        assert args.model_output is None  # resolved from --variant
        assert args.split_from is None
        assert args.variant_scripts is None

    def test_the_architecture_defaults_are_the_contracts(self):
        args = parse("train-effect-model")
        assert args.e_dim == 64
        assert args.e_noise == 0.05
        assert args.keyword_expand_p == 0.25
        assert args.context_dropout == 0.15

    def test_the_auxiliary_weights_are_the_contracts(self):
        args = parse("train-effect-model")
        assert args.mlm_weight == 0.1
        assert args.mlm_mask_prob == 0.15
        assert args.api_weight == 0.05

    def test_the_schedule_defaults_are_the_contracts(self):
        args = parse("train-effect-model")
        assert args.curriculum_epoch == 3
        assert args.batch_size == 32
        assert args.grad_accum == 1
        assert args.steps_per_epoch == 5000
        assert args.epochs == 40
        assert args.patience == 5

    def test_the_context_cache_is_off_by_default(self):
        """It is a documented fallback for the GPU budget, not the default."""
        args = parse("train-effect-model")
        assert args.context_cache is False
        assert args.cache_refresh == 500
        assert parse("train-effect-model", "--context-cache").context_cache

    def test_the_mixture_defaults_to_the_eight_class_one(self):
        assert parse("train-effect-model").kind_mix is None

    def test_the_variant_choices_are_the_five(self):
        for variant in ("full", "identity", "state-only", "no-state", "taxonomy"):
            assert parse(
                "train-effect-model", "--variant", variant,
            ).variant == variant
        with pytest.raises(SystemExit):
            parse("train-effect-model", "--variant", "oracle")

    def test_full_is_the_default_variant(self):
        assert parse("train-effect-model").variant == "full"

    def test_withhold_keyword_defaults_to_none(self):
        assert parse("train-effect-model").withhold_keyword is None
        assert parse(
            "train-effect-model", "--withhold-keyword", "cascade",
        ).withhold_keyword == "cascade"


class TestVariantAcceptsCorpusAtThePreflightCheck:
    """Task 7 fix round 1, Finding 2's CLI-layer gap.

    ``require_split_from`` was fixed to accept ``--corpus`` as an alternative
    to ``--split-from`` for a non-``full`` variant, but ``cli.py``'s own
    ``run_train_effect_model`` calls it a second time, earlier, as a
    pre-flight check that turns ``MissingSplitError`` into a clean exit code
    before ``train_config_from``/``run`` ever run. That call was not updated
    alongside the library fix, so ``python -m effects train-effect-model
    --variant identity --corpus DIR`` was still rejected at the door — a gap
    the existing ``require_split_from`` unit tests cannot see, because they
    call the function directly and never touch this second call site.

    The real trainer entry point (``train_effect_model.run``) is
    monkeypatched out in the accepted case, so getting past the pre-flight
    check is what these tests prove — not a full training run — and neither
    case imports torch or reads a real corpus off disk.
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

    def test_a_variant_with_neither_route_is_still_refused(self):
        from effects.infrastructure.cli import run_train_effect_model

        args = parse("train-effect-model", "--variant", "identity")

        assert run_train_effect_model(args) == 2


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
        from effects.application.build_corpus import BuildCorpusError
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
        args = parse("encode-abilities")
        assert args.variant == "full"
        assert args.checkpoint is None  # resolved from --variant
        assert args.output_root == "output/effects/abilities/"
        assert args.clean is False

    def test_the_vocabulary_paths_default_to_the_checkpoints(self):
        """FR-097: resolved from the loaded checkpoint, not from a constant."""
        args = parse("encode-abilities")
        assert args.vocab_path is None
        assert args.keyword_definitions is None

    def test_the_variant_selects_the_checkpoint_and_the_suffix_together(self):
        from effects.application.encode_abilities import EncodeAbilitiesConfig

        config = EncodeAbilitiesConfig(variant="no-state")
        assert config.resolved_checkpoint().parent.name == "no-state"

    def test_clean_is_a_flag(self):
        assert parse("encode-abilities", "--clean").clean is True


class TestEvaluateEffectModel:
    def test_the_defaults_are_the_contracts(self):
        args = parse("evaluate-effect-model")
        assert args.checkpoint == "models/effects/effect-model/latest.pt"
        assert args.records_dir == "output/effects/records/"
        assert args.variant_checkpoints is None
        assert args.sealed_encoder_checkpoint == "models/sealed/encoder/latest.pt"

    def test_variant_checkpoint_is_repeatable(self):
        args = parse(
            "evaluate-effect-model",
            "--variant-checkpoint", "identity=a.pt",
            "--variant-checkpoint", "no-state=b.pt",
        )
        assert args.variant_checkpoints == ["identity=a.pt", "no-state=b.pt"]

    def test_there_is_no_split_flag(self):
        """Splits come from the checkpoint — never a flag, never recomputed."""
        for name in ("--split", "--split-from", "--held-out-cards"):
            with pytest.raises(SystemExit):
                parse("evaluate-effect-model", name, "x")

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
    def test_every_us1_subcommand_is_registered(self):
        parser = build_parser()
        action = next(
            a for a in parser._actions if isinstance(a.choices, dict)
        )
        assert {
            "build-vocab", "extract-keyword-definitions", "train-effect-model",
            "encode-abilities", "evaluate-effect-model",
        } <= set(action.choices)

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

        assert run_train_effect_model(args) == 2


class TestCorpusExclusiveFlagsAtThePreflightCheck:
    """``validate_corpus_flags`` ran only inside ``run()`` and raised a bare
    ``ValueError`` nothing caught, so ``train-effect-model --corpus DIR
    --records-dir X`` printed a stack trace where its sibling guard one line
    over printed a message and exited 2.

    Tested through the dispatcher rather than through the library function,
    because that is exactly how the last gap of this shape survived its own
    fix round: the library was right and the entry point never called it.
    """

    def test_a_flag_the_manifest_records_is_refused_with_an_exit_code(self):
        from effects.infrastructure.cli import run_train_effect_model

        args = parse(
            "train-effect-model", "--corpus", "output/effects/corpus",
            "--records-dir", "output/effects/other-records",
        )

        assert run_train_effect_model(args) == 2

    def test_the_refusal_happens_before_the_trainer_is_entered(self, monkeypatch):
        from effects.infrastructure.cli import run_train_effect_model

        def explode(config):
            raise AssertionError("run() must not be reached")

        monkeypatch.setattr("effects.application.train_effect_model.run", explode)
        args = parse(
            "train-effect-model", "--corpus", "output/effects/corpus",
            "--holdout-permille", "30",
        )

        assert run_train_effect_model(args) == 2


class TestBuildCorpus:
    """Every ``build-corpus`` default, pinned.

    The defaults are contract here more sharply than anywhere else on this
    surface: each one is written into the manifest and each one changes the
    dataset's digest, so a default that drifts changes what a corpus — and
    every checkpoint pinned to it — means, silently and after the fact.
    """

    def test_the_path_defaults_are_the_contracts(self):
        args = parse("build-corpus")
        assert args.records_dir == "output/effects/records/"
        assert args.output == "output/effects/corpus/"
        assert args.cards_folders is None  # resolved to both converted trees
        assert args.vocab_path == "models/effects/vocab.txt"
        assert args.variant_scripts is None

    def test_the_holdout_defaults_match_the_trainers(self):
        """The same two values must be passed here, at `holdout-cards` and at
        training, or the depleted corpus and the split disagree."""
        args = parse("build-corpus")
        assert args.holdout_permille == parse("train-effect-model").holdout_permille
        assert (
            args.holdout_max_carriers
            == parse("train-effect-model").holdout_max_carriers
        )
        assert (args.holdout_permille, args.holdout_max_carriers) == (20, 8)

    def test_the_selection_defaults_are_the_contracts(self):
        args = parse("build-corpus")
        assert args.text_cap == 200
        assert args.card_disjoint_text_cap == 50
        assert args.game_disjoint_games == 1000
        assert args.training_records == 0  # no ceiling
        assert args.class_mix is None  # the training mixture
        assert args.seed == 42

    def test_the_run_defaults_are_the_contracts(self):
        args = parse("build-corpus")
        assert args.workers is None  # CPU count
        assert args.verify is False
        assert parse("build-corpus", "--verify").verify is True

    def test_the_parsed_defaults_and_the_configs_own_agree(self):
        """Two default surfaces for one flag set, and only this compares them."""
        from pathlib import Path

        from effects.application.build_corpus import BuildCorpusConfig

        args = parse("build-corpus")
        config = BuildCorpusConfig(records_dir=Path("."))
        assert config.output == Path(args.output)
        assert config.vocab_path == args.vocab_path
        assert config.holdout_permille == args.holdout_permille
        assert config.holdout_max_carriers == args.holdout_max_carriers
        assert config.text_cap == args.text_cap
        assert config.card_disjoint_text_cap == args.card_disjoint_text_cap
        assert config.game_disjoint_target == args.game_disjoint_games
        assert config.training_records == args.training_records
        assert config.class_mix == args.class_mix
        assert config.seed == args.seed
        assert config.workers == args.workers
        assert config.verify == args.verify
        assert config.cards_folders == ("output/cardsfolder", "output/tokenscripts")

    def test_the_class_mix_is_parsed_through_the_trainers_own_syntax(self):
        args = parse("build-corpus", "--class-mix", "rewrite=0.5,combat=0.5")
        assert args.class_mix == {"rewrite": 0.5, "combat": 0.5}

    def test_a_class_the_mixture_has_no_name_for_is_refused_at_the_parser(self):
        with pytest.raises(SystemExit):
            parse("build-corpus", "--class-mix", "oracle=1.0")

    def test_cards_folder_is_repeatable(self):
        args = parse("build-corpus", "--cards-folder", "a/", "--cards-folder", "b/")
        assert args.cards_folders == ["a/", "b/"]

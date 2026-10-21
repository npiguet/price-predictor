"""Where the effects pipeline reads and writes by default, relative to the repo root.

One definition per path, in the spelling an operator types (trailing slash on a
directory), because the CLI prints these in ``--help`` and
``specs/023-ability-effect-model/contracts/cli.md`` pins them as written. Kept
apart from the CLI and free of imports, so an application module can take a
default from here without importing the parser, and the parser can format its
help without importing an application module that loads torch.
"""

from __future__ import annotations

#: The raw shard root every collector appends to and every reader walks.
DEFAULT_RECORDS_DIR = "output/effects/records/"
#: Both converted trees: cards, and the tokens kept out of the card tree.
DEFAULT_CARDS_FOLDERS = ("output/cardsfolder/", "output/tokenscripts/")
DEFAULT_KEYWORD_DEFINITIONS = "output/effects/keyword-definitions.json"
DEFAULT_VOCAB_PATH = "models/effects/vocab.txt"
DEFAULT_SCRIPT_VOCAB_PATH = "models/effects/vocab-script.txt"
DEFAULT_VARIANT_SCRIPTS = "output/effects/variant-scripts/"
DEFAULT_ABILITIES_ROOT = "output/effects/abilities/"
DEFAULT_CHECKPOINT = "models/effects/effect-model/latest.pt"
#: Where the probe scripts write their report directories and probe sets.
DEFAULT_REPORTS_DIR = "output/effects/reports/"

#: Kept apart from DEFAULT_RECORDS_DIR: this is the *curated dataset*
#: directory build-corpus writes to, not the raw shard root it reads from
#: (specs/2026-09-05-ability-effect-model.md § Curated corpus).
DEFAULT_CORPUS_OUTPUT = "output/effects/corpus/"

#: Forge's *source* card scripts, not the converted ones: a variant is a Forge
#: script Forge must be able to load.
DEFAULT_FORGE_CARDS_PATH = "../forge/forge-gui/res/cardsfolder/"

#: Forge's own token scripts, beside the card scripts `convert` reads. The
#: remap needs the raw tree rather than `output/tokenscripts/`: it tells
#: same-name token scripts apart by their Colors/Types/PT lines, which
#: conversion does not preserve.
DEFAULT_FORGE_TOKENSCRIPTS = "../forge/forge-gui/res/tokenscripts"

#: Where the sealed pipeline's per-card winnability labels live by default;
#: ``--win-rates`` names another table. The decodability battery skips when it
#: is absent rather than failing: the labels are an operator's training data,
#: not a repository artifact.
DEFAULT_WIN_RATES = "output/sealed/cards-win-rates.txt"

#: The sealed pipeline's own artifacts the scorer smoke test reads: the encoder
#: whose vectors pooled ``e`` is concatenated with, and the match outcomes
#: Phase A trains on. Restated here rather than imported, because ``effects``
#: never imports ``sealed.application``.
DEFAULT_SEALED_ENCODER = "models/sealed/encoder/latest.pt"
DEFAULT_MATCH_OUTCOMES = "output/sealed/match-outcomes.txt"

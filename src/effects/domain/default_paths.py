"""Where the effects pipeline reads and writes by default, relative to the repo root.

Spelled as an operator types them (trailing slash on a directory), because
``--help`` prints them and ``specs/023-ability-effect-model/contracts/cli.md``
pins them as written. Importing nothing but ``pathlib``, so the parser can
read them without loading torch.
"""

from __future__ import annotations

from pathlib import Path

#: The raw shard root every collector appends to and every reader walks.
DEFAULT_RECORDS_DIR = "output/effects/records/"
#: Both converted trees: cards, and the tokens kept out of the card tree.
DEFAULT_CARDS_FOLDERS = ("output/cardsfolder/", "output/tokenscripts/")
#: The same two trees as the paths a config holds.
DEFAULT_CARDS_FOLDER_PATHS: tuple[Path, ...] = tuple(
    Path(folder) for folder in DEFAULT_CARDS_FOLDERS
)
DEFAULT_KEYWORD_DEFINITIONS = "output/effects/keyword-definitions.json"
DEFAULT_VOCAB_PATH = "models/effects/vocab.txt"
DEFAULT_SCRIPT_VOCAB_PATH = "models/effects/vocab-script.txt"
DEFAULT_VARIANT_SCRIPTS = "output/effects/variant-scripts/"
DEFAULT_ABILITIES_ROOT = "output/effects/abilities/"
DEFAULT_CHECKPOINT = "models/effects/effect-model/latest.pt"
DEFAULT_REPORTS_DIR = "output/effects/reports/"
#: The curated dataset build-corpus writes, not the raw shard root it reads.
DEFAULT_CORPUS_OUTPUT = "output/effects/corpus/"
#: Forge's source card scripts: a variant is a script Forge must load.
DEFAULT_FORGE_CARDS_PATH = "../forge/forge-gui/res/cardsfolder/"
#: The sealed pipeline's per-card win rates; an operator's data, not a repo
#: artifact, so the decodability battery skips when it is absent.
DEFAULT_WIN_RATES = "output/sealed/cards-win-rates.txt"

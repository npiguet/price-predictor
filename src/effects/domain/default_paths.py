"""Where the effects pipeline reads by default, relative to the repo root.

Only sources are named here: the converted trees of generation 3 (the one
layout path every command shares), Forge's card scripts and MTGJSON's dump.
Every other path a command reads or writes is a required flag
(``.claude/skills/artifact-layout/SKILL.md`` rule 2). Spelled as an operator
types them (trailing slash on a directory), because ``--help`` prints them.
Importing nothing but ``pathlib``, so the parser can read them without
loading torch.
"""

from __future__ import annotations

from pathlib import Path

#: Both converted trees of generation 3: cards, and the tokens kept out of the card tree.
DEFAULT_CARDS_FOLDERS = (
    "output/effects/gen3/converted/cardsfolder/",
    "output/effects/gen3/converted/tokenscripts/",
)
#: The same two trees as the paths a config holds.
DEFAULT_CARDS_FOLDER_PATHS: tuple[Path, ...] = tuple(
    Path(folder) for folder in DEFAULT_CARDS_FOLDERS
)
#: Forge's source card scripts: a variant is a script Forge must load.
DEFAULT_FORGE_CARDS_PATH = "../forge/forge-gui/res/cardsfolder/"
#: MTGJSON's card dump, read for Forge's printed card names.
DEFAULT_PRINTINGS_PATH = "resources/AllPrintings.json"

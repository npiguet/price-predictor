"""Taking a variant round's scripts back out of Forge's custom-cards directory.

Every worker of a variant round, and the sidecar pass before it, copies the
variant tree into Forge's custom-cards directory before Forge reads its card
database (``ForgeEnvironmentInitializer.stageCustomCards``), and lists what it
copied in a manifest there. The next staging clears that list first, but
between two variant rounds the scripts would stay loaded into every Forge
startup on the machine (FR-058b; python R8). A round removes them when it ends.
"""

from __future__ import annotations

import logging
from pathlib import Path

import price_predictor.infrastructure.forge_jvm as forge_jvm

logger = logging.getLogger(__name__)

#: The staging's manifest, ``ForgeEnvironmentInitializer.STAGED_MANIFEST``:
#: not a ``.txt``, because Forge reads every ``.txt`` there as a card.
STAGED_MANIFEST = ".staged-by-effects.list"


def custom_cards_dir() -> Path:
    """Forge's ``USER_CUSTOM_CARDS_DIR`` as a worker resolves it: the worker
    makes ``forge-gui/`` the assets dir, and a headless GUI's user dir is
    ``<assets>/data/``."""
    return forge_jvm.forge_dir() / "forge-gui" / "data" / "custom" / "cards"


def unstage_variants(directory: Path | None = None) -> int:
    """Delete every file the last staging copied into ``directory`` (default
    :func:`custom_cards_dir`), as its manifest names them, and empty the
    manifest; returns how many were deleted. A card a person put there is not
    in the manifest and stays."""
    directory = custom_cards_dir() if directory is None else Path(directory)
    manifest = directory / STAGED_MANIFEST
    if not manifest.is_file():
        return 0
    removed = 0
    for name in manifest.read_text(encoding="utf-8").splitlines():
        name = name.strip()
        if name and (directory / name).is_file():
            (directory / name).unlink()
            removed += 1
    manifest.write_text("", encoding="utf-8")
    logger.info("Unstaged %d variant script(s) from %s", removed, directory)
    return removed

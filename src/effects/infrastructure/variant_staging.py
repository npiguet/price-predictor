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

#: The staging's manifest, ``ForgeEnvironmentInitializer.STAGED_MANIFEST``.
STAGED_MANIFEST = forge_jvm.STAGED_MANIFEST


def custom_cards_dir() -> Path:
    """Forge's ``USER_CUSTOM_CARDS_DIR`` as a worker resolves it."""
    return forge_jvm.custom_cards_dir()


def unstage_variants(directory: Path | None = None) -> int:
    """Delete every file the last staging copied into ``directory`` (default
    :func:`custom_cards_dir`), as its manifest names them, and empty the
    manifest; returns how many were deleted. A card a person put there is not
    in the manifest and stays. A run killed too hard to get here leaves its
    scripts staged: the next command's first JVM clears them
    (``forge_jvm.clear_stale_staging``)."""
    directory = custom_cards_dir() if directory is None else Path(directory)
    removed = forge_jvm.unstage_custom_cards(directory)
    logger.info("Unstaged %d variant script(s) from %s", removed, directory)
    return removed

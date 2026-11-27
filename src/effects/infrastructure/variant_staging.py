"""Taking a variant round's scripts back out of Forge's custom-cards directory.

Every worker of a variant round, and the sidecar pass before it, copies the
variant tree into Forge's custom-cards directory before Forge reads its card
database (``ForgeEnvironmentInitializer.stageCustomCards``), and lists what it
copied in a manifest there. The next staging clears that list first, but
between two variant rounds the scripts would stay loaded into every Forge
startup on the machine (FR-058b). A round removes them when it ends.
"""

from __future__ import annotations

import logging
from pathlib import Path

import price_predictor.infrastructure.custom_card_staging as custom_card_staging

logger = logging.getLogger(__name__)

#: The staging's manifest, ``ForgeEnvironmentInitializer.STAGED_MANIFEST``.
STAGED_MANIFEST = custom_card_staging.STAGED_MANIFEST


def custom_cards_dir() -> Path:
    """Forge's ``USER_CUSTOM_CARDS_DIR`` as a worker resolves it."""
    return custom_card_staging.custom_cards_dir()


def claim_staging() -> None:
    """Record this process as the staging's owner before anything is staged,
    so another command's first JVM leaves the staging alone while this run
    lasts (``custom_card_staging.clear_stale_staging``); :func:`unstage_variants`
    drops the record."""
    custom_card_staging.claim_staging()


def unstage_variants(directory: Path | None = None) -> int:
    """Delete every file the last staging copied into ``directory`` (default
    :func:`custom_cards_dir`), as its manifest names them, empty the manifest
    and drop the owner record; returns how many were deleted. A card a person
    put there is not in the manifest and stays. A manifest that cannot be
    read is reported and left. A run killed too hard to get here leaves its
    scripts staged: the next command's first JVM clears them once this
    process is gone (``custom_card_staging.clear_stale_staging``)."""
    directory = custom_cards_dir() if directory is None else Path(directory)
    try:
        removed = custom_card_staging.unstage_custom_cards(directory)
    except (OSError, ValueError) as exc:
        logger.warning("Could not unstage the variant scripts in %s: %s", directory, exc)
        return 0
    logger.info("Unstaged %d variant script(s) from %s", removed, directory)
    return removed

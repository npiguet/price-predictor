"""The UMAP check (FR-110): the ability cache projected to two dimensions.

Reported, never a verdict: the picture is for a person to read, and a run
that cannot draw it has not failed.
"""

from __future__ import annotations

import logging

import numpy as np

from effects.application.geometry_checks import CachedVectors
from effects.domain.evaluation_report import CheckName, CheckResult, CheckStatus

logger = logging.getLogger(__name__)


def umap_projection(matrix: np.ndarray, *, seed: int = 42) -> np.ndarray | None:
    """A 2-D UMAP projection, or None when the library or the data is absent:
    a run that cannot draw the picture has not failed."""
    if matrix.shape[0] < 10:
        return None
    try:
        import umap
    except ImportError:
        logger.warning("umap-learn is not installed; skipping the projection")
        return None
    # A seed forces UMAP single-threaded; saying so keeps it from warning.
    reducer = umap.UMAP(n_components=2, random_state=seed, n_jobs=1)
    return reducer.fit_transform(matrix)


def check_umap(cached: CachedVectors) -> CheckResult:
    projection = umap_projection(cached.matrix)
    if projection is None:
        return CheckResult(
            CheckName.UMAP, CheckStatus.SKIPPED,
            "too few vectors, or umap-learn is not installed",
        )
    return CheckResult(
        CheckName.UMAP, CheckStatus.REPORTED,
        f"projected {projection.shape[0]} unique ability texts to 2-D",
    )

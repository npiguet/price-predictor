"""The UMAP check (FR-110): the ability cache projected to two dimensions,
coloured by what each text does.

Reported, never a verdict: the picture is for a person to read, and a run
that cannot draw it has not failed. What it does is the line's script API
type (``Pump``, ``DealDamage``, ``ChangeZone`` …), the one effect category
every sidecar line records. The projection is written twice beside the
evaluation, as a picture and as a table, so a cluster seen in one can be
looked up in the other.
"""

from __future__ import annotations

import csv
import logging
import math
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from effects.application.geometry_checks import CachedVectors
from effects.domain.evaluation_report import CheckName, CheckResult, CheckStatus

logger = logging.getLogger(__name__)

UMAP_PNG = "umap.png"
UMAP_CSV = "umap.csv"

#: API types that get a panel of their own, the most frequent first; every
#: other type shares :data:`OTHER_API_TYPE`'s.
TOP_API_TYPES = 12
OTHER_API_TYPE = "other"

#: Panels per row of the picture.
PANEL_COLUMNS = 4
#: The panel's own texts, and every other text behind them for context: one
#: hue per panel rather than one per type, since a dozen hues in one scatter
#: cannot be told apart.
HIGHLIGHT = "#2a78d6"
CONTEXT = "#c3c2b7"
INK = "#0b0b0b"
MUTED = "#898781"


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


def api_type_panels(
    api_types: Sequence[str], *, top: int = TOP_API_TYPES,
) -> tuple[list[str], list[str]]:
    """``(panels, panel_of)``: the panel names in order, the ``top`` most
    frequent types (ties by name, so the order never depends on the cache's)
    then :data:`OTHER_API_TYPE` if any type is left over, and each text's
    panel."""
    counts = Counter(api_types)
    ranked = sorted(counts, key=lambda name: (-counts[name], name))
    kept = ranked[:top]
    panels = kept + ([OTHER_API_TYPE] if len(ranked) > top else [])
    named = set(kept)
    return panels, [name if name in named else OTHER_API_TYPE for name in api_types]


def write_umap_csv(
    path: Path, projection: np.ndarray, api_types: Sequence[str], texts: Sequence[str],
) -> None:
    """One row per text: its two coordinates, its own API type (not its
    panel, so a reader can regroup) and the text."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("x", "y", "api_type", "text"))
        for (x, y), api_type, text in zip(projection, api_types, texts, strict=True):
            writer.writerow((f"{x:.6f}", f"{y:.6f}", api_type, text))


def write_umap_png(
    path: Path, projection: np.ndarray, panels: Sequence[str], panel_of: Sequence[str],
) -> None:
    """Small multiples, one panel per API type: the type's texts over every
    other text, every panel on the same axes so a region is one place in all.

    Drawn on the Agg canvas directly rather than through ``pyplot``, so no
    global backend is chosen for whoever imports this, and written without
    the version stamp, so one projection writes one file byte for byte.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    columns = min(PANEL_COLUMNS, len(panels))
    rows = math.ceil(len(panels) / columns)
    figure = Figure(figsize=(3.2 * columns, 3.2 * rows + 0.8), dpi=100)
    FigureCanvasAgg(figure)
    axes = figure.subplots(rows, columns, sharex=True, sharey=True, squeeze=False)
    of = np.asarray(panel_of)
    for ax, name in zip(axes.flat, panels):
        mine = of == name
        ax.scatter(*projection[~mine].T, s=1, c=CONTEXT, linewidths=0)
        ax.scatter(*projection[mine].T, s=2, c=HIGHLIGHT, linewidths=0)
        ax.set_title(f"{name} ({int(mine.sum())})", fontsize=9, color=INK)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(CONTEXT)
    for ax in list(axes.flat)[len(panels):]:
        ax.set_visible(False)
    figure.suptitle(
        f"Ability cache, UMAP: {len(panel_of)} unique texts by script API type",
        fontsize=11, color=INK,
    )
    figure.legend(
        handles=[
            Line2D([], [], marker="o", linestyle="", color=HIGHLIGHT,
                   label="texts of the panel's API type"),
            Line2D([], [], marker="o", linestyle="", color=CONTEXT,
                   label="every other text"),
        ],
        loc="upper center", bbox_to_anchor=(0.5, 0.97), ncol=2, frameon=False,
        fontsize=8, labelcolor=MUTED,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(path, metadata={"Software": None})


def check_umap(cached: CachedVectors, output_dir: Path) -> CheckResult:
    """Project every unique cached text and write :data:`UMAP_PNG` and
    :data:`UMAP_CSV` under ``output_dir``; the detail names both."""
    projection = umap_projection(cached.matrix)
    if projection is None:
        return CheckResult(
            CheckName.UMAP, CheckStatus.SKIPPED,
            "too few vectors, or umap-learn is not installed",
        )
    texts = cached.texts
    api_types = [cached.api_type_of(text) for text in texts]
    panels, panel_of = api_type_panels(api_types)
    output_dir.mkdir(parents=True, exist_ok=True)
    png, table = output_dir / UMAP_PNG, output_dir / UMAP_CSV
    write_umap_csv(table, projection, api_types, texts)
    write_umap_png(png, projection, panels, panel_of)
    distinct = len(set(api_types))
    return CheckResult(
        CheckName.UMAP, CheckStatus.REPORTED,
        f"projected {projection.shape[0]} unique ability texts to 2-D, one panel "
        f"per script API type ({min(distinct, TOP_API_TYPES)} of {distinct}, the "
        f"rest as {OTHER_API_TYPE!r}): {png} and {table}",
        {"texts": float(projection.shape[0]), "api_types": float(distinct)},
    )

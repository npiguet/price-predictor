"""The figure style the experiment records share.

The pairs used together (blue/orange, blue/red, each vs gray) were checked in
OKLab under deutan/protan simulation; red carries sign, which bar direction
re-encodes.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from _lib.paths import EXPERIMENT_IMAGES  # noqa: E402

BLUE, ORANGE, GREEN, RED, GRAY = "#3b6fb6", "#e08f3c", "#4a9a62", "#c05555", "#8a8a8a"

PROBE_RC = {
    "font.size": 10, "axes.titlesize": 11, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
    "figure.dpi": 110,
}


def use_probe_style() -> None:
    plt.rcParams.update(PROBE_RC)


def figure_saver(prefix: str, directory: Path = EXPERIMENT_IMAGES) -> Callable:
    """``save(fig, name)`` writing ``{prefix}-{name}.png`` and ``.svg`` into ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)

    def save(fig, name: str) -> None:
        for ext in ("png", "svg"):
            fig.savefig(directory / f"{prefix}-{name}.{ext}", bbox_inches="tight",
                        dpi=150 if ext == "png" else None)
        plt.close(fig)
        print("wrote", name)

    return save

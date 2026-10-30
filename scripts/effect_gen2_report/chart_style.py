"""What the gen-2 charts and diagrams share: where they are written, the width colours,
how a colour is lightened and which charts a command line asks for."""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
from _lib.style import BLUE, GREEN, ORANGE  # noqa: E402
from lineage import WIDTHS  # noqa: E402

RECORD = Path(__file__).resolve().parents[2] / "experiments" / "2026-09-18-effect-model-gen2"
DATA = RECORD / "data"
CHARTS = RECORD / "charts"

#: The colour of each width of e, never used for anything else.
WIDTH_COLOUR = dict(zip(WIDTHS, (BLUE, ORANGE, GREEN)))


def shade(colour: str, amount: float) -> tuple:
    """``colour`` blended toward white; 1 keeps it, 0 is white."""
    r, g, b = matplotlib.colors.to_rgb(colour)
    return (1 - amount * (1 - r), 1 - amount * (1 - g), 1 - amount * (1 - b))


def save(fig, name: str, **options) -> None:
    """Write ``name.png`` into the record's charts folder. Not ``_lib.style.figure_saver``,
    which writes an SVG beside it and prefixes the name, into the shared images folder."""
    CHARTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(CHARTS / f"{name}.png", bbox_inches="tight", dpi=150, facecolor="white",
                **options)
    plt.close(fig)
    print("wrote", name)


def draw_requested(drawers: list[Callable[[], None]]) -> None:
    """Call every drawer, or only the ones named on the command line."""
    only = set(sys.argv[1:])
    for draw in drawers:
        if not only or draw.__name__ in only:
            draw()

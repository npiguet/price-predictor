"""What the gen-2 charts and diagrams share: where they are written, the width colours
and how a colour is lightened."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RECORD = Path(__file__).resolve().parents[2] / "experiments" / "2026-09-18-effect-model-gen2"
DATA = RECORD / "data"
CHARTS = RECORD / "charts"

#: The colour of each width of e, never used for anything else.
WIDTH_COLOUR = {64: "#3b6fb6", 128: "#e08f3c", 256: "#4a9a62"}


def shade(colour: str, amount: float) -> tuple:
    """``colour`` blended toward white; 1 keeps it, 0 is white."""
    r, g, b = matplotlib.colors.to_rgb(colour)
    return (1 - amount * (1 - r), 1 - amount * (1 - g), 1 - amount * (1 - b))


def save(fig, name: str, **options) -> None:
    CHARTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(CHARTS / f"{name}.png", bbox_inches="tight", dpi=150, facecolor="white",
                **options)
    plt.close(fig)
    print("wrote", name)

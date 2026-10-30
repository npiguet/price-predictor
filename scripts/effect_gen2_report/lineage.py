"""The gen-2 training runs and how they descend from one another.

``build_data.py`` reads each run's directory; ``make_charts.py`` lays its charts out
by the same lineage: per width of ``e``, an arm trained with lethality copies, run A
resuming it at the full learning rate, and run B continuing A's kept epoch at a tenth.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Run:
    """One training run: a directory whose ``train.log`` holds its epochs."""

    name: str
    directory: str
    width: int
    copies: int
    lr_scale: float
    parent: str | None = None


RUN_LIST = [
    Run("e64 (earlier inputs)", "2026-10-gen2-e64-earlier-inputs", 64, 0, 1.0),
    Run("e64", "2026-10-gen2-e64", 64, 0, 1.0),
    Run("e128", "2026-10-gen2-e128", 128, 0, 1.0),
    Run("e64-lethality", "2026-10-gen2-e64-lethality", 64, 8, 1.0),
    Run("e128-lethality", "2026-10-gen2-e128-lethality", 128, 8, 1.0),
    Run("e256-lethality", "2026-10-gen2-e256-lethality", 256, 8, 1.0),
    Run("A64", "2026-10-gen2-e64-lethality-ext20", 64, 8, 1.0, "e64-lethality"),
    Run("B64", "2026-10-gen2-e64-lethality-lr0.1", 64, 8, 0.1, "A64"),
    Run("A128", "2026-10-gen2-e128-lethality-ext20", 128, 8, 1.0, "e128-lethality"),
    Run("B128", "2026-10-gen2-e128-lethality-lr0.1", 128, 8, 0.1, "A128"),
    Run("A256", "2026-10-gen2-e256-lethality-ext20", 256, 8, 1.0, "e256-lethality"),
    Run("B256", "2026-10-gen2-e256-lethality-lr0.1", 256, 8, 0.1, "A256"),
]
GEN1 = Run("gen-1", "2026-09-17-full-textless-corpus", 64, 0, 1.0)
RUNS = {run.name: run for run in RUN_LIST}
WIDTHS = (64, 128, 256)
WIDTH_OF = {run.name: run.width for run in RUN_LIST}


def stage(run: Run) -> str:
    """``none`` without lethality copies, else ``arm``, ``A`` or ``B`` by continuation depth."""
    if not run.copies:
        return "none"
    depth = 0
    while run.parent:
        run, depth = RUNS[run.parent], depth + 1
    return ("arm", "A", "B")[depth]


#: Per width, the arm with copies and its two continuations.
LINES = {w: tuple(r.name for stage_name in ("arm", "A", "B") for r in RUN_LIST
                  if r.width == w and r.copies and stage(r) == stage_name) for w in WIDTHS}
#: Per width, the arm without copies on the corrected inputs (the later of a width's two).
NO_COPIES = {r.width: r.name for r in RUN_LIST if not r.copies}
#: Every probed checkpoint in lineage order: an arm is probed once, a continuation at its
#: lowest card-disjoint epoch and at its last.
CHECKPOINTS = [name for r in RUN_LIST
               for name in ([f"{r.name} best", f"{r.name} last"] if r.parent else [r.name])]

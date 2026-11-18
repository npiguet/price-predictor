"""The board-abilities table: per field, each subset's loss and the cost of noise in the card slots.

Nothing here does work at import time.
"""

from __future__ import annotations

from collections.abc import Mapping
from math import isnan

from _lib.markdown import md_table

from effect_knowledge_probes.board import SUBSETS
from effect_knowledge_probes.common import fmt


def _shown(values) -> bool:
    """A row is worth a line when some subset measured more than rounding."""
    return any(v is not None and not isnan(v) and abs(v) >= 0.0005 for v in values)


def _per_field(result: Mapping[str, dict], present: list[str], fields: list[str],
               value_of) -> str:
    rows = [[field, *(value_of(result[name]["ablation"].get(field, {})) for name in present)]
            for field in fields]
    return md_table(["field", *present],
                    [[row[0], *map(fmt, row[1:])] for row in rows if _shown(row[1:])],
                    "l" + "r" * len(present))


def render(result: Mapping[str, dict]) -> str:
    """Per field, each subset's loss and the cost of noise in the card slots."""
    present = [name for name in SUBSETS if result.get(name, {}).get("ablation")]
    fields = sorted({field for name in present for field in result[name]["ablation"]})
    lines = ["# Board abilities", "",
             "Records per subset: " + ", ".join(
                 f"{name} {sum(result[name]['n'].values())}" for name in SUBSETS
                 if name in result) + ".", "",
             "## Loss per field", "",
             _per_field(result, present, fields, lambda entry: entry.get("base_loss")),
             "", "## Loss increase with noise in the card slots", "",
             _per_field(result, present, fields,
                        lambda entry: entry.get("noise", {}).get("cards"))]
    return "\n".join(lines) + "\n"

"""The line and pair ladders: ``e`` read for an ability line, and for a trigger with its cause.

Nothing here does work at import time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from effect_knowledge_probes import ladder
from effect_knowledge_probes.line_labelling import LineLabels

#: Items a ladder needs in one stratum and half.
MIN_LADDER_ITEMS = 20


def line_split(carriers: set[str], held_out_cards: set[str]) -> str:
    """``held-out`` when every card printing the text is held out, else ``trained``.

    Every game naming a held-out card is kept out of training, so a text printed
    on one held-out card and one trained card was still trained on, through the
    trained card. Only a text no trained card prints was never seen.
    """
    return "held-out" if carriers and carriers <= held_out_cards else "trained"


@dataclass(frozen=True)
class LineSources:
    """What the line ladder reads besides the labels.

    ``cache`` is ``(items, acting_text)``: ``common.line_items`` over the
    checkpoint's cache, every line it encodes read once per run, and the join's
    ``key -> text``. ``act_hidden`` is the trunk's ``[ACT]`` output by text.
    """

    cache: tuple
    act_hidden: dict
    held_out_cards: set


class _Row(NamedTuple):
    text: str
    e: np.ndarray
    label: float
    weight: float
    split: str


def _line_rows(spec, line_labels: list[LineLabels], sources: LineSources) -> list[_Row]:
    """One row per labelled line of ``spec`` whose text has an encoded ``e``."""
    items, acting_text = sources.cache
    by_text = {item.text: item for item in items}
    rows = []
    for entry in line_labels:
        label = entry.labels.get(spec.name)
        if label is None:
            continue
        text = acting_text(entry.key)
        item = by_text.get(text)
        if item is None or item.e is None:
            continue
        rows.append(_Row(text, item.e, label, entry.weights.get(spec.name, 1.0),
                         line_split(item.carriers, sources.held_out_cards)))
    return rows


def _acting_rungs(result: dict, data: ladder.LadderInput, texts: list[str],
                  act_mean: dict) -> None:
    """Add rung 2 (``[ACT]``) and rung 1 over the lines that were seen acting."""
    acted = [i for i, t in enumerate(texts) if t in act_mean]
    if len(acted) < MIN_LADDER_ITEMS:
        return
    subset = ladder.LadderInput(
        kind=data.kind, y=data.y[acted], groups=data.groups[acted],
        features={"1": data.features["1"][acted],
                  "2": np.stack([act_mean[texts[i]] for i in acted])},
        weight=data.weight[acted],
    )
    sub = ladder.run_ladder(subset, rungs=("1", "2"))
    result["rungs"]["2"] = sub["rungs"]["2"]
    result["rungs"]["1-acting"] = sub["rungs"]["1"]
    result["n_acting"] = sub["n"]


def _stratum_ladder(spec, chosen: list[_Row], e_dim: int, act_mean: dict) -> dict:
    """The ladder over one split's rows: ``e``, its width control, then ``[ACT]``."""
    texts = [r.text for r in chosen]
    data = ladder.LadderInput(
        kind=spec.kind, y=np.array([r.label for r in chosen], dtype=np.float64),
        groups=np.array(texts, dtype=object),
        features={"1": np.stack([r.e for r in chosen]),
                  "1w": np.stack([ladder.width_vector(t, e_dim) for t in texts])},
        weight=np.array([r.weight for r in chosen], dtype=np.float64),
    )
    result = ladder.run_ladder(data, rungs=("1", "1w"))
    _acting_rungs(result, data, texts, act_mean)
    return result


def _target_ladder(spec, rows: list[_Row], e_dim: int, act_mean: dict) -> dict:
    strata = {}
    for split in ("held-out", "trained"):
        chosen = [r for r in rows if r.split == split]
        if len(chosen) >= MIN_LADDER_ITEMS:
            strata[split] = _stratum_ladder(spec, chosen, e_dim, act_mean)
    return {"scope": "line", "kind": spec.kind, "strata": strata}


def line_ladder(line_labels: list[LineLabels], probe_model, targets,
                sources: LineSources) -> dict:
    """The short ladder for every line target: ``e``, its width control, ``[ACT]``."""
    act_mean = {text: np.mean(rows, axis=0) for text, rows in sources.act_hidden.items()}
    return {spec.name: _target_ladder(spec, _line_rows(spec, line_labels, sources),
                                      probe_model.e_dim, act_mean)
            for spec in targets if spec.scope == "line"}

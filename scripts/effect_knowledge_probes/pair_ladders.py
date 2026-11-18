"""The pair ladder: ``e`` read for a trigger line together with its cause.

Nothing here does work at import time.
"""

from __future__ import annotations

import numpy as np

from effect_knowledge_probes import ladder
from effect_knowledge_probes.line_ladders import MIN_LADDER_ITEMS


def _union_groups(pairs: list[tuple[str, str]]) -> list[str]:
    """Fold groups for text pairs: connected components, so no text straddles."""
    parent: dict[str, str] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        parent[find(a)] = find(b)
    return [find(a) for a, _ in pairs]


def _encoded(item) -> bool:
    return item is not None and item.e is not None


def _pair_rows(pairs: list[dict], cache) -> list[tuple]:
    """``(a, b, label)`` for each pair whose both lines have an encoded ``e``."""
    from effects.domain.provenance import ProvenanceKey

    items, acting_text = cache
    by_text = {item.text: item for item in items}
    rows = []
    for pair in pairs:
        a = by_text.get(acting_text(ProvenanceKey.from_dict(pair["a"])))
        b = by_text.get(acting_text(ProvenanceKey.from_dict(pair["b"])))
        if _encoded(a) and _encoded(b):
            rows.append((a, b, pair["label"]))
    return rows


def _pair_groups(rows: list[tuple]) -> tuple[list[str], str]:
    """Fold groups for the pairs: components, unless one holds over half of them."""
    groups = _union_groups([(a.text, b.text) for a, b, _ in rows])
    if max(groups.count(g) for g in set(groups)) > len(rows) / 2:
        return [a.text for a, _, _ in rows], "trigger-text"
    return groups, "component"


def pair_ladder(pairs: list[dict], probe_model, cache) -> dict:
    """Interaction pairs: a trigger line with its cause, read from both ``e``."""
    rows = _pair_rows(pairs, cache)
    if len(rows) < MIN_LADDER_ITEMS:
        return {"scope": "pair", "kind": "binary", "strata": {}}
    groups, grouping = _pair_groups(rows)
    e_dim = probe_model.e_dim
    data = ladder.LadderInput(
        kind="binary", y=np.array([r[2] for r in rows], dtype=np.float64),
        groups=np.array(groups, dtype=object),
        features={
            "1": np.stack([np.concatenate([a.e, b.e]) for a, b, _ in rows]),
            "1w": np.stack([np.concatenate([ladder.width_vector(a.text, e_dim),
                                            ladder.width_vector(b.text, e_dim)])
                            for a, b, _ in rows]),
        },
    )
    result = ladder.run_ladder(data, rungs=("1", "1w"))
    result["grouping"] = grouping
    return {"scope": "pair", "kind": "binary", "strata": {"all": result}}

"""The knowledge probes' interaction pairs read their keys through the sidecar join."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from effect_knowledge_probes import common, run

from effects.domain.provenance import ProvenanceKey
from tests.unit.effects.builders import GEN3_SIDECAR_FOLDERS

AIR_ELEMENTAL = "cardsfolder/a/air_elemental.txt"
KEYWORD = {"script_file": AIR_ELEMENTAL, "face": 0, "trait_kind": "keyword",
           "index_within_kind": 0}


def test_a_pair_keyed_by_a_built_key_of_a_one_line_keyword_is_kept_with_its_lines_text(
    monkeypatch,
) -> None:
    """Since the key change a prowess or ward trigger keys as ``keyword n`` + ``built``,
    which no line declares: the join reads it as the keyword's line."""
    sidecars = common.sidecar_cache(GEN3_SIDECAR_FOLDERS)
    items = common.line_items(GEN3_SIDECAR_FOLDERS, "script")
    acting_text = common.acting_text_function(sidecars, "script")
    built = {**KEYWORD, "built": {"kind": "trigger", "index": 0}}
    text = acting_text(ProvenanceKey.from_dict(KEYWORD))
    seen: list = []

    def keep(data, rungs):
        seen.append(data)
        return {"rungs": {}, "n": len(data.y)}

    monkeypatch.setattr(run.ladder, "run_ladder", keep)
    for item in items:
        item.e = np.zeros(4, dtype=np.float32)
    pairs = [{"a": built, "b": KEYWORD, "label": i % 2} for i in range(run.MIN_LADDER_ITEMS)]
    out = run.pair_ladder(pairs, SimpleNamespace(e_dim=4), (items, acting_text))
    assert out["strata"]["all"]["n"] == run.MIN_LADDER_ITEMS
    assert text and any(item.text == text for item in items)

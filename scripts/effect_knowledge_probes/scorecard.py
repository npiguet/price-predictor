"""One checkpoint's knowledge-probe results, as ``scorecard.json`` holds them (FR-088).

``run.py`` fills one per probing run and writes it beside the tables;
``compare.py`` reads several back. The ladder results inside ``families`` stay
the nested dicts ``ladder.run_ladder`` returns: their shape is the ladder's,
and the scorecard only files them by family and target.

Nothing here does work at import time.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: Sections a run fills only when it ran them, absent from the JSON otherwise.
_OPTIONAL = ("sweeps", "ablation", "board_abilities", "method_c")


@dataclass
class Scorecard:
    checkpoint: str
    probe_set_digest: str
    corpus_digest: str
    date: str
    e_dim: int
    surface: str
    join_rate: float | None
    #: ``family -> {"targets": {target: {"scope", "kind", "strata": {...}}}}``.
    families: dict[str, dict]
    sweeps: dict | None = None
    ablation: dict | None = None
    board_abilities: dict | None = None
    method_c: dict | None = None
    #: ``{"wall_seconds", "peak_gpu_bytes"}``, so the budget is checked, not assumed.
    runtime: dict = field(default_factory=dict)

    def target(self, family: str, name: str, *, scope: str, kind: str) -> dict:
        """The entry one target's results are filed under, created on first use."""
        return self.families[family]["targets"].setdefault(
            name, {"scope": scope, "kind": kind, "strata": {}})

    def put(self, family: str, name: str, entry: dict) -> None:
        self.families[family]["targets"][name] = entry

    def to_json(self) -> dict:
        out = {
            "checkpoint": self.checkpoint, "probe_set_digest": self.probe_set_digest,
            "corpus_digest": self.corpus_digest, "date": self.date,
            "e_dim": self.e_dim, "surface": self.surface, "join_rate": self.join_rate,
            "families": self.families,
        }
        for name in _OPTIONAL:
            value = getattr(self, name)
            if value is not None:
                out[name] = value
        out["runtime"] = self.runtime
        return out

    @classmethod
    def from_json(cls, data: dict) -> Scorecard:
        return cls(
            checkpoint=data.get("checkpoint", ""),
            probe_set_digest=data.get("probe_set_digest", ""),
            corpus_digest=data.get("corpus_digest", ""), date=data.get("date", ""),
            e_dim=data.get("e_dim", 0), surface=data.get("surface", ""),
            join_rate=data.get("join_rate"), families=data.get("families", {}),
            **{name: data.get(name) for name in _OPTIONAL},
            runtime=data.get("runtime", {}),
        )

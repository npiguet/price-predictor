"""The frozen probe set: enumerated once per curated corpus, read by every run (FR-076).

``run.py --freeze-probe-set`` writes it; every probing run, and every one-off
probe that reads its items, reads it back through :func:`load_probe_set`,
which refuses a set frozen against another corpus. Its ``digest`` is the
sha256 of its canonical JSON (sorted keys, no whitespace) with ``digest``
itself left out, and every scorecard records it.

Nothing here does work at import time.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path


def canonical_digest(payload: dict) -> str:
    """sha256 of the canonical JSON (sorted keys, no whitespace), ``digest`` excluded."""
    body = {k: v for k, v in payload.items() if k != "digest"}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def probe_set_path(directory: Path, corpus_digest: str) -> Path:
    """Where the set frozen against the corpus of ``corpus_digest`` lives."""
    return Path(directory) / f"knowledge-probes-set-{corpus_digest[:12]}.json"


def load_probe_set(path: Path, corpus_digest: str) -> ProbeSet:
    """The frozen set for this corpus; refuses a missing or mismatched one.

    The digest is checked over what :class:`ProbeSet` reads back, so a field it
    would drop fails here rather than vanishing from the next set written.
    """
    if not path.exists():
        raise SystemExit(
            f"no frozen probe set at {path}: run with --freeze-probe-set first "
            "(once per curated corpus)")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("corpus_digest") != corpus_digest:
        raise SystemExit(
            f"{path} was frozen against corpus digest "
            f"{data.get('corpus_digest')}, not this corpus's {corpus_digest}")
    probe_set = ProbeSet.from_json(data)
    if data.get("digest") != canonical_digest(probe_set.to_json()):
        raise SystemExit(f"{path} does not match its own digest; it was edited")
    return probe_set


# ── the set's items ─────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class FrozenLineItem:
    """One ability line's labels, keyed by the provenance of its first carrier.

    ``weights`` holds a per-label weight where a label is an estimate over few
    records (the state-dependence spread); absent from the JSON when empty.
    """

    key: dict
    labels: dict[str, float]
    weights: dict[str, float] = field(default_factory=dict)

    def to_json(self) -> dict:
        out = {"key": self.key, "labels": self.labels}
        if self.weights:
            out["weights"] = self.weights
        return out

    @classmethod
    def from_json(cls, data: dict) -> FrozenLineItem:
        return cls(key=data["key"], labels=data["labels"], weights=data.get("weights", {}))

    def provenance_key(self):
        from effects.domain.provenance import ProvenanceKey

        return ProvenanceKey.from_dict(self.key)


@dataclass(frozen=True, slots=True)
class FrozenRecordItem:
    """One board-dependent item: a record of a probe game, its target and label."""

    record_id: str
    stratum: str
    family: str
    target: str
    label: float
    entity: str | None = None
    #: A trigger's cause, as the acting key's dict; filled in once the join ran.
    partner: dict | None = None
    partner_entity: str | None = None
    group: str = ""

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict) -> FrozenRecordItem:
        return cls(**{f.name: data[f.name] for f in fields(cls)})

    def record_item(self):
        """The item as ``labels.RecordItem``, its partner a provenance key again."""
        from effect_knowledge_probes.labels import RecordItem
        from effects.domain.provenance import ProvenanceKey

        partner = None if self.partner is None else ProvenanceKey.from_dict(self.partner)
        return RecordItem(
            record_id=self.record_id, target=self.target, label=self.label,
            entity=self.entity, partner=partner, partner_entity=self.partner_entity,
            group=self.group,
        )


@dataclass
class ProbeSet:
    """Everything one corpus's probing runs read, frozen with its digest."""

    corpus: str
    corpus_digest: str
    surface: str
    #: ``stratum -> probe game ids``, and the shards they were read from.
    games: dict[str, list[str]]
    sources: dict[str, list[str]]
    items: list[FrozenLineItem]
    #: ``{a, b, label, source}``: a trigger line's key, its cause's, and
    #: whether the cause fired it.
    pairs: list[dict]
    record_items: list[FrozenRecordItem]
    #: ``sweep -> [{record_id, ...}]``, the records each board sweep edits.
    sweeps: dict[str, list[dict]]
    join_rate: float
    join: dict[str, int]
    settings: dict
    digest: str = ""

    def to_json(self) -> dict:
        """The set as it is written; ``digest`` only once it has one."""
        out = {f.name: getattr(self, f.name) for f in fields(self)}
        out["items"] = [item.to_json() for item in self.items]
        out["record_items"] = [item.to_json() for item in self.record_items]
        if not self.digest:
            del out["digest"]
        return out

    @classmethod
    def from_json(cls, data: dict) -> ProbeSet:
        plain = {f.name: data[f.name] for f in fields(cls) if f.name != "digest"}
        return cls(**{
            **plain, "digest": data.get("digest", ""),
            "items": [FrozenLineItem.from_json(item) for item in data["items"]],
            "record_items": [FrozenRecordItem.from_json(item) for item in data["record_items"]],
        })

    def sealed(self) -> ProbeSet:
        """The set with its digest computed over everything else."""
        return replace(self, digest=canonical_digest(self.to_json()))

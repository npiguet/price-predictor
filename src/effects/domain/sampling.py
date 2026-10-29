"""The eight sampling classes, and how a shard's records are drawn from them.

Within a class, records are drawn by a weighted shuffle without replacement
over the resident shard (FR-086), weighted by how rarely their acting text was
seen. Torch-free, so curation workers can classify a record without the model.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from effects.domain.records import EffectRecord, Moment, PlayabilitySubkind, RecordKind


class SamplingClass(StrEnum):
    """The granularity of the corpus mixture and the field losses (FR-085).

    Declared in report order.
    """

    RESOLUTION_EFFECT = "resolution-effect"
    COMBAT = "combat"
    CONTINUOUS = "continuous"
    PLAYABILITY_DECISION = "playability-decision"
    RESOLUTION_COST = "resolution-cost"
    TRIGGER = "trigger"
    REWRITE = "rewrite"
    PLAYABILITY_LEGALITY = "playability-legality"


SAMPLING_CLASSES: tuple[SamplingClass, ...] = tuple(SamplingClass)


#: The default ``build-corpus --class-mix``: the proportions the training
#: stratum is written in.
DEFAULT_KIND_MIX: dict[SamplingClass, float] = {
    SamplingClass.RESOLUTION_EFFECT: 0.30,
    SamplingClass.COMBAT: 0.20,
    SamplingClass.CONTINUOUS: 0.12,
    SamplingClass.PLAYABILITY_DECISION: 0.10,
    SamplingClass.RESOLUTION_COST: 0.08,
    SamplingClass.TRIGGER: 0.08,
    SamplingClass.REWRITE: 0.07,
    SamplingClass.PLAYABILITY_LEGALITY: 0.05,
}


#: Classes whose records have no acting ability text, so rarity weighting has
#: nothing to key on and they sample uniformly instead (FR-087).
UNIFORM_CLASSES: frozenset[SamplingClass] = frozenset({
    SamplingClass.COMBAT, SamplingClass.PLAYABILITY_LEGALITY,
})


#: Rarity weight ceiling, as a multiple of the weight of the text at
#: :data:`RARITY_PERCENTILE` of effective games (FR-054).
RARITY_CAP = 20.0
#: The percentile of effective games the ceiling is set against; not the
#: maximum, which one ubiquitous text (``Flying``) would set alone.
RARITY_PERCENTILE = 99.0


def sampling_class(record: EffectRecord) -> SamplingClass:
    """The mixture class a record belongs to."""
    match record.kind:
        case RecordKind.RESOLUTION:
            return (
                SamplingClass.RESOLUTION_COST
                if record.moment is Moment.ACTIVATION
                else SamplingClass.RESOLUTION_EFFECT
            )
        case RecordKind.COMBAT:
            return SamplingClass.COMBAT
        case RecordKind.CONTINUOUS:
            return SamplingClass.CONTINUOUS
        case RecordKind.TRIGGER:
            return SamplingClass.TRIGGER
        case RecordKind.REWRITE:
            return SamplingClass.REWRITE
        case RecordKind.PLAYABILITY:
            return (
                SamplingClass.PLAYABILITY_DECISION
                if record.subkind is PlayabilitySubkind.DECISION
                else SamplingClass.PLAYABILITY_LEGALITY
            )
    raise ValueError(f"no sampling class for record kind {record.kind}")


def parse_kind_mix(value: str | None) -> dict[SamplingClass, float]:
    """``build-corpus --class-mix`` as ``class=share,…``; the default when absent."""
    if not value:
        return dict(DEFAULT_KIND_MIX)
    mix: dict[SamplingClass, float] = {}
    for part in value.split(","):
        if not part.strip():
            continue
        name, _, share = part.partition("=")
        name = name.strip()
        if name not in SamplingClass:
            raise ValueError(
                f"unknown sampling class {name!r}; "
                f"known: {[klass.value for klass in SamplingClass]}"
            )
        mix[SamplingClass(name)] = float(share)
    return mix


def effective_games(
    records: Iterable[EffectRecord], text_of,
) -> dict[str, int]:
    """Distinct games contributing a record of each unique ability text.

    Games rather than records: one long game can produce hundreds of records
    of one ability, all on one board.
    """
    games: defaultdict[str, set[str]] = defaultdict(set)
    for record in records:
        text = text_of(record)
        if text is not None:
            games[text].add(record.game_id)
    return {text: len(ids) for text, ids in games.items()}


def _percentile(values: Sequence[float], percentile: float) -> float:
    """Nearest-rank percentile of a non-empty sequence."""
    ordered = sorted(values)
    rank = max(1, -(-len(ordered) * percentile // 100))
    return ordered[min(int(rank), len(ordered)) - 1]


def rarity_weights(
    games_per_text: dict[str, int], *, cap: float = RARITY_CAP,
    percentile: float = RARITY_PERCENTILE,
) -> dict[str, float]:
    """``effective_games ** -0.5``, capped at ``cap`` times the p99 text's weight.

    The cap stops a single-game ability from dominating a batch (FR-054).
    """
    if not games_per_text:
        return {}
    raw = {
        text: float(count) ** -0.5 if count > 0 else 0.0
        for text, count in games_per_text.items()
    }
    counts = [float(count) for count in games_per_text.values() if count > 0]
    if not counts:
        return raw
    ceiling = cap * _percentile(counts, percentile) ** -0.5
    return {text: min(weight, ceiling) for text, weight in raw.items()}


def sample_weights(
    records: Sequence[EffectRecord], text_of, *,
    rarity: Mapping[str, int] | None = None,
    class_of=None,
) -> list[float]:
    """Per-record sampling weight within its class (FR-054).

    A record with no acting text weighs 1 before normalization (FR-087).
    Capped and normalized to mean one within each class, so rarity reorders
    records inside a class without moving mass between classes.

    Args:
        rarity: a corpus-wide ``text -> games`` table, preferred to the shard's
            own count; a text it does not name falls back to the shard's count
            rather than zero, which would dominate every batch.
        class_of: ``record -> class``; absent, the records are one class.
    """
    texts = [text_of(record) for record in records]
    shard_counts = effective_games(records, text_of)
    classes = [
        class_of(record) if class_of is not None else "" for record in records
    ]
    out = [1.0] * len(records)
    for name in set(classes):
        members = [i for i, value in enumerate(classes) if value == name]
        present = {texts[i] for i in members if texts[i] is not None}
        weights = rarity_weights({
            text: (
                shard_counts[text] if rarity is None
                else rarity.get(text, shard_counts[text])
            )
            for text in present
        })
        for i in members:
            if texts[i] is not None:
                out[i] = weights.get(texts[i], 1.0)
        mean = sum(out[i] for i in members) / len(members)
        if mean > 0:
            for i in members:
                out[i] /= mean
    return out


@dataclass(frozen=True, slots=True)
class BatchPlan:
    """Which records a batch draws, grouped by the game they came from.

    For cost only: records from one game share context texts, so one encode
    serves many records.
    """

    by_game: dict[str, list[EffectRecord]] = field(default_factory=dict)

    @property
    def records(self) -> list[EffectRecord]:
        return [r for group in self.by_game.values() for r in group]


def weighted_order(weights: Sequence[float], rng: random.Random) -> list[int]:
    """Indices in a weighted shuffle: each drawn once, heavier ones earlier.

    Efraimidis–Spirakis: key ``u ** (1 / w)`` per item, sorted descending, is a
    sample without replacement in proportion to ``w``.
    """
    keys = [
        rng.random() ** (1.0 / max(float(weight), 1e-9)) for weight in weights
    ]
    return sorted(range(len(keys)), key=keys.__getitem__, reverse=True)


def batches_without_replacement(
    records: Sequence[EffectRecord], weights: Sequence[float], *,
    batch_size: int, rng: random.Random,
) -> Iterator[BatchPlan]:
    """Endless full batches over ``records``: one weighted shuffle per pass.

    A batch smaller than ``batch_size`` is yielded only when the whole shard
    is smaller; otherwise the ragged tail is dropped and the next pass
    starts, so every step trains on a full batch.
    """
    if not records:
        return
    size = min(batch_size, len(records))
    while True:
        order = weighted_order(weights, rng)
        for start in range(0, len(order) - size + 1, size):
            plan: dict[str, list[EffectRecord]] = defaultdict(list)
            for index in order[start:start + size]:
                plan[records[index].game_id].append(records[index])
            yield BatchPlan(dict(plan))

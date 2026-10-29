"""The eight sampling classes, and how a shard's records are drawn from them.

A record's class is the granularity the corpus mixture and the field losses
both speak in (FR-085): a field whose classes are all absent from the corpus
contributes no loss. Within a class, records are drawn by a weighted shuffle
without replacement, weighted by how rarely their acting text was seen.

**Rarity weighting counts games, not records.** One long game can produce
hundreds of records of one ability; weighting by records would make that ability
look well-observed when it has been seen in one board state. The weights feed a
weighted shuffle *without replacement* over the resident shard (FR-086): the
per-batch class draw is gone, because a sixteen-record shard resampled with
replacement was replayed 640 times.

Torch-free, so the curation workers and the record readers can classify a
record without loading the model.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from effects.domain.records import EffectRecord, Moment, PlayabilitySubkind, RecordKind


class SamplingClass(StrEnum):
    """The eight sampling classes, which are also the granularity FR-085 speaks in.

    A field whose classes are all absent from the corpus contributes no loss. A
    string enum, so a class keys the manifest's JSON and ``--class-mix``
    exactly as the bare string always did. Declared in report order.
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


#: The eight-class mixture (root spec § Training), now a ``build-corpus``
#: input: it sets the proportions the training stratum is written in, and the
#: trainer samples whatever the corpus holds rather than re-mixing per batch.
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
#: The percentile of effective games the ceiling is set against. The 99th
#: rather than the maximum: one ubiquitous text (``Flying``) would otherwise
#: set the ceiling for the whole class, and a corpus that gains one more game
#: of it would move every rare text's weight.
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

    Games rather than records: one long game can produce hundreds of records of
    one ability, and weighting by records would call that ability well-observed
    when it has been seen in a single board state.
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

    The exponent flattens the corpus's long tail without erasing it: an ability
    seen in one game gets more weight than one seen in a hundred, but not a
    hundred times more. The cap stops a single-game ability from dominating a
    batch, and is expressed against the text at the ``percentile``-th
    percentile of effective games (FR-054) — a common text, so the cap scales
    with the corpus, but not the single most common one, whose count would set
    every other text's ceiling on its own.
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

    A record with no acting ability text weighs 1 before normalization:
    ``combat`` and ``playability``/``attackers``/``blockers`` sample uniformly,
    because there is no ability text for rarity to key on (FR-087).

    The ceiling is set and the weights normalized **within each sampling
    class**: each class's weights are capped against its own p99 text and
    scaled to a mean of one, so rarity reorders records inside a class without
    moving mass between classes — the class mixture is the corpus's, fixed by
    ``build-corpus --class-mix``.

    Args:
        rarity: a corpus-wide ``text -> games`` table from a curated dataset's
            manifest, read in preference to the resident shard's own count. A
            text the table does not name falls back to the shard's count for
            it rather than to zero, so a shard collected after the dataset was
            built still weights sanely instead of dominating every batch.
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

    Grouping matters for cost, not correctness: records from one game share a
    board, so their context entities share ability texts, and one encode of a
    text then serves many records instead of one.
    """

    by_game: dict[str, list[EffectRecord]] = field(default_factory=dict)

    @property
    def records(self) -> list[EffectRecord]:
        return [r for group in self.by_game.values() for r in group]


def weighted_order(weights: Sequence[float], rng: random.Random) -> list[int]:
    """Indices in a weighted shuffle: each drawn once, heavier ones earlier.

    Efraimidis–Spirakis: key ``u ** (1 / w)`` per item, sorted descending, is a
    sample without replacement in proportion to ``w``. One shuffle per pass is
    what lets a shard be seen whole rather than resampled from with
    replacement, which is how a sixteen-record shard was replayed 640 times.
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

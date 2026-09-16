"""What a curated dataset records about how it was built (FR-143).

The manifest is the dataset's identity. A training run reads its split, its
rarity table and its caps from here rather than deriving them, and a checkpoint
records ``digest()`` so ``evaluate-effect-model`` can refuse a dataset that has
been rebuilt since (FR-147) — a rebuild is a different split, and scoring the
gates against it would score them partly on games the model trained on.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class SourceShard:
    """One raw shard the dataset was built from, by path and size.

    Size rather than a content hash: a shard is append-only and uniquely named,
    so its length is what changes when a collection run extends the corpus, and
    hashing tens of gigabytes to learn that would cost more than the build.
    """

    name: str
    size: int


@dataclass(frozen=True, slots=True)
class ClassCounts:
    """What one sampling class contributed, read against kept.

    ``unique_texts`` is the number no record count can stand in for: it is what
    says whether the per-text cap trimmed the head or flattened the tail
    (FR-145).
    """

    read: int
    kept: int
    dropped_by_cap: int
    unique_texts: int


@dataclass(frozen=True, slots=True)
class CorpusManifest:
    """Every decision ``build-corpus`` made, and what it made them from."""

    seed: int
    #: The encoding surface the rarity table's text keys were built on.
    #: `surface_of(vocab_path)` decides it, and a table built on the other
    #: surface keys every text differently while looking exactly as valid.
    surface: str
    vocab_path: str
    #: The stage-four variant tree the build resolved against, or "" when
    #: none was given. Which trees resolved decides which keys became texts,
    #: and so which texts the rarity table names and the cap reaches.
    variant_scripts: str
    holdout_permille: int
    holdout_max_carriers: int
    text_cap: int
    card_disjoint_text_cap: int
    game_disjoint_target: int
    training_records: int
    class_mix: dict[str, float]
    #: The class proportions the training output actually holds, counted from
    #: what the write pass kept. ``class_mix`` is what the build was asked
    #: for; this is what it delivered, and the two part company whenever a
    #: class runs out or an availability estimate misses. Recorded rather than
    #: left to be recomputed, because a reader comparing a mixture against the
    #: corpus would have to re-read the whole dataset to do it (FR-139,
    #: FR-143).
    delivered_mix: dict[str, float]
    held_out_cards: tuple[str, ...]
    #: The held-out ability texts themselves — what a ``--corpus`` training
    #: run's ``HeldOutCards.texts`` is built from. Distinct from
    #: ``held_out_cards``: the check-holdout guard sizes the card-disjoint
    #: stratum from ``.texts`` alone (FR-088), so a manifest that omitted this
    #: would make every ``--corpus`` run read that stratum as empty.
    held_out_texts: tuple[str, ...]
    card_disjoint_games: tuple[str, ...]
    game_disjoint_games: tuple[str, ...]
    rarity: dict[str, int]
    sources: tuple[SourceShard, ...]
    per_class: dict[str, ClassCounts]
    #: Records written to each output: "training", "card-disjoint",
    #: "game-disjoint", "gate-one", and "dropped-held-out" for held-out games
    #: the card-disjoint cap declined. The per-stratum sibling of
    #: ``per_class``'s per-class breakdown (FR-143, FR-145). "gate-one"
    #: overlaps "card-disjoint" -- its records are written to both -- so these
    #: values do not sum to a record total.
    per_stratum: dict[str, int]
    #: Distinct ability texts in every output in ``OUTPUTS`` (not
    #: "dropped-held-out", which writes nothing). Read together with
    #: ``per_stratum``, this is the pair FR-145 asks an operator to read: a
    #: stratum can hold plenty of records and still have flattened its tail,
    #: and only the text count says so. Not to be confused with
    #: ``ClassCounts.unique_texts``, which is the same idea sliced by
    #: sampling class within the training output alone.
    unique_texts: dict[str, int]
    shortfall: dict[str, int]
    #: Records refused by ``effects.domain.record_quality`` in both passes,
    #: by reason (FR-148). Empty on a manifest written before the rule existed.
    quality_dropped: dict[str, int] = field(default_factory=dict)
    #: Games per top-level source directory under ``--records-dir``
    #: ("depleted", "full-strength", …; "." for shards at the root), and how
    #: many of them name a held-out card. A depleted directory whose held-out
    #: count is not zero is a leak in collection, and this is where it shows
    #: (FR-149).
    games_by_source: dict[str, int] = field(default_factory=dict)
    held_out_games_by_source: dict[str, int] = field(default_factory=dict)
    #: ``--shard-records``: the size output shards were repacked to. 0 on a
    #: manifest from before repacking, whose shards mirror their sources.
    shard_records: int = 0
    #: ``--validation-sample``: records per stratum in ``validation/samples/``.
    validation_sample: int = 0
    #: ``--max-events-per-record`` the quality rule used.
    max_events_per_record: int = 0
    #: A **watch** statistic, not a refusal: records the build kept whose
    #: events the collector could not attribute to a producing clause (FR-148).
    #: Attribution failing does not make the outcome someone else's -- the
    #: bracket collector records what happened inside the ability's own
    #: resolution -- so these records are trained on. A rising share is a
    #: statement about the collector, not about the corpus.
    unattributed_records: int = 0

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["sources"] = [asdict(shard) for shard in self.sources]
        data["per_class"] = {
            name: asdict(counts) for name, counts in self.per_class.items()
        }
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CorpusManifest:
        return cls(
            seed=int(data["seed"]),
            surface=str(data["surface"]),
            vocab_path=str(data["vocab_path"]),
            variant_scripts=str(data.get("variant_scripts", "")),
            holdout_permille=int(data["holdout_permille"]),
            holdout_max_carriers=int(data["holdout_max_carriers"]),
            text_cap=int(data["text_cap"]),
            card_disjoint_text_cap=int(data["card_disjoint_text_cap"]),
            game_disjoint_target=int(data["game_disjoint_target"]),
            training_records=int(data["training_records"]),
            class_mix={k: float(v) for k, v in data["class_mix"].items()},
            delivered_mix={k: float(v) for k, v in data["delivered_mix"].items()},
            held_out_cards=tuple(data["held_out_cards"]),
            held_out_texts=tuple(data["held_out_texts"]),
            card_disjoint_games=tuple(data["card_disjoint_games"]),
            game_disjoint_games=tuple(data["game_disjoint_games"]),
            rarity={k: int(v) for k, v in data["rarity"].items()},
            sources=tuple(
                SourceShard(name=s["name"], size=int(s["size"]))
                for s in data["sources"]
            ),
            per_class={
                name: ClassCounts(**{k: int(v) for k, v in counts.items()})
                for name, counts in data["per_class"].items()
            },
            per_stratum={k: int(v) for k, v in data["per_stratum"].items()},
            unique_texts={k: int(v) for k, v in data["unique_texts"].items()},
            shortfall={k: int(v) for k, v in data["shortfall"].items()},
            quality_dropped={k: int(v) for k, v in data.get("quality_dropped", {}).items()},
            games_by_source={k: int(v) for k, v in data.get("games_by_source", {}).items()},
            held_out_games_by_source={
                k: int(v) for k, v in data.get("held_out_games_by_source", {}).items()
            },
            shard_records=int(data.get("shard_records", 0)),
            validation_sample=int(data.get("validation_sample", 0)),
            max_events_per_record=int(data.get("max_events_per_record", 0)),
            unattributed_records=int(data.get("unattributed_records", 0)),
        )

    def digest(self) -> str:
        """A stable hash over every decision, insensitive to listing order.

        Sorted keys and a sorted source list, so two builds of the same dataset
        agree whatever order the filesystem enumerated shards in.
        """
        payload = self.as_dict()
        payload["sources"] = sorted(payload["sources"], key=lambda s: s["name"])
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.blake2b(canonical.encode("utf-8"), digest_size=16).hexdigest()

    def drift(
        self, current: tuple[SourceShard, ...],
    ) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
        """Shards added, removed and resized since the build, each sorted.

        What ``--verify`` reports (FR-144). A resized shard is one a collection
        run appended to after this dataset read it.
        """
        was = {shard.name: shard.size for shard in self.sources}
        now = {shard.name: shard.size for shard in current}
        added = tuple(sorted(set(now) - set(was)))
        removed = tuple(sorted(set(was) - set(now)))
        resized = tuple(
            sorted(name for name in set(was) & set(now) if was[name] != now[name])
        )
        return added, removed, resized

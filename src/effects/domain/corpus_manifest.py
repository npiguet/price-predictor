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
from dataclasses import asdict, dataclass
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
    holdout_permille: int
    holdout_max_carriers: int
    text_cap: int
    card_disjoint_text_cap: int
    game_disjoint_target: int
    training_records: int
    class_mix: dict[str, float]
    held_out_cards: tuple[str, ...]
    card_disjoint_games: tuple[str, ...]
    game_disjoint_games: tuple[str, ...]
    rarity: dict[str, int]
    sources: tuple[SourceShard, ...]
    per_class: dict[str, ClassCounts]
    shortfall: dict[str, int]

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
            holdout_permille=int(data["holdout_permille"]),
            holdout_max_carriers=int(data["holdout_max_carriers"]),
            text_cap=int(data["text_cap"]),
            card_disjoint_text_cap=int(data["card_disjoint_text_cap"]),
            game_disjoint_target=int(data["game_disjoint_target"]),
            training_records=int(data["training_records"]),
            class_mix={k: float(v) for k, v in data["class_mix"].items()},
            held_out_cards=tuple(data["held_out_cards"]),
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
            shortfall={k: int(v) for k, v in data["shortfall"].items()},
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

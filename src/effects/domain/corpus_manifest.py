"""What a curated dataset records about how it was built (FR-143).

The manifest is the dataset's identity. A training run reads its split, its
rarity table and its caps from here rather than deriving them, and a checkpoint
records ``digest()`` so ``evaluate-effect-model`` can refuse a dataset that has
been rebuilt since (FR-147) — a rebuild is a different split, and scoring the
gates against it would score them partly on games the model trained on.
"""

from __future__ import annotations

from dataclasses import MISSING, Field, asdict, dataclass, field, fields
from typing import Any

from effects.domain.canonical_digest import canonical_digest
from effects.domain.text_holdout import HoldoutUnit


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


def _optional_float(value) -> float | None:
    return None if value is None else float(value)


def _optional_int(value) -> int | None:
    return None if value is None else int(value)


def _float_map(values: dict) -> dict[str, float]:
    return {k: float(v) for k, v in values.items()}


def _int_map(values: dict) -> dict[str, int]:
    return {k: int(v) for k, v in values.items()}


def _nested_int_map(values: dict) -> dict[str, dict[str, int]]:
    return {k: _int_map(inner) for k, inner in values.items()}


def _doubly_nested_int_map(values: dict) -> dict[str, dict[str, dict[str, int]]]:
    return {k: _nested_int_map(inner) for k, inner in values.items()}


def _sources(values: list) -> tuple[SourceShard, ...]:
    return tuple(SourceShard(name=s["name"], size=int(s["size"])) for s in values)


def _class_counts(values: dict) -> dict[str, ClassCounts]:
    return {name: ClassCounts(**_int_map(counts)) for name, counts in values.items()}


#: Field metadata: how ``from_dict`` reads the field's JSON value back.
_READ = "read"
#: Field metadata: the value a required field reads as when a manifest lacks it.
_ABSENT = "absent"
#: Field metadata marking a field ``as_dict`` leaves out while it holds its
#: default: every gen-2 field, and the one gen-1 field gen-2 stopped writing.
_OMITTED_AT_DEFAULT = "omitted_at_default"


def _reads(read, **extra) -> dict[str, Any]:
    """A field's metadata: its reader, plus any of the flags above."""
    return {_READ: read, **extra}


def _gen1_implied(read) -> dict[str, Any]:
    return _reads(read, **{_OMITTED_AT_DEFAULT: True})


def _default_of(spec: Field) -> Any:
    """The value a manifest lacking ``spec`` reads as; KeyError if it has none."""
    if _ABSENT in spec.metadata:
        return spec.metadata[_ABSENT]
    if spec.default is not MISSING:
        return spec.default
    if spec.default_factory is not MISSING:
        return spec.default_factory()
    raise KeyError(spec.name)


@dataclass(frozen=True, slots=True)
class CorpusManifest:
    """Every decision ``build-corpus`` made, and what it made them from."""

    seed: int = field(metadata=_reads(int))
    #: The encoding surface the rarity table's text keys were built on.
    #: `surface_of(vocab_path)` decides it, and a table built on the other
    #: surface keys every text differently while looking exactly as valid.
    surface: str = field(metadata=_reads(str))
    vocab_path: str = field(metadata=_reads(str))
    #: The stage-four variant tree the build resolved against, or "" when
    #: none was given. Which trees resolved decides which keys became texts,
    #: and so which texts the rarity table names and the cap reaches.
    variant_scripts: str = field(metadata=_reads(str, absent=""))
    holdout_permille: int = field(metadata=_reads(int))
    holdout_max_carriers: int = field(metadata=_reads(int))
    text_cap: int = field(metadata=_reads(int))
    card_disjoint_text_cap: int = field(metadata=_reads(int))
    training_records: int = field(metadata=_reads(int))
    class_mix: dict[str, float] = field(metadata=_reads(_float_map))
    #: The class proportions the training output actually holds, counted from
    #: what the write pass kept. ``class_mix`` is what the build was asked
    #: for; this is what it delivered, and the two part company whenever a
    #: class runs out or an availability estimate misses. Recorded rather than
    #: left to be recomputed, because a reader comparing a mixture against the
    #: corpus would have to re-read the whole dataset to do it (FR-139,
    #: FR-143).
    delivered_mix: dict[str, float] = field(metadata=_reads(_float_map))
    held_out_cards: tuple[str, ...] = field(metadata=_reads(tuple))
    #: The held-out ability texts themselves — what a ``--corpus`` training
    #: run's ``HeldOutCards.texts`` is built from. Distinct from
    #: ``held_out_cards``: the check-holdout guard sizes the card-disjoint
    #: stratum from ``.texts`` alone (FR-088), so a manifest that omitted this
    #: would make every ``--corpus`` run read that stratum as empty.
    held_out_texts: tuple[str, ...] = field(metadata=_reads(tuple))
    card_disjoint_games: tuple[str, ...] = field(metadata=_reads(tuple))
    game_disjoint_games: tuple[str, ...] = field(metadata=_reads(tuple))
    rarity: dict[str, int] = field(metadata=_reads(_int_map))
    sources: tuple[SourceShard, ...] = field(metadata=_reads(_sources))
    per_class: dict[str, ClassCounts] = field(metadata=_reads(_class_counts))
    #: Records written to each output: "training", "card-disjoint",
    #: "game-disjoint", "gate-one", and "dropped-held-out" for held-out games
    #: the card-disjoint cap declined. The per-stratum sibling of
    #: ``per_class``'s per-class breakdown (FR-143, FR-145). "gate-one"
    #: overlaps "card-disjoint" -- its records are written to both -- so these
    #: values do not sum to a record total.
    per_stratum: dict[str, int] = field(metadata=_reads(_int_map))
    #: Distinct ability texts in every output in ``OUTPUTS`` (not
    #: "dropped-held-out", which writes nothing). Read together with
    #: ``per_stratum``, this is the pair FR-145 asks an operator to read: a
    #: stratum can hold plenty of records and still have flattened its tail,
    #: and only the text count says so. Not to be confused with
    #: ``ClassCounts.unique_texts``, which is the same idea sliced by
    #: sampling class within the training output alone.
    unique_texts: dict[str, int] = field(metadata=_reads(_int_map))
    shortfall: dict[str, int] = field(metadata=_reads(_int_map))
    #: Records refused by ``effects.domain.record_quality`` in both passes,
    #: by reason (FR-148). Empty on a manifest written before the rule existed.
    quality_dropped: dict[str, int] = field(
        default_factory=dict, metadata=_reads(_int_map),
    )
    #: Games per top-level source directory under ``--records-dir``
    #: ("depleted", "full-strength", …; "." for shards at the root), and how
    #: many of them name a held-out card. A depleted directory whose held-out
    #: count is not zero is a leak in collection, and this is where it shows
    #: (FR-149).
    games_by_source: dict[str, int] = field(
        default_factory=dict, metadata=_reads(_int_map),
    )
    held_out_games_by_source: dict[str, int] = field(
        default_factory=dict, metadata=_reads(_int_map),
    )
    #: ``--shard-records``: the size output shards were repacked to. 0 on a
    #: manifest from before repacking, whose shards mirror their sources.
    shard_records: int = field(default=0, metadata=_reads(int))
    #: ``--validation-sample``: records per stratum in ``validation/samples/``.
    validation_sample: int = field(default=0, metadata=_reads(int))
    #: ``--max-events-per-record`` the quality rule used.
    max_events_per_record: int = field(default=0, metadata=_reads(int))
    #: A **watch** statistic, not a refusal: records the build kept whose
    #: events the collector could not attribute to a producing clause (FR-148).
    #: Attribution failing does not make the outcome someone else's -- the
    #: bracket collector records what happened inside the ability's own
    #: resolution -- so these records are trained on. A rising share is a
    #: statement about the collector, not about the corpus.
    unattributed_records: int = field(default=0, metadata=_reads(int))
    #: Legacy, with ``token_keys_ambiguous`` and ``forge_tokenscripts``: the
    #: retired token-key remap's tallies (FR-151), kept because manifests on
    #: disk carry them and the digest hashes them; always written empty now.
    token_keys_remapped: int = field(default=0, metadata=_reads(int))
    token_keys_ambiguous: dict[str, int] = field(
        default_factory=dict, metadata=_reads(_int_map),
    )
    #: Script file -> resolution records refused because every acting key
    #: mapped to no rendered line (FR-148), largest 100 only. The refusal is
    #: dominated by Forge's implicit permanent spell, which every permanent
    #: carries, so the untrimmed tally is one entry per permanent script the
    #: corpus ever resolved. Read by an operator, not by code.
    no_acting_text_scripts: dict[str, int] = field(
        default_factory=dict, metadata=_reads(_int_map),
    )
    #: Legacy, as ``token_keys_remapped``: kept for digest stability.
    forge_tokenscripts: str = field(default="", metadata=_reads(str))
    #: ``--game-disjoint-games``, the sampled-count rule gen-2 replaced with a
    #: per-game threshold (FR-044). No longer written; a manifest that carries
    #: it is still read, and keeps its digest.
    game_disjoint_target: int | None = field(
        default=None, metadata=_gen1_implied(_optional_int),
    )

    # ── gen-2 (feature 024). Each is omitted from ``as_dict`` while it holds
    # the value a gen-1 manifest implies, so a gen-1 manifest re-serializes to
    # the same digest and every checkpoint pinned to it stays evaluable. ──

    #: ``template`` or ``text`` (FR-042); absent reads as ``text``.
    holdout_unit: HoldoutUnit = field(
        default=HoldoutUnit.TEXT, metadata=_gen1_implied(HoldoutUnit),
    )
    #: ``class -> family -> {available, share, written, repeats, shortfall}``.
    families: dict[str, dict[str, dict[str, int]]] = field(
        default_factory=dict, metadata=_gen1_implied(_doubly_nested_int_map),
    )
    #: ``class -> family -> {outcome signature: records written}``.
    signatures: dict[str, dict[str, dict[str, int]]] = field(
        default_factory=dict, metadata=_gen1_implied(_doubly_nested_int_map),
    )
    #: ``class -> {on_policy, off_policy}`` training records, by ``random_seat``.
    policy_counts: dict[str, dict[str, int]] = field(
        default_factory=dict, metadata=_gen1_implied(_nested_int_map),
    )
    #: ``legality subkind -> {real, what_if, unknown}`` training records.
    legality_counts: dict[str, dict[str, int]] = field(
        default_factory=dict, metadata=_gen1_implied(_nested_int_map),
    )
    #: Surveyed records per legality mode. A legality record's family is its
    #: rarest mode, so every reader placing records in families — the build's
    #: write pass, the trainer's epoch line, the evaluator — reads it here.
    legality_mode_counts: dict[str, int] = field(
        default_factory=dict, metadata=_gen1_implied(_int_map),
    )
    #: Game-disjoint games placed only under ``--game-disjoint-keyword-share``.
    keyword_threshold_games: tuple[str, ...] = field(
        default=(), metadata=_gen1_implied(tuple),
    )
    #: Held-out texts with no gate-one resolution record.
    held_out_texts_without_resolution: tuple[str, ...] = field(
        default=(), metadata=_gen1_implied(tuple),
    )
    #: ``held-out text -> games`` for the texts recorded in fewer than five.
    held_out_texts_under_five_games: dict[str, int] = field(
        default_factory=dict, metadata=_gen1_implied(_int_map),
    )
    #: The game-disjoint and reuse flags as run; None on a gen-1 manifest.
    game_disjoint_share: float | None = field(
        default=None, metadata=_gen1_implied(_optional_float),
    )
    game_disjoint_keyword_share: float | None = field(
        default=None, metadata=_gen1_implied(_optional_float),
    )
    game_disjoint_keywords: tuple[str, ...] = field(
        default=(), metadata=_gen1_implied(tuple),
    )
    reuse_cap: int | None = field(default=None, metadata=_gen1_implied(_optional_int))
    #: Training records the plan asked of each class, copies included, read
    #: against ``per_class[...].kept``: a class delivered under its target is
    #: a plan that counted records the write pass could not write.
    class_targets: dict[str, int] = field(
        default_factory=dict, metadata=_gen1_implied(_int_map),
    )

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["sources"] = [asdict(shard) for shard in self.sources]
        data["per_class"] = {
            name: asdict(counts) for name, counts in self.per_class.items()
        }
        for name, default in _GEN1_DEFAULTS.items():
            if getattr(self, name) == default:
                data.pop(name, None)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CorpusManifest:
        """Each field through its own reader, an older manifest's gaps at default.

        A required field missing from ``data`` raises ``KeyError`` unless its
        metadata names the value an absent one reads as.
        """
        values = {}
        for spec in fields(cls):
            raw = data[spec.name] if spec.name in data else _default_of(spec)
            values[spec.name] = spec.metadata[_READ](raw)
        return cls(**values)

    #: Fields the digest leaves out: operator-facing tallies that say nothing
    #: about which records the dataset holds. The digest is what FR-147 pins a
    #: checkpoint to, so adding a field to this dataclass must not move it --
    #: a manifest written before the field existed has to keep hashing the way
    #: it did, or every checkpoint trained against it stops being evaluable.
    _OUTSIDE_THE_DIGEST = ("no_acting_text_scripts",)

    def digest(self) -> str:
        """A stable hash over every decision, insensitive to listing order.

        Sorted keys and a sorted source list, so two builds of the same dataset
        agree whatever order the filesystem enumerated shards in.
        ``_OUTSIDE_THE_DIGEST`` names the fields that are reported rather than
        decided, and they are dropped here rather than never recorded.
        """
        payload = self.as_dict()
        for field_name in self._OUTSIDE_THE_DIGEST:
            payload.pop(field_name, None)
        payload["sources"] = sorted(payload["sources"], key=lambda s: s["name"])
        return canonical_digest(payload).hex()

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


_FIELDS = {spec.name: spec for spec in fields(CorpusManifest)}

#: Every field ``as_dict`` leaves out, and the value at which it does.
_GEN1_DEFAULTS = {
    name: _default_of(spec) for name, spec in _FIELDS.items()
    if spec.metadata.get(_OMITTED_AT_DEFAULT)
}

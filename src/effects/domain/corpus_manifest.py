"""What a curated dataset records about how it was built (FR-143).

A training run reads its split, rarity table and caps from here rather than
deriving them, and a checkpoint records ``digest()`` so ``evaluate-effect-model``
can refuse a dataset rebuilt since (FR-147).
"""

from __future__ import annotations

from dataclasses import MISSING, Field, asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, NamedTuple

from effects.domain.ability_tokenizer import Surface
from effects.domain.canonical_digest import canonical_digest
from effects.domain.split import HoldoutRule
from effects.domain.text_holdout import HoldoutUnit
from price_predictor.domain.dataclass_defaults import field_default


class ManifestDrift(NamedTuple):
    """The source shards that changed since a build, each sorted by name."""

    added: tuple[str, ...]
    removed: tuple[str, ...]
    resized: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SourceShard:
    """One raw shard the dataset was built from, by path and size: a shard is
    append-only and uniquely named, so its length is what changes, and hashing
    tens of gigabytes would cost more than the build."""

    name: str
    size: int


@dataclass(frozen=True, slots=True)
class ClassCounts:
    """What one sampling class contributed. ``unique_texts`` says whether the
    per-text cap trimmed the head or flattened the tail (FR-145)."""

    read: int
    kept: int
    dropped_by_cap: int
    unique_texts: int


@dataclass(frozen=True, slots=True)
class FamilyRow:
    """One rule family's row of ``families`` (FR-053): what its class budget
    gave it and what the training output holds. ``share`` is the equal split
    before redistribution, so ``shortfall`` is ``max(0, share - written)``."""

    available: int
    share: int
    planned: int
    #: Copies included; ``repeats`` is how many of them are copies.
    written: int
    repeats: int
    shortfall: int


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


def _reads(
    read, *, absent: Any = MISSING, omitted_at_default: bool = False,
) -> dict[str, Any]:
    """A field's metadata: its reader, the value an absent one reads as when
    ``absent`` is given, and whether ``as_dict`` omits it at its default."""
    metadata: dict[str, Any] = {_READ: read}
    if absent is not MISSING:
        metadata[_ABSENT] = absent
    if omitted_at_default:
        metadata[_OMITTED_AT_DEFAULT] = True
    return metadata


def _gen1_implied(read) -> dict[str, Any]:
    return _reads(read, omitted_at_default=True)


def _default_of(spec: Field) -> Any:
    """The value a manifest lacking ``spec`` reads as; KeyError if it has none."""
    if _ABSENT in spec.metadata:
        return spec.metadata[_ABSENT]
    try:
        return field_default(spec)
    except ValueError:
        raise KeyError(spec.name) from None


@dataclass(frozen=True, slots=True)
class CorpusManifest:
    """Every decision ``build-corpus`` made, and what it made them from."""

    seed: int = field(metadata=_reads(int))
    #: The encoding surface (``surface_of(vocab_path)``) the rarity table's
    #: text keys were built on; the other surface keys every text differently.
    surface: Surface = field(metadata=_reads(Surface))
    vocab_path: str = field(metadata=_reads(str))
    #: The variant tree the build resolved keys against, or "" when none.
    variant_scripts: str = field(metadata=_reads(str, absent=""))
    holdout_permille: int = field(metadata=_reads(int))
    holdout_max_carriers: int = field(metadata=_reads(int))
    text_cap: int = field(metadata=_reads(int))
    card_disjoint_text_cap: int = field(metadata=_reads(int))
    training_records: int = field(metadata=_reads(int))
    class_mix: dict[str, float] = field(metadata=_reads(_float_map))
    #: The class proportions the training output holds, beside ``class_mix``,
    #: the ones requested (FR-139, FR-143).
    delivered_mix: dict[str, float] = field(metadata=_reads(_float_map))
    held_out_cards: tuple[str, ...] = field(metadata=_reads(tuple))
    #: The held-out ability texts a ``--corpus`` run builds
    #: ``HeldOutCards.texts`` from, which sizes the card-disjoint stratum
    #: (FR-088).
    held_out_texts: tuple[str, ...] = field(metadata=_reads(tuple))
    card_disjoint_games: tuple[str, ...] = field(metadata=_reads(tuple))
    game_disjoint_games: tuple[str, ...] = field(metadata=_reads(tuple))
    rarity: dict[str, int] = field(metadata=_reads(_int_map))
    sources: tuple[SourceShard, ...] = field(metadata=_reads(_sources))
    per_class: dict[str, ClassCounts] = field(metadata=_reads(_class_counts))
    #: Records per :class:`~effects.domain.strata.Stratum` (FR-143, FR-145).
    #: "gate-one" overlaps "card-disjoint" and the dropped strata are never
    #: written, so the values do not sum to a record total.
    per_stratum: dict[str, int] = field(metadata=_reads(_int_map))
    #: Distinct ability texts per written stratum (``OUTPUT_STRATA``), read
    #: beside ``per_stratum``: only the text count shows a flattened tail.
    unique_texts: dict[str, int] = field(metadata=_reads(_int_map))
    shortfall: dict[str, int] = field(metadata=_reads(_int_map))
    #: Records refused by ``effects.domain.record_quality`` in both passes,
    #: by reason (FR-148). Empty on a manifest written before the rule existed.
    quality_dropped: dict[str, int] = field(
        default_factory=dict, metadata=_reads(_int_map),
    )
    #: Games per top-level source directory under ``--records-dir`` ("." for
    #: the root), and how many name a held-out card: a nonzero count under a
    #: depleted directory is a collection leak (FR-149).
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
    #: Watched, not refused: kept records whose events the collector could not
    #: attribute to a clause (FR-148). They are still the ability's own
    #: outcomes, so they train.
    unattributed_records: int = field(default=0, metadata=_reads(int))
    #: Legacy, with ``token_keys_ambiguous`` and ``forge_tokenscripts``: the
    #: retired token-key remap's tallies (FR-151), kept because manifests on
    #: disk carry them and the digest hashes them; always written empty now.
    token_keys_remapped: int = field(default=0, metadata=_reads(int))
    token_keys_ambiguous: dict[str, int] = field(
        default_factory=dict, metadata=_reads(_int_map),
    )
    #: Script file -> resolution records refused because every acting key
    #: mapped to no rendered line (FR-148), largest 100 only: Forge's implicit
    #: permanent spell would otherwise add an entry per permanent. For an
    #: operator, not code.
    no_acting_text_scripts: dict[str, int] = field(
        default_factory=dict, metadata=_reads(_int_map),
    )
    #: Legacy, as ``token_keys_remapped``: kept for digest stability.
    forge_tokenscripts: str = field(default="", metadata=_reads(str))
    #: Gen-1's ``--game-disjoint-games``, replaced by a per-game threshold
    #: (FR-044). No longer written; a manifest carrying it still reads, and
    #: keeps its digest.
    game_disjoint_target: int | None = field(
        default=None, metadata=_gen1_implied(_optional_int),
    )

    # Gen-2 (feature 024): each is omitted from ``as_dict`` at the value a gen-1
    # manifest implies, so a gen-1 manifest keeps its digest.

    #: ``template`` or ``text`` (FR-042); absent reads as ``text``.
    holdout_unit: HoldoutUnit = field(
        default=HoldoutUnit.TEXT, metadata=_gen1_implied(HoldoutUnit),
    )
    #: ``class -> family ->`` a :class:`FamilyRow`'s fields. Stored as plain
    #: numbers rather than rows: a manifest built before ``planned`` existed
    #: lacks it, and reading one in as a row would change its digest.
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
    #: Surveyed records per legality mode: every reader places a legality
    #: record in its rarest mode's family from this table.
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
    #: against ``per_class[...].kept``.
    class_targets: dict[str, int] = field(
        default_factory=dict, metadata=_gen1_implied(_int_map),
    )

    @property
    def holdout(self) -> HoldoutRule:
        """The rule the held-out texts were chosen by; stored as three fields,
        the form every manifest so far was written in."""
        return HoldoutRule(self.holdout_permille, self.holdout_max_carriers, self.holdout_unit)

    def variant_scripts_or(self, given: Path | None) -> Path | None:
        """The variant-script tree a run over this dataset reads: ``given``,
        else the one the build resolved keys against, else None.

        Training and evaluation both take it, since a run reading no tree sees
        every variant ability the build kept as no text.
        """
        if given is not None:
            return Path(given)
        return Path(self.variant_scripts) if self.variant_scripts else None

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

    #: Operator-facing tallies the digest leaves out, so a manifest written
    #: before one existed keeps the digest its checkpoints pin (FR-147).
    _OUTSIDE_THE_DIGEST = ("no_acting_text_scripts",)

    def digest(self) -> str:
        """A stable hash over every decision: sorted keys and sources, so the
        order the filesystem enumerated shards in cannot move it."""
        payload = self.as_dict()
        for field_name in self._OUTSIDE_THE_DIGEST:
            payload.pop(field_name, None)
        payload["sources"] = sorted(payload["sources"], key=lambda s: s["name"])
        return canonical_digest(payload).hex()

    def drift(self, current: tuple[SourceShard, ...]) -> ManifestDrift:
        """Shards added, removed and resized since the build, each sorted: what
        ``--verify`` reports (FR-144)."""
        was = {shard.name: shard.size for shard in self.sources}
        now = {shard.name: shard.size for shard in current}
        added = tuple(sorted(set(now) - set(was)))
        removed = tuple(sorted(set(was) - set(now)))
        resized = tuple(
            sorted(name for name in set(was) & set(now) if was[name] != now[name])
        )
        return ManifestDrift(added, removed, resized)


_FIELDS = {spec.name: spec for spec in fields(CorpusManifest)}

#: Every field ``as_dict`` leaves out, and the value at which it does.
_GEN1_DEFAULTS = {
    name: _default_of(spec) for name, spec in _FIELDS.items()
    if spec.metadata.get(_OMITTED_AT_DEFAULT)
}

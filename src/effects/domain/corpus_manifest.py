"""What a curated dataset records about how it was built (FR-143).

A training run reads its split, rarity table and caps from here rather than
deriving them, and a checkpoint records ``digest()`` so ``evaluate-effect-model``
can refuse a dataset rebuilt since (FR-147).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import MISSING, Field, asdict, dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

from effects.domain.ability_tokenizer import Surface
from effects.domain.canonical_digest import canonical_digest
from effects.domain.rule_budgets import RuleBudgets, RuleRow
from effects.domain.rule_catalogue import MapVerdict, RuleMapEntry
from effects.domain.split import HoldoutRule
from effects.domain.text_holdout import HoldoutUnit
from price_predictor.domain.dataclass_defaults import field_default

if TYPE_CHECKING:
    from effects.domain.corpus_plan import GameStrata


class ManifestDrift(NamedTuple):
    """The source shards that changed since a build, each sorted by name."""

    added: tuple[str, ...]
    removed: tuple[str, ...]
    resized: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SourceShard:
    """One raw shard the dataset was built from, by path and size: a shard is
    append-only and uniquely named, so its length is what changes, and hashing
    tens of gigabytes would cost more than the build.

    ``name`` is the shard's posix path relative to the corpus root, the one
    spelling the listing, both passes and the manifest share; the directory
    it reports under and the part it writes are read off it here.
    """

    name: str
    size: int

    @property
    def source(self) -> str:
        """The top-level directory the shard sits in, ``"."`` at the root."""
        return self.name.split("/", 1)[0] if "/" in self.name else "."

    @property
    def part_name(self) -> str:
        """A flat, **injective** name for the shard's write-pass part.

        Injective because ``write_shard`` truncates, so a collision is a silent
        overwrite; escaping ``_`` first makes it so, where a plain ``/ -> __``
        would map ``a_/b`` and ``a/_b`` alike.
        """
        return self.name.replace("_", "_u").replace("/", "_s")

    def path(self, root: Path) -> Path:
        """Where the shard is under ``root``, the corpus root it was listed from."""
        return Path(root) / self.name


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
    """One effect family's row of ``families`` (FR-053): what its class budget
    gave it and what the training output holds. ``share`` is the equal split
    before redistribution, so ``shortfall`` is ``max(0, share - written)``."""

    available: int
    share: int
    #: None on a manifest built before the column existed, which leaves it
    #: out of the row again on the way back, so its digest does not move.
    planned: int | None
    #: Copies included; ``repeats`` is how many of them are copies.
    written: int
    repeats: int
    shortfall: int

    def as_dict(self) -> dict[str, int]:
        row = asdict(self)
        if self.planned is None:
            del row["planned"]
        return row

    @classmethod
    def from_dict(cls, values: dict) -> FamilyRow:
        planned = values.get("planned")
        return cls(
            available=int(values["available"]), share=int(values["share"]),
            planned=None if planned is None else int(planned),
            written=int(values["written"]), repeats=int(values["repeats"]),
            shortfall=int(values["shortfall"]),
        )


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


def _family_rows(values: dict) -> dict[str, dict[str, FamilyRow]]:
    return {
        klass: {family: FamilyRow.from_dict(row) for family, row in rows.items()}
        for klass, rows in values.items()
    }


def _tuple_map(values: dict) -> dict[str, tuple[str, ...]]:
    return {k: tuple(inner) for k, inner in values.items()}


def _nested_tuple_map(values: dict) -> dict[str, dict[str, tuple[str, ...]]]:
    return {k: _tuple_map(inner) for k, inner in values.items()}


def _rule_rows(values: dict) -> dict[str, dict[str, dict[str, RuleRow]]]:
    return {
        klass: {rule: {polarity: RuleRow.from_dict(row) for polarity, row in rows.items()}
                for rule, rows in by_rule.items()}
        for klass, by_rule in values.items()
    }


def _rule_map(values: dict) -> dict[str, RuleMapEntry]:
    return {
        rule: RuleMapEntry(rule=rule, verdict=MapVerdict(entry["verdict"]),
                           outputs=tuple(entry["outputs"]))
        for rule, entry in values.items()
    }


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
    #: ``class -> family -> FamilyRow``.
    families: dict[str, dict[str, FamilyRow]] = field(
        default_factory=dict, metadata=_gen1_implied(_family_rows),
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

    # Gen-3 (feature 025, FR-060, FR-060a): written by every gen-3 build and
    # omitted from ``as_dict`` at their defaults like the gen-2 fields, so an
    # older manifest keeps its digest; no gen-3 reader accepts one.

    #: The survey's per-rule counts in ``rule-counts.json``'s shape (FR-008):
    #: ``records``, ``rules`` (rule -> polarity -> source -> records),
    #: ``not_countable``, ``none`` and ``review``.
    rule_counts: dict[str, Any] = field(
        default_factory=dict, metadata=_gen1_implied(dict),
    )
    #: The rules-to-fields map as the build read it (research R4): every
    #: tagged rule's verdict and outputs. Budgets, the rule mask, the freeze
    #: and evaluation read verdicts and outputs here, never ``rules/heads/``.
    rule_map: dict[str, RuleMapEntry] = field(
        default_factory=dict, metadata=_gen1_implied(_rule_map),
    )
    #: The map's content hash, which a checkpoint records.
    rule_map_digest: str = field(default="", metadata=_gen1_implied(str))
    #: ``rule -> outputs``: of each mapped rule's outputs, in the map's
    #: order, those at least one of its tagged records carries
    #: (``rule_output_carriage``; FR-060a). The rule mask and the rule tests
    #: read these, never the map's own list; a rule its records carry no
    #: output of maps to an empty tuple.
    effective_outputs: dict[str, tuple[str, ...]] = field(
        default_factory=dict, metadata=_gen1_implied(_tuple_map),
    )
    #: ``class -> rule -> polarity -> RuleRow`` (FR-060).
    rules: dict[str, dict[str, dict[str, RuleRow]]] = field(
        default_factory=dict, metadata=_gen1_implied(_rule_rows),
    )
    #: ``class -> budgeted rules whose polarity halves stay unequal`` (FR-054a).
    one_sided_rules: dict[str, tuple[str, ...]] = field(
        default_factory=dict, metadata=_gen1_implied(_tuple_map),
    )
    #: ``class -> condition carriers seen on one side only`` (FR-056).
    one_sided_conditions: dict[str, tuple[str, ...]] = field(
        default_factory=dict, metadata=_gen1_implied(_tuple_map),
    )
    #: ``rule -> polarity -> source -> records`` for every tagged rule the map
    #: does not mark ``trained`` (FR-054).
    unbudgeted_rules: dict[str, dict[str, dict[str, int]]] = field(
        default_factory=dict, metadata=_gen1_implied(_doubly_nested_int_map),
    )
    #: ``stratum -> half -> rules`` with both polarities in the inventories
    #: and one missing from that half (research R3).
    left_out_of_half: dict[str, dict[str, tuple[str, ...]]] = field(
        default_factory=dict, metadata=_gen1_implied(_nested_tuple_map),
    )
    #: ``turn_action`` records written to no stratum, by subkind (FR-059).
    turn_action_left_out: dict[str, int] = field(
        default_factory=dict, metadata=_gen1_implied(_int_map),
    )
    #: The closed subtype and keyword lists (FR-052, FR-053), in the head's
    #: column order: ``closed_lists.ordered_subtypes`` and ``ordered_keywords``,
    #: which is what ``output_vocabularies`` holds, so
    #: ``scripts/scan_output_vocabularies.py --manifest`` copies them verbatim.
    subtypes: tuple[str, ...] = field(default=(), metadata=_gen1_implied(tuple))
    keywords: tuple[str, ...] = field(default=(), metadata=_gen1_implied(tuple))
    #: ``stratum -> half -> game ids``, sorted (FR-070).
    halves: dict[str, dict[str, tuple[str, ...]]] = field(
        default_factory=dict, metadata=_gen1_implied(_nested_tuple_map),
    )

    @property
    def budgeted_rules(self) -> frozenset[str]:
        """The rules the recorded map marks ``trained`` (FR-054)."""
        return frozenset(
            rule for rule, entry in self.rule_map.items() if entry.verdict is MapVerdict.TRAINED)

    def rule_outputs(self, rule: str) -> tuple[str, ...]:
        """The outputs the recorded map names for ``rule``; empty for a rule
        it does not hold (FR-060a)."""
        entry = self.rule_map.get(rule)
        return () if entry is None else entry.outputs

    @property
    def effective_rule_outputs(self) -> dict[str, tuple[str, ...]]:
        """``rule -> effective outputs`` for every mapped rule: what the rule
        mask and the rule tests read (FR-060a)."""
        return {rule: self.effective_outputs.get(rule, ()) for rule in self.rule_map}

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
        data["families"] = {
            klass: {family: row.as_dict() for family, row in rows.items()}
            for klass, rows in self.families.items()
        }
        data["rules"] = {
            klass: {rule: {polarity: row.as_dict() for polarity, row in rows.items()}
                    for rule, rows in by_rule.items()}
            for klass, by_rule in self.rules.items()
        }
        data["rule_map"] = {
            rule: {"verdict": str(entry.verdict), "outputs": list(entry.outputs)}
            for rule, entry in self.rule_map.items()
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


def _unbudgeted_rules(
    budgets: RuleBudgets, rule_map: Mapping[str, RuleMapEntry],
) -> dict[str, dict[str, dict[str, int]]]:
    """Every tagged rule the map does not mark ``trained`` (FR-054), with the
    training records the budgets counted for it; a rule tagged only on records
    no class trains on (a validation game's, a ``turn_action`` record's) is
    named with none, so the manifest names every unbudgeted rule."""
    names = set(budgets.unbudgeted) | {
        rule for rule, entry in rule_map.items() if entry.verdict is not MapVerdict.TRAINED
    }
    return {rule: budgets.unbudgeted.get(rule, {}) for rule in sorted(names)}


def rule_curation_fields(
    *,
    budgets: RuleBudgets,
    strata: GameStrata,
    rule_map: Mapping[str, RuleMapEntry],
    rule_map_digest: str,
    rule_counts: Mapping[str, Any],
    carried: Mapping[tuple[str, str], int],
    one_sided_conditions: Mapping[str, Iterable[str]],
    turn_action_left_out: Mapping[str, int],
    subtypes: Iterable[str],
    keywords: Iterable[str],
) -> dict[str, Any]:
    """The gen-3 fields of :class:`CorpusManifest`, as keyword arguments.

    Built from what the build already holds: the rule budgets, the placed
    strata, the map snapshot, the survey's rule counts (the ``rule-counts.json``
    object), its ``(rule, output) -> records carrying it`` tally, which
    decides each rule's effective outputs, its one-sided conditions
    (``corpus_curation.one_sided_conditions``) and turn-action tally, and the
    subtype and keyword names it read, which are put in the head's column
    order here.
    """
    from effects.domain.closed_lists import ordered_keywords, ordered_subtypes

    return {
        "rule_counts": dict(rule_counts),
        "rule_map": dict(sorted(rule_map.items())),
        "rule_map_digest": rule_map_digest,
        "effective_outputs": {
            rule: tuple(o for o in entry.outputs if carried.get((rule, o), 0) > 0)
            for rule, entry in sorted(rule_map.items())
        },
        "rules": budgets.rows,
        "one_sided_rules": dict(budgets.one_sided),
        "one_sided_conditions": {
            klass: tuple(carriers) for klass, carriers in sorted(one_sided_conditions.items())
        },
        "unbudgeted_rules": _unbudgeted_rules(budgets, rule_map),
        "left_out_of_half": {
            str(stratum): {str(half): tuple(rules) for half, rules in by_half.items()}
            for stratum, by_half in strata.left_out_of_half.items()
        },
        "turn_action_left_out": dict(sorted(turn_action_left_out.items())),
        "subtypes": ordered_subtypes(subtypes),
        "keywords": ordered_keywords(keywords),
        "halves": {
            str(stratum): {str(half): tuple(sorted(games)) for half, games in by_half.items()}
            for stratum, by_half in strata.halves.items()
        },
    }

"""What a curated dataset records about how it was built (FR-143).

A training run reads its split, rarity table and caps from here rather than
deriving them, and a checkpoint records ``digest()`` so ``evaluate-effect-model``
can refuse a dataset rebuilt since (FR-147).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

from effects.domain.ability_tokenizer import Surface
from effects.domain.canonical_digest import canonical_digest
from effects.domain.rule_budgets import RuleBudgets, RuleRow
from effects.domain.rule_catalogue import MapVerdict, RuleMapEntry
from effects.domain.split import HoldoutRule

if TYPE_CHECKING:
    from effects.domain.corpus_plan import GameStrata


#: The generation of corpus a manifest describes, written as its ``generation``
#: member: the only one this code reads.
MANIFEST_GENERATION = 3

#: Members only a gen-2 build wrote, which tell a gen-2 manifest from a gen-1
#: one when neither carries a ``generation`` member.
_GEN2_MEMBERS = ("families", "signatures", "policy_counts")


class ManifestError(ValueError):
    """A manifest this code does not read: one of an older generation, or one
    missing a member."""


def _recorded_generation(data: dict[str, Any]) -> str:
    """The generation a manifest's own members name: its ``generation`` member,
    else gen-2 where a gen-2 build's members are present, else gen-1."""
    if "generation" in data:
        return f"gen-{data['generation']}"
    return "gen-2" if any(name in data for name in _GEN2_MEMBERS) else "gen-1"


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
class CuratedShard:
    """One shard the build wrote into a stratum, as the write pass counted it:
    its file name inside the stratum's directory and its records."""

    name: str
    records: int


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
    planned: int
    #: Copies included; ``repeats`` is how many of them are copies.
    written: int
    repeats: int
    shortfall: int

    def as_dict(self) -> dict[str, int]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> FamilyRow:
        return cls(
            available=int(values["available"]), share=int(values["share"]),
            planned=int(values["planned"]), written=int(values["written"]),
            repeats=int(values["repeats"]), shortfall=int(values["shortfall"]),
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


def _curated_shards(values: dict) -> dict[str, tuple[CuratedShard, ...]]:
    return {
        stratum: tuple(CuratedShard(**shard) for shard in shards)
        for stratum, shards in values.items()
    }


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


def _each_rule_row[R, S](
    by_class: Mapping[str, Mapping[str, Mapping[str, R]]], convert: Callable[[R], S],
) -> dict[str, dict[str, dict[str, S]]]:
    """``class -> rule -> polarity -> row``, each row through ``convert``: the
    one walk over a stratum's rule rows, read or written."""
    return {
        klass: {rule: {polarity: convert(row) for polarity, row in rows.items()}
                for rule, rows in by_rule.items()}
        for klass, by_rule in by_class.items()
    }


def _rule_rows(values: dict) -> dict[str, dict[str, dict[str, RuleRow]]]:
    return _each_rule_row(values, RuleRow.from_dict)


def _leave_out_drawn(payload: dict) -> None:
    """Drop each rule row's ``drawn`` from a payload: which rule a copy was
    drawn for reports what the budgets did and decides nothing (``added``,
    inside the digest, holds the copies), so it never moves the digest."""
    strata = [payload.get("rules", {}), *payload.get("validation_rules", {}).values()]
    rows = (
        row for rules in strata for by_rule in rules.values()
        for by_polarity in by_rule.values() for row in by_polarity.values()
    )
    for row in rows:
        row.pop("drawn", None)


def _rule_map(values: dict) -> dict[str, RuleMapEntry]:
    return {
        rule: RuleMapEntry(rule=rule, verdict=MapVerdict(entry["verdict"]),
                           outputs=tuple(entry["outputs"]))
        for rule, entry in values.items()
    }


#: Field metadata: how ``from_dict`` reads the field's JSON value back.
_READ = "read"


def _reads(read) -> dict[str, Any]:
    """A field's metadata: its reader."""
    return {_READ: read}


@dataclass(frozen=True, slots=True)
class CorpusManifest:
    """Every decision ``build-corpus`` made, and what it made them from."""

    seed: int = field(metadata=_reads(int))
    #: The encoding surface (the vocabulary header's) the rarity table's
    #: text keys were built on; the other surface keys every text differently.
    surface: Surface = field(metadata=_reads(Surface))
    vocab_path: str = field(metadata=_reads(str))
    #: The variant tree the build resolved keys against, or "" when none.
    variant_scripts: str = field(metadata=_reads(str))
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
    #: by reason (FR-148).
    quality_dropped: dict[str, int] = field(
        default_factory=dict, metadata=_reads(_int_map),
    )
    #: The same records, ``reason -> sampling class -> records``: a class
    #: losing its records to one reason (``game-end`` takes the combats that
    #: end a game) is read here.
    quality_dropped_by_class: dict[str, dict[str, int]] = field(
        default_factory=dict, metadata=_reads(_nested_int_map),
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
    #: ``--shard-records``: the size output shards were repacked to; 0 leaves
    #: the shards mirroring their sources.
    shard_records: int = field(default=0, metadata=_reads(int))
    #: ``--validation-sample``: records per stratum in ``validation/samples/``.
    validation_sample: int = field(default=0, metadata=_reads(int))
    #: ``--max-events-per-record`` the quality rule used.
    max_events_per_record: int = field(default=0, metadata=_reads(int))
    #: Watched, not refused: kept records whose events the collector could not
    #: attribute to a clause (FR-148). They are still the ability's own
    #: outcomes, so they train.
    unattributed_records: int = field(default=0, metadata=_reads(int))
    #: Script file -> resolution records refused because every acting key
    #: mapped to no rendered line (FR-148), largest 100 only: Forge's implicit
    #: permanent spell would otherwise add an entry per permanent. For an
    #: operator, not code.
    no_acting_text_scripts: dict[str, int] = field(
        default_factory=dict, metadata=_reads(_int_map),
    )
    #: Which generation of corpus this manifest describes: ``build-corpus``
    #: writes :data:`MANIFEST_GENERATION`, and ``from_dict`` refuses a manifest
    #: that records another or none.
    generation: int = field(default=MANIFEST_GENERATION, metadata=_reads(int))

    # Feature 024.

    #: ``class -> family -> FamilyRow``.
    families: dict[str, dict[str, FamilyRow]] = field(
        default_factory=dict, metadata=_reads(_family_rows),
    )
    #: ``class -> family -> {outcome signature: records written}``.
    signatures: dict[str, dict[str, dict[str, int]]] = field(
        default_factory=dict, metadata=_reads(_doubly_nested_int_map),
    )
    #: ``class -> {on_policy, off_policy}`` training records, by ``random_seat``.
    policy_counts: dict[str, dict[str, int]] = field(
        default_factory=dict, metadata=_reads(_nested_int_map),
    )
    #: ``legality subkind -> {real, what_if, unknown}`` training records.
    legality_counts: dict[str, dict[str, int]] = field(
        default_factory=dict, metadata=_reads(_nested_int_map),
    )
    #: Surveyed records per legality mode: every reader places a legality
    #: record in its rarest mode's family from this table.
    legality_mode_counts: dict[str, int] = field(
        default_factory=dict, metadata=_reads(_int_map),
    )
    #: Game-disjoint games placed only under ``--game-disjoint-keyword-share``.
    keyword_threshold_games: tuple[str, ...] = field(
        default=(), metadata=_reads(tuple),
    )
    #: Held-out texts with no gate-one resolution record.
    held_out_texts_without_resolution: tuple[str, ...] = field(
        default=(), metadata=_reads(tuple),
    )
    #: ``held-out text -> games`` for the texts recorded in fewer than five.
    held_out_texts_under_five_games: dict[str, int] = field(
        default_factory=dict, metadata=_reads(_int_map),
    )
    #: The game-disjoint and reuse flags as run; None where a flag was not set.
    game_disjoint_share: float | None = field(
        default=None, metadata=_reads(_optional_float),
    )
    game_disjoint_keyword_share: float | None = field(
        default=None, metadata=_reads(_optional_float),
    )
    game_disjoint_keywords: tuple[str, ...] = field(
        default=(), metadata=_reads(tuple),
    )
    reuse_cap: int | None = field(default=None, metadata=_reads(_optional_int))
    #: Training records the plan asked of each class, copies included, read
    #: against ``per_class[...].kept``.
    class_targets: dict[str, int] = field(
        default_factory=dict, metadata=_reads(_int_map),
    )

    # Gen-3 (feature 025, FR-060, FR-060a): written by every gen-3 build.

    #: The survey's per-rule counts in ``rule-counts.json``'s shape (FR-008):
    #: ``records``, ``rules`` (rule -> polarity -> source -> records),
    #: ``not_countable``, ``none`` and ``review``.
    rule_counts: dict[str, Any] = field(
        default_factory=dict, metadata=_reads(dict),
    )
    #: The rules-to-fields map as the build read it (research R4): every
    #: tagged rule's verdict and outputs. Budgets, the rule mask, the freeze
    #: and evaluation read verdicts and outputs here, never ``rules/heads/``.
    rule_map: dict[str, RuleMapEntry] = field(
        default_factory=dict, metadata=_reads(_rule_map),
    )
    #: The map's content hash, which a checkpoint records.
    rule_map_digest: str = field(default="", metadata=_reads(str))
    #: ``rule -> outputs``: of each mapped rule's outputs, in the map's
    #: order, those at least one of its tagged records carries
    #: (``rule_output_carriage``; FR-060a). The rule mask and the rule tests
    #: read these, never the map's own list; a rule its records carry no
    #: output of maps to an empty tuple.
    effective_outputs: dict[str, tuple[str, ...]] = field(
        default_factory=dict, metadata=_reads(_tuple_map),
    )
    #: ``class -> rule -> polarity -> RuleRow`` (FR-060).
    rules: dict[str, dict[str, dict[str, RuleRow]]] = field(
        default_factory=dict, metadata=_reads(_rule_rows),
    )
    #: ``class -> budgeted rules whose polarity halves stay unequal`` (FR-054a).
    one_sided_rules: dict[str, tuple[str, ...]] = field(
        default_factory=dict, metadata=_reads(_tuple_map),
    )
    #: ``class -> condition carriers seen on one side only`` (FR-056).
    one_sided_conditions: dict[str, tuple[str, ...]] = field(
        default_factory=dict, metadata=_reads(_tuple_map),
    )
    #: ``rule -> polarity -> source -> records`` for every tagged rule the map
    #: does not mark ``trained`` (FR-054).
    unbudgeted_rules: dict[str, dict[str, dict[str, int]]] = field(
        default_factory=dict, metadata=_reads(_doubly_nested_int_map),
    )
    #: ``stratum -> half -> rules`` with both polarities in the inventories
    #: and one missing from that half (research R3).
    left_out_of_half: dict[str, dict[str, tuple[str, ...]]] = field(
        default_factory=dict, metadata=_reads(_nested_tuple_map),
    )
    #: Rules some record is tagged with and no stratum writes a copy of (the
    #: tags of refused and turn-action records sit under the ``refused``
    #: source of ``rule_counts``): a rule the counts name that no ``rules`` row
    #: of the training or a validation stratum writes. Sorted. Outside the
    #: digest: it reports, it decides nothing.
    tagged_nowhere_written: tuple[str, ...] = field(default=(), metadata=_reads(tuple))
    #: ``turn_action`` records written to no stratum, by subkind (FR-059).
    turn_action_left_out: dict[str, int] = field(
        default_factory=dict, metadata=_reads(_int_map),
    )
    #: The closed subtype and keyword lists (FR-052, FR-053), in the head's
    #: column order: ``output_list_survey.ordered_subtypes`` and ``ordered_keywords``,
    #: which is what ``output_vocabularies`` holds, so
    #: ``scripts/scan_output_vocabularies.py --manifest`` copies them verbatim.
    subtypes: tuple[str, ...] = field(default=(), metadata=_reads(tuple))
    keywords: tuple[str, ...] = field(default=(), metadata=_reads(tuple))
    #: ``stratum -> half -> game ids``, sorted (FR-070).
    halves: dict[str, dict[str, tuple[str, ...]]] = field(
        default_factory=dict, metadata=_reads(_nested_tuple_map),
    )
    #: ``validation stratum -> class -> rule -> polarity -> RuleRow``: each
    #: validation stratum's rule budgets, as ``rules`` holds the training
    #: stratum's (FR-068a).
    validation_rules: dict[str, dict[str, dict[str, dict[str, RuleRow]]]] = field(
        default_factory=dict, metadata=_reads(
            lambda values: {stratum: _rule_rows(rows) for stratum, rows in values.items()}),
    )
    #: ``class -> collection flag -> kept records`` over the records read
    #: (``fork``, ``interventional``, ``directed``, ``aimed``, ``degraded``,
    #: and ``runtime_only`` for a resolution acting only through runtime-only
    #: keys); a flag no record of the class carries is left out.
    record_flags: dict[str, dict[str, int]] = field(
        default_factory=dict, metadata=_reads(_nested_int_map),
    )
    #: Placement's precision thresholds (research E5), config fields with no
    #: flag: ``min_games_per_half``, ``max_item_error``, ``max_component_error``
    #: (None where infinite).
    placement_thresholds: dict[str, float | None] = field(
        default_factory=dict, metadata=_reads(dict),
    )
    #: sha256 of the keyword-definition file the build read; "" without one.
    keyword_definitions_digest: str = field(default="", metadata=_reads(str))
    #: ``"<member> <repr of the value>" -> times read`` for every value of a
    #: raw record outside the wire's closed lists (``shard_io.off_list_counts``).
    decode_off_list: dict[str, int] = field(
        default_factory=dict, metadata=_reads(_int_map),
    )
    #: Each raw shard whose damaged last gzip member ended its read.
    damaged_tails: tuple[str, ...] = field(default=(), metadata=_reads(tuple))
    #: Keys the survey read as no line, by why (``sidecar_io.JoinGaps``):
    #: ``unresolved``, ``undescribed``, ``ambiguous``, ``runtime_only`` and
    #: ``dropped`` by script file, ``unconfigured`` by source tree, each with
    #: how often it was asked; an empty kind left out. ``runtime_only`` and
    #: ``dropped`` are expected answers, not damage.
    join_gaps: dict[str, dict[str, int]] = field(
        default_factory=dict, metadata=_reads(_nested_int_map),
    )
    #: ``kind -> events`` of kept records no output column reads (a hidden
    #: keyword change, a name change, a mana cost change) and the subtype and
    #: keyword changes the closed output lists lack
    #: (``effect_targets.RecordTargets.unsupervised``).
    unsupervised_events: dict[str, int] = field(
        default_factory=dict, metadata=_reads(_int_map),
    )
    #: The commit of the code that built the corpus (``git rev-parse HEAD``),
    #: "" outside a checkout.
    code_version: str = field(default="", metadata=_reads(str))
    #: sha256 over every sidecar the build joined against, by tree, relative
    #: path and size: a reconversion since the build moves it.
    sidecar_digest: str = field(default="", metadata=_reads(str))
    #: Gen-3 games dropped whole as cut, their shard holding no game-end marker
    #: (WP7); their records are ``per_stratum["dropped-cut"]``. Inside the
    #: digest: which games are cut decides which records the corpus holds.
    cut_games: int = field(default=0, metadata=_reads(int))
    #: ``output stratum -> its shards``, each by file name inside the
    #: stratum's directory with the records the write pass put in it, sorted
    #: by name; a stratum holding none lists none. What a reader compares the
    #: shards it finds with (finding 11.4). Outside the digest: the names
    #: follow from the repack, decided by members inside it.
    shards: dict[str, tuple[CuratedShard, ...]] = field(
        default_factory=dict, metadata=_reads(_curated_shards),
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
        return HoldoutRule(self.holdout_permille, self.holdout_max_carriers)

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
        data["rules"] = _each_rule_row(self.rules, RuleRow.as_dict)
        data["rule_map"] = {
            rule: {"verdict": str(entry.verdict), "outputs": list(entry.outputs)}
            for rule, entry in self.rule_map.items()
        }
        data["validation_rules"] = {
            stratum: _each_rule_row(by_class, RuleRow.as_dict)
            for stratum, by_class in self.validation_rules.items()
        }
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CorpusManifest:
        """Each member through its own reader.

        A manifest of another generation, or none, raises :class:`ManifestError`
        naming it, as does one missing a member: only the manifest this code
        writes is read.
        """
        recorded = _recorded_generation(data)
        if recorded != f"gen-{MANIFEST_GENERATION}":
            raise ManifestError(
                f"the manifest is a {recorded} manifest; this code reads "
                f"gen-{MANIFEST_GENERATION} only. Rebuild the corpus with "
                "`python -m effects build-corpus`."
            )
        missing = [spec.name for spec in fields(cls) if spec.name not in data]
        if missing:
            raise ManifestError(
                f"the manifest lacks {', '.join(missing)}; rebuild the corpus "
                "with `python -m effects build-corpus`."
            )
        return cls(**{
            spec.name: spec.metadata[_READ](data[spec.name]) for spec in fields(cls)
        })

    #: Operator-facing tallies the digest leaves out: fields no decision reads,
    #: what the build met reading (a tally a pool worker's sidecar cache
    #: shapes, a shard's damaged tail), so neither where the raw corpus sits
    #: nor what ran before moves the digest the selection set and every
    #: checkpoint find the corpus by (FR-147).
    _OUTSIDE_THE_DIGEST = (
        "no_acting_text_scripts", "decode_off_list", "damaged_tails", "join_gaps",
        "unsupervised_events", "code_version", "sidecar_digest",
        "tagged_nowhere_written", "shards",
    )

    def digest(self) -> str:
        """A stable hash over every decision: sorted keys and sources, so the
        order the filesystem enumerated shards in cannot move it."""
        payload = self.as_dict()
        for field_name in self._OUTSIDE_THE_DIGEST:
            payload.pop(field_name, None)
        _leave_out_drawn(payload)
        payload["sources"] = sorted(payload["sources"], key=lambda s: s["name"])
        return canonical_digest(payload).hex()

    def shards_digest(self) -> str:
        """A hash of the shard listing alone, which :meth:`digest` leaves out:
        a resume compares it, so a rebuilt corpus deciding the same thing but
        naming or sizing its shards differently is not trained on as the one
        the run began on."""
        return canonical_digest({
            stratum: [asdict(shard) for shard in shards]
            for stratum, shards in self.shards.items()
        }).hex()

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


@dataclass(frozen=True, slots=True)
class RuleCuration:
    """What the build already holds that the manifest's gen-3 fields are built
    from: the rule budgets, the placed strata, the map snapshot, the survey's
    rule counts (the ``rule-counts.json`` object), its ``(rule, output) ->
    records carrying it`` tally, which decides each rule's effective outputs,
    its one-sided conditions (``condition_balance.one_sided_conditions``) and
    turn-action tally, and the subtype and keyword names it read."""

    budgets: RuleBudgets
    strata: GameStrata
    rule_map: Mapping[str, RuleMapEntry]
    rule_map_digest: str
    rule_counts: Mapping[str, Any]
    carried: Mapping[tuple[str, str], int]
    one_sided_conditions: Mapping[str, Iterable[str]]
    turn_action_left_out: Mapping[str, int]
    subtypes: Iterable[str]
    keywords: Iterable[str]


def rule_curation_fields(curation: RuleCuration) -> dict[str, Any]:
    """The gen-3 fields of :class:`CorpusManifest`, as keyword arguments, the
    subtype and keyword names put in the head's column order here."""
    from effects.domain.output_list_survey import ordered_keywords, ordered_subtypes

    budgets, strata, rule_map = curation.budgets, curation.strata, curation.rule_map
    carried, one_sided_conditions = curation.carried, curation.one_sided_conditions
    return {
        "rule_counts": dict(curation.rule_counts),
        "rule_map": dict(sorted(rule_map.items())),
        "rule_map_digest": curation.rule_map_digest,
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
        "turn_action_left_out": dict(sorted(curation.turn_action_left_out.items())),
        "subtypes": ordered_subtypes(curation.subtypes),
        "keywords": ordered_keywords(curation.keywords),
        "halves": {
            str(stratum): {str(half): tuple(sorted(games)) for half, games in by_half.items()}
            for stratum, by_half in strata.halves.items()
        },
    }

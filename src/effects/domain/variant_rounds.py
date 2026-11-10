"""The rules ``collect-variants`` plays its two rounds by (root spec § 8.15).

What a variant's games are, when a variant is done, and, for the steered
round, which (rule, polarity) pairs are short, how the round's variants are
split over them, which script lines a rule's tags act through and when a rule
retires. The orchestration that writes scripts, reads shards and drives Forge
is ``application/collect_variants.py``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum

from effects.domain.coverage import CoverageUnit
from effects.domain.provenance import ProvenanceKey, SourceTree
from effects.domain.records import EffectRecord, acting_keys
from effects.domain.rule_catalogue import (
    DetectorDeclaration,
    MapVerdict,
    Polarity,
    RuleMapEntry,
    ScenarioCatalogue,
)
from effects.domain.script_text import CHAIN_KEYS, params, segments


class VariantRound(StrEnum):
    """Which variant round ``collect-variants --round`` runs (FR-058a, FR-058c)."""

    UNTARGETED = "untargeted"
    STEERED = "steered"


DEFAULT_VARIANT_GAMES = 2
DEFAULT_NO_PROGRESS_ROUNDS = 3
DEFAULT_STEER_FLOOR = 100
DEFAULT_STEER_VARIANTS = 2000
DEFAULT_STEER_MIN_GAIN = 1
DEFAULT_STEER_ITERATIONS = 5

#: The polarities a steered round fills; ``other`` is a third branch no
#: floor applies to.
STEERED_POLARITIES: tuple[Polarity, ...] = (Polarity.FIRES, Polarity.DOES_NOT_FIRE)


def variant_script_file(file_name: str) -> str:
    """The provenance ``script_file`` a variant's records name it by: its file
    in the flat variant tree."""
    return f"{SourceTree.VARIANT_SCRIPTS}/{file_name}"


@dataclass
class VariantGames(CoverageUnit):
    """One variant's progress toward ``--variant-games`` games (FR-058a).

    ``rule`` is the rule a steered variant is aimed at, None in the
    untargeted round.
    """

    name: str
    script_file: str
    rule: str | None = None
    games: set[str] = field(default_factory=set)
    retired: bool = False
    rounds_without_progress: int = 0
    rounds_unseen: int = 0

    @property
    def progress(self) -> int:
        """Distinct games."""
        return len(self.games)


@dataclass
class VariantShardGames:
    """What one shard says of the variants: per variant script file the games
    holding a record that acts through it, the variant files any snapshot
    showed, and the records read."""

    games: dict[str, set[str]] = field(default_factory=dict)
    seen: set[str] = field(default_factory=set)
    records: int = 0

    def add(self, record: EffectRecord) -> None:
        """Count one record: its acting key, or its decision candidate's key,
        names the variant's file (FR-058a); every entity's lines name the
        variants it shows."""
        self.records += 1
        for key in acting_keys(record):
            if key.tree == SourceTree.VARIANT_SCRIPTS:
                self.games.setdefault(key.script_file, set()).add(record.game_id)
        for entity in record.state.entities:
            self.seen.update(
                key.script_file for key in entity.line_keys
                if key.tree == SourceTree.VARIANT_SCRIPTS
            )

    def merge(self, other: VariantShardGames) -> None:
        for script_file, games in other.games.items():
            self.games.setdefault(script_file, set()).update(games)
        self.seen |= other.seen
        self.records += other.records


def credit_games(
    units: Mapping[str, VariantGames], found: VariantShardGames,
) -> set[str]:
    """Add ``found``'s games to the units; returns the names of the units its
    snapshots showed."""
    shown: set[str] = set()
    for name, unit in units.items():
        unit.games |= found.games.get(unit.script_file, set())
        if unit.script_file in found.seen or unit.script_file in found.games:
            shown.add(name)
    return shown


def missing_games(units: Mapping[str, VariantGames], target: int) -> dict[str, float]:
    """Each unfinished variant's deck weight: the games it still lacks."""
    return {
        name: float(unit.shortfall(target))
        for name, unit in units.items() if not unit.done(target)
    }


# ── the steered round ──────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ShortPair:
    """A (rule, polarity) with fewer than ``--steer-floor`` records over all
    three sources (FR-058c)."""

    rule: str
    polarity: Polarity
    records: int
    shortfall: int


def pair_records(counts: Mapping, rule: str, polarity: Polarity) -> int:
    """A (rule, polarity)'s records over every source in a ``rule-counts``
    document's ``rules`` section; zero for a rule it omits."""
    sides = counts.get("rules", {}).get(rule, {})
    return sum(sides.get(str(polarity), {}).values())


def detected_rules(catalogue: ScenarioCatalogue) -> frozenset[str]:
    """The rules some scenario declares a detector for (not ``none``)."""
    return frozenset(
        rule for scenario in catalogue.scenarios
        if isinstance(scenario.detector, DetectorDeclaration)
        for rule in scenario.rules
    )


def short_pairs(
    counts: Mapping,
    heads: Mapping[str, RuleMapEntry],
    detected: frozenset[str],
    *,
    floor: int,
    retired: Iterable[str] = (),
) -> list[ShortPair]:
    """Every short pair of a rule the map marks ``trained`` with a declared
    detector, not retired, in rule order; a rule the counts omit is short in
    both polarities."""
    retired = frozenset(retired)
    found: list[ShortPair] = []
    for rule in sorted(detected - retired):
        entry = heads.get(rule)
        if entry is None or entry.verdict is not MapVerdict.TRAINED:
            continue
        for polarity in STEERED_POLARITIES:
            records = pair_records(counts, rule, polarity)
            if records < floor:
                found.append(ShortPair(rule, polarity, records, floor - records))
    return found


def variant_shares(pairs: list[ShortPair], total: int) -> dict[tuple[str, Polarity], int]:
    """``--steer-variants`` split over the pairs by shortfall, at least one
    each (FR-058c): ``total × shortfall ÷ summed shortfall``, rounded down."""
    summed = sum(pair.shortfall for pair in pairs)
    return {
        (pair.rule, pair.polarity): max(1, math.floor(total * pair.shortfall / summed))
        for pair in pairs
    }


def gained(
    before: Mapping, after: Mapping, pairs: Iterable[ShortPair],
) -> dict[str, int]:
    """Per rule, the most records any of its short polarities gained from
    ``before`` to ``after``."""
    best: dict[str, int] = {}
    for pair in pairs:
        gain = (pair_records(after, pair.rule, pair.polarity)
                - pair_records(before, pair.rule, pair.polarity))
        best[pair.rule] = max(best.get(pair.rule, gain), gain)
    return best


def rules_to_retire(gains: Mapping[str, int], *, min_gain: int) -> set[str]:
    """The rules whose short polarities gained fewer than ``--steer-min-gain``
    records this iteration (FR-058d)."""
    return {rule for rule, gain in gains.items() if gain < min_gain}


#: Line-kind prefixes of a Forge source script whose rest is one trait.
_TRAIT_PREFIXES: tuple[str, ...] = ("A:", "T:", "S:", "R:", "K:")


def _trait_body(line: str) -> str | None:
    """A source line's trait text without its kind prefix (``SVar:Name:``
    included); None for any other line."""
    stripped = line.strip()
    if stripped.startswith("SVar:"):
        name_and_body = stripped[len("SVar:"):]
        _, separator, body = name_and_body.partition(":")
        return body if separator else None
    for prefix in _TRAIT_PREFIXES:
        if stripped.startswith(prefix):
            return stripped[len(prefix):]
    return None


def _signature(segment: str) -> tuple:
    """A segment's parameters, a chain label's value elided, so a source line
    and the sidecar's relabelled chain compare equal; a bare keyword is its
    own text."""
    found = params(segment)
    if not found:
        return ("keyword", segment.strip())
    return tuple(sorted(
        (key, "*" if key in CHAIN_KEYS else " ".join(value.split()))
        for key, value in found.items()
    ))


def acting_lines(source_lines: list[str], script_text: str) -> list[int]:
    """The indices of the source script lines that make up a sidecar line's
    chained ``script_text``: each trait or SVar line whose parameters equal
    one of its segments'."""
    wanted = {_signature(segment) for segment in segments(script_text)}
    return [
        index for index, line in enumerate(source_lines)
        if (body := _trait_body(line)) is not None and _signature(body) in wanted
    ]


@dataclass(frozen=True, slots=True)
class CarrierLines:
    """One carrier of a rule from the counts: a card's script file and the
    keys of the lines the rule's tags act through on it."""

    script_file: str
    keys: tuple[ProvenanceKey, ...] = ()


def counted_carriers(counts: Mapping, rule: str) -> list[CarrierLines]:
    """A rule's carriers as ``rule-counts`` lists them, most records first."""
    found: list[CarrierLines] = []
    for carrier in counts.get("carriers", {}).get(rule, ()):
        script_file = carrier["script_file"]
        found.append(CarrierLines(script_file, tuple(
            ProvenanceKey.from_dict(line["key"], script_file=script_file)
            for line in carrier.get("lines", ())
        )))
    return found

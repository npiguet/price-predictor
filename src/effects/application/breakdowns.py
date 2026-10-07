"""The evaluation report's breakdowns: one scored record set, grouped many ways.

Gen-2's arms are compared by hand, and what is compared is mostly how a field's
loss moves across slices of the validation strata rather than one pooled
number. A pooled loss over the card-disjoint stratum is dominated by the few
texts with the most records; the per-text mean weighs every held-out text once,
which is the question gate 1's population was built to ask (FR-064). The
rarity buckets and rule families then say *where* in the tail an arm gains or
loses, and the policy and decision slices separate what the random seat and
the real legality declarations add (FR-066, FR-067).

Every function here is pure grouping over :class:`RecordResult` rows. Scoring
a record — running the model, reading its losses — is ``gate_one.measure``'s
job; nothing here imports torch, so the grouping rules are testable on plain
rows.

A field's mean is taken over the records that supervise it, never over all
records. A field a record does not supervise has no loss for it, and counting
it as zero would make a slice look better the fewer of its records carry the
field.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field

from effects.domain.rarity import RARITY_BUCKETS, rarity_bucket

#: The loss key every scored record carries: the per-entity loss summed over
#: the gate and every supervised field, as the trainer's validation reads it.
TOTAL = "total"

#: The legality subkinds a ``what_if`` classes (FR-031).
LEGALITY_SUBKINDS = frozenset({"attackers", "blockers"})

REAL, WHAT_IF, UNKNOWN = "real", "what-if", "unknown"
ON_POLICY, OFF_POLICY = "on", "off"
WITHHELD, TRAINED = "withheld", "trained"


@dataclass(frozen=True, slots=True)
class RecordResult:
    """One scored record: what identifies it, what groups it, what it scored.

    ``losses`` holds :data:`TOTAL` and every field the record supervised, by
    field name; an unsupervised field is absent rather than zero. ``keywords``
    is every keyword on the acting line or carried by an entity of the board,
    in the gate's spelling (``first_strike``), for the withheld-keyword slice.
    """

    record_id: str
    game_id: str
    kind: str
    losses: Mapping[str, float]
    subkind: str | None = None
    text: str | None = None
    family: str | None = None
    random_seat: bool = False
    what_if: bool | None = None
    keywords: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True, slots=True)
class FieldMeans:
    """Per-field means over one group, with how many records each mean read."""

    means: dict[str, float]
    counts: dict[str, int]

    @property
    def records(self) -> int:
        return self.counts.get(TOTAL, 0)


def field_means(results: Iterable[RecordResult]) -> FieldMeans:
    """Each field's mean loss over the records that supervise it."""
    sums: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for result in results:
        for name, value in result.losses.items():
            sums[name] += value
            counts[name] += 1
    return FieldMeans(
        {name: sums[name] / counts[name] for name in sums}, dict(counts),
    )


def per_text_means(results: Iterable[RecordResult]) -> FieldMeans:
    """Each field's mean over texts, every text's own mean weighted once.

    Records with no acting text cannot be grouped by one and are left out. The
    count beside each mean is the number of texts that supervised the field.
    """
    by_text: dict[str, list[RecordResult]] = defaultdict(list)
    for result in results:
        if result.text is not None:
            by_text[result.text].append(result)
    text_means = [field_means(rows) for rows in by_text.values()]
    return _mean_of_means(text_means)


def _mean_of_means(groups: Iterable[FieldMeans]) -> FieldMeans:
    sums: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for group in groups:
        for name, value in group.means.items():
            sums[name] += value
            counts[name] += 1
    return FieldMeans(
        {name: sums[name] / counts[name] for name in sums}, dict(counts),
    )


@dataclass(frozen=True, slots=True)
class GroupResult:
    """One group's per-record and per-text figures."""

    per_record: FieldMeans
    per_text: FieldMeans


def _group(
    results: Iterable[RecordResult], key: Callable[[RecordResult], str | None],
) -> dict[str, GroupResult]:
    groups: dict[str, list[RecordResult]] = defaultdict(list)
    for result in results:
        name = key(result)
        if name is not None:
            groups[name].append(result)
    return {
        name: GroupResult(field_means(rows), per_text_means(rows))
        for name, rows in sorted(groups.items())
    }


def overall(results: Iterable[RecordResult]) -> GroupResult:
    """The whole set, per record and per text (FR-064)."""
    rows = list(results)
    return GroupResult(field_means(rows), per_text_means(rows))


def by_rarity_bucket(
    results: Iterable[RecordResult], games_by_text: Mapping[str, float],
) -> dict[str, GroupResult]:
    """Grouped by the rarity bucket of each record's text (FR-064).

    ``games_by_text`` is the number of distinct games each text acted in,
    from the curated corpus's rarity table. A text it does not name falls back
    to the games the scored records themselves span, so a held-out text the
    table never counted is still bucketed by what the stratum saw of it.
    Buckets come back in ascending order of games.
    """
    rows = [r for r in results if r.text is not None]
    seen: dict[str, set[str]] = defaultdict(set)
    for result in rows:
        seen[result.text].add(result.game_id)

    def bucket(result: RecordResult) -> str:
        games = games_by_text.get(result.text)
        if games is None:
            games = len(seen[result.text])
        return rarity_bucket(games)

    grouped = _group(rows, bucket)
    return {name: grouped[name] for name in RARITY_BUCKETS if name in grouped}


def by_family(results: Iterable[RecordResult]) -> dict[str, GroupResult]:
    """Grouped by rule family (FR-064)."""
    return _group(results, lambda r: r.family)


def mean_over_families(families: Mapping[str, GroupResult]) -> GroupResult:
    """Every family weighted once, so a common family cannot drown a rare one."""
    return GroupResult(
        _mean_of_means(group.per_record for group in families.values()),
        _mean_of_means(group.per_text for group in families.values()),
    )


def memorization_gap(
    game_disjoint: Mapping[str, GroupResult],
    card_disjoint: Mapping[str, GroupResult],
) -> dict[str, dict[str, float]]:
    """Per family: game-disjoint minus card-disjoint, on the fields both have.

    Both strata are scored per record. The game-disjoint stratum holds texts
    the model trained on, the card-disjoint stratum texts it never saw, so a
    loss that rises from one to the other is what the model only remembered
    (FR-065). A family present in one stratum alone has no gap to report.
    """
    gaps: dict[str, dict[str, float]] = {}
    for family in sorted(set(game_disjoint) & set(card_disjoint)):
        seen = game_disjoint[family].per_record.means
        unseen = card_disjoint[family].per_record.means
        gaps[family] = {
            name: seen[name] - unseen[name]
            for name in sorted(set(seen) & set(unseen))
        }
    return gaps


def policy_slice(results: Iterable[RecordResult]) -> dict[str, GroupResult]:
    """Records acted by the random seat, apart from the rest (FR-066)."""
    return _group(results, lambda r: OFF_POLICY if r.random_seat else ON_POLICY)


def decision_slice(results: Iterable[RecordResult]) -> dict[str, GroupResult]:
    """Legality records, real declarations apart from what-if queries (FR-067).

    ``unknown`` is a gen-1 record, which never counts as real.
    """
    def kind(result: RecordResult) -> str | None:
        if result.subkind not in LEGALITY_SUBKINDS:
            return None
        if result.what_if is None:
            return UNKNOWN
        return WHAT_IF if result.what_if else REAL

    return _group(results, kind)


def keyword_slice(
    results: Iterable[RecordResult], withheld: str,
) -> dict[str, GroupResult]:
    """Records carrying the withheld keyword beside records carrying others.

    The withheld group is every record where the keyword is on the acting line
    or carried by a board entity (FR-068). The trained group is every record
    carrying some keyword and not that one, so the comparison is keyword
    records against keyword records rather than against vanilla boards.
    """
    target = normalize_keyword(withheld)

    def group(result: RecordResult) -> str | None:
        if target in result.keywords:
            return WITHHELD
        return TRAINED if result.keywords else None

    return _group(results, group)


def normalize_keyword(name: str) -> str:
    """A keyword in the gate's spelling: lower case, underscores, no value.

    ``Ward:2`` and ``ward`` are one keyword; ``First Strike`` is
    ``first_strike``.
    """
    return name.split(":", 1)[0].strip().lower().replace(" ", "_")

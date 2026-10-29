"""The evaluation report's breakdowns: one scored record set, grouped many ways.

Gen-2's arms are compared by hand, and what is compared is mostly how a field's
loss moves across slices of the validation strata rather than one pooled
number. A pooled loss over the card-disjoint stratum is dominated by the few
texts with the most records; the per-text mean weighs every held-out text once,
which is the question gate 1's population was built to ask (FR-064). The
rarity buckets and rule families then say *where* in the tail an arm gains or
loses, and the policy and decision slices separate what the random seat and
the real legality declarations add (FR-066, FR-067).

Every function here is pure grouping over :class:`RecordResult` rows, and the
rendering of each group as a report line. Scoring a record — running the
model, reading its losses — is ``record_scoring.score_records``'s job; nothing
here imports torch, so the grouping rules are testable on plain rows.

A field's mean is taken over the records that supervise it, never over all
records. A field a record does not supervise has no loss for it, and counting
it as zero would make a slice look better the fewer of its records carry the
field.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field

from effects.domain.evaluation_report import CheckName, CheckResult, CheckStatus
from effects.domain.keyword_names import normalize_keyword
from effects.domain.rarity import RARITY_BUCKETS, rarity_bucket
from effects.domain.records import WHAT_IF_SUBKINDS, LegalityDecision
from effects.domain.strata import Stratum

#: The loss key every scored record carries: the per-entity loss summed over
#: the gate and every supervised field, as the trainer's validation reads it.
TOTAL = "total"

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
    return _mean_per_key(result.losses for result in results)


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
    return _mean_per_key(group.means for group in groups)


def _mean_per_key(rows: Iterable[dict[str, float]]) -> FieldMeans:
    """Each key's mean over the rows that carry it, beside how many did.

    A row without a key is not a zero for it: a record that does not
    supervise a field, like a text none of whose records did, says nothing
    about that field.
    """
    sums: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        for name, value in row.items():
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
    """Legality records, real declarations apart from what-if queries (FR-067)."""
    def kind(result: RecordResult) -> str | None:
        if result.subkind not in WHAT_IF_SUBKINDS:
            return None
        return LegalityDecision.of(result)

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


# ── the report lines ────────────────────────────────────────────────────


def breakdown_checks(
    scored: dict[str, list],
    *,
    games_by_text: dict[str, float],
    withheld_keyword: str | None,
) -> list[CheckResult]:
    """Every breakdown of the scored strata, one report line per group.

    ``scored`` maps a stratum to its :class:`RecordResult` rows, so the
    report's shape is testable without a model.
    """
    card = scored.get(Stratum.CARD_DISJOINT, [])
    game = scored.get(Stratum.GAME_DISJOINT, [])
    checks = [*_card_disjoint_checks(card, games_by_text)]
    checks += _memorization_gap_checks(card, game)
    for stratum, rows in ((Stratum.CARD_DISJOINT, card), (Stratum.GAME_DISJOINT, game)):
        checks += _slice_checks(
            f"{stratum} policy", policy_slice(rows), present=OFF_POLICY,
            skipped=f"{stratum} off-policy",
            why="no record in this stratum was acted by the random seat",
        )
        checks += _slice_checks(
            f"{stratum} legality", decision_slice(rows),
            present=LegalityDecision.REAL,
            skipped=f"{stratum} real-decision legality",
            why="no legality record in this stratum is classed a real decision",
        )
    if withheld_keyword:
        checks += _slice_checks(
            f"{CheckName.ZERO_SHOT_KEYWORD} {withheld_keyword}",
            keyword_slice([*card, *game], withheld_keyword),
            present=WITHHELD, skipped=CheckName.ZERO_SHOT_KEYWORD,
            why=(
                f"no validation record carries the withheld keyword "
                f"{withheld_keyword!r}"
            ),
        )
    return checks


def _card_disjoint_checks(card: list, games_by_text: dict[str, float]) -> list[CheckResult]:
    """The card-disjoint stratum overall, by rarity, by family, and over families."""
    if not card:
        return [CheckResult(
            Stratum.CARD_DISJOINT, CheckStatus.SKIPPED,
            "the card-disjoint stratum holds no record to break down",
        )]
    checks = [_group_check(Stratum.CARD_DISJOINT, overall(card))]
    for bucket, group in by_rarity_bucket(card, games_by_text).items():
        checks.append(_group_check(f"{Stratum.CARD_DISJOINT} rarity {bucket}", group))
    families = by_family(card)
    for family, group in families.items():
        checks.append(_group_check(f"{Stratum.CARD_DISJOINT} family {family}", group))
    if families:
        checks.append(_group_check(
            f"{Stratum.CARD_DISJOINT} mean over families",
            mean_over_families(families),
            detail=f"{len(families)} families, each weighted once",
        ))
    return checks


def _memorization_gap_checks(card: list, game: list) -> list[CheckResult]:
    """Per family, how much worse the unseen cards score than the seen ones."""
    gaps = memorization_gap(by_family(game), by_family(card))
    if not gaps:
        return [CheckResult(
            CheckName.MEMORIZATION_GAP, CheckStatus.SKIPPED,
            "no family has records in both validation strata",
        )]
    return [
        CheckResult(
            f"{CheckName.MEMORIZATION_GAP} {family}", CheckStatus.REPORTED,
            "game-disjoint minus card-disjoint loss, per field", values,
        )
        for family, values in gaps.items()
    ]


def _slice_checks(
    prefix: str, groups: dict, *, present, skipped: str, why: str,
) -> list[CheckResult]:
    """Every group of one slice as a line, or one skip naming why it is absent.

    ``present`` is the group whose absence means the slice measured nothing —
    the off-policy rows, the real decisions, the withheld keyword — so the
    other groups alone would be a comparison with one side missing.
    """
    if present not in groups:
        return [CheckResult(skipped, CheckStatus.SKIPPED, why)]
    return [
        _group_check(f"{prefix} {name}", group) for name, group in groups.items()
    ]


def _group_check(name: str, group: GroupResult, *, detail: str | None = None) -> CheckResult:
    """One group's report line: per-record field means, then per-text ones.

    Per-text means are prefixed ``text:``; the detail carries both counts, so a
    mean over three records reads as one. A mean of group means passes its own
    ``detail``, since its counts are groups rather than records.
    """
    values = {
        **group.per_record.means,
        **{f"text:{key}": value for key, value in group.per_text.means.items()},
    }
    texts = group.per_text.counts.get(TOTAL, 0)
    return CheckResult(
        name, CheckStatus.REPORTED,
        detail or f"{group.per_record.records} records, {texts} texts", values,
    )

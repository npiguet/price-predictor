"""The evaluation report's breakdowns: one scored record set, grouped many ways.

Pure grouping over :class:`RecordResult` rows, torch-free; scoring is
``record_scoring.score_records``'s job, and a half's control score is reported
by ``half_score_report``. The per-text mean weighs every held-out text
once, where a pooled loss is dominated by the texts with the most records
(FR-064). The scored records are distinct: a validation stratum repeats a
record's row for every added copy, and every record is read once (FR-068a).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field

from effects.domain.check_label import CheckLabel, TableRow
from effects.domain.evaluation_report import CheckName, CheckResult, CheckStatus
from effects.domain.keyword_names import normalize_keyword
from effects.domain.rarity import RARITY_BUCKETS, TextGames
from effects.domain.records import LEGALITY_SUBKINDS, LegalityDecision
from effects.domain.strata import Stratum

#: The loss key every scored record carries: the per-entity loss summed over
#: the gate and every supervised field, as the trainer's validation reads it.
TOTAL = "total"

ON_POLICY, OFF_POLICY = "on", "off"
WITHHELD, TRAINED = "withheld", "trained"


@dataclass(frozen=True, slots=True)
class RecordResult:
    """One scored record: what groups it and what it scored.

    ``game_id`` is what a text the rarity table never counted is bucketed by.
    ``losses`` holds :data:`TOTAL` and every field the record supervised; an
    unsupervised field is absent rather than zero. ``keywords`` is every
    keyword on the acting line or the board, in the gate's spelling
    (``first_strike``).
    """

    game_id: str
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

    Records with no acting text are left out; the counts are of texts.
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

    A missing key is not a zero: counting it as one would make a slice look
    better the fewer of its records carry the field.
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
    results: Iterable[RecordResult], games_by_text: Mapping[str, int],
) -> dict[str, GroupResult]:
    """Grouped by the rarity bucket of each record's text (FR-064), buckets in
    ascending order of games.

    A text the corpus's rarity table ``games_by_text`` does not name falls back
    to the games the scored records themselves span.
    """
    rows = [r for r in results if r.text is not None]
    games = TextGames.of(rows, lambda result: result.text, games_by_text)
    grouped = _group(rows, lambda result: games.bucket(result.text))
    return {name: grouped[name] for name in RARITY_BUCKETS if name in grouped}


def by_family(results: Iterable[RecordResult]) -> dict[str, GroupResult]:
    """Grouped by effect family (FR-064)."""
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
    """Per family: game-disjoint minus card-disjoint per-record loss, on the
    fields and families both strata have (FR-065)."""
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
        if result.subkind not in LEGALITY_SUBKINDS:
            return None
        return LegalityDecision.of(result)

    return _group(results, kind)


def keyword_slice(
    results: Iterable[RecordResult], withheld: str,
) -> dict[str, GroupResult]:
    """Records carrying the withheld keyword beside records carrying others
    (FR-068), so the comparison is not against vanilla boards."""
    target = normalize_keyword(withheld)

    def group(result: RecordResult) -> str | None:
        if target in result.keywords:
            return WITHHELD
        return TRAINED if result.keywords else None

    return _group(results, group)


def breakdown_checks(
    scored: dict[str, list],
    *,
    games_by_text: Mapping[str, int],
    withheld_keyword: str | None,
) -> list[CheckResult]:
    """Every breakdown of the scored strata (stratum -> rows), one report line
    per group."""
    card = scored.get(Stratum.CARD_DISJOINT, [])
    game = scored.get(Stratum.GAME_DISJOINT, [])
    checks = [*_card_disjoint_checks(card, games_by_text)]
    checks += _memorization_gap_checks(card, game)
    for stratum, rows in ((Stratum.CARD_DISJOINT, card), (Stratum.GAME_DISJOINT, game)):
        checks += _slice_checks(policy_slice(rows), SliceSpec(
            CheckLabel.of(stratum, "policy"), present=OFF_POLICY,
            skipped=CheckLabel.of(stratum, "off-policy"),
            why="no record in this stratum was acted by the random seat",
        ))
        checks += _slice_checks(decision_slice(rows), SliceSpec(
            CheckLabel.of(stratum, "legality"), present=LegalityDecision.REAL,
            skipped=CheckLabel.of(stratum, "real-decision legality"),
            why="no legality record in this stratum is classed a real decision",
        ))
    if withheld_keyword:
        checks += _slice_checks(keyword_slice([*card, *game], withheld_keyword), SliceSpec(
            CheckLabel.of(CheckName.ZERO_SHOT_KEYWORD, withheld_keyword),
            present=WITHHELD, skipped=CheckLabel.of(CheckName.ZERO_SHOT_KEYWORD),
            why=(
                f"no validation record carries the withheld keyword "
                f"{withheld_keyword!r}"
            ),
        ))
    return checks


def _card_disjoint_checks(card: list, games_by_text: Mapping[str, int]) -> list[CheckResult]:
    """The card-disjoint stratum overall, by rarity, by family, and over families."""
    if not card:
        return [CheckResult(
            Stratum.CARD_DISJOINT, CheckStatus.SKIPPED,
            "the card-disjoint stratum holds no record to break down",
        )]
    checks = [_group_check(CheckLabel.of(Stratum.CARD_DISJOINT), overall(card))]
    for bucket, group in by_rarity_bucket(card, games_by_text).items():
        checks.append(_group_check(CheckLabel.of(Stratum.CARD_DISJOINT, "rarity", bucket), group))
    families = by_family(card)
    for family, group in families.items():
        checks.append(_group_check(CheckLabel.of(Stratum.CARD_DISJOINT, "family", family), group))
    if families:
        checks.append(_group_check(
            CheckLabel.of(Stratum.CARD_DISJOINT, "mean over families"),
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
    return [_gap_check(family, values) for family, values in gaps.items()]


def _gap_check(family: str, gaps: Mapping[str, float]) -> CheckResult:
    """One family's gap: the total on the line, every field's in the table."""
    return CheckResult(
        str(CheckLabel.of(CheckName.MEMORIZATION_GAP, family)), CheckStatus.REPORTED,
        "game-disjoint minus card-disjoint loss; the per-field gaps are in the table",
        {TOTAL: gaps[TOTAL]} if TOTAL in gaps else {},
        tuple(TableRow(name, value) for name, value in gaps.items()),
    )


@dataclass(frozen=True, slots=True)
class SliceSpec:
    """How one slice reads: its lines' head, the group whose absence skips the
    comparison, the skip line's name and why it is skipped."""

    head: CheckLabel
    present: str
    skipped: CheckLabel
    why: str


def _slice_checks(groups: dict, spec: SliceSpec) -> list[CheckResult]:
    """Every group of one slice as a line, or one skip when the ``present``
    group is missing and the comparison would have one side."""
    if spec.present not in groups:
        return [CheckResult(str(spec.skipped), CheckStatus.SKIPPED, spec.why)]
    return [_group_check(spec.head.then(name), group) for name, group in groups.items()]


def _group_check(
    label: CheckLabel, group: GroupResult, *, detail: str | None = None,
) -> CheckResult:
    """One group's report line: the total loss per record and per text
    (``text:``-prefixed) with both counts in the detail unless ``detail`` is
    given. Every field's mean is in the table, beside the records it read."""
    per_record, per_text = group.per_record, group.per_text
    values = {
        key: value for key, value in (
            (TOTAL, per_record.means.get(TOTAL)), (f"text:{TOTAL}", per_text.means.get(TOTAL)),
        ) if value is not None
    }
    table = (
        *(TableRow(name, value, per_record.counts[name])
          for name, value in per_record.means.items()),
        *(TableRow(f"text:{name}", value, per_text.counts[name])
          for name, value in per_text.means.items()),
    )
    texts = per_text.counts.get(TOTAL, 0)
    return CheckResult(
        str(label), CheckStatus.REPORTED,
        detail or f"{per_record.records} records, {texts} texts", values, table,
    )

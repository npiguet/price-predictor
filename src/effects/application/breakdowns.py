"""The evaluation report's breakdowns: one scored record set, grouped many ways,
and a half's control score as report lines.

Pure grouping over :class:`RecordResult` rows, torch-free; scoring is
``record_scoring.score_records``'s job, and a half's score
``control_score.score_half``'s. The per-text mean weighs every held-out text
once, where a pooled loss is dominated by the texts with the most records
(FR-064). The scored records are distinct: a validation stratum repeats a
record's row for every added copy, and every record is read once (FR-068a).
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from effects.domain.evaluation_report import CheckName, CheckResult, CheckStatus
from effects.domain.keyword_names import normalize_keyword
from effects.domain.rarity import RARITY_BUCKETS, TextGames
from effects.domain.records import LEGALITY_SUBKINDS, LegalityDecision
from effects.domain.selection_set import COMPONENTS, DEVIANCE, RULE_TESTS, deviance_by_head
from effects.domain.strata import Stratum

if TYPE_CHECKING:
    from effects.application.control_score import ControlScore

#: Left-out item names a report line spells out before it counts the rest.
NAMED_LEFT_OUT = 12

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
    #: The record's generation, which says how a missing ``what_if`` reads.
    generation: int | None = None
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


def _card_disjoint_checks(card: list, games_by_text: Mapping[str, int]) -> list[CheckResult]:
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
    """Every group of one slice as a line, or one skip when the ``present``
    group is missing and the comparison would have one side."""
    if present not in groups:
        return [CheckResult(skipped, CheckStatus.SKIPPED, why)]
    return [
        _group_check(f"{prefix} {name}", group) for name, group in groups.items()
    ]


def _group_check(name: str, group: GroupResult, *, detail: str | None = None) -> CheckResult:
    """One group's report line: per-record field means, then ``text:``-prefixed
    per-text ones, with both counts in the detail unless ``detail`` is given."""
    values = {
        **group.per_record.means,
        **{f"text:{key}": value for key, value in group.per_text.means.items()},
    }
    texts = group.per_text.counts.get(TOTAL, 0)
    return CheckResult(
        name, CheckStatus.REPORTED,
        detail or f"{group.per_record.records} records, {texts} texts", values,
    )


# ── a half's control score ──────────────────────────────────────────────


def half_score_checks(
    score: ControlScore, *, half: str, games: Mapping[str, int],
    selection_control: float | None = None, note: str = "",
) -> list[CheckResult]:
    """One half of every stratum, scored as the control score scores the
    selection half (FR-070): per stratum its four components and their mean,
    then each component's items, the rule tests per rule (FR-074), and last
    the control score the strata mix to beside the selection half's the
    checkpoint recorded. ``games`` is each stratum's games in the half."""
    prefix = f"{half}-half"
    checks: list[CheckResult] = []
    for stratum, scored in sorted(score.strata.items()):
        where = f"{prefix} {stratum}"
        missing = [name for name in COMPONENTS if name in scored.components
                   and math.isnan(scored.components[name])]
        detail = f"{games.get(stratum, 0)} games; score = the mean of the four components"
        if missing:
            detail += f"; nothing to score in {', '.join(missing)}"
        checks.append(CheckResult(where, CheckStatus.REPORTED, detail, {
            **dict(scored.components), "score": scored.score}))
        for component in COMPONENTS:
            if component not in scored.components:
                continue
            items = dict(scored.items.get(component, {}))
            left_out = tuple(scored.left_out.get(component, ()))
            name = (f"{CheckName.RULE_TESTS} {stratum} {half}" if component == RULE_TESTS
                    else f"{where} {component}")
            checks.append(CheckResult(name, CheckStatus.REPORTED,
                                      _items_detail(component, items, left_out), items))
    values = {"control": score.control} if len(score.strata) == 2 else {}
    if selection_control is not None:
        values["selection_control"] = float(selection_control)
    if values:
        detail = (f"stratum weight {score.stratum_weight:g} on the card-disjoint score"
                  + ("; selection_control is the best the checkpoint recorded"
                     if selection_control is not None else "")
                  + (f"; {note}" if note else ""))
        checks.append(CheckResult(f"{prefix} control", CheckStatus.REPORTED, detail, values))
    return checks


def _items_detail(component: str, items: Mapping[str, float], left_out: tuple[str, ...]) -> str:
    """How many items a component scored, each rescaled and clipped at -1, and
    which it left out."""
    unit = "rules" if component == RULE_TESTS else "items"
    detail = f"{len(items)} {unit} scored, rescaled and clipped below at -1"
    if component == DEVIANCE and items:
        heads = deviance_by_head(items)
        detail += "; by head: " + ", ".join(
            f"{head} {mean:.3f} ({count})" for head, (mean, count) in heads.items())
    if left_out:
        named = ", ".join(left_out[:NAMED_LEFT_OUT])
        more = len(left_out) - NAMED_LEFT_OUT
        detail += f"; {len(left_out)} left out: {named}" + (f" and {more} more" if more > 0 else "")
    return detail

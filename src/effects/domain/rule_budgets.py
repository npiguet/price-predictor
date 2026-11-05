"""Per-rule shares within each sampling class, and the added copies that fill them
(FR-054, FR-054a, FR-055; research R2).

Runs per class after :func:`effects.domain.corpus_plan.plan_selection`, whose
copies stay each record's natural share:

1. a rule's share is the class target divided by the budgeted rules tagged on
   the class's records, a budgeted rule being one the map snapshot marks
   ``trained``; each polarity's half is the share halved;
2. a (rule, polarity)'s written count starts at the natural copies of its
   tagged records, and a record tagged with several rules counts toward each;
3. each half is filled on its own: what a short side cannot reach stays its
   shortfall and is never given to the other side;
4. a record's added copies never take its total past ``reuse_cap``, and a
   text's added copies never take its total in the class past ``text_cap``;
5. each added copy goes to the record's rule furthest below its half.

Copies are added in rounds, one per record per round, records in seeded
order, so repeats spread over a rule's records rather than piling onto the
first ones; a record leaves once none of its rules is below its half.

Not ``budget_allocation.allocate``: it hands a short member's share to its
siblings, which FR-054a forbids between polarity halves, and it needs
disjoint members, which multi-tagged records are not.

Torch-free: ``build-corpus`` plans in its main process.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from effects.domain.rule_catalogue import Polarity
from effects.domain.rule_detectors import RuleSource

#: The polarities that draw a half of a rule's share (FR-054a).
HALF_POLARITIES: tuple[Polarity, ...] = (Polarity.FIRES, Polarity.DOES_NOT_FIRE)


@dataclass(frozen=True, slots=True)
class BudgetedRecord:
    """One tagged training record, as the survey keeps it for the budgets.

    ``record_hash`` is ``corpus_curation.record_hash(record_id, seed=--seed)``,
    the keyed hash ``seeded_draw.stable_order`` sorts by, so sorting records
    by it is the seeded order; it is also the write pass's key for the record.
    """

    game: str
    record_hash: int
    klass: str
    #: The acting text ``--text-cap`` counts; None for a record with none,
    #: which no text cap bounds, as in gen-2's keyless member.
    text: str | None
    #: Copies gen-2's class, family and signature budgets give the record.
    natural: int
    source: RuleSource
    #: ``(rule, polarity)``, one per rule the record's detectors tag.
    tags: tuple[tuple[str, Polarity], ...]


@dataclass(frozen=True, slots=True)
class RuleRow:
    """One (class, rule, polarity) row of the manifest's ``rules`` (FR-060).

    ``available`` counts distinct tagged records and ``written`` their copies,
    natural and added, each by :class:`RuleSource`; ``added`` is the part of
    ``written`` the budgets drew. ``share`` is the polarity's half, None on
    ``other`` and on an unbudgeted rule, which draw none; ``shortfall`` is
    ``max(0, share - written)``, zero where ``share`` is None. A half is filled
    from every source alike, so ``share`` and ``shortfall`` are not split.
    """

    available: dict[str, int]
    share: int | None
    written: dict[str, int]
    added: dict[str, int]
    shortfall: int

    @property
    def written_total(self) -> int:
        return sum(self.written.values())

    def as_dict(self) -> dict:
        return {
            "available": dict(sorted(self.available.items())),
            "share": self.share,
            "written": dict(sorted(self.written.items())),
            "added": dict(sorted(self.added.items())),
            "shortfall": self.shortfall,
        }

    @classmethod
    def from_dict(cls, values: Mapping) -> RuleRow:
        share = values.get("share")
        return cls(
            available={k: int(v) for k, v in values["available"].items()},
            share=None if share is None else int(share),
            written={k: int(v) for k, v in values["written"].items()},
            added={k: int(v) for k, v in values["added"].items()},
            shortfall=int(values["shortfall"]),
        )


@dataclass(frozen=True, slots=True)
class RuleBudgets:
    """What the budgets drew, and what the manifest reports about it."""

    #: ``game -> record hash -> drawn rule per added copy``; a record with no
    #: added copy is absent. Grouped by game so the write pass can shard it
    #: by whatever game hash its writers split on.
    copies: dict[str, dict[int, tuple[str, ...]]]
    #: ``class -> rule -> polarity -> RuleRow``, budgeted and unbudgeted rules.
    rows: dict[str, dict[str, dict[str, RuleRow]]]
    #: ``class -> one rule's share``; a class with no budgeted rule is absent.
    shares: dict[str, int]
    #: ``class -> budgeted rules whose halves stay unequal`` (FR-054a).
    one_sided: dict[str, tuple[str, ...]]
    #: ``rule -> polarity -> source -> distinct tagged records`` over every
    #: class, for each tagged rule the map does not mark ``trained`` (FR-054).
    unbudgeted: dict[str, dict[str, dict[str, int]]]

    def added(self, game: str, record_hash: int) -> tuple[str, ...]:
        """The drawn rule of each added copy of one record; empty for none."""
        return self.copies.get(game, {}).get(record_hash, ())


def plan_rule_budgets(
    records: Iterable[BudgetedRecord],
    *,
    class_targets: Mapping[str, int],
    budgeted: Iterable[str],
    reuse_cap: int,
    text_cap: int,
) -> RuleBudgets:
    """The added copies of every class, and its rule rows.

    Args:
        class_targets: records each class is asked to write, as
            ``SelectionPlan.targets`` holds them.
        budgeted: the rules the map snapshot marks ``trained``.
        text_cap: ``--text-cap``; ``<= 0`` is no cap.
    """
    budgeted = frozenset(budgeted)
    by_class: dict[str, list[BudgetedRecord]] = defaultdict(list)
    for record in records:
        by_class[record.klass].append(record)

    copies: dict[str, dict[int, tuple[str, ...]]] = {}
    rows: dict[str, dict[str, dict[str, RuleRow]]] = {}
    shares: dict[str, int] = {}
    one_sided: dict[str, tuple[str, ...]] = {}
    unbudgeted: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    for klass in sorted(by_class):
        plan = _ClassPlan(
            by_class[klass], budgeted=budgeted, target=class_targets.get(klass, 0),
            reuse_cap=reuse_cap, text_cap=text_cap,
        )
        plan.fill()
        for index, drawn in plan.drawn.items():
            record = plan.records[index]
            copies.setdefault(record.game, {})[record.record_hash] = tuple(drawn)
        rows[klass] = plan.rows()
        if plan.rules:
            shares[klass] = plan.share
        sided = plan.one_sided()
        if sided:
            one_sided[klass] = sided
        for (rule, polarity, source), n in plan.available.items():
            if rule not in budgeted:
                unbudgeted[rule][polarity][source] += n
    return RuleBudgets(
        copies=copies, rows=rows, shares=shares, one_sided=one_sided,
        unbudgeted={
            rule: {str(p): dict(sorted(by_source.items()))
                   for p, by_source in sorted(by_polarity.items())}
            for rule, by_polarity in sorted(unbudgeted.items())
        },
    )


@dataclass(slots=True)
class _ClassPlan:
    """One class's budgets in progress."""

    records: Sequence[BudgetedRecord]
    budgeted: frozenset[str]
    target: int
    reuse_cap: int
    text_cap: int
    #: Budgeted rules tagged in the class, sorted.
    rules: tuple[str, ...] = ()
    share: int = 0
    #: ``(rule, polarity, source) -> distinct tagged records``.
    available: Counter = field(default_factory=Counter)
    #: ``(rule, polarity, source) -> copies written``, natural then added.
    written: Counter = field(default_factory=Counter)
    added: Counter = field(default_factory=Counter)
    #: ``(rule, polarity) -> copies written``, every source.
    totals: Counter = field(default_factory=Counter)
    #: Index into ``records`` -> drawn rule per added copy.
    drawn: dict[int, list[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.rules = tuple(sorted({
            rule for record in self.records for rule, _ in record.tags
            if rule in self.budgeted
        }))
        self.share = self.target // len(self.rules) if self.rules else 0
        for record in self.records:
            for rule, polarity in record.tags:
                self.available[rule, polarity, record.source] += 1
                self.written[rule, polarity, record.source] += record.natural
                self.totals[rule, polarity] += record.natural

    @property
    def half(self) -> int:
        return self.share // 2

    def fill(self) -> None:
        """Add copies in rounds of at most one per record, records in seeded order."""
        if self.half <= 0 or self.reuse_cap <= 0:
            return
        records = self.records
        room = [self.reuse_cap - record.natural for record in records]
        text_room = self._text_room()
        active = sorted(
            (i for i, record in enumerate(records) if room[i] > 0 and self._halves(record)),
            key=lambda i: _seeded(records[i]),
        )
        while active:
            still: list[int] = []
            for i in active:
                record = records[i]
                if text_room.get(record.text, 1) <= 0:
                    continue
                rule = self._furthest_below(record)
                if rule is None:
                    continue
                self._add(i, rule)
                room[i] -= 1
                if record.text in text_room:
                    text_room[record.text] -= 1
                if room[i] > 0:
                    still.append(i)
            active = still

    def _text_room(self) -> dict[str, int]:
        """Added copies each text may still take: what keeps its total in the
        class at or under ``text_cap``; no entry is no cap."""
        if self.text_cap <= 0:
            return {}
        natural: Counter = Counter()
        for record in self.records:
            if record.text is not None:
                natural[record.text] += record.natural
        return {text: max(0, self.text_cap - n) for text, n in natural.items()}

    def _halves(self, record: BudgetedRecord) -> list[tuple[str, Polarity]]:
        """The record's tags that draw a half."""
        return [
            (rule, polarity) for rule, polarity in record.tags
            if rule in self.budgeted and polarity in HALF_POLARITIES
        ]

    def _furthest_below(self, record: BudgetedRecord) -> str | None:
        """The record's rule whose half is furthest from filled; ties to the
        smaller rule id, so the draw is a function of the corpus alone."""
        best: tuple[int, str] | None = None
        for rule, polarity in self._halves(record):
            deficit = self.half - self.totals[rule, polarity]
            if deficit > 0 and (best is None or (-deficit, rule) < (-best[0], best[1])):
                best = (deficit, rule)
        return None if best is None else best[1]

    def _add(self, index: int, rule: str) -> None:
        record = self.records[index]
        self.drawn.setdefault(index, []).append(rule)
        for tagged, polarity in record.tags:
            self.written[tagged, polarity, record.source] += 1
            self.added[tagged, polarity, record.source] += 1
            self.totals[tagged, polarity] += 1

    def rows(self) -> dict[str, dict[str, RuleRow]]:
        out: dict[str, dict[str, RuleRow]] = {}
        cells = {(rule, polarity) for rule, polarity, _ in self.available}
        cells |= {(rule, polarity) for rule in self.rules for polarity in HALF_POLARITIES}
        for rule, polarity in sorted(cells):
            share = (
                self.half
                if rule in self.budgeted and polarity in HALF_POLARITIES else None
            )
            written = self._by_source(self.written, rule, polarity)
            out.setdefault(rule, {})[str(polarity)] = RuleRow(
                available=self._by_source(self.available, rule, polarity),
                share=share,
                written=written,
                added=self._by_source(self.added, rule, polarity),
                shortfall=0 if share is None else max(0, share - sum(written.values())),
            )
        return out

    def _by_source(self, counts: Counter, rule: str, polarity: Polarity) -> dict[str, int]:
        return {
            str(source): counts[rule, polarity, source]
            for source in RuleSource if counts[rule, polarity, source]
        }

    def one_sided(self) -> tuple[str, ...]:
        """Budgeted rules whose two halves end filled to different counts:
        one side reached a count the other could not (FR-054a)."""
        return tuple(
            rule for rule in self.rules
            if len({min(self.half, self.totals[rule, polarity])
                    for polarity in HALF_POLARITIES}) > 1
        )


def _seeded(record: BudgetedRecord) -> tuple:
    """The seeded record order: the keyed hash first, then fields that make
    the order total whatever order the survey's shards arrived in."""
    return (record.record_hash, record.game, record.klass, record.text or "",
            record.natural, str(record.source), record.tags)

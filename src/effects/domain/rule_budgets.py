"""Per-rule shares within each sampling class, and the added copies that fill them
(FR-054, FR-054a, FR-055; research R2).

Runs per class after :func:`effects.domain.corpus_plan.plan_selection`, whose
copies stay each record's natural share:

1. a rule's share is the class target divided by the budgeted rules tagged on
   the class's records, a budgeted rule being one the map snapshot marks
   ``trained``; each polarity's half is the share halved;
2. a (rule, polarity)'s written count starts at the natural copies of its
   tagged records, and a record tagged with several rules counts toward each,
   its added copies too; a protection record's copy (FR-058) counts toward
   its drawn rule alone, since the copy shows that rule's protection only;
3. each half is filled on its own: what a short side cannot reach stays its
   shortfall and is never given to the other side;
4. a record's added copies never take its total past ``reuse_cap``, and a
   text's added copies never take its total in the class past ``text_cap``,
   its total counting the natural copies of every record of the text in the
   class, tagged or not (``text_natural``);
5. each added copy goes to the record's rule furthest below its half.

Copies are added in rounds, one per record per round, over every tagged
record of the class, fork records included, in seeded order, so repeats spread
over a rule's records rather than piling onto the first ones; a record leaves
once none of its rules is below its half.

Not ``budget_allocation.allocate``: it hands a short member's share to its
siblings, which FR-054a forbids between polarity halves, and it needs
disjoint members, which multi-tagged records are not.

Torch-free: ``build-corpus`` plans in its main process.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from effects.domain.corpus_types import CopyMap, RecordHash, RuleCell
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
    record_hash: RecordHash
    klass: str
    #: The acting text ``--text-cap`` counts; None for a record with none,
    #: which no text cap bounds, as in gen-2's keyless member.
    text: str | None
    #: Copies gen-2's class, family and signature budgets give the record.
    natural: int
    source: RuleSource
    #: ``(rule, polarity)``, one per rule the record's detectors tag.
    tags: tuple[RuleCell, ...]
    #: A record offering protection copies, tagged with every protection
    #: rule it offers: each copy is drawn for one and counts toward it alone.
    protection: bool = False


@dataclass(frozen=True, slots=True)
class RuleRow:
    """One (class, rule, polarity) row of the manifest's ``rules`` (FR-060).

    ``available`` counts distinct tagged records and ``written`` their copies,
    natural and added, each by :class:`RuleSource`; ``added`` is the part of
    ``written`` the budgets drew, and ``drawn`` the part of ``added`` drawn
    for this very rule, the copies whose ``curation.drawn_rule`` it is: a
    natural record's copy counts toward every rule the record is tagged with,
    so the rest of ``added`` is a copy drawn for another rule of the record.
    ``available`` leaves out ``fork`` and is a report figure no budget
    decision reads: a probe's, an intervention's or a protection fork's
    record re-records a moment of its live game, so a combat counts once
    however many probes re-ran it in ``available``, while every copy of
    such a record is written and counted in ``written``, which the halves'
    fills read.
    ``share`` is the polarity's half, None on ``other`` and on an unbudgeted
    rule, which draw none; ``shortfall`` is ``max(0, share - written)``, zero
    where ``share`` is None. A half is filled from every source alike, so
    ``share`` and ``shortfall`` are not split.
    """

    available: dict[str, int]
    share: int | None
    written: dict[str, int]
    added: dict[str, int]
    drawn: dict[str, int]
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
            "drawn": dict(sorted(self.drawn.items())),
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
            drawn={k: int(v) for k, v in values["drawn"].items()},
            shortfall=int(values["shortfall"]),
        )


@dataclass(frozen=True, slots=True)
class RuleBudgets:
    """What the budgets drew, and what the manifest reports about it."""

    #: ``game -> record hash -> drawn rule per added copy``; a record with no
    #: added copy is absent. Grouped by game so the write pass can shard it
    #: by whatever game hash its writers split on.
    copies: CopyMap
    #: ``class -> rule -> polarity -> RuleRow``, budgeted and unbudgeted rules.
    rows: dict[str, dict[str, dict[str, RuleRow]]]
    #: ``class -> one rule's share``; a class with no budgeted rule is absent.
    shares: dict[str, int]
    #: ``class -> budgeted rules whose halves stay unequal`` (FR-054a).
    one_sided: dict[str, tuple[str, ...]]
    #: ``rule -> polarity -> source -> distinct tagged records`` over every
    #: class, ``fork`` left out as in ``RuleRow.available``, for each tagged
    #: rule the map does not mark ``trained`` (FR-054).
    unbudgeted: dict[str, dict[str, dict[str, int]]]


@dataclass(frozen=True, slots=True)
class BudgetInputs:
    """What :func:`plan_rule_budgets` plans against.

    ``class_targets``: records each class is asked to write, as
    ``SelectionPlan.targets`` holds them. ``budgeted``: the rules the map
    snapshot marks ``trained``. ``text_cap``: ``--text-cap``; ``<= 0`` is no
    cap. ``text_natural``: ``class -> text -> natural copies`` of every record
    of the text in the class, untagged ones included, which the text cap
    counts; a text it omits counts its tagged records' alone.
    """

    class_targets: Mapping[str, int]
    budgeted: Iterable[str]
    reuse_cap: int
    text_cap: int
    text_natural: Mapping[str, Mapping[str, int]] | None = None


def plan_rule_budgets(records: Iterable[BudgetedRecord], inputs: BudgetInputs) -> RuleBudgets:
    """The added copies of every class, and its rule rows."""
    budgeted = frozenset(inputs.budgeted)
    by_class: dict[str, list[BudgetedRecord]] = defaultdict(list)
    for record in records:
        by_class[record.klass].append(record)

    copies: dict[str, dict[RecordHash, tuple[str, ...]]] = {}
    rows: dict[str, dict[str, dict[str, RuleRow]]] = {}
    shares: dict[str, int] = {}
    one_sided: dict[str, tuple[str, ...]] = {}
    unbudgeted: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    for klass in sorted(by_class):
        plan = _ClassPlan(
            by_class[klass], budgeted=budgeted, target=inputs.class_targets.get(klass, 0),
            reuse_cap=inputs.reuse_cap, text_cap=inputs.text_cap,
            text_natural=dict((inputs.text_natural or {}).get(klass, {})),
        )
        plan.fill()
        _add_copies(copies, plan)
        rows[klass] = plan.rows()
        if plan.rules:
            shares[klass] = plan.share
        sided = plan.one_sided()
        if sided:
            one_sided[klass] = sided
        for (rule, polarity, source), n in plan.unbudgeted_available():
            unbudgeted[rule][polarity][source] += n
    return RuleBudgets(
        copies=copies, rows=rows, shares=shares, one_sided=one_sided,
        unbudgeted=_sorted_counts(unbudgeted),
    )


def _add_copies(copies: dict[str, dict[RecordHash, tuple[str, ...]]], plan: _ClassPlan) -> None:
    """Each of the class's records with an added copy, under its game."""
    for index, drawn in plan.drawn.items():
        record = plan.records[index]
        copies.setdefault(record.game, {})[record.record_hash] = tuple(drawn)


def _sorted_counts(by_rule: Mapping[str, Mapping]) -> dict[str, dict[str, dict[str, int]]]:
    """``rule -> polarity -> source -> records``, each level sorted."""
    return {
        rule: {str(p): dict(sorted(by_source.items()))
               for p, by_source in sorted(by_polarity.items())}
        for rule, by_polarity in sorted(by_rule.items())
    }


@dataclass(slots=True)
class _ClassPlan:
    """One class's budgets in progress."""

    records: Sequence[BudgetedRecord]
    budgeted: frozenset[str]
    target: int
    reuse_cap: int
    text_cap: int
    #: ``text -> natural copies`` of every record of the text in the class.
    text_natural: dict[str, int] = field(default_factory=dict)
    #: Budgeted rules tagged in the class, sorted.
    rules: tuple[str, ...] = ()
    share: int = 0
    #: ``(rule, polarity, source) -> distinct tagged records``, ``fork`` left out.
    available: Counter = field(default_factory=Counter)
    #: ``(rule, polarity, source) -> copies written``, natural then added.
    written: Counter = field(default_factory=Counter)
    added: Counter = field(default_factory=Counter)
    #: The part of ``added`` drawn for the cell's own rule.
    drawn_for: Counter = field(default_factory=Counter)
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
            # A fork's record adds a zero, so its cell still gets a row.
            counted = record.source is not RuleSource.FORK
            for rule, polarity in record.tags:
                self.available[rule, polarity, record.source] += counted
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
            active = [i for i in active if self._draw(i, room, text_room)]

    def _draw(self, i: int, room: list[int], text_room: dict[str, int]) -> bool:
        """Add one copy of record ``i`` if its text and a rule allow; whether
        the record stays in the rounds."""
        record = self.records[i]
        if text_room.get(record.text, 1) <= 0:
            return False
        rule = self._furthest_below(record)
        if rule is None:
            return False
        self._add(i, rule)
        room[i] -= 1
        if record.text in text_room:
            text_room[record.text] -= 1
        return room[i] > 0

    def _text_room(self) -> dict[str, int]:
        """Added copies each text may still take: what keeps its total in the
        class, untagged records' natural copies included, at or under
        ``text_cap``; no entry is no cap."""
        if self.text_cap <= 0:
            return {}
        natural: Counter = Counter()
        for record in self.records:
            if record.text is not None:
                natural[record.text] += record.natural
        return {text: max(0, self.text_cap - max(n, self.text_natural.get(text, 0)))
                for text, n in natural.items()}

    def _halves(self, record: BudgetedRecord) -> list[RuleCell]:
        """The record's tags that draw a half."""
        return [
            (rule, polarity) for rule, polarity in record.tags
            if rule in self.budgeted and polarity in HALF_POLARITIES
        ]

    def _furthest_below(self, record: BudgetedRecord) -> str | None:
        """The record's rule whose half is furthest from filled; ties to the
        smaller rule id, so the draw is a function of the corpus alone."""
        below = [
            (self.totals[rule, polarity] - self.half, rule)
            for rule, polarity in self._halves(record)
            if self.totals[rule, polarity] < self.half
        ]
        return min(below)[1] if below else None

    def _add(self, index: int, rule: str) -> None:
        record = self.records[index]
        self.drawn.setdefault(index, []).append(rule)
        for tagged, polarity in _counted_tags(record, rule):
            self.written[tagged, polarity, record.source] += 1
            self.added[tagged, polarity, record.source] += 1
            self.totals[tagged, polarity] += 1
            self.drawn_for[tagged, polarity, record.source] += tagged == rule

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
                drawn=self._by_source(self.drawn_for, rule, polarity),
                shortfall=0 if share is None else max(0, share - sum(written.values())),
            )
        return out

    def _by_source(self, counts: Counter, rule: str, polarity: Polarity) -> dict[str, int]:
        return {
            str(source): counts[rule, polarity, source]
            for source in RuleSource if counts[rule, polarity, source]
        }

    def unbudgeted_available(self) -> list[tuple[tuple[str, Polarity, RuleSource], int]]:
        """``((rule, polarity, source), records)`` of each rule not budgeted,
        the counts above zero only (a fork's record adds none)."""
        return [(cell, n) for cell, n in self.available.items()
                if cell[0] not in self.budgeted and n]

    def one_sided(self) -> tuple[str, ...]:
        """Budgeted rules whose two halves end filled to different counts:
        one side reached a count the other could not (FR-054a)."""
        return tuple(
            rule for rule in self.rules
            if len({min(self.half, self.totals[rule, polarity])
                    for polarity in HALF_POLARITIES}) > 1
        )


def _counted_tags(record: BudgetedRecord, rule: str) -> tuple[RuleCell, ...]:
    """The tags one added copy of ``record`` drawn for ``rule`` counts toward:
    every tag of a natural record's copy, the drawn rule's alone on a
    protection copy."""
    if not record.protection:
        return record.tags
    return tuple(tag for tag in record.tags if tag[0] == rule)


def _seeded(record: BudgetedRecord) -> tuple:
    """The seeded record order: the keyed hash first, then fields that make
    the order total whatever order the survey's shards arrived in."""
    return (record.record_hash, record.game, record.klass, record.text or "",
            record.natural, str(record.source), record.tags)

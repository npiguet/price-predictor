"""The training supply, the validation weights and the rule budgets of each
stratum, planned over the survey once placement has put each game.

Only training games' records size the class, family and text quotas and the
copy plans; each validation stratum is planned as training is, a class's
target being its records, each written once (FR-059a, FR-068a).
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from effects.application.corpus_survey import Survey, SurveyedRecord
from effects.domain.corpus_curation import validation_multipliers
from effects.domain.corpus_plan import GameStrata, Keys, SelectionPlan, TrainingSupply
from effects.domain.rule_budgets import BudgetedRecord, BudgetInputs, RuleBudgets, plan_rule_budgets
from effects.domain.rule_catalogue import Polarity
from effects.domain.strata import Stratum

if TYPE_CHECKING:
    from effects.application.build_corpus_config import BuildCorpusConfig

__all__ = [
    "VALIDATION_STRATA", "BudgetPlanner", "PlacedSurvey", "training_supply",
    "validation_weights",
]

logger = logging.getLogger(__name__)


#: The validation strata, in the order each is planned and logged.
VALIDATION_STRATA: tuple[Stratum, ...] = (Stratum.CARD_DISJOINT, Stratum.GAME_DISJOINT)


@dataclass(frozen=True, slots=True)
class PlacedSurvey:
    """The survey, and the strata placement put each of its games in."""

    survey: Survey
    strata: GameStrata

    def trains(self, record: SurveyedRecord) -> bool:
        """A candidate record training is planned over: in a game placement
        trained, acting through no held-out text."""
        return not record.held_out_text and self.strata.stratum_of(record.game) is Stratum.TRAINING

    def candidates_in(self, stratum: Stratum) -> list[SurveyedRecord]:
        """The candidate records of the games placed in ``stratum``, in survey order."""
        return [
            record for record in self.survey.totals.candidates
            if self.strata.stratum_of(record.game) is stratum
        ]


def training_supply(placed: PlacedSurvey) -> TrainingSupply:
    """The survey's supply with the candidate games placement trained folded
    in, their legality cells folded as the survey's were."""
    totals = placed.survey.totals
    supply = TrainingSupply(totals.supply.heap_cap)
    supply.merge(totals.supply)
    for record in totals.candidates:
        if placed.trains(record):
            supply.add(record.cell.resolved(totals.mode_counts), record.keys, record.record_hash)
    return supply


def validation_weights(
    placed: PlacedSurvey, config: BuildCorpusConfig,
) -> dict[Stratum, dict[str, float]]:
    """Each validation stratum's ``class_weight`` per class, which weighs it
    into the training stratum's ``--class-mix`` (FR-059a), logged beside the
    class counts they reweigh."""
    weights: dict[Stratum, dict[str, float]] = {}
    for stratum in VALIDATION_STRATA:
        counts = Counter(str(record.klass) for record in placed.candidates_in(stratum))
        weights[stratum] = validation_multipliers(
            counts, {str(k): v for k, v in config.mix().items()}, reuse_cap=config.reuse_cap,
        )
        for name in sorted(counts):
            logger.info(
                "%-14s %-22s %7d record(s), class weight %.2f",
                stratum, name, counts[name], weights[stratum][name],
            )
    return weights


@dataclass(frozen=True, slots=True)
class BudgetPlanner:
    """The training stratum's rule budgets over gen-2's natural share, and
    each validation stratum's over its records' one natural copy each
    (FR-059a, FR-068a): every stratum is built with the same rule budgets, a
    class's target being the records it writes before any copy is added. A
    validation class's weight in the mix is no copy: it is the rows'
    ``class_weight``."""

    placed: PlacedSurvey
    key_text: dict[Keys, str]
    config: BuildCorpusConfig
    #: The rules the map marks ``trained`` (``RuleCatalogue.budgeted``).
    budgeted: frozenset[str]

    def plan(self, selection: SelectionPlan) -> tuple[RuleBudgets, dict[Stratum, RuleBudgets]]:
        """The training stratum's budgets, then each validation stratum's."""
        budgets = self.training(selection)
        return budgets, {stratum: self.validation(stratum) for stratum in VALIDATION_STRATA}

    def training(self, selection: SelectionPlan) -> RuleBudgets:
        """Over the tagged records and the trained candidates offering a tag
        or a protection copy, each at its natural share."""
        totals = self.placed.survey.totals
        mode_counts = totals.mode_counts

        def natural(record: SurveyedRecord) -> int:
            return selection.copies(record.cell.resolved(mode_counts), record.keys,
                                    record.record_hash)

        training = [*totals.tagged, *(
            record for record in totals.candidates
            if self.placed.trains(record) and (record.tags or record.protection)
        )]
        return self._planned(
            _budgeted_records(training, self.key_text, natural), selection.targets,
            selection.text_planned,
        )

    def validation(self, stratum: Stratum) -> RuleBudgets:
        """Over the stratum's candidates, each written once."""
        records = self.placed.candidates_in(stratum)
        targets, text_natural = _held_once(records, self.key_text)
        return self._planned(
            _budgeted_records(records, self.key_text, _once), targets, text_natural,
        )

    def _planned(
        self, records: Iterable[BudgetedRecord], targets: Mapping[str, int],
        text_natural: Mapping[str, Mapping[str, int]],
    ) -> RuleBudgets:
        return plan_rule_budgets(records, BudgetInputs(
            class_targets=targets, budgeted=self.budgeted, reuse_cap=self.config.reuse_cap,
            text_cap=self.config.text_cap, text_natural=text_natural,
        ))


def _once(_record: SurveyedRecord) -> int:
    """A validation record's natural copies: one, whatever its class."""
    return 1


def _held_once(
    records: Iterable[SurveyedRecord], key_text: dict[Keys, str],
) -> tuple[dict[str, int], dict[str, Counter[str]]]:
    """A validation stratum's class targets and natural copies per text, each
    record written once."""
    targets: Counter[str] = Counter()
    text_natural: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        targets[str(record.klass)] += 1
        text = key_text.get(record.keys)
        if text is not None:
            text_natural[str(record.klass)][text] += 1
    return dict(targets), text_natural


def _budgeted_records(
    records: Iterable[SurveyedRecord], key_text: dict[Keys, str],
    natural: Callable[[SurveyedRecord], int],
) -> Iterator[BudgetedRecord]:
    """Each tagged record as the budgets read it, and beside it each record
    offering protection copies as a record of its own, natural share zero,
    tagged ``fires`` with every protection rule it offers, each copy counting
    toward its drawn rule alone (FR-058)."""
    for record in records:
        text = key_text.get(record.keys)
        if record.tags:
            yield BudgetedRecord(
                game=record.game, record_hash=record.record_hash, klass=str(record.klass),
                text=text, natural=natural(record), source=record.source, tags=record.tags,
            )
        if record.protection:
            yield BudgetedRecord(
                game=record.game, record_hash=record.protection_hash,
                klass=str(record.klass), text=text, natural=0, source=record.source,
                tags=tuple((rule, Polarity.FIRES) for rule in record.protection),
                protection=True,
            )

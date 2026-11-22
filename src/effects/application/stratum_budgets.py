"""The training supply, the validation weights and the rule budgets of each
stratum, planned over the survey once placement has put each game.

Only training games' records size the class, family and text quotas and the
copy plans; each validation stratum is planned as training is, a class's
target being its records, each written once (FR-059a, FR-068a).
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator
from typing import TYPE_CHECKING

from effects.application.corpus_survey import Survey, SurveyedRecord
from effects.domain.corpus_curation import validation_multipliers
from effects.domain.corpus_plan import GameStrata, Keys, SelectionPlan, TrainingSupply
from effects.domain.rule_budgets import BudgetedRecord, BudgetInputs, RuleBudgets, plan_rule_budgets
from effects.domain.rule_catalogue import Polarity
from effects.domain.strata import Stratum

if TYPE_CHECKING:
    from effects.application.build_corpus_config import BuildCorpusConfig

__all__ = ["rule_budgets", "training_supply", "validation_weights"]

logger = logging.getLogger(__name__)


def training_supply(survey: Survey, strata: GameStrata) -> TrainingSupply:
    """The survey's supply with the candidate games placement trained folded
    in, their legality cells folded as the survey's were."""
    totals = survey.totals
    supply = TrainingSupply(totals.supply.heap_cap)
    supply.merge(totals.supply)
    for record in totals.candidates:
        if not record.held_out_text and strata.stratum_of(record.game) is Stratum.TRAINING:
            supply.add(record.cell.resolved(totals.mode_counts), record.keys, record.record_hash)
    return supply


def validation_weights(
    survey: Survey, strata: GameStrata, config: BuildCorpusConfig,
) -> dict[Stratum, dict[str, float]]:
    """Each validation stratum's ``class_weight`` per class, which weighs it
    into the training stratum's ``--class-mix`` (FR-059a), logged beside the
    class counts they reweigh."""
    weights: dict[Stratum, dict[str, float]] = {}
    for stratum in (Stratum.CARD_DISJOINT, Stratum.GAME_DISJOINT):
        counts = Counter(
            str(record.klass) for record in survey.totals.candidates
            if strata.stratum_of(record.game) is stratum
        )
        weights[stratum] = validation_multipliers(
            counts, {str(k): v for k, v in config.mix().items()}, reuse_cap=config.reuse_cap,
        )
        for name in sorted(counts):
            logger.info(
                "%-14s %-22s %7d record(s), class weight %.2f",
                stratum, name, counts[name], weights[stratum][name],
            )
    return weights


def rule_budgets(
    survey: Survey, strata: GameStrata, selection: SelectionPlan, key_text: dict[Keys, str],
    *, config: BuildCorpusConfig, budgeted: frozenset[str],
) -> tuple[RuleBudgets, dict[Stratum, RuleBudgets]]:
    """The training stratum's rule budgets over gen-2's natural share, and
    each validation stratum's over its records' one natural copy each
    (FR-059a, FR-068a): every stratum is built with the same rule budgets, a
    class's target being the records it writes before any copy is added. A
    validation class's weight in the mix is no copy: it is the rows'
    ``class_weight``."""
    totals = survey.totals
    mode_counts = totals.mode_counts

    def natural(record: SurveyedRecord) -> int:
        return selection.copies(record.cell.resolved(mode_counts), record.keys,
                                record.record_hash)

    training = [*totals.tagged, *(
        record for record in totals.candidates
        if not record.held_out_text and strata.stratum_of(record.game) is Stratum.TRAINING
        and (record.tags or record.protection)
    )]
    def plan(records, targets, text_natural) -> RuleBudgets:
        return plan_rule_budgets(records, BudgetInputs(
            class_targets=targets, budgeted=budgeted, reuse_cap=config.reuse_cap,
            text_cap=config.text_cap, text_natural=text_natural,
        ))

    budgets = plan(
        _budgeted_records(training, key_text, natural), selection.targets,
        selection.text_planned,
    )
    validation: dict[Stratum, RuleBudgets] = {}
    for stratum in (Stratum.CARD_DISJOINT, Stratum.GAME_DISJOINT):
        records = [r for r in totals.candidates if strata.stratum_of(r.game) is stratum]
        targets, text_natural = _held_once(records, key_text)
        validation[stratum] = plan(
            _budgeted_records(records, key_text, _once), targets, text_natural,
        )
    return budgets, validation


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

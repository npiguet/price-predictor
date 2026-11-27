"""A cost half's rule-test row is counted at the entities its payment trains
at (FR8's path).

The freeze counts a row of a rule read at entities only where the record
carries one of the rule's effective outputs at the tag's outcome entities
(``RuleTestRules.tests_of``). A cost half's tag names the paid entities, so a
sacrifice's row is read at the sacrificed creature. The record is the real
gen-3 fixture's; no shipped family tags a cost half, so the tag comes from the
stand-in family of ``test_cost_half_rule_mask``.
"""

from __future__ import annotations

from effects.application.freeze_selection_set import RuleTestRules
from effects.domain.effect_head_layout import HeadField
from effects.domain.rule_catalogue import Polarity
from effects.domain.rule_detectors import tag_record
from tests.unit.effects.domain.test_cost_half_rule_mask import RULE, _index
from tests.unit.effects.domain.test_cost_half_targets import _BODYGUARD, _record


def _rules(outputs: tuple[str, ...]) -> RuleTestRules:
    index = _index()
    return RuleTestRules(tag=lambda record: tag_record(record, index).tags,
                         declared=frozenset({RULE}), trained=frozenset({RULE}),
                         entity_read={RULE: outputs})


def test_a_sacrifices_row_is_read_at_the_sacrificed_creature():
    rules = _rules((HeadField.ZONE_OUTCOME,))
    assert rules.tests_of(_record(_BODYGUARD)) == [(RULE, Polarity.FIRES, ("E28",))]


def test_an_output_the_payment_did_not_write_reads_nothing():
    rules = _rules((HeadField.TAP_STATE,))
    assert rules.tests_of(_record(_BODYGUARD)) == [(RULE, Polarity.FIRES, ())]

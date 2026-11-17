"""``LegalityDecision``: one spelling of real / what-if."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from effects.domain.records import LegalityDecision, PlayabilitySubkind
from tests.unit.effects.builders import java_golden_records


@pytest.mark.parametrize(("what_if", "decision"), [
    (False, LegalityDecision.REAL),
    (True, LegalityDecision.WHAT_IF),
])
def test_a_blockers_records_what_if_decides_it(what_if, decision):
    record = SimpleNamespace(what_if=what_if, subkind=PlayabilitySubkind.BLOCKERS)
    assert LegalityDecision.of(record) is decision


def test_an_attackers_record_is_a_real_declaration():
    """No ``what_if`` on attackers: one record per creature of the
    declaration made, none for a query."""
    attackers = SimpleNamespace(what_if=None, subkind=PlayabilitySubkind.ATTACKERS)
    assert LegalityDecision.of(attackers) is LegalityDecision.REAL


def test_a_golden_attackers_record_reads_real():
    from effects.infrastructure.record_codec import record_from_dict

    records = [record_from_dict(raw) for raw in java_golden_records()]
    attackers = [r for r in records if r.subkind is PlayabilitySubkind.ATTACKERS]
    assert attackers
    assert {LegalityDecision.of(r) for r in attackers} == {LegalityDecision.REAL}


def test_the_manifest_and_the_slices_share_one_hyphenated_spelling():
    assert LegalityDecision.WHAT_IF == "what-if"

"""``LegalityDecision``: one spelling of real / what-if / unknown."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from effects.domain.records import LegalityDecision


@pytest.mark.parametrize(("what_if", "decision"), [
    (False, LegalityDecision.REAL),
    (True, LegalityDecision.WHAT_IF),
    (None, LegalityDecision.UNKNOWN),
])
def test_a_records_what_if_decides_it(what_if, decision):
    assert LegalityDecision.of(SimpleNamespace(what_if=what_if)) is decision


def test_the_manifest_and_the_slices_share_one_hyphenated_spelling():
    assert LegalityDecision.WHAT_IF == "what-if"

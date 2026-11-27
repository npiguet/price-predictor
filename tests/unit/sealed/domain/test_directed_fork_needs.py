"""DR1: ``match-outcomes`` recomputes each listed rule's
directed-fork probability from the forks its workers kept, as ``collect-variants``
does from its shards, so a rule whose side fills within a run stops forking."""

from __future__ import annotations

import pytest

from price_predictor.domain.worker_status import SlotStatuses, directed_fork_tally_document
from sealed.domain.directed_fork_needs import DirectedForkNeeds, DirectedRuleNeed
from tests.unit.worker_status_line import directed_counts, status_line

#: 118.12b and 704.5f as `rules_catalogue.py directed-rules --probabilities-output`
#: writes them from `rules/directed-fork-counts.json` (fires 0 records, its one
#: registered side; does_not_fire 6814 records, over the count).
NEEDS = DirectedForkNeeds(count_per_side=100, rules={
    "118.12b": DirectedRuleNeed("fires", 0),
    "704.5f": DirectedRuleNeed("does_not_fire", 6814),
})


def _tally(kept: dict[str, int]) -> dict[str, dict[str, int]]:
    """The supervisor's summed tally after one worker's status line reported ``kept``."""
    slot = SlotStatuses()
    slot.add_line(status_line("a", directed={
        rule: directed_counts(matched=n + 5, drawn=n, taken=n, kept=n)
        for rule, n in kept.items()}))
    return directed_fork_tally_document([slot])


def test_the_starting_probabilities_follow_the_file_s_records():
    assert NEEDS.probabilities() == {"118.12b": 1.0, "704.5f": 0.0}


def test_a_rule_at_1_whose_side_reaches_the_count_through_kept_forks_is_at_0():
    assert NEEDS.probabilities(_tally({"118.12b": 100}))["118.12b"] == 0.0


def test_a_rule_partly_filled_by_kept_forks_takes_the_share_still_missing():
    assert NEEDS.probabilities(_tally({"118.12b": 40}))["118.12b"] == pytest.approx(0.6)


def test_kept_forks_of_an_over_filled_rule_change_nothing():
    assert NEEDS.probabilities(_tally({"704.5f": 7}))["704.5f"] == 0.0


def test_a_tallied_rule_the_file_does_not_name_gets_no_probability():
    assert set(NEEDS.probabilities(_tally({"205.1a": 3}))) == {"118.12b", "704.5f"}

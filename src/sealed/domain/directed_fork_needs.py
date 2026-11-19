"""What each listed directed-fork rule still needs within a ``match-outcomes``
run, and the probability that follows (D8, finding 6.1).

A rule's registered variants all make one side (``fires`` or
``does_not_fire``), so each fork it keeps adds a game to that side. The
supervisor starts from the records the side has in the counts
``--directed-fork-probabilities`` names, adds the forks the workers' status
lines report kept (``price_predictor.domain.worker_status``) and
recomputes the rule's probability with the formula ``collect-variants`` uses.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from price_predictor.domain.directed_fork_probabilities import directed_fork_probability

#: The tally's count of the forks a rule kept.
KEPT = "kept"


@dataclass(frozen=True)
class DirectedRuleNeed:
    """The one side a rule's registered variants make, and the records the
    counts give it there."""

    side: str
    records: int


@dataclass(frozen=True)
class DirectedForkNeeds:
    """Each rule's need, and the records a side needs."""

    count_per_side: int
    rules: Mapping[str, DirectedRuleNeed]

    def probabilities(
        self, tally: Mapping[str, Mapping[str, int]] | None = None,
    ) -> dict[str, float]:
        """Each rule's probability once the forks ``tally`` (``{rule: {count
        name: n}}``, the run's summed tally) reports kept are added to its side;
        without a tally, the starting probabilities. A tallied rule this does
        not name gets none."""
        tally = tally or {}
        return {
            rule: directed_fork_probability(
                [need.records + tally.get(rule, {}).get(KEPT, 0)], self.count_per_side)
            for rule, need in self.rules.items()
        }

"""The directed-fork probabilities a ``collect-variants`` run hands its
workers: recomputed from the forks its shards hold, before the first round
and after each one (D8)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from effects.domain.directed_sides import DirectedForkGames, directed_side_counts, forkable_sides
from price_predictor.domain.directed_fork_probabilities import directed_fork_probability

if TYPE_CHECKING:
    from effects.application.collect_variants import CollectVariantsConfig
    from effects.domain.rule_catalogue import Polarity
    from effects.infrastructure.collector_connector import CollectorSupervisor

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DirectedSteering:
    """The listed rules' sides and the count each side needs; a run listing
    no rule has no sides and sets nothing."""

    sides: dict[str, dict[str, Polarity]] = field(default_factory=dict)
    count: int = 1

    @classmethod
    def of(cls, config: CollectVariantsConfig) -> DirectedSteering:
        rules = config.caps.directed_fork_rules
        if not rules:
            return cls()
        from effects.infrastructure.rules_catalogue_io import (
            load_registered_variants,
            load_scenarios,
        )

        sides = forkable_sides(load_scenarios(config.rules_dir), rules,
                               load_registered_variants(config.rules_dir))
        return cls(sides, config.steer_floor)

    def steer(self, supervisor: CollectorSupervisor, forks: DirectedForkGames) -> None:
        """Hand ``supervisor`` the probabilities ``forks`` give, for the
        workers of the rounds that follow, and print them."""
        if not self.sides:
            return
        counts = directed_side_counts(self.sides, forks)
        probabilities = {rule: directed_fork_probability(c, self.count)
                         for rule, c in counts.items()}
        supervisor.set_directed_rule_probabilities(probabilities)
        logger.info("directed fork probabilities: %s", ", ".join(
            f"{rule} {p:.2f} ({min(counts[rule])}/{self.count})"
            for rule, p in probabilities.items()))

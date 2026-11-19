"""What a collection run counts of its directed forks, and the probabilities
that follow (D8).

A directed fork's records carry ``aimed_at`` and ``directed = {variant,
operation}``. ``aimed_at`` names every rule of the fork's registration: one,
or several rules sharing a board, a binding and an edit (the three meld rules,
Aura swap's two), whose one fork counts under each. A registration forks at
most once per game, so a rule's kept forks are the distinct games holding a
record naming it. A fork's side is the
polarity its variant has in the catalogue. The sides of a rule are the
polarities of its registered variants that the catalogue classifies into an
operation: a catalogue variant the Java registry does not register is never
forked, so its side would never fill, and the side the live games already show
is not made by a fork and is not counted.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass, field

from effects.domain.directed_fork_probabilities import directed_fork_probability
from effects.domain.fork_operation import ForkOperation
from effects.domain.records import EffectRecord
from effects.domain.rule_catalogue import Polarity, ScenarioCatalogue

#: The polarities a fork's records are counted under; ``other`` names no side.
COUNTED_SIDES = frozenset({Polarity.FIRES, Polarity.DOES_NOT_FIRE})


def forkable_sides(
    catalogue: ScenarioCatalogue, rules: Iterable[str],
    registered: Mapping[str, Collection[str]],
) -> dict[str, dict[str, Polarity]]:
    """``rule -> {variant label -> polarity}`` over each rule's non-base
    variants classified into an operation and registered for it."""
    by_rule = catalogue.by_rule()
    return {
        rule: {
            variant.label: variant.polarity
            for scenario in by_rule.get(rule, ())
            for variant in scenario.variants[1:]
            if isinstance(variant.operation, ForkOperation)
            and variant.label in registered.get(rule, ())
        }
        for rule in rules
    }


@dataclass
class DirectedForkGames:
    """The games holding a directed fork's record, per (rule, variant label):
    a record naming several rules credits its game to each."""

    games: dict[tuple[str, str], set[str]] = field(default_factory=dict)

    def add(self, record: EffectRecord) -> None:
        if record.directed is None:
            return
        for rule in record.aimed_at or ():
            self.games.setdefault((rule, record.directed.variant), set()).add(record.game_id)

    def merge(self, other: DirectedForkGames) -> None:
        for key, games in other.games.items():
            self.games.setdefault(key, set()).update(games)

    def drop_games(self, cut: Collection[str]) -> None:
        """Leave the cut games out: ``build-corpus`` drops them."""
        kept = {key: games - set(cut) for key, games in self.games.items()}
        self.games = {key: games for key, games in kept.items() if games}

    def count(self, rule: str, labels: Collection[str]) -> int:
        """The games with a fork of ``rule`` by any variant of ``labels``."""
        return sum(len(games) for (r, label), games in self.games.items()
                   if r == rule and label in labels)


def directed_side_counts(
    sides: Mapping[str, Mapping[str, Polarity]], forks: DirectedForkGames,
) -> dict[str, list[int]]:
    """Each rule's forks per counted side; a rule whose variants name no
    counted side is left out."""
    out: dict[str, list[int]] = {}
    for rule, variants in sides.items():
        polarities = sorted({p for p in variants.values() if p in COUNTED_SIDES})
        if polarities:
            out[rule] = [
                forks.count(rule, {label for label, p in variants.items() if p is polarity})
                for polarity in polarities
            ]
    return out


def directed_rule_probabilities(
    sides: Mapping[str, Mapping[str, Polarity]], forks: DirectedForkGames, count: int,
) -> dict[str, float]:
    """Each rule's probability from the forks it has on each of its sides."""
    return {rule: directed_fork_probability(counts, count)
            for rule, counts in directed_side_counts(sides, forks).items()}

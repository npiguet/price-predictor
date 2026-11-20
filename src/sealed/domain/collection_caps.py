"""The caps a collecting run applies, and how they reach the worker.

The one definition of the effect-record cap flags' values, names and
defaults: ``sealed`` never imports ``effects``, so it lives here, where both
read it (``effects.domain.collection_caps`` re-exports it). The JVM's
``CollectionCaps.defaults()`` restates the defaults in Java;
``tests/unit/effects/domain/test_collection_caps.py`` pins the two.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields

DEFAULT_MANA_CAP = 1
DEFAULT_PLAYABILITY_RATE = 0.01
DEFAULT_ZONE_CANDIDATE_RATE = 0.0001
DEFAULT_PLAIN_CANDIDATE_RATE = 0.001
DEFAULT_INTERVENTIONS_PER_GAME = 2
DEFAULT_PROBES_PER_GAME = 2
#: Every damage-step keyword gate 2 checks, in the spelling the JVM compares a
#: normalized Forge keyword against; ``effects``' gate-2 table is pinned to it
#: by ``test_collection_caps.py``.
DEFAULT_PROBE_KEYWORDS = (
    "first_strike,double_strike,deathtouch,lifelink,trample,indestructible,"
    "wither,infect"
)
DEFAULT_RANDOM_SEAT_SHARE = 0.0
#: Only ever travels beside a share of 0: a share above zero without an
#: explicit ``P`` is refused (FR-026).
DEFAULT_RANDOM_SEAT_PROBABILITY = 0.0
#: Directed forks (FR-012): at most two a game, and a quarter of the
#: boundaries where an enabled rule's precondition holds.
DEFAULT_DIRECTED_FORKS_PER_GAME = 2
DEFAULT_DIRECTED_FORK_PROBABILITY = 0.25
#: The records one game may write, its forks' included (FR-041a); past it the
#: worker withholds the rest and names the game.
DEFAULT_MAX_RECORDS_PER_GAME = 5000
#: A listed rule's own directed-fork probability travels as this prefix and
#: the rule id (``effect.directed.probability.704.5f``).
DIRECTED_RULE_PROBABILITY_PREFIX = "effect.directed.probability."


@dataclass(frozen=True, slots=True)
class CollectionCaps:
    """What one worker may collect, per game; enforced in the JVM, which is
    why the caps travel as system properties."""

    #: Records per unique (mana ability, mana produced) per game.
    mana_cap: int = DEFAULT_MANA_CAP
    #: Stage 1 of decision sampling (FR-033b): share of enumerated decision
    #: candidates evaluated; the legality subkinds are coalesced instead.
    playability_rate: float = DEFAULT_PLAYABILITY_RATE
    #: Stage 2: share of all enumerated candidates written, both stages
    #: together, for a candidate failing ``zone`` (and otherwise only cost or
    #: timing), and for a plain one; a special candidate is written whenever
    #: stage 1 keeps it.
    zone_candidate_rate: float = DEFAULT_ZONE_CANDIDATE_RATE
    plain_candidate_rate: float = DEFAULT_PLAIN_CANDIDATE_RATE
    interventions_per_game: int = DEFAULT_INTERVENTIONS_PER_GAME
    probes_per_game: int = DEFAULT_PROBES_PER_GAME
    #: Comma-separated keywords to probe; empty takes no probe fork at all.
    probe_keywords: str = DEFAULT_PROBE_KEYWORDS
    #: Share of matches with one random seat (FR-021).
    random_seat_share: float = DEFAULT_RANDOM_SEAT_SHARE
    #: ``P``: the chance the random seat draws uniformly from the legal options.
    random_seat_probability: float = DEFAULT_RANDOM_SEAT_PROBABILITY
    #: The rules a ``collect-coverage`` or ``collect-variants`` run aims at,
    #: stamped on every record as ``aimed_at`` (FR-009a); empty for a run aimed
    #: at no rule in particular, None for a run whose records carry no
    #: ``aimed_at`` at all. Travels by its presence: None sends no property.
    aimed_at: tuple[str, ...] | None = None
    #: Whether each worker appends its games' dropped-event counts beside its
    #: shard, for the supervisor to sum and print when the run ends (FR-041).
    count_dropped_events: bool = False
    #: The rules directed forks are taken for (``collect-variants``; never
    #: ``collect-coverage``, whose ``--directed-fork-rules`` only aims its
    #: records); empty takes none. Comma-joined, blank when empty.
    directed_fork_rules: tuple[str, ...] = ()
    directed_forks_per_game: int = DEFAULT_DIRECTED_FORKS_PER_GAME
    #: The chance a boundary where an enabled rule's precondition holds forks.
    directed_fork_probability: float = DEFAULT_DIRECTED_FORK_PROBABILITY
    #: A listed rule's own probability, in [0, 1]; a rule without one takes
    #: ``directed_fork_probability``, and one at 0 is never forked.
    directed_rule_probabilities: dict[str, float] = field(default_factory=dict)
    #: The records one game may write (FR-041a): a game whose AIs loop is cut
    #: off there, its withheld records counted and the game named.
    max_records_per_game: int = DEFAULT_MAX_RECORDS_PER_GAME

    def __post_init__(self) -> None:
        for rule, probability in self.directed_rule_probabilities.items():
            if not 0.0 <= probability <= 1.0 or math.isnan(probability):
                raise ValueError(
                    f"the directed-fork probability of rule {rule} must be in [0, 1], "
                    f"got {probability}"
                )

    def as_system_properties(self) -> dict[str, str]:
        """The `-D effect.*` properties the worker reads them back from."""
        properties = {
            property_name(field.name): _property_value(getattr(self, field.name))
            for field in fields(self)
            if field.name not in ("aimed_at", "directed_rule_probabilities")
        }
        properties.update(directed_rule_probability_properties(
            self.directed_rule_probabilities))
        if self.aimed_at is not None:
            properties[property_name("aimed_at")] = ",".join(self.aimed_at)
        return properties


def property_name(field_name: str) -> str:
    """The ``-D`` property a cap travels as: ``effect.`` and the field name,
    its underscores as dots."""
    return f"effect.{field_name.replace('_', '.')}"


#: The property ``--count-dropped-events`` travels to the worker as.
COUNT_DROPPED_EVENTS_PROPERTY = property_name("count_dropped_events")


def directed_rule_probability_properties(probabilities: Mapping[str, float]) -> dict[str, str]:
    """Each rule's own probability as its ``-D`` property."""
    return {DIRECTED_RULE_PROBABILITY_PREFIX + rule: str(p) for rule, p in probabilities.items()}


def _property_value(value: object) -> str:
    """A flag as the ``true``/``false`` the worker's flag reader expects, and a
    list comma-joined, as its list reader splits it."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, tuple):
        return ",".join(value)
    return str(value)


def _comma_list(value: str) -> list[str]:
    return [item for item in value.split(",") if item]


#: How a cap's property value reads back, by its field's annotation.
_READERS: dict[str, Callable[[str], object]] = {
    "int": int,
    "float": float,
    "str": str,
    "bool": lambda value: value == "true",
    "tuple[str, ...]": _comma_list,
    "tuple[str, ...] | None": _comma_list,
}


def caps_manifest(properties: Mapping[str, object]) -> dict[str, object]:
    """The ``-D`` cap properties as ``run.json`` records the resolved caps: the
    field names and types, ``aimed_at`` null for a run aimed at no rule (what
    ``effects``' ``caps_document`` writes for the same caps)."""
    document: dict[str, object] = {"aimed_at": None}
    for spec in fields(CollectionCaps):
        name = property_name(spec.name)
        if spec.name != "directed_rule_probabilities" and name in properties:
            document[spec.name] = _READERS[spec.type](str(properties[name]))
    document["directed_rule_probabilities"] = {
        name[len(DIRECTED_RULE_PROBABILITY_PREFIX):]: float(str(value))
        for name, value in properties.items()
        if name.startswith(DIRECTED_RULE_PROBABILITY_PREFIX)
    }
    return document

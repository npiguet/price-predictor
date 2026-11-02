"""The caps a collecting run applies, and how they reach the worker."""

from __future__ import annotations

from dataclasses import dataclass, fields

from effects.domain.damage_step_keywords import DAMAGE_STEP_KEYWORDS

DEFAULT_MANA_CAP = 1
DEFAULT_PLAYABILITY_RATE = 0.01
DEFAULT_ZONE_CANDIDATE_RATE = 0.0001
DEFAULT_PLAIN_CANDIDATE_RATE = 0.001
DEFAULT_INTERVENTIONS_PER_GAME = 2
DEFAULT_PROBES_PER_GAME = 2
#: In gate 2's table spelling, which is what the JVM compares a normalized
#: Forge keyword against.
DEFAULT_PROBE_KEYWORDS = ",".join(row.keyword for row in DAMAGE_STEP_KEYWORDS)
DEFAULT_RANDOM_SEAT_SHARE = 0.0
#: Only ever travels beside a share of 0: a share above zero without an
#: explicit ``P`` is refused (FR-026).
DEFAULT_RANDOM_SEAT_PROBABILITY = 0.0


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

    def as_system_properties(self) -> dict[str, str]:
        """The `-D effect.*` properties the worker reads them back from."""
        properties = {
            _property_name(field.name): _property_value(getattr(self, field.name))
            for field in fields(self) if field.name != "aimed_at"
        }
        if self.aimed_at is not None:
            properties[_property_name("aimed_at")] = ",".join(self.aimed_at)
        return properties


def _property_name(field_name: str) -> str:
    return f"effect.{field_name.replace('_', '.')}"


def _property_value(value: object) -> str:
    """A flag as the ``true``/``false`` the worker's flag reader expects."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)

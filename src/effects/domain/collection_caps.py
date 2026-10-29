"""The caps a collecting run applies, and how they reach the worker."""

from __future__ import annotations

from dataclasses import dataclass, fields

from effects.domain.damage_step_keywords import DAMAGE_STEP_KEYWORDS

DEFAULT_MANA_CAP = 1
DEFAULT_PLAYABILITY_RATE = 0.1
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
    #: Share of `decision`-subkind logging points kept; the legality subkinds
    #: are coalesced instead.
    playability_rate: float = DEFAULT_PLAYABILITY_RATE
    interventions_per_game: int = DEFAULT_INTERVENTIONS_PER_GAME
    probes_per_game: int = DEFAULT_PROBES_PER_GAME
    #: Comma-separated keywords to probe; empty takes no probe fork at all.
    probe_keywords: str = DEFAULT_PROBE_KEYWORDS
    #: Share of matches with one random seat (FR-021).
    random_seat_share: float = DEFAULT_RANDOM_SEAT_SHARE
    #: ``P``: the chance the random seat draws uniformly from the legal options.
    random_seat_probability: float = DEFAULT_RANDOM_SEAT_PROBABILITY

    def as_system_properties(self) -> dict[str, str]:
        """The `-D effect.*` properties the worker reads them back from."""
        return {
            f"effect.{field.name.replace('_', '.')}": str(getattr(self, field.name))
            for field in fields(self)
        }

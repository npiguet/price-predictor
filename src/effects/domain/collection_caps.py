"""The caps a collecting run applies, and how they reach the worker.

Every one is a **per-worker-process** quantity that the Python supervisor cannot
observe: how many records one mana ability has already contributed, how often a
decision point has been sampled, how many forks a game has spent. The supervisor
sets them and the JVM enforces them, which is why they travel as system
properties rather than as arguments.

Kept here rather than in the CLI so the three commands that collect — the sealed
match supervisor, the coverage collector and the variant collector — cannot
drift into passing different names for the same knob.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

#: Records per unique mana ability, per game. Lands are the reason: a Mountain
#: taps a dozen times a game for the same R, and the repeats are the same
#: observation over a board that barely moved.
DEFAULT_MANA_CAP = 1
DEFAULT_PLAYABILITY_RATE = 0.1
DEFAULT_INTERVENTIONS_PER_GAME = 2
DEFAULT_PROBES_PER_GAME = 2


@dataclass(frozen=True, slots=True)
class CollectionCaps:
    """What one worker may collect, per game."""

    #: Records per unique (mana ability, mana produced) per game. A dual land
    #: producing G and producing U are two observations of one line, so the
    #: produced mana is part of the key rather than of the count.
    mana_cap: int = DEFAULT_MANA_CAP
    #: Share of `decision`-subkind logging points kept. The AI evaluates every
    #: candidate at every priority; the legality subkinds are not sampled,
    #: because identical answers are coalesced instead.
    playability_rate: float = DEFAULT_PLAYABILITY_RATE
    #: Stage three. Forks that force an ability no game plays.
    interventions_per_game: int = DEFAULT_INTERVENTIONS_PER_GAME
    #: Stage three. Forks that re-run a combat with a keyword stripped.
    probes_per_game: int = DEFAULT_PROBES_PER_GAME
    #: Comma-separated keywords to probe. Empty takes no fork at all, whatever
    #: the build state — the runtime switch is deliberately separate from the
    #: build decision gate 2 drives.
    probe_keywords: str = ""

    @classmethod
    def from_args(cls, args) -> CollectionCaps:
        """Read the flags `_add_cap_flags` installed, whichever command they sit on.

        Falls back per field rather than requiring all of them: the trainer and
        the evaluator carry no cap flags and still build configs from their own
        namespaces. ``field.default`` rather than a class attribute, because
        ``slots=True`` leaves none.
        """
        return cls(**{
            spec.name: getattr(args, spec.name, spec.default)
            for spec in fields(cls)
        })

    def as_system_properties(self) -> dict[str, str]:
        """The `-D` properties the worker reads them back from.

        Named `effect.*` like the other collection properties, so a worker
        started without instrumentation ignores the lot.
        """
        return {
            f"effect.{field.name.replace('_', '.')}": str(getattr(self, field.name))
            for field in fields(self)
        }

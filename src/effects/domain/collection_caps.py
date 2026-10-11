"""The caps a collecting run applies, and how they reach the worker.

Every one is a **per-worker-process** quantity that the Python supervisor cannot
observe: how many records one mana ability has already contributed, how often a
decision point has been sampled, how many forks a game has spent. The supervisor
sets them and the JVM enforces them, which is why they travel as system
properties rather than as arguments.

Kept here rather than in the CLI so the three commands that collect — the sealed
match supervisor, the coverage collector and the variant collector — cannot
drift into passing different names for the same knob.

Snapshot depth is not among them. Every record collects all four inclusion
tiers — referenced objects, the core board, the unreferenced stack, and
unreferenced hands and graveyards — because a depth that varied by collector or
by run made ``state.tiers`` a proxy for how a record was collected, which is
collection metadata the schema bars from being a model input.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

from effects.domain.damage_step_keywords import DAMAGE_STEP_KEYWORDS

#: Records per unique mana ability, per game. Lands are the reason: a Mountain
#: taps a dozen times a game for the same R, and the repeats are the same
#: observation over a board that barely moved.
DEFAULT_MANA_CAP = 1
DEFAULT_PLAYABILITY_RATE = 0.1
DEFAULT_INTERVENTIONS_PER_GAME = 2
DEFAULT_PROBES_PER_GAME = 2
#: Every damage-step keyword gate 2 checks, in its table's order and spelling,
#: which is the spelling the JVM compares a normalized Forge keyword against.
DEFAULT_PROBE_KEYWORDS = ",".join(row.keyword for row in DAMAGE_STEP_KEYWORDS)
#: Share of what-if legality queries kept; real decisions are always written.
#: Off: the AI re-asks one combat many times while planning, so a query is
#: kept in proportion to how long the AI deliberated over its board.
DEFAULT_LEGALITY_RATE = 0.0
#: Share of matches with one random seat (FR-021). Off unless a run asks.
DEFAULT_RANDOM_SEAT_SHARE = 0.0
#: The random seat's ``P``. No default a run could rely on: a share above zero
#: without it is refused before any game (FR-026), so the 0 here only ever
#: travels beside a share of 0.
DEFAULT_RANDOM_SEAT_PROBABILITY = 0.0


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
    #: Comma-separated keywords to probe; all eight damage-step keywords unless
    #: a run narrows the list. Empty takes no fork at all, whatever the build
    #: state — the runtime switch is deliberately separate from the build
    #: decision gate 2 drives.
    probe_keywords: str = DEFAULT_PROBE_KEYWORDS
    #: Share of the `legality` playability subkinds kept, sampled *after* the
    #: dedup so a coalesced answer is not counted twice. Its own rate rather
    #: than `playability_rate`'s, because the two subkinds arrive at wildly
    #: different volumes from the same priority pass.
    legality_rate: float = DEFAULT_LEGALITY_RATE
    #: Share of matches in which one seat, chosen at random per match, is the
    #: random seat (FR-021). Per worker for the same reason as every cap: the
    #: match draw happens in the JVM.
    random_seat_share: float = DEFAULT_RANDOM_SEAT_SHARE
    #: ``P``: the chance the random seat draws uniformly from the legal
    #: options at a decision point instead of taking the Forge AI's choice.
    random_seat_probability: float = DEFAULT_RANDOM_SEAT_PROBABILITY

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

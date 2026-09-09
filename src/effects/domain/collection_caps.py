"""The caps a collecting run applies, and how they reach the worker.

Every one is a **per-worker-process** quantity that the Python supervisor cannot
observe: how many records one mana ability has already contributed, how often a
decision point has been sampled, how many forks a game has spent. The supervisor
sets them and the JVM enforces them, which is why they travel as system
properties rather than as arguments.

Kept here rather than in the CLI so the three commands that collect — the sealed
match supervisor, the coverage collector and the variant collector — cannot
drift into passing different names for the same knob.

``snapshot_tiers`` is not a cap but a depth, and it lives here anyway because
it travels the same way and drifted the same way: the JVM read
``effect.snapshot.tiers``, ``MatchWorkerMain`` documented it, and no supervisor
could set it, so stage three's hand-and-graveyard tier took a code edit to
request. A knob one side reads and the other never writes is invisible from
both, which is what the field set of this class is for.
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
DEFAULT_LEGALITY_RATE = 0.1
DEFAULT_SNAPSHOT_TIERS = "1,2,3"

#: The four inclusion depths, in the order they widen: 1 referenced objects
#: (target, affected, source), 2 core — global, battlefield and command-zone
#: effect cards, 3 the unreferenced stack, 4 unreferenced hands and graveyards.
#: Cumulative, so a run names a prefix and never a subset.
SNAPSHOT_TIERS = (1, 2, 3, 4)


def parse_snapshot_tiers(text: str) -> tuple[int, ...]:
    """The tier vector a `--snapshot-tiers` string names, or raise.

    Refused here rather than in the JVM because both Java failure modes are
    worse than an argument error: an unparseable vector falls back to the
    default *silently*, and a parseable non-prefix throws inside
    ``SnapshotBuilder`` once a game is already running, hours into a pass on
    one worker while the others keep collecting. The rule is the builder's:
    a prefix of ``[1,2,3,4]`` holding at least 1 and 2, because tiers are
    cumulative and the battlefield goes into every snapshot regardless — a
    vector omitting 2 would tell a reader the board was uncollected while the
    board sits in ``entities``.
    """
    try:
        tiers = tuple(
            int(part.strip()) for part in text.split(",") if part.strip()
        )
    except ValueError:
        raise ValueError(
            f"snapshot tiers must be comma-separated integers, got {text!r}"
        ) from None
    if tiers != SNAPSHOT_TIERS[:len(tiers)] or len(tiers) < 2:
        raise ValueError(
            "snapshot tiers must be a prefix of "
            f"{list(SNAPSHOT_TIERS)} holding at least 1 and 2, got "
            f"{list(tiers)}"
        )
    return tiers


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
    #: Share of the `legality` playability subkinds kept, sampled *after* the
    #: dedup so a coalesced answer is not counted twice. Its own rate rather
    #: than `playability_rate`'s, because the two subkinds arrive at wildly
    #: different volumes from the same priority pass.
    legality_rate: float = DEFAULT_LEGALITY_RATE
    #: The snapshot inclusion depth, comma-separated, as a prefix of
    #: ``1,2,3,4``. A run-level value rather than a per-collector one: a depth
    #: chosen per call site makes ``state.tiers`` a proxy for *how* a record was
    #: collected, and the first corpus put tier 4 on the interventional records
    #: and nowhere else — a perfect predictor of a flag the schema forbids the
    #: model to see. Stage three wants ``1,2,3,4``, which is hands and
    #: graveyards.
    snapshot_tiers: str = DEFAULT_SNAPSHOT_TIERS

    def __post_init__(self) -> None:
        # The one cap whose value the JVM can reject: parse it here so a
        # mistyped vector fails the command rather than the fifth game.
        parse_snapshot_tiers(self.snapshot_tiers)

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

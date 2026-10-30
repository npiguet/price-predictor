"""What a validate-corpus check reports, and the shared ways of judging a rate.

A :class:`Finding` with no threshold is watched: it prints ``[WATCH]`` and
never fails a run. :func:`floor_finding` and :func:`ceiling_finding` are the two
ways a measured share is judged, and :class:`Share` is how one is counted.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import Protocol

from effects.domain.records import EffectRecord

#: How many offending examples a finding's detail carries.
EXAMPLES = 5


@dataclass(frozen=True, slots=True)
class Thresholds:
    """Where each measured rate stops being acceptable.

    A ``float | None`` field defaulting to ``None`` is watched, not judged. Each
    field is a ``validate-corpus`` flag, its ``help`` metadata the flag's help.
    """

    min_keyed_rate: float = field(default=0.95, metadata={"help": (
        "Share of records naming a line that must resolve it to a printed key "
        "(default: %(default)s)"
    )})
    max_duplicate_rate: float = field(default=0.02, metadata={"help": (
        "Share of a kind's records that may repeat an earlier record of the "
        "same kind in the same game (default: %(default)s)"
    )})
    #: A worker killed mid-game truncates its last block, leaving a few
    #: unpaired halves.
    max_unpaired_link_rate: float = field(default=0.02, metadata={"help": (
        "Share of link ids that may lack one activation and one resolution "
        "half (default: %(default)s)"
    )})
    #: Two forty-card sealed decks plus tokens sit far below this.
    max_names_per_game: int = field(default=120, metadata={"help": (
        "Distinct card names one game_id may show before it reads as more "
        "than one game (default: %(default)s)"
    )})
    min_cost_evidence: int = field(default=200, metadata={"help": (
        "Activation records needed before a cost channel reading zero means "
        "the collector rather than the pool (default: %(default)s)"
    )})
    #: A deferred half can legitimately lag by a turn; a game boundary jumps
    #: many.
    turn_jump_tolerance: int = field(default=1, metadata={"help": (
        "Turns a snapshot may sit below the highest already seen in its game "
        "before that reads as a game boundary; deferred mana-reservoir flushes "
        "are exempt (default: %(default)s)"
    )})
    max_duplicate_event_rate: float = field(default=0.02, metadata={"help": (
        "Share of events that may repeat another event in the same record, "
        "which --max-duplicate-rate's whole-record comparison cannot see "
        "(default: %(default)s)"
    )})
    #: Far above ~1:1 the sampler stopped drawing negatives, and positives
    #: alone teach the base rate.
    max_trigger_fired_share: float = field(default=0.65, metadata={"help": (
        "Share of trigger records that may report fired=true; the negative "
        "sampler targets ~1:1 (default: %(default)s)"
    )})
    min_zone_change_from_zone_rate: float | None = field(default=None, metadata={
        "help": (
            "Share of zone_change events that must carry from_zone; unset, it "
            "is reported as [WATCH] without failing the run"
        ),
    })
    min_attributed_rate: float | None = field(default=None, metadata={"help": (
        "Share of resolution events that must name a producing clause; unset, "
        "it is reported as [WATCH] without failing the run"
    )})
    min_fork_attributed_rate: float | None = field(default=None, metadata={"help": (
        "Share of events on fork records that must name a producing clause, "
        "apart from --min-attributed-rate; unset, it is reported as [WATCH]"
    )})
    #: Some hooks genuinely name no causing object, so the healthy share is
    #: below 1 by an unknown amount.
    min_cause_rate: float | None = field(default=None, metadata={"help": (
        "Share of the events whose schema row declares a cause that must "
        "populate it; unset, it is reported as [WATCH]"
    )})
    #: A replacement scripted with ``ReplacementResult$`` returns
    #: ``replaced``/``updated`` having run no ability, so the healthy share is
    #: below 1 by an unknown amount.
    min_rewrite_replaced_by_rate: float | None = field(default=None, metadata={
        "help": (
            "Share of the rewrite records whose result ran an ability that must "
            "name it; unset, it is reported as [WATCH]"
        ),
    })
    #: A fork is taken at the moment it mirrors, so no disagreement is healthy.
    max_mirror_turn_disagreement_rate: float = field(default=0.0, metadata={"help": (
        "Share of fork records whose turn may differ from the record they "
        "mirror, a stale snapshot (default: %(default)s)"
    )})


@dataclass(frozen=True, slots=True)
class Finding:
    """One invariant's result: what it measured, and whether that is allowed.
    A ``watched`` finding has no threshold and cannot fail a run."""

    name: str
    ok: bool
    measured: str
    detail: tuple[str, ...] = ()
    watched: bool = False

    def lines(self) -> tuple[str, ...]:
        tag = "WATCH" if self.watched else ("PASS" if self.ok else "FAIL")
        head = f"[{tag}] {self.name}: {self.measured}"
        return (head, *(f"         {line}" for line in self.detail))


class Findings(list[Finding]):
    """Every finding of one pass, in report order."""

    def broken(self) -> list[Finding]:
        return [finding for finding in self if not finding.ok]

    def summary(self) -> list[str]:
        """The closing lines: what broke, or that every judged invariant held."""
        broken = self.broken()
        watched = sum(1 for finding in self if finding.watched)
        judged = len(self) - watched
        if broken:
            return [
                f"{len(broken)} of {judged} judged invariants broken:",
                *(f"  {finding.name}" for finding in broken),
            ]
        return [f"all {judged} judged invariants hold ({watched} watched, not judged)"]


def share_of(part: int, whole: int) -> float:
    """``part`` as a share of ``whole``, and 0.0 of an empty one."""
    return part / whole if whole else 0.0


@dataclass(slots=True)
class Share:
    """How many of the cases ``seen`` were a hit."""

    hit: int = 0
    seen: int = 0

    def note(self, hit: bool) -> None:
        self.hit += hit
        self.seen += 1

    @property
    def rate(self) -> float:
        return share_of(self.hit, self.seen)

    @classmethod
    def total(cls, shares: Iterable[Share]) -> Share:
        """Every case of ``shares`` as one."""
        out = cls()
        for share in shares:
            out.hit += share.hit
            out.seen += share.seen
        return out


def histogram(counts: Iterable[tuple[str, int]]) -> str:
    """``name count, name count``, in the order given, or ``none`` when empty."""
    return ", ".join(f"{name} {count}" for name, count in counts) or "none"


def keep_example(examples: list[str], example: str) -> None:
    """Record ``example`` unless ``examples`` already holds :data:`EXAMPLES`."""
    if len(examples) < EXAMPLES:
        examples.append(example)


def floor_finding(
    name: str,
    floor: float | None,
    seen: int,
    rate: float,
    measured: Callable[[str], str],
    detail: Iterable[str],
) -> Finding:
    """A rate judged against a floor, watched when the floor is ``None``.

    ``measured`` is handed the floor's label to place in its own sentence. An
    empty window holds whatever the floor.
    """
    limit = "watched, no floor" if floor is None else f"floor {floor:.1%}"
    return Finding(
        name=name,
        ok=floor is None or not seen or rate >= floor,
        watched=floor is None,
        measured=measured(limit),
        detail=tuple(detail),
    )


def ceiling_finding(
    name: str,
    ceiling: float,
    rate: float,
    measured: Callable[[str], str],
    detail: Iterable[str],
) -> Finding:
    """A rate judged against a ceiling, :func:`floor_finding`'s mirror.

    ``measured`` is handed the ceiling's label to place in its own sentence.
    An empty window's rate is 0.0, so it holds whatever the ceiling.
    """
    return Finding(
        name=name,
        ok=rate <= ceiling,
        measured=measured(f"limit {ceiling:.1%}"),
        detail=tuple(detail),
    )


class Check(Protocol):
    """One invariant: running counts filled by the pass, then its findings
    (several where they read the same counts)."""

    def add(self, record: EffectRecord) -> None: ...

    def findings(self, limits: Thresholds) -> Iterator[Finding]: ...

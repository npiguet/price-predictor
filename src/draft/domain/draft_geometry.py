"""Booster↔seat/pick geometry for a recorded draft (FR-016).

A ``DraftRecord`` is self-contained: the *ordering* of its ``boosters`` array
plus a small set of fixed conventions pins which seat made every pick, with no
external state. This module is the single owner of those conventions; the
data-gen supervisor (US1) and the training loader (US2) both reconstruct seat
observation histories through it.

Conventions (all 0-indexed seats; packs and picks are 1-indexed in the public
API to match the spec's prose):

- ``pod_size = len(seats)``, ``packs = len(boosters) / pod_size``,
  ``pack_size P = len(boosters[0].picks)`` — derived, never stored.
- ``boosters[k]`` belongs to ``pack_number = floor(k / pod_size) + 1`` and was
  *opened* by ``opening_seat = k mod pod_size``.
- Passing direction ``dir_p`` is ``+1`` for packs 1 & 3 (odd, pass left) and
  ``-1`` for pack 2 (even, pass right). The pick at 0-indexed position ``j`` of
  ``boosters[k]`` was made by seat ``(opening_seat + j * dir_p) mod pod_size``.

Pure functions, no torch — mirrors ``sealed/domain`` purity conventions.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Seat:
    """One drafter in the pod (``seats[i]`` of a record)."""

    agent: str
    deck: list[str]
    deck_score: float | None


@dataclass(frozen=True, slots=True)
class Booster:
    """One opened pack (``boosters[k]`` of a record).

    ``picks`` is the cards in the order they were drafted from this physical
    pack, fully drained so ``len(picks) == pack_size``.
    """

    set_code: str
    picks: list[str]


@dataclass(frozen=True, slots=True)
class DraftRecord:
    """One JSONL line: a complete, self-contained draft (data-model §1)."""

    draft_id: str
    run_id: str
    timestamp: str
    seats: list[Seat]
    boosters: list[Booster]


@dataclass(frozen=True, slots=True)
class DraftGeometry:
    """Derived sizes for one draft, plus the forward/inverse pick maps.

    Construct via :meth:`from_record` (or directly from sizes). All seat
    indices are 0-based; ``pack`` and ``pick`` arguments are 1-based.
    """

    pod_size: int
    packs: int
    pack_size: int

    @classmethod
    def from_record(cls, record: DraftRecord) -> "DraftGeometry":
        pod_size = len(record.seats)
        n_boosters = len(record.boosters)
        if pod_size == 0:
            raise ValueError("DraftRecord has no seats")
        if n_boosters % pod_size != 0:
            raise ValueError(
                f"boosters ({n_boosters}) is not a multiple of pod_size "
                f"({pod_size}); record geometry is inconsistent"
            )
        packs = n_boosters // pod_size
        pack_size = len(record.boosters[0].picks)
        return cls(pod_size=pod_size, packs=packs, pack_size=pack_size)

    # -- forward map (booster index → pack / opening seat / pick author) ----- #

    def pack_number(self, booster_index: int) -> int:
        """1-based pack number of ``boosters[booster_index]``."""
        return booster_index // self.pod_size + 1

    def opening_seat(self, booster_index: int) -> int:
        """0-based seat that opened ``boosters[booster_index]``."""
        return booster_index % self.pod_size

    @staticmethod
    def direction(pack_number: int) -> int:
        """Passing direction ``dir_p``: +1 for odd packs (1, 3, …), -1 for even."""
        return 1 if pack_number % 2 == 1 else -1

    def seat_of_pick(self, booster_index: int, offset: int) -> int:
        """0-based seat that made the pick at 0-based ``offset`` of a booster."""
        dir_p = self.direction(self.pack_number(booster_index))
        return (self.opening_seat(booster_index) + offset * dir_p) % self.pod_size

    # -- inverse map ((seat, pack, pick) → booster index + offset) ----------- #

    def booster_for_pick(self, seat: int, pack: int, pick: int) -> tuple[int, int]:
        """Booster index + 0-based offset for a target ``(seat, pack, pick)``.

        ``pack`` and ``pick`` are 1-based. Returns ``(booster_index, offset)``
        where ``offset = pick - 1`` and the booster is the physical pack seat
        ``seat`` is holding at that pick.
        """
        if not 1 <= pack <= self.packs:
            raise ValueError(f"pack {pack} out of range 1..{self.packs}")
        if not 1 <= pick <= self.pack_size:
            raise ValueError(f"pick {pick} out of range 1..{self.pack_size}")
        offset = pick - 1
        dir_p = self.direction(pack)
        s_open = (seat - offset * dir_p) % self.pod_size
        booster_index = (pack - 1) * self.pod_size + s_open
        return booster_index, offset

    def legal_actions(
        self, record: DraftRecord, seat: int, pack: int, pick: int,
    ) -> list[str]:
        """Cards still in the pack when ``seat`` makes ``(pack, pick)``.

        The legal action set is ``boosters[k].picks[offset:]`` — the card taken
        at this pick plus every card passed on (drafted later from this pack).
        """
        k, offset = self.booster_for_pick(seat, pack, pick)
        return list(record.boosters[k].picks[offset:])

    def taken_card(
        self, record: DraftRecord, seat: int, pack: int, pick: int,
    ) -> str:
        """The card ``seat`` actually took at ``(pack, pick)``."""
        k, offset = self.booster_for_pick(seat, pack, pick)
        return record.boosters[k].picks[offset]

    def drafted_pool(self, record: DraftRecord, seat: int) -> list[str]:
        """Reconstruct one seat's full drafted pool (``packs × pack_size`` cards).

        Walks every booster position and collects the cards this seat authored,
        in (pack, pick) order. Used by the US1 supervisor to rebuild the pool a
        deck is then built and scored from.
        """
        pool: list[str] = []
        for pack in range(1, self.packs + 1):
            for pick in range(1, self.pack_size + 1):
                pool.append(self.taken_card(record, seat, pack, pick))
        return pool

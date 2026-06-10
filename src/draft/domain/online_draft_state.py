"""Online (request-stream) typed-token draft-state reconstruction (data-model §2.4).

A live model-piloted seat is shown one pack at a time: at pick ``(pack, pick)``
the worker sends the cards currently in the seat's held pack and asks the policy
for a choice. This tracker rebuilds — incrementally, from that stream of
sightings plus the seat's own committed picks — the *identical* typed-token
:class:`~draft.domain.draft_state.DraftState` that
:func:`draft.domain.draft_state.build_state` produces for the same
``(seat, pack, pick)`` from a *finished* :class:`DraftRecord`.

It is a thin sibling of ``build_state`` (research §third-instance check): it
reuses ``CardInstance`` / ``DraftState`` / ``_recency`` and the pod/pack/pick
conventions of :mod:`draft.domain.draft_geometry`, and is pinned to ``build_state``
by a gating equivalence test (SC-003). It is *not* a silent duplicate — its
input is a partial request stream, which no existing reconstruction consumes
(``build_state`` and the training loader both require a complete record).

Pure domain logic: no torch, no IO. One tracker per ``(draft_id, seat)``.

TODO(shared-reconstruction): typed-token reconstruction now lives in three
places — ``draft_state.build_state`` (full-record oracle),
``train_draft_agent._Loader._emit_seat`` (full-record incremental), and this
request-stream tracker. A shared ``wheel-diff / pack-flush / recency`` kernel
across all three is a non-blocking follow-up (research §third-instance check),
deferred because the input contracts differ (complete record vs. partial
stream) and the training walk is numpy-vectorized for throughput.

Unlike the offline walk, the taken card at each pick is **not** positionally
``vis[0]`` (the live model picks any card), so the wheel diff is computed as a
multiset difference — "what I saw a wheel ago, minus the card I took, minus
what is still here" = the cards others took — which is order-independent and
equals ``build_state``'s positional slice on a draw-ordered record.
"""

from __future__ import annotations

from collections import Counter

from draft.domain.draft_state import (
    TYPE_PACK,
    TYPE_PASSED,
    TYPE_POOL,
    TYPE_TAKEN,
    CardInstance,
    DraftState,
    _recency,
)


class OnlineDraftStateTracker:
    """Incremental per-seat reconstruction of ``DraftState`` from pick requests.

    Lifecycle: created lazily on a seat's first request, advanced by
    :meth:`observe` (which returns the state the policy consumes) followed by
    :meth:`commit` (which records the card the policy chose), and discarded when
    the draft completes or is abandoned.
    """

    def __init__(self) -> None:
        # Persistent state, mirroring build_state's locals (which it rebuilds per
        # call); here they accumulate across the seat's picks instead.
        self._pool: list[str] = []                      # self-drafted, ordered
        self._taken: list[str] = []                     # known taken by others
        self._passed: dict[str, tuple[int, int]] = {}   # current-pack unresolved
        self._last_seen: dict[str, tuple[int, int]] = {}  # name -> last (pack,pick)

        # Per-pack sightings + committed picks, parallel by pick offset, so the
        # wheel diff at pick i can read what this booster held pod picks earlier.
        self._pack_sightings: list[list[str]] = []
        self._pack_picks: list[str] = []
        self._current_pack: int | None = None

        # Real pack size, captured from the opening pack (pack 1, pick 1); used
        # by _recency exactly as build_state uses geometry.pack_size.
        self._pack_size: int | None = None

        # The sighting awaiting a commit: (pack, pick, pack-contents).
        self._pending: tuple[int, int, list[str]] | None = None

    def observe(
        self,
        *,
        pack_number: int,
        pick_number: int,
        pack: list[str],
        pod_size: int,
    ) -> DraftState:
        """Advance to ``(pack_number, pick_number)`` and return its typed-token state.

        ``pack`` is the cards currently in the seat's held pack (the legal
        actions). The returned :class:`DraftState` carries ``target_index = -1``;
        the taken card is revealed by the following :meth:`commit`.
        """
        if self._pack_size is None:
            # The opening pack is fully intact, so its length is the real pack
            # size (build_state's geometry.pack_size). Model seats see every one
            # of their picks, so the first request is always (pack 1, pick 1).
            self._pack_size = len(pack)

        # Pack boundary: flush still-unresolved PASSED to TAKEN (FR-019b) before
        # the new pack's first pick, exactly as build_state flushes for p < pack.
        if self._current_pack is not None and pack_number != self._current_pack:
            for name in self._passed:
                self._taken.append(name)
            self._passed.clear()
        if pack_number != self._current_pack:
            self._current_pack = pack_number
            self._pack_sightings = []
            self._pack_picks = []

        offset = pick_number - 1

        # Wheel diff (FR-019a): if this booster has wheeled back (held pod picks
        # ago), the cards I then passed that are gone now were taken by others.
        if offset >= pod_size:
            prev = offset - pod_size  # 0-based index of the sighting a wheel ago
            vis_prev = self._pack_sightings[prev]
            took_prev = self._pack_picks[prev]
            others = Counter(vis_prev)
            others.subtract(Counter(pack))
            others[took_prev] -= 1
            for name, n in others.items():
                for _ in range(max(0, n)):
                    self._passed.pop(name, None)
                    self._taken.append(name)

        self._pack_sightings.append(list(pack))

        # Snapshot last_seen *before* this sighting so recency is "prior to this
        # pick" (FR-021); the target sighting itself does not count.
        prior_last_seen = dict(self._last_seen)

        # PACK actions: distinct cards in pick order. Wheel survivors that were
        # in PASSED move back into PACK (drop them from PASSED).
        pack_actions: list[str] = []
        seen: set[str] = set()
        for name in pack:
            if name not in seen:
                seen.add(name)
                pack_actions.append(name)
        for name in pack_actions:
            self._passed.pop(name, None)

        instances: list[CardInstance] = []

        def add(name: str, token_type: int) -> None:
            packs_ago, pick_ago = _recency(
                prior_last_seen, name, pack_number, pick_number, self._pack_size,
            )
            instances.append(CardInstance(name, token_type, packs_ago, pick_ago))

        for name in self._pool:
            add(name, TYPE_POOL)
        for name in pack_actions:
            add(name, TYPE_PACK)
        for name in self._passed:
            add(name, TYPE_PASSED)
        for name in self._taken:
            add(name, TYPE_TAKEN)

        self._pending = (pack_number, pick_number, list(pack))
        return DraftState(
            pack_number=pack_number,
            pick_number=pick_number,
            pack_actions=pack_actions,
            target_index=-1,  # revealed by commit()
            cards=instances,
        )

    def commit(self, card: str) -> None:
        """Record the card the policy chose for the last :meth:`observe`.

        Advances persistent state past that pick exactly as build_state advances
        a completed pick: the card joins the seat's pool, every sighted card's
        last-seen updates, and the rest of the pack becomes PASSED.
        """
        if self._pending is None:
            raise RuntimeError("commit() called before observe()")
        pack_number, pick_number, vis = self._pending
        self._pending = None

        self._pool.append(card)
        for c in vis:
            self._last_seen[c] = (pack_number, pick_number)
        self._passed.pop(card, None)
        # PASSED = everything in the pack except the one taken (one occurrence).
        remaining = list(vis)
        try:
            remaining.remove(card)
        except ValueError:
            pass
        for c in remaining:
            self._passed[c] = (pack_number, pick_number)

        self._pack_picks.append(card)

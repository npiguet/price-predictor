"""The sliding window of per-label ``deck_score``s behind the anchor margin, its
best and its stall counter (data-model §7).
"""

from __future__ import annotations

from collections import deque

from draft.domain.draft_geometry import DraftRecord

# --------------------------------------------------------------------------- #
# Anchor window (data-model §7, FR-017/FR-019/FR-021)
# --------------------------------------------------------------------------- #

class AnchorWindow:
    """Sliding window of per-draft, per-label ``deck_score``s behind the margin.

    The anchor and learner labels are bound once at construction and never
    re-bound during a run (FR-021) — the moment the anchor moves, the margin
    stops meaning "improvement over a fixed point".
    """

    def __init__(self, maxlen: int, learner_label: str, anchor_label: str) -> None:
        self._window: deque[dict[str, list[float]]] = deque(maxlen=maxlen)
        self._maxlen = maxlen
        self.learner_label = learner_label
        self.anchor_label = anchor_label
        self.best_margin: float | None = None
        self.best_round: int | None = None
        # The stall counter shared by --patience and --lr-decay-patience
        # (FR-035). It only starts once a best exists, i.e. once the window is
        # full — before that there is nothing to be stalled against.
        self.rounds_since_best = 0

    def add(self, record: DraftRecord) -> None:
        """Append one draft's scored seats, evicting the oldest when full."""
        entry: dict[str, list[float]] = {}
        for seat in record.seats:
            if seat.deck_score is None:
                continue  # failed build — excluded from every mean
            entry.setdefault(seat.agent, []).append(float(seat.deck_score))
        self._window.append(entry)

    @property
    def window_drafts(self) -> int:
        return len(self._window)

    def label_mean(self, label: str) -> float | None:
        """Mean ``deck_score`` for ``label`` across the window; ``None`` if unseen."""
        total = 0.0
        count = 0
        for entry in self._window:
            for score in entry.get(label, ()):
                total += score
                count += 1
        return total / count if count else None

    def label_means(self) -> dict[str, float]:
        """Every label's windowed mean, for the ``progress`` line's raw components."""
        totals: dict[str, list[float]] = {}
        for entry in self._window:
            for label, scores in entry.items():
                totals.setdefault(label, []).extend(scores)
        return {
            label: sum(scores) / len(scores)
            for label, scores in totals.items() if scores
        }

    @property
    def margin(self) -> float | None:
        """``mean(learner) − mean(anchor)``; ``None`` until both are populated."""
        learner = self.label_mean(self.learner_label)
        anchor = self.label_mean(self.anchor_label)
        if learner is None or anchor is None:
            return None
        return learner - anchor

    @property
    def is_full(self) -> bool:
        """Whether the window holds a full ``anchor_window`` drafts (FR-033)."""
        return len(self._window) >= self._maxlen

    def observe_round(self, round_index: int) -> bool:
        """Record this round's margin as the new best if it is; return whether it was.

        **Window-full guard** (FR-033): nothing is recorded until the window is
        full. A margin over a partly-filled window is computed from fewer drafts
        and is correspondingly noisy — without the guard an early lucky round
        would pin the run's best for its whole length. The margin is still
        *printed* throughout; it just cannot win.
        """
        if not self.is_full:
            return False
        margin = self.margin
        if margin is None:
            return False
        if self.best_margin is None or margin > self.best_margin:
            self.best_margin = margin
            self.best_round = round_index
            self.rounds_since_best = 0
            return True
        self.rounds_since_best += 1
        return False

    def note_lr_decay(self) -> None:
        """Reset the stall counter after an LR decay (FR-035's cooldown).

        This is what lets annealing pre-empt stopping: with the counter reset on
        every decay, an armed ``--patience`` cannot fire until no further decay
        is possible, i.e. at the LR floor.
        """
        self.rounds_since_best = 0

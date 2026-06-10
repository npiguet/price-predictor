"""Draft-side adapter for ``draft analyze-generated-decks``.

Sources decks from a ``drafts.jsonl`` corpus instead of a generated-decks file:
each seat's built ``deck`` becomes a :class:`GeneratedDeck` (label = the seat's
agent, set code = the draft's set), which feeds the shared composition engine in
:mod:`sealed.application.analyze_generated_decks`. Because the corpus also
carries each seat's ``deck_score`` (which generated-decks files lack), this
module additionally summarizes per-label deck score — the agent-vs-Forge
headline — printed above the composition report.

``draft`` importing ``sealed`` is the allowed one-way dependency.
"""

from __future__ import annotations

from collections.abc import Iterable
from statistics import mean, median

from draft.domain.draft_geometry import DraftRecord
from sealed.domain.deck import Deck
from sealed.infrastructure.pool_file_reader import GeneratedDeck


def available_agents(records: Iterable[DraftRecord]) -> list[str]:
    """Sorted distinct agent/label values present across all seats."""
    return sorted({seat.agent for record in records for seat in record.seats})


def generated_decks_from_drafts(
    records: Iterable[DraftRecord],
    agent: str | None = None,
) -> list[GeneratedDeck]:
    """One :class:`GeneratedDeck` per built seat across all records.

    The seat's ``agent`` is the deck label and the draft's set (from its first
    booster) is the set code. Seats whose build failed (empty ``deck``) are
    skipped, matching the ``deck_score is None`` convention. When ``agent`` is
    given, only seats piloted by that agent are included.
    """
    decks: list[GeneratedDeck] = []
    for record in records:
        if not record.boosters:
            continue
        set_code = record.boosters[0].set_code
        for seat in record.seats:
            if agent is not None and seat.agent != agent:
                continue
            if not seat.deck:
                continue
            decks.append(GeneratedDeck(
                label=seat.agent, set_code=set_code, deck=Deck.of(seat.deck),
            ))
    return decks


def deck_score_summary(
    records: Iterable[DraftRecord],
    agent: str | None = None,
) -> list[tuple[str, float, float, int]]:
    """Per-label ``(mean, median, n)`` deck score, sorted by label.

    Aggregates ``seat.deck_score`` across every record, skipping seats whose
    score is ``None`` (failed build). When ``agent`` is given, only that agent's
    seats are summarized. Labels with no scored seat are omitted.
    """
    scores_by_label: dict[str, list[float]] = {}
    for record in records:
        for seat in record.seats:
            if agent is not None and seat.agent != agent:
                continue
            if seat.deck_score is None:
                continue
            scores_by_label.setdefault(seat.agent, []).append(seat.deck_score)
    return [
        (label, mean(scores), median(scores), len(scores))
        for label, scores in sorted(scores_by_label.items())
    ]


def format_deck_score_section(
    summary: list[tuple[str, float, float, int]],
) -> str:
    """Render the ``=== Deck score (per label) ===`` block."""
    lines = ["=== Deck score (per label) ==="]
    if not summary:
        lines.append("  (no scored decks)")
        return "\n".join(lines)
    width = max(len(label) for label, _, _, _ in summary)
    for label, mean_score, median_score, n in summary:
        lines.append(
            f"  {label:<{width}} : mean={mean_score:.2f} "
            f"median={median_score:.2f} (n={n})"
        )
    return "\n".join(lines)

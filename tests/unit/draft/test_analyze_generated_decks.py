"""Draft corpus → decks adapter + per-label deck-score summary + CLI wiring."""

from __future__ import annotations

from draft.application.analyze_generated_decks import (
    deck_score_summary,
    format_deck_score_section,
    generated_decks_from_drafts,
)
from draft.domain.draft_geometry import Booster, DraftRecord, Seat
from draft.infrastructure.cli import build_parser, run_analyze_generated_decks


def _record(draft_id, seats, set_code="BLB") -> DraftRecord:
    return DraftRecord(
        draft_id=draft_id, run_id="r", timestamp="t", seats=seats,
        boosters=[Booster(set_code, ["A"]), Booster(set_code, ["B"])],
    )


def test_generated_decks_skips_failed_builds_and_stamps_label_and_set() -> None:
    record = _record("d1", [
        Seat("draft-agent", ["Plains"] * 40, 6.0),
        Seat("forge-full", [], None),            # failed build → skipped
    ])
    decks = generated_decks_from_drafts([record])
    assert len(decks) == 1
    assert decks[0].label == "draft-agent"
    assert decks[0].set_code == "BLB"
    assert decks[0].cards == tuple(["Plains"] * 40)


def test_deck_score_summary_per_label_skips_none() -> None:
    records = [
        _record("d1", [
            Seat("draft-agent", ["Plains"] * 40, 6.0),
            Seat("forge-full", ["Forest"] * 40, 5.0),
        ]),
        _record("d2", [
            Seat("draft-agent", ["Plains"] * 40, 8.0),
            Seat("forge-full", [], None),        # no score → excluded
        ]),
    ]
    summary = dict((label, (m, med, n)) for label, m, med, n in deck_score_summary(records))
    assert summary["draft-agent"] == (7.0, 7.0, 2)   # mean/median of [6,8]
    assert summary["forge-full"] == (5.0, 5.0, 1)


def test_format_deck_score_section_renders_labels() -> None:
    text = format_deck_score_section([("draft-agent", 6.41, 6.38, 412)])
    assert "=== Deck score (per label) ===" in text
    assert "draft-agent" in text and "mean=6.41" in text and "(n=412)" in text
    # Empty summary degrades gracefully.
    assert "(no scored decks)" in format_deck_score_section([])


def test_draft_subcommand_dispatches() -> None:
    args = build_parser().parse_args(["analyze-generated-decks"])
    assert args.func is run_analyze_generated_decks
    assert args.drafts_path == "output/draft/drafts.jsonl"
    assert args.no_rarity is False

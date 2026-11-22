"""A record whose own changes end the game is refused as ``game-end`` (decision
D13): Forge's state-based check stops at the game's end before it destroys any
creature, so such a record shows no death the rules would cause."""

from __future__ import annotations

import pytest

from effects.domain.provenance import KeyResolution
from effects.domain.record_quality import QualityDefect, quality_defect, record_refusal
from effects.domain.records import EffectRecord

_RUN = "3550376f-3c30-4231-90cd-cc0f1b308e87"


@pytest.fixture(scope="module")
def fixture_records() -> dict[str, EffectRecord]:
    from effects.infrastructure.shard_io import read_shard
    from tests.unit.effects.builders import GEN3_RECORDS

    return {r.record_id: r for r in read_shard(GEN3_RECORDS)}


@pytest.mark.parametrize("record_id", [
    # A natural combat whose damage step holds player_lost and player_won.
    f"{_RUN}.0-mwqvqdyrb9c0.627",
    # Its damage-step probe, which ends the game the same way.
    f"{_RUN}.0-mwqvqdyrb9c0.628",
    # A burn spell's lethal resolution: its life change takes the player to
    # 0, while the game-end record beside it holds the events.
    f"{_RUN}.2-mwqvqdxrusnv.4638",
    # That game-end record: refused as the game's end, not as `no-ability`.
    f"{_RUN}.2-mwqvqdxrusnv.4639",
])
def test_a_record_that_ends_the_game_is_refused_as_game_end(fixture_records, record_id):
    record = fixture_records[record_id]
    assert quality_defect(record) is QualityDefect.GAME_END
    assert record_refusal(record, lambda key: KeyResolution.LINE) is QualityDefect.GAME_END


def test_a_combat_that_leaves_both_players_in_the_game_is_kept(fixture_records):
    """A natural combat of thirteen events, none of them the game's end."""
    record = fixture_records[f"{_RUN}.0-mwqvqdyrb9c0.240"]
    assert quality_defect(record) is None


def test_the_refusal_is_spelled_as_the_manifest_counts_it():
    assert QualityDefect.GAME_END == "game-end"

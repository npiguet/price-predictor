"""The per-entity head has no ``game_lost`` output (decision D13): every record
that holds the game's end is refused by ``build-corpus``, so no kept record
could set it."""

from __future__ import annotations

from effects.domain.effect_head_layout import FIELDS_BY_NAME, HeadField
from effects.domain.effect_targets import derive_targets

_RUN = "3550376f-3c30-4231-90cd-cc0f1b308e87"


def test_the_head_has_no_game_lost_field():
    assert "game_lost" not in FIELDS_BY_NAME
    assert "game_lost" not in {str(field) for field in HeadField}


def test_a_record_holding_the_games_end_derives_no_game_result():
    """The fixture's combat whose damage step holds ``player_lost`` and
    ``player_won``: its targets name neither player for the game's end."""
    from effects.infrastructure.shard_io import read_shard
    from tests.unit.effects.builders import GEN3_RECORDS

    record = next(r for r in read_shard(GEN3_RECORDS)
                  if r.record_id == f"{_RUN}.0-mwqvqdyrb9c0.627")
    targets = derive_targets(record)
    assert not any("game_lost" in entry.fields for entry in targets.values())

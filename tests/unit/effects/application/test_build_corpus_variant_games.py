"""A game with a variant on its board whose sidecar prints a held-out text is
held out whole, as a held-out card's game is (finding 14.5)."""

from __future__ import annotations

from dataclasses import replace

from effects.domain.provenance import ProvenanceKey
from effects.infrastructure.shard_io import read_records
from tests.unit.effects.application.build_corpus_support import (
    _HELD_OUT_TEXT,
    _SHOCK_KEY,
    _resolution,
    _write_card,
    _write_corpus,
)
from tests.unit.effects.builders import lines_of, make_entity, write_ended_shard

_VARIANT_FILE = "variant-scripts/held_variant.txt"
_VARIANT_KEY = ProvenanceKey(_VARIANT_FILE, 0, "spell", 0)
_OTHER_FILE = "variant-scripts/plain_variant.txt"
_OTHER_KEY = ProvenanceKey(_OTHER_FILE, 0, "spell", 0)
#: Games of each kind: enough that some of the plain ones train, whatever the
#: hash puts in the game-disjoint stratum.
_GAMES = 8


def _game_with_variant_on_board(game: str, variant: ProvenanceKey) -> list:
    """Three resolutions through a plain card, and a variant's line on the
    board of one of them: never the acting line."""
    records = [_resolution(f"{game}.{n}", game, ability=(_SHOCK_KEY,)) for n in range(3)]
    board = make_entity("E0", "A Variant", lines=lines_of(printed=(variant,)))
    state = replace(records[0].state, entities=(*records[0].state.entities, board))
    records[0] = replace(records[0], state=state)
    return records


def _build(tmp_path):
    root = tmp_path / "source"
    fixture = _write_corpus(root)
    _write_card(root, _VARIANT_FILE, "Held Variant", _HELD_OUT_TEXT)
    _write_card(root, _OTHER_FILE, "Plain Variant", "deals 4 damage to each opponent")
    write_ended_shard(fixture.records / "full-strength" / "run.0-v.jsonl.gz", [
        *(record for n in range(_GAMES) for record in
          _game_with_variant_on_board(f"g-held-{n}", _VARIANT_KEY)),
        *(record for n in range(_GAMES) for record in
          _game_with_variant_on_board(f"g-plain-{n}", _OTHER_KEY)),
    ])
    return fixture.build(
        tmp_path / "curated", variant_scripts=str(root / "variant-scripts"),
    )


def test_a_training_game_with_a_held_out_variant_on_its_board_is_not_trained_on(tmp_path):
    store = _build(tmp_path)

    trained = {record.game_id for record in read_records(store.training_dir)}

    assert not {game for game in trained if game.startswith("g-held-")}
    assert {game for game in trained if game.startswith("g-plain-")}, (
        "a variant printing no held-out text holds nothing out")

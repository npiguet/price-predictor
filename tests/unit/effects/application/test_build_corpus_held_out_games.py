"""Games that resolve or print a held-out text."""

from __future__ import annotations

from dataclasses import replace

from effects.domain.provenance import ProvenanceKey, ProvenanceSidecar, SidecarLine
from effects.infrastructure.shard_io import (
    read_records,
)
from effects.infrastructure.sidecar_io import sidecar_path_for
from tests.unit.effects.application.build_corpus_support import (
    _HELD_OUT_CARD,
    _HELD_OUT_KEY,
    _HELD_OUT_TEXT,
    _SHOCK_KEY,
    _Card,
    _resolution,
    _rewrite,
    _write_card,
)
from tests.unit.effects.file_builders import write_ended_shard, write_sidecar
from tests.unit.effects.record_builders import lines_of, make_entity, make_snapshot


def test_a_held_out_game_resolving_no_held_out_text_is_dropped(tmp_path, a_corpus):
    """Only a resolution counts a game toward its held-out text (FR-142).

    Gate 1 and the gate-one slice score resolution records, so a game whose
    held-out text appears only on a rewrite holds nothing gate 1 measures.
    """
    write_ended_shard(a_corpus.records / "full-strength" / "run.0-rewrite.jsonl.gz", [
        _rewrite("g-held-3.0", "g-held-3", ability=(_HELD_OUT_KEY,)),
        _resolution(
            "g-held-3.1", "g-held-3", ability=(_SHOCK_KEY,),
            entity_names=(_HELD_OUT_CARD,),
        ),
    ])
    store = a_corpus.build(tmp_path / "curated")
    manifest = store.load()

    assert "g-held-3" not in manifest.card_disjoint_games
    assert "g-tainted" in manifest.card_disjoint_games
    trained = {r.game_id for r in read_records(store.training_dir)}
    assert "g-held-3" not in trained


def test_a_token_printing_a_held_out_text_holds_its_game_out(tmp_path, a_corpus):
    """A token script is a carrier, so a game it appears in is held out."""
    token_file = "tokenscripts/r_2_2_goblin_held.txt"
    token_key = ProvenanceKey(token_file, 0, "spell", 1)
    _write_card(tmp_path / "source", _Card(
        token_file, "Goblin Held", _HELD_OUT_TEXT, index_within_kind=1,
    ))
    write_ended_shard(a_corpus.records / "full-strength" / "run.0-token.jsonl.gz", [
        replace(
            _resolution("g-token.0", "g-token", ability=(_SHOCK_KEY,)),
            state=make_snapshot(entities=(
                make_entity("E1", "Goblin Token", lines=lines_of(printed=(token_key,))),
            )),
        ),
    ])
    store = a_corpus.build(tmp_path / "curated")

    trained = {r.game_id for r in read_records(store.training_dir)}
    assert "g-token" not in trained
    assert "g-token" not in store.load().game_disjoint_games


def test_a_variant_printing_a_held_out_text_trains_on_nothing(tmp_path, a_corpus):
    """A game whose record acts through a variant printing a held-out text is
    held out whole, not trained on with the one record dropped:
    the variant's text is the very text the card-disjoint stratum measures."""
    variant_file = "variant-scripts/held_out_bears_variant_1.txt"
    variant_key = ProvenanceKey(variant_file, 0, "spell", 0)
    root = tmp_path / "source"
    path = root / "variant-scripts" / "held_out_bears_variant_1.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_HELD_OUT_TEXT + "\n", encoding="utf-8")
    write_sidecar(
        ProvenanceSidecar(
            card="Held Out Bears Variant", script_file=variant_file,
            lines=(SidecarLine(
                line_index=0, line_kind="spell", provenance=(variant_key,),
                script_text=_HELD_OUT_TEXT, prose_offset=0,
            ),),
        ),
        sidecar_path_for(path),
    )
    write_ended_shard(a_corpus.records / "variants" / "run.0-v.jsonl.gz", [
        _resolution("g-var.0", "g-var", ability=(variant_key,)),
        _resolution("g-var.1", "g-var", ability=(_SHOCK_KEY,)),
    ])
    # Every resolution record the class can supply is wanted, so only the
    # held-out text can keep the variant's record out.
    store = a_corpus.build(
        tmp_path / "curated", variant_scripts=str(root / "variant-scripts"), text_cap=0,
        class_mix={"resolution-effect": 1.0},
    )

    training = list(read_records(store.training_dir))
    assert not [r for r in training if r.game_id == "g-var"]
    assert "g-var" not in store.load().game_disjoint_games


# ── gen-3 curation: rule budgets, curation, turn actions ──────────

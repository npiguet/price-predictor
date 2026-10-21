"""``stratum_records``: gate 1's held-out records and what training contained."""

from __future__ import annotations

from types import SimpleNamespace

from effects.domain.provenance import ProvenanceKey
from effects.domain.records import EffectRecord, Moment, RecordKind, ResolutionPayload
from effects.domain.state_snapshot import GlobalState, StateSnapshot
from effects.infrastructure.effect_model_store import SplitProvenance
from effects.infrastructure.model_runner import stratum_records
from effects.infrastructure.record_io import write_shard

_SNAPSHOT = StateSnapshot(
    global_=GlobalState(turn=1, phase="main1", active="P0", priority="P0", stack_size=0),
    players=(), entities=(),
)


def _key(name: str) -> ProvenanceKey:
    return ProvenanceKey(f"cardsfolder/x/{name}.txt", 0, "spell", 0)


def _record(game: str, number: int, ability) -> EffectRecord:
    return EffectRecord(
        record_id=f"{game}.{number}", run_id="r", timestamp="t", game_id=game,
        kind=RecordKind.RESOLUTION, moment=Moment.RESOLUTION, actor_player="P0",
        state=_SNAPSHOT, payload=ResolutionPayload(), ability=ability,
    )


def _checkpoint(card_disjoint, game_disjoint) -> SimpleNamespace:
    return SimpleNamespace(provenance=SplitProvenance(
        held_out_cards=(), card_disjoint_games=tuple(card_disjoint),
        game_disjoint_games=tuple(game_disjoint),
    ))


def test_the_split_sorts_held_out_records_from_training_acting_lines(tmp_path):
    held = _record("held", 1, (_key("bolt"),))
    write_shard(tmp_path / "s.jsonl.gz", [
        held,
        _record("train", 1, (_key("shock"),)),
        _record("train", 2, (_key("shock"),)),
        _record("train", 3, None),
        _record("game-disjoint", 1, (_key("growth"),)),
    ])
    config = SimpleNamespace(records_dir=tmp_path)

    card_disjoint, training_acting, why_not = stratum_records(
        config, _checkpoint(["held"], ["game-disjoint"]),
    )

    assert why_not == ""
    assert [r.record_id for r in card_disjoint] == ["held.1"]
    assert training_acting == {(_key("shock"),)}


def test_no_card_disjoint_games_says_why(tmp_path):
    write_shard(tmp_path / "s.jsonl.gz", [_record("train", 1, (_key("shock"),))])
    card_disjoint, training_acting, why_not = stratum_records(
        SimpleNamespace(records_dir=tmp_path), _checkpoint(["absent"], []),
    )
    assert card_disjoint == [] and training_acting == set()
    assert "none of the checkpoint's 1 card-disjoint games" in why_not

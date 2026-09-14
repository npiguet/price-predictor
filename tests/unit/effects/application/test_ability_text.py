"""``text_for_key``, extracted out of ``ability_text_of`` (build-corpus task 4).

``build-corpus`` surveys by provenance key and only folds keys to ability text
once, in the main process, against the one ``SidecarCache`` it already builds
for the holdout. ``text_for_key`` is that fold's single definition; this test
pins it against ``ability_text_of``, which is defined in terms of it, so the
extraction cannot drift the two apart.
"""

from __future__ import annotations

from effects.domain.provenance import ProvenanceKey, SidecarLine
from effects.domain.records import EffectRecord, Moment, RecordKind, ResolutionPayload
from effects.domain.state_snapshot import EntityState, GlobalState, StateSnapshot


def _entity(name: str, **overrides) -> EntityState:
    defaults = {
        "id": f"E-{name}", "name": name, "zone": "battlefield",
        "controller": "P0",
    }
    defaults.update(overrides)
    return EntityState(**defaults)


def _record(game_id: str, entities=(), ability=None, **overrides) -> EffectRecord:
    defaults: dict = {
        "record_id": f"{game_id}.1", "run_id": "run", "timestamp": "t",
        "game_id": game_id, "kind": RecordKind.RESOLUTION,
        "moment": Moment.RESOLUTION, "actor_player": "P0",
        "ability": ability,
        "state": StateSnapshot(
            global_=GlobalState(
                turn=1, phase="main1", active="P0", priority="P0", stack_size=0,
            ),
            players=(),
            entities=tuple(entities),
        ),
        "payload": ResolutionPayload(),
    }
    defaults.update(overrides)
    return EffectRecord(**defaults)


class FakeSidecarCache:
    """The smallest double for ``SidecarCache``: an explicit key -> line map.

    ``text_for_key`` (and, through it, ``ability_text_of``) calls only
    ``line_for`` and ``prose_for``, so this fake implements exactly those two.
    No sidecar fake existed anywhere in the test tree before this one; a later
    task that needs a sidecar double should import this rather than write a
    third.
    """

    def __init__(
        self,
        lines: dict[ProvenanceKey, SidecarLine | None],
        prose: dict[ProvenanceKey, str | None] | None = None,
    ) -> None:
        self._lines = lines
        self._prose = dict(prose or {})

    def line_for(self, key: ProvenanceKey) -> SidecarLine | None:
        return self._lines[key]

    def prose_for(self, key: ProvenanceKey) -> str | None:
        return self._prose.get(key)


def test_text_for_key_and_ability_text_of_agree_on_one_key():
    from effects.application.train_effect_model import ability_text_of, text_for_key

    key = ProvenanceKey("cardsfolder/f/flying_test.txt", 0, "keyword", 0)
    record = _record("g1", ability=(key,))
    sidecars = FakeSidecarCache(
        lines={
            key: SidecarLine(
                line_index=0, line_kind="keyword", provenance=(key,),
                script_text="Flying",
            ),
        },
        prose={key: "Flying"},
    )

    assert text_for_key(key, sidecars, "script") == ability_text_of(
        record, sidecars, "script"
    )

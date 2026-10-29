"""``text_for_key`` and the per-record acting text (build-corpus task 4).

``build-corpus`` surveys by provenance key and only folds keys to ability text
once, in the main process, against the one ``SidecarCache`` it already builds
for the holdout. ``text_for_key`` is that fold's single definition; this test
pins it against ``KeyResolver.acting_text``, the per-record text the trainer
weights by, so the two cannot drift apart.
"""

from __future__ import annotations

import pytest

from effects.domain.provenance import ProvenanceKey, ProvenanceSidecar, SidecarLine
from effects.domain.records import EffectRecord, RecordKind
from effects.domain.state_snapshot import EntityState
from effects.infrastructure.sidecar_io import (
    SidecarCache,
    UnconfiguredTree,
    UnconvertedScript,
    write_sidecar,
)
from tests.unit.effects.builders import make_entity, make_record, make_snapshot


def _entity(name: str, **overrides) -> EntityState:
    return make_entity(f"E-{name}", name, **overrides)


def _record(game_id: str, entities=(), ability=None, **overrides) -> EffectRecord:
    return make_record(
        game_id=game_id, ability=ability,
        state=make_snapshot(entities=tuple(entities)), **overrides,
    )


class FakeSidecarCache:
    """The smallest double for ``SidecarCache``: an explicit key -> line map.

    ``text_for_key`` (and, through it, ``KeyResolver``) calls only
    ``line_for`` and ``prose_for``; ``get``, which only a charm root's options
    read, raises as a script with no sidecar does.
    No sidecar fake existed anywhere in the test tree before this one; a later
    task that needs a sidecar double should import this rather than write a
    third.

    A key's ``script_file`` decides which of the real cache's two failure
    shapes ``line_for`` mirrors when the key itself is not registered: a
    script_file no registered key names at all reads as no text, the same way
    a script the converted corpus never held does; a script_file some other
    registered key *does* name is a sidecar that exists and disagrees with
    this key, so it raises the way a real reconversion mismatch would.

    The real cache raises only for a key *inside* a range the sidecar declared;
    outside one the key is runtime-only. The fake has no declared ranges, so it
    raises on every unregistered key of a known script, and a test wanting the
    runtime-only answer registers that key with a ``None`` line instead.
    """

    def __init__(
        self,
        lines: dict[ProvenanceKey, SidecarLine | None],
        prose: dict[ProvenanceKey, str | None] | None = None,
    ) -> None:
        self._lines = lines
        self._prose = dict(prose or {})
        self._known_scripts = {key.script_file for key in lines}

    def line_for(self, key: ProvenanceKey) -> SidecarLine | None:
        if key in self._lines:
            return self._lines[key]
        if key.script_file not in self._known_scripts:
            return None
        raise KeyError(
            f"provenance key {key} appears in neither the lines nor the "
            f"dropped_keys of {key.script_file}; the sidecar does not "
            "describe the card this record was collected against "
            "(reconversion between collection and training?)"
        )

    def prose_for(self, key: ProvenanceKey) -> str | None:
        return self._prose.get(key)

    def get(self, script_file: str):
        raise KeyError(f"{script_file}: the fake holds lines, not sidecars")


def test_text_for_key_and_the_acting_text_agree_on_one_key():
    from effects.application.ability_text import KeyResolver, text_for_key

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

    assert text_for_key(key, sidecars, "script") == KeyResolver(
        sidecars, "script",
    ).acting_text(record)


def test_a_key_in_neither_list_raises_rather_than_reading_as_no_text(tmp_path):
    """The contract's fail-loudly case: the sidecar does not describe the card.

    Plague Sliver's own static grants a trigger to every other sliver, and that
    trigger keys to this script, which declares no trigger — runtime-only, and
    no text. The mismatch is a gap *inside* a declared range: statics 0 and 2
    are declared here, so static 1 is an index the converter parsed around and
    put in neither list, which only a reconversion produces.
    """
    from effects.application.ability_text import text_for_key
    from effects.infrastructure.sidecar_io import sidecar_path_for

    script = "cardsfolder/p/plague_sliver.txt"
    txt = tmp_path / "cardsfolder" / "p" / "plague_sliver.txt"
    txt.parent.mkdir(parents=True)
    txt.write_text("name: plague sliver\n", encoding="utf-8")
    rendered = ProvenanceKey(script, 0, "static", 0)
    write_sidecar(
        ProvenanceSidecar(
            card="plague sliver", script_file=script,
            lines=(SidecarLine(line_index=0, line_kind="static",
                               provenance=(rendered,), script_text="all slivers have"),),
            dropped_keys=(
                ProvenanceKey(script, 0, "spell", 0),
                ProvenanceKey(script, 0, "static", 2),
            ),
        ),
        sidecar_path_for(txt),
    )
    sidecars = SidecarCache({"cardsfolder": tmp_path / "cardsfolder"})

    assert text_for_key(rendered, sidecars, "script") == "all slivers have"
    assert text_for_key(ProvenanceKey(script, 0, "spell", 0), sidecars, "script") is None
    assert text_for_key(ProvenanceKey(script, 0, "trigger", 0), sidecars, "script") is None
    with pytest.raises(KeyError, match="neither the lines nor the dropped_keys"):
        text_for_key(ProvenanceKey(script, 0, "static", 1), sidecars, "script")


def test_an_unconfigured_tree_still_reads_as_no_text(tmp_path):
    """A variant key without ``--variant-scripts`` resolves to nothing, as pinned
    by ``test_without_the_variant_tree_a_variant_text_resolves_to_nothing``."""
    from effects.application.ability_text import text_for_key

    sidecars = SidecarCache({"cardsfolder": tmp_path})
    key = ProvenanceKey("variant-scripts/x.txt", 0, "spell", 0)
    with pytest.raises(UnconfiguredTree):
        sidecars.path_for(key.script_file)
    assert text_for_key(key, sidecars, "script") is None


def test_an_unconverted_script_reads_as_no_text(tmp_path):
    from effects.application.ability_text import text_for_key

    sidecars = SidecarCache({"cardsfolder": tmp_path})
    key = ProvenanceKey("cardsfolder/f/food_token.txt", 0, "spell", 0)
    with pytest.raises(UnconvertedScript):
        sidecars.get(key.script_file)
    assert text_for_key(key, sidecars, "script") is None


def test_a_decisions_acting_text_is_its_candidates():
    """A ``decision`` record names no acting line of its own; ``[ACT]`` reads
    its candidate's, and rarity has to key on the same text or the whole
    decision class weighs as if it had none."""
    from effects.application.ability_text import KeyResolver
    from effects.domain.records import (
        Candidate,
        PlayabilityDecisionPayload,
        PlayabilitySubkind,
    )

    key = ProvenanceKey("cardsfolder/s/shock.txt", 0, "spell", 0)
    record = _record(
        "g1", kind=RecordKind.PLAYABILITY, moment=None,
        subkind=PlayabilitySubkind.DECISION,
        payload=PlayabilityDecisionPayload(candidates=(Candidate(
            ability=(key,), can_play=True, affordable=True, has_legal_target=True,
        ),)),
    )
    sidecars = FakeSidecarCache(
        lines={key: SidecarLine(
            line_index=0, line_kind="spell", provenance=(key,),
            script_text="DealDamage",
        )},
    )

    assert KeyResolver(sidecars, "script").acting_text(record) == "DealDamage"


def test_every_line_on_disk_is_a_known_text_spelled_as_the_join_spells_it(tmp_path):
    """The identity baseline's index is built from these, so a text it looks
    up at training time must be spelled exactly as ``resolve_key`` spells it."""
    from effects.application.ability_text import known_texts, text_for_key
    from effects.infrastructure.sidecar_io import sidecar_path_for

    script = "cardsfolder/s/shock.txt"
    txt = tmp_path / "cardsfolder" / "s" / "shock.txt"
    txt.parent.mkdir(parents=True)
    txt.write_text("name: shock\nspell[0]: deals 2 damage to any target.\n", encoding="utf-8")
    keys = (ProvenanceKey(script, 0, "spell", 0), ProvenanceKey(script, 0, "static", 0))
    write_sidecar(
        ProvenanceSidecar(
            card="shock", script_file=script,
            lines=(
                SidecarLine(line_index=1, line_kind="spell", provenance=(keys[0],),
                            script_text="SP$ DealDamage | NumDmg$ 2"),
                SidecarLine(line_index=1, line_kind="static", provenance=(keys[1],),
                            script_text="SP$ DealDamage | NumDmg$ 2"),
            ),
        ),
        sidecar_path_for(txt),
    )
    roots = {"cardsfolder": tmp_path / "cardsfolder"}
    sidecars = SidecarCache(roots)

    for surface in ("script", "prose"):
        texts = known_texts(roots, surface)
        assert texts == sorted(set(texts))
        assert texts == sorted({text_for_key(key, sidecars, surface) for key in keys})

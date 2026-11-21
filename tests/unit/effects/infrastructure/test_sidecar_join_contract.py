"""The sidecar join's three rules (provenance-sidecar contract § Rules), over
the converter's own sidecars and real fixture records.

A key a line declares resolves to that line; a ``built`` key no line declares
resolves as the keyword's bare key; a bare key naming a keyword of several
lines is all of them on the board and ``AMBIGUOUS`` acting. No key is
declared by two lines of a face: a reader refuses such a sidecar.

A charm's mode keys (``trigger 0 option m``, an activated charm's
``spell 1`` and its options) resolve to their own lines, over real converter
sidecars.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import replace
from pathlib import Path

import pytest

from effects.application.ability_text import KeyResolver
from effects.domain.ability_tokenizer import Surface
from effects.domain.effect_head_input import slot_lines
from effects.domain.provenance import (
    BuiltTrait,
    KeyResolution,
    ProvenanceKey,
    SidecarError,
)
from effects.domain.records import EffectRecord, RecordKind
from effects.domain.rule_detectors.acting_line import acting_line, acting_root_params
from effects.infrastructure.shard_io import read_shard
from effects.infrastructure.sidecar_io import SidecarCache, read_sidecar, sidecar_from_dict
from tests.unit.effects.builders import GEN3_RECORDS, GEN3_SIDECAR_FOLDERS

_TRIBUTE_TREE = Path(__file__).resolve().parents[3] / "fixtures" / "effects" / "tribute-sidecars"
_CHARM_TREE = Path(__file__).resolve().parents[3] / "fixtures" / "effects" / "charm-sidecars"
_SAGA = "cardsfolder/b/braidss_frightful_return.txt"
#: Braids's Frightful Return's saga keyword, which builds one trigger per chapter.
_SAGA_KEY = ProvenanceKey(_SAGA, 0, "keyword", 0)


def _chapter(index: int) -> ProvenanceKey:
    return ProvenanceKey(_SAGA, 0, "keyword", 0, built=BuiltTrait("trigger", index))


def _cache(folders=GEN3_SIDECAR_FOLDERS) -> SidecarCache:
    return SidecarCache({folder.name: folder for folder in folders})


def _record(path: Path, number: int) -> EffectRecord:
    """The fixture's record ``number`` (0-based), as the collector wrote it."""
    return next(itertools.islice(read_shard(path), number, None))


def _fixture_json(script: str, folder: Path = GEN3_SIDECAR_FOLDERS[0]) -> dict:
    relative = script.removeprefix("cardsfolder/").replace(".txt", ".provenance.json")
    return json.loads((folder / relative).read_text(encoding="utf-8"))


class TestAKeyALineDeclaresResolvesToThatLine:
    def test_a_chapter_iii_key_reads_the_iii_line(self):
        cache = _cache()
        line = cache.line_for(_chapter(2))
        assert line.line_kind == "chapter"
        assert line.provenance == (_chapter(2),)
        assert line.script_text.startswith("Chapter$ 3 |")
        assert cache.prose_for(_chapter(2)).startswith("III — target opponent may sacrifice")

    def test_every_chapter_reads_its_own_line(self):
        sidecar = _cache().get(_SAGA)
        chapters = [line for line in sidecar.lines if line.line_kind == "chapter"]
        assert [sidecar.line_for(_chapter(n)) for n in range(3)] == chapters

    def test_a_detector_reads_the_chapter_s_own_script(self):
        """Finding 7.9: a script parameter is read off the joined line, chapter
        III's trigger, not chapter I's. A real gen-3 resolution, edited to act
        through chapter III."""
        record = _record(GEN3_RECORDS, 31)
        assert record.kind.value == "resolution"
        acting = replace(record, ability=(_chapter(2),))
        assert acting_root_params(acting, _cache())["Chapter"] == "3"


class TestABuiltKeyNoLineDeclaresReadsTheKeywordLine:
    def test_a_cycling_key_with_built_reads_the_cycling_line(self):
        """Gempalm Avenger as the current converter writes it: its cycling
        line declares the bare keyword key only, while the collector names
        the cycling ability the keyword built."""
        data = _fixture_json("cardsfolder/g/gempalm_avenger.txt")
        cycling = data["lines"][0]
        cycling["provenance"] = [key for key in cycling["provenance"]
                                 if key["trait_kind"] == "keyword"]
        sidecar = sidecar_from_dict(data)
        script = "cardsfolder/g/gempalm_avenger.txt"
        built = ProvenanceKey(script, 0, "keyword", 0, built=BuiltTrait("spell", 0))
        assert sidecar.resolution_of(built) is KeyResolution.LINE
        assert sidecar.line_for(built).script_text == "Cycling:2 W"


class TestABareKeyOfAKeywordOfSeveralLines:
    def test_on_the_board_it_is_its_chapter_lines_in_order(self):
        """A real gen-3 record's Braids's Frightful Return on the battlefield,
        whose keyword line names the saga's bare key: one slot per chapter
        line, then its read-ahead line."""
        record = _record(GEN3_RECORDS, 276)
        saga = next(e for e in record.state.entities if e.name == "Braids's Frightful Return")
        resolver = KeyResolver(_cache(), Surface.SCRIPT)
        keys = [line.key for line in slot_lines(saga, resolver.board_keys)]
        assert keys == [_chapter(0), _chapter(1), _chapter(2),
                        ProvenanceKey(_SAGA, 0, "keyword", 1)]
        texts = [resolver.text_of(key) for key in keys[:3]]
        assert [text.split(" |")[0] for text in texts] == [
            "Chapter$ 1", "Chapter$ 2", "Chapter$ 3"]

    def test_a_line_printing_two_chapters_is_one_slot(self):
        """Life of Toshiro Umezawa's "I, II" line: one slot, read through
        its first chapter's key."""
        script = "cardsfolder/l/life_of_toshiro_umezawa_memory_of_toshiro.txt"
        sidecar = _cache().get(script)
        keys = sidecar.board_keys(ProvenanceKey(script, 0, "keyword", 0))
        assert len(keys) == len([line for line in sidecar.lines if line.line_kind == "chapter"])
        assert [sidecar.row_for(key) for key in keys] == sorted(
            {sidecar.row_for(key) for key in keys})

    def test_a_key_a_line_declares_is_itself_on_the_board(self):
        assert _cache().get(_SAGA).board_keys(_chapter(1)) == (_chapter(1),)

    def test_acting_it_is_ambiguous_counted_and_without_text(self):
        """Review focus 1: a real gen-3 record acting through the saga's bare
        key, as a record written before the key change did, joined against the
        current converter's sidecar: no text, never chapter I's, and
        counted."""
        trigger = next(r for r in read_shard(GEN3_RECORDS) if r.kind is RecordKind.TRIGGER)
        record = replace(trigger, ability=(_SAGA_KEY,))
        key = record.ability[0]
        assert (key.script_file, key.trait_kind, key.built) == (_SAGA, "keyword", None)
        cache = _cache()
        assert cache.resolution_of(key) is KeyResolution.AMBIGUOUS
        assert KeyResolver(cache, Surface.SCRIPT).acting_text(record) is None
        assert acting_line(record, cache) is None
        # Three lookups, each through the join: the resolution, the text, the line.
        assert cache.join_gaps().ambiguous == {key.script_file: 3}
        assert any("ambiguous" in line for line in cache.join_gaps().report_lines())


class TestABareKeyALineDeclaresBesideItsBuiltLines:
    def test_a_tributes_unpaid_trigger_line_is_a_board_slot_after_the_keywords_line(self):
        """Fanatic of Xenagos: line 5 declares ``keyword 1``, line 6 a ``built``
        trigger key of it; both are slots, in line order."""
        script = "cardsfolder/f/fanatic_of_xenagos.txt"
        sidecar = SidecarCache({"cardsfolder": _TRIBUTE_TREE / "cardsfolder"}).get(script)
        bare = ProvenanceKey(script, 0, "keyword", 1)
        keys = sidecar.board_keys(bare)
        assert keys == (bare, replace(bare, built=BuiltTrait("trigger", 0)))
        assert [sidecar.row_for(key) for key in keys] == [1, 2]

    def test_a_keyword_declaring_its_built_key_on_its_own_line_is_one_slot(self):
        script = "cardsfolder/a/aven_augur.txt"
        key = ProvenanceKey(script, 0, "keyword", 0)
        assert _cache().get(script).board_keys(key) == (key,)


class TestEveryTextlessAnswerIsCounted:
    def test_a_dropped_and_a_runtime_only_key_are_each_counted_once(self):
        script = "cardsfolder/a/aven_augur.txt"
        cache = _cache()
        assert cache.join(ProvenanceKey(script, 0, "spell", 0)) == (KeyResolution.DROPPED, None)
        runtime = ProvenanceKey(script, 0, "keyword", 9)
        assert cache.join(runtime) == (KeyResolution.RUNTIME_ONLY, None)
        gaps = cache.join_gaps()
        assert (gaps.dropped, gaps.runtime_only) == ({script: 1}, {script: 1})
        assert any("runtime_only" in line for line in gaps.report_lines())


def _charm_cache() -> SidecarCache:
    return SidecarCache({"cardsfolder": _CHARM_TREE / "cardsfolder"})


def _joined_prose(key: ProvenanceKey) -> tuple[int, str | None]:
    """The line index the join gives ``key`` and that line's converted prose."""
    cache = _charm_cache()
    joined = cache.join(key)
    assert joined.resolution is KeyResolution.LINE
    return joined.line.line_index, cache.prose_for(key)


_FANTASTIC_FOUR = "cardsfolder/t/the_fantastic_four.txt"
_WAR_RIDERS = "cardsfolder/v/varchilds_war_riders.txt"
_BOW = "cardsfolder/b/bow_of_nylea.txt"


class TestACharmModeKeyResolvesToItsOptionLine:
    @pytest.mark.parametrize(("option", "line_index", "prose"), [
        (0, 5, "create a 0/4 colorless wall creature token with defender."),
        (1, 6, "CARDNAME deal 3 damage to each opponent."),
        (2, 7, "put two +1/+1 counters on target creature."),
        (3, 8, "draw a card."),
    ])
    def test_the_fantastic_four_trigger_options(self, option, line_index, prose):
        key = ProvenanceKey(_FANTASTIC_FOUR, 0, "trigger", 0, option=option)
        assert _joined_prose(key) == (line_index, prose)

    @pytest.mark.parametrize(("option", "line_index", "prose"), [
        (0, 7, "have an opponent create a 1/1 red survivor creature token"
               " for each age counter on CARDNAME."),
        (1, 8, "sacrifice CARDNAME."),
    ])
    def test_varchild_s_war_riders_trigger_options(self, option, line_index, prose):
        key = ProvenanceKey(_WAR_RIDERS, 0, "trigger", 0, option=option)
        assert _joined_prose(key) == (line_index, prose)

    def test_the_trigger_key_without_an_option_reads_the_triggered_line(self):
        key = ProvenanceKey(_WAR_RIDERS, 0, "trigger", 0)
        assert _joined_prose(key) == (6, (
            "cumulative upkeep—have an opponent create a 1/1 red survivor creature token."))


class TestAnActivatedCharmKeyResolvesToItsActivatedLine:
    def test_bow_of_nylea_s_spell_1_is_the_activated_line(self):
        key = ProvenanceKey(_BOW, 0, "spell", 1)
        assert _joined_prose(key) == (3, "{1}{G}, {T}: choose one")
        assert _charm_cache().line_for(key).line_kind == "activated"

    def test_its_first_option_is_the_counter_line_not_the_activated_one(self):
        key = ProvenanceKey(_BOW, 0, "spell", 1, option=0)
        assert _joined_prose(key) == (4, "put a +1/+1 counter on target creature.")


class TestNoKeyIsDeclaredByTwoLines:
    def test_a_sidecar_declaring_a_key_twice_raises_naming_its_path_and_key(self, tmp_path):
        """A sidecar written before the key change: chapter I's key on chapter
        II's line too."""
        data = _fixture_json(_SAGA)
        data["lines"][1]["provenance"].append(data["lines"][0]["provenance"][0])
        path = tmp_path / "braidss_frightful_return.provenance.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(SidecarError) as raised:
            read_sidecar(path)
        assert str(path) in str(raised.value)
        assert _chapter(0).label in str(raised.value)

    def test_one_line_naming_a_key_twice_is_one_declaration(self):
        data = _fixture_json(_SAGA)
        data["lines"][0]["provenance"] *= 2
        assert sidecar_from_dict(data).line_for(_chapter(0)).line_index == 3


class TestOneJoinAnswersEveryLookup:
    """``SidecarCache.join`` is the one key-to-line lookup: every other
    reader's answer, and every tally of a key with no text, comes from it."""

    def test_the_join_gives_a_key_its_resolution_and_line(self):
        cache = _cache()
        joined = cache.join(_chapter(2))
        assert joined.resolution is KeyResolution.LINE
        assert joined.line == cache.line_for(_chapter(2))
        assert cache.join(_SAGA_KEY) == (KeyResolution.AMBIGUOUS, None)

    def test_an_unconfigured_tree_is_counted_and_warned_once(self, caplog):
        """A variant key on a run given no ``--variant-scripts``: every reader,
        the trainer's text included, counts it, and the run says so once."""
        cache = _cache()
        key = ProvenanceKey("variant-scripts/braidss_frightful_return_v1.txt", 0, "keyword", 0)
        with caplog.at_level("WARNING", logger="effects.infrastructure.sidecar_io"):
            assert cache.join(key) == (KeyResolution.UNCONFIGURED_TREE, None)
            assert cache.line_for(key) is None
            assert KeyResolver(cache, Surface.SCRIPT).text_of(key) is None
        assert cache.join_gaps().unconfigured == {"variant-scripts": 3}
        warned = [r for r in caplog.records if "variant-scripts" in r.getMessage()]
        assert len(warned) == 1

    def test_an_unconverted_script_is_counted_by_the_join_alone(self):
        """A token keyed under the wrong tree: counted per lookup through the
        join; reading the sidecar itself is not a lookup and counts nothing."""
        cache = _cache()
        key = ProvenanceKey("cardsfolder/b/bird_token.txt", 0, "spell", 0)
        with pytest.raises(KeyError, match="never converted"):
            cache.get(key.script_file)
        assert cache.join(key) == (KeyResolution.UNCONVERTED, None)
        assert cache.resolution_of(key) is KeyResolution.UNCONVERTED
        assert cache.join_gaps().unresolved == {key.script_file: 2}


class TestOneFolderPerTree:
    def test_two_cards_folders_of_one_tree_are_refused_naming_both(self, tmp_path):
        """The fixture's card tree and a kept-aside copy: both key as
        ``cardsfolder/...``, so one would silently shadow the other."""
        from effects.infrastructure.sidecar_io import sidecar_roots

        fixture, kept_aside = GEN3_SIDECAR_FOLDERS[0], tmp_path / "old-cardsfolder"
        with pytest.raises(ValueError, match="cardsfolder") as raised:
            sidecar_roots([fixture, kept_aside])
        assert str(fixture) in str(raised.value)
        assert str(kept_aside) in str(raised.value)

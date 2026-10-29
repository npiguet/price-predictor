"""The provenance join on disk (T036).

The join's cases are the whole contract — a key resolves, a key was
deduplicated away, a trait exists only at runtime, or the sidecar does not
describe the card at all, and only the last may fail — and are pinned on the
domain sidecar in ``tests/unit/effects/domain/test_provenance_resolution.py``.
Here: the alignment rule, the file form, and the cache that reads it.
"""

from __future__ import annotations

import errno
import json
from pathlib import Path

import pytest

from effects.domain.provenance import (
    ProvenanceKey,
    ProvenanceSidecar,
    RoleSpan,
    SidecarLine,
    SubAbilityLink,
)
from effects.infrastructure import sidecar_io as _sidecar_io
from effects.infrastructure.sidecar_io import (
    SidecarCache,
    prose_for,
    read_sidecar,
    sidecar_from_dict,
    sidecar_path_for,
    sidecar_to_dict,
)
from tests.unit.effects.builders import write_sidecar

_SCRIPT = "cardsfolder/a/ajanis_pridemate.txt"
_TRIGGER = ProvenanceKey(_SCRIPT, 0, "trigger", 0)
_STATIC = ProvenanceKey(_SCRIPT, 0, "static", 0)
_DROPPED = ProvenanceKey(_SCRIPT, 0, "static", 2)
#: A gap inside the declared ``static`` range: the fixture declares index 0 on
#: a line and index 2 as dropped, so index 1 is in neither list while the
#: converter demonstrably parsed around it. The only remaining mismatch shape.
_ABSENT = ProvenanceKey(_SCRIPT, 0, "static", 1)


def _sidecar() -> ProvenanceSidecar:
    """One card whose line 3 merged a trigger and a static, plus a dropped trait."""
    return ProvenanceSidecar(
        card="Ajani's Pridemate",
        script_file=_SCRIPT,
        lines=(
            SidecarLine(
                line_index=0, line_kind="keyword", provenance=(),
                script_text="Flying",
            ),
            SidecarLine(
                line_index=3,
                line_kind="triggered",
                provenance=(_TRIGGER, _STATIC),
                sub_ability_links=(SubAbilityLink(path=(0,), label="DBPutCounter"),),
                script_api_type="PutCounter",
                script_param_keys=("Defined", "CounterType", "CounterNum"),
                script_text=(
                    "DB$ PutCounter | Defined$ Self | CounterType$ P1P1 | "
                    "CounterNum$ 1"
                ),
                role_spans=(
                    RoleSpan(0, 34, "trigger-condition"),
                    RoleSpan(35, 71, "effect"),
                ),
            ),
        ),
        dropped_keys=(_DROPPED,),
    )


class TestRowAlignment:
    def test_row_i_is_the_ith_entry_of_lines(self):
        assert _sidecar().row_for(_TRIGGER) == 1

    def test_the_row_is_the_position_not_the_rendered_line_index(self):
        """``line_index`` points into the .txt; the cache row is the array slot."""
        sidecar = _sidecar()
        assert sidecar.lines[1].line_index == 3
        assert sidecar.row_for(_TRIGGER) == 1

    def test_a_merged_line_carries_several_keys_that_share_one_row(self):
        sidecar = _sidecar()
        assert sidecar.row_for(_TRIGGER) == sidecar.row_for(_STATIC) == 1
        assert len(sidecar.lines[1].provenance) == 2


class TestSerialization:
    def test_a_sidecar_round_trips(self):
        sidecar = _sidecar()
        assert sidecar_from_dict(sidecar_to_dict(sidecar)) == sidecar

    def test_per_line_keys_omit_the_script_file(self):
        data = sidecar_to_dict(_sidecar())
        assert "script_file" not in data["lines"][1]["provenance"][0]
        assert data["script_file"] == _SCRIPT

    def test_a_read_key_carries_the_headers_script_file(self):
        restored = sidecar_from_dict(sidecar_to_dict(_sidecar()))
        assert restored.lines[1].provenance[0].script_file == _SCRIPT

    def test_dropped_keys_survive(self):
        restored = sidecar_from_dict(sidecar_to_dict(_sidecar()))
        assert restored.dropped_keys == (_DROPPED,)

    def test_role_spans_and_sub_ability_links_survive(self):
        restored = sidecar_from_dict(sidecar_to_dict(_sidecar()))
        line = restored.lines[1]
        assert line.role_spans[0].role == "trigger-condition"
        assert line.sub_ability_links[0].path == (0,)

    def test_the_file_is_written_beside_the_converted_text(self, tmp_path):
        txt = tmp_path / "a" / "ajanis_pridemate.txt"
        path = sidecar_path_for(txt)
        assert path.name == "ajanis_pridemate.provenance.json"
        assert path.parent == txt.parent

    def test_a_written_sidecar_reads_back(self, tmp_path):
        path = sidecar_path_for(tmp_path / "a" / "ajanis_pridemate.txt")
        write_sidecar(_sidecar(), path)
        assert read_sidecar(path) == _sidecar()

    def test_a_written_sidecar_is_valid_json(self, tmp_path):
        path = sidecar_path_for(tmp_path / "x.txt")
        write_sidecar(_sidecar(), path)
        assert json.loads(path.read_text(encoding="utf-8"))["card"] == (
            "Ajani's Pridemate"
        )

    def test_a_missing_sidecar_raises_rather_than_reading_as_empty(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_sidecar(tmp_path / "never_converted.provenance.json")


class TestSidecarCache:
    def _write_trees(self, tmp_path):
        """The same filename in two trees, with different content."""
        cards = tmp_path / "cardsfolder"
        tokens = tmp_path / "tokenscripts"
        write_sidecar(
            _sidecar(), sidecar_path_for(cards / "a" / "ajanis_pridemate.txt"),
        )
        token_script = "tokenscripts/ajanis_pridemate.txt"
        write_sidecar(
            ProvenanceSidecar(
                card="Ajani's Pridemate token",
                script_file=token_script,
                lines=(
                    SidecarLine(
                        line_index=0, line_kind="keyword",
                        provenance=(ProvenanceKey(token_script, 0, "static", 0),),
                    ),
                ),
            ),
            sidecar_path_for(tokens / "ajanis_pridemate.txt"),
        )
        return SidecarCache({"cardsfolder": cards, "tokenscripts": tokens})

    def test_one_filename_resolves_differently_per_tree(self, tmp_path):
        cache = self._write_trees(tmp_path)
        assert cache.get(_SCRIPT).card == "Ajani's Pridemate"
        assert cache.get("tokenscripts/ajanis_pridemate.txt").card == (
            "Ajani's Pridemate token"
        )

    def test_the_letter_keyed_and_flat_layouts_both_resolve(self, tmp_path):
        cache = self._write_trees(tmp_path)
        assert cache.path_for(_SCRIPT).parts[-2] == "a"
        assert cache.path_for("tokenscripts/ajanis_pridemate.txt").parts[-2] == (
            "tokenscripts"
        )

    def test_a_card_is_read_once(self, tmp_path):
        cache = self._write_trees(tmp_path)
        assert cache.get(_SCRIPT) is cache.get(_SCRIPT)

    def test_an_unconfigured_tree_fails_loudly(self, tmp_path):
        cache = self._write_trees(tmp_path)
        with pytest.raises(KeyError, match="no converted root configured"):
            cache.get("variant-scripts/whatever.txt")

    def test_a_script_that_was_never_converted_raises_key_error(self, tmp_path):
        """A token entity's key names a card the converted corpus does not hold.

        The collector files a token's provenance under ``cardsfolder/`` from its
        sanitized name, while tokens convert into ``output/tokenscripts/`` under
        Forge's own script filenames, so the path resolves to nothing. It is a
        KeyError rather than a FileNotFoundError because callers already treat an
        unresolvable key as a context ability that contributes no text, and a
        distinct exception type made that handling miss this case.
        """
        cache = self._write_trees(tmp_path)
        with pytest.raises(KeyError, match="never converted"):
            cache.get("cardsfolder/b/bird_token.txt")

    def test_an_unresolvable_key_is_counted_rather_than_swallowed(self, tmp_path):
        # 0.07% of the corpus is unresolvable today. A conversion that broke
        # would raise that share without changing the shape of a run, so the
        # count is what an operator has to be able to see.
        cache = self._write_trees(tmp_path)
        for _ in range(3):
            with pytest.raises(KeyError):
                cache.get("cardsfolder/b/bird_token.txt")
        with pytest.raises(KeyError):
            cache.get("cardsfolder/c/clue_token.txt")
        assert cache.unresolved == {
            "cardsfolder/b/bird_token.txt": 3,
            "cardsfolder/c/clue_token.txt": 1,
        }

    def test_line_for_returns_none_for_a_never_converted_script(self, tmp_path):
        cache = self._write_trees(tmp_path)
        key = ProvenanceKey("cardsfolder/b/bird_token.txt", 0, "static", 0)
        assert cache.line_for(key) is None

    def test_line_or_none_reads_every_gap_as_no_line(self, tmp_path):
        """One policy for a reader that tolerates gaps: a dropped key, a script
        never converted, a tree this run has no root for, and a key the sidecar
        does not describe all read as no line."""
        cache = self._write_trees(tmp_path)
        assert cache.line_or_none(_TRIGGER).script_api_type == "PutCounter"
        for key in (
            _DROPPED,
            ProvenanceKey("cardsfolder/b/bird_token.txt", 0, "static", 0),
            ProvenanceKey("variant-scripts/shock_variant.txt", 0, "spell", 0),
            _ABSENT,
        ):
            assert cache.line_or_none(key) is None

    def test_line_or_none_counts_the_keys_a_sidecar_does_not_describe(self, tmp_path):
        """That case is the join's fail-loudly one: tolerated here, never hidden."""
        cache = self._write_trees(tmp_path)
        for _ in range(2):
            cache.line_or_none(_ABSENT)
        cache.line_or_none(_DROPPED)
        assert cache.undescribed == {_SCRIPT: 2}

    def test_script_text_is_the_line_s_normalized_script(self, tmp_path):
        """Normalized, the form the holdout hashes; None wherever the join
        gives no line or the line carries no script."""
        cache = self._write_trees(tmp_path)
        assert cache.script_text(_STATIC) == (
            "DB$ PutCounter | Defined$ Self | CounterType$ P1P1 | CounterNum$ 1"
        )
        assert cache.script_text(_DROPPED) is None
        token = ProvenanceKey("tokenscripts/ajanis_pridemate.txt", 0, "static", 0)
        assert cache.script_text(token) is None

    def test_the_cache_joins_a_record_key_to_its_row(self, tmp_path):
        cache = self._write_trees(tmp_path)
        assert cache.get(_TRIGGER.script_file).row_for(_TRIGGER) == 1
        assert cache.line_for(_TRIGGER).script_api_type == "PutCounter"


class TestTheVariantTreeHasNoProse:
    """A variant's sidecar sits beside a Forge *source* script, not a converted
    one, so reading a line of it as prose would encode script grammar as if it
    were card text — and it would look like a working run."""

    def _cache(self, tmp_path):
        variants = tmp_path / "variant-scripts"
        script = "variant-scripts/lightning_bolt_variant_0.txt"
        key = ProvenanceKey(script, 0, "spell", 0)
        write_sidecar(
            ProvenanceSidecar(
                card="Lightning Bolt Variant 0",
                script_file=script,
                lines=(
                    SidecarLine(
                        line_index=0, line_kind="spell", provenance=(key,),
                        script_text="SP$ DealDamage | NumDmg$ 7",
                    ),
                ),
            ),
            sidecar_path_for(variants / "lightning_bolt_variant_0.txt"),
        )
        # The .txt beside it is the source script the collector played.
        (variants / "lightning_bolt_variant_0.txt").write_text(
            "Name:Lightning Bolt Variant 0\nTypes:Instant\n", encoding="utf-8",
        )
        return SidecarCache({"variant-scripts": variants}), key

    def test_a_variant_key_has_no_prose(self, tmp_path):
        cache, key = self._cache(tmp_path)
        assert cache.prose_for(key) is None

    def test_the_line_still_resolves(self, tmp_path):
        """Only the prose surface is absent; the join itself is ordinary."""
        cache, key = self._cache(tmp_path)
        assert cache.get(key.script_file).row_for(key) == 0
        assert cache.line_for(key).script_text == "SP$ DealDamage | NumDmg$ 7"


class TestTheProseSplit:
    """Where a line's prose starts is the converter's answer, not a re-parse.

    A continuation line of a multi-line description has no ``kind[n]: ``
    prefix, so cutting it at its first ``": "`` would take ``{T}: `` off it as
    though it were one and shift every role span on it.
    """

    _RENDERED = ["static: creatures you control have:", "{T}: add {G}."]

    def _line(self, index, prose_offset):
        return SidecarLine(
            line_index=index, line_kind="static", provenance=(),
            prose_offset=prose_offset,
        )

    def test_a_continuation_line_keeps_its_colon(self):
        assert prose_for(self._line(1, 0), self._RENDERED) == "{T}: add {G}."

    def test_a_prefixed_line_starts_after_its_prefix(self):
        line = self._line(0, len("static: "))
        assert prose_for(line, self._RENDERED) == "creatures you control have:"

    def test_a_sidecar_without_the_offset_falls_back_to_the_first_colon(self):
        """Sidecars on disk predate the field and must still read."""
        assert prose_for(self._line(0, None), self._RENDERED) == (
            "creatures you control have:"
        )

    def test_the_offset_round_trips_and_is_absent_when_unknown(self):
        sidecar = ProvenanceSidecar(
            card="Test", script_file=_SCRIPT,
            lines=(self._line(0, 8), self._line(1, None)),
        )
        data = sidecar_to_dict(sidecar)
        assert data["lines"][0]["prose_offset"] == 8
        assert "prose_offset" not in data["lines"][1]
        assert sidecar_from_dict(data) == sidecar


class TestATransientOpenFailureIsRetried:
    """A two-hour build must not die on one file another process held open.

    A survey worker died fifteen minutes into a corpus rebuild on
    ``OSError: [Errno 22] Invalid argument`` opening a sidecar that was intact
    and parsed fine afterwards. Nothing about the file was wrong, so the read
    is retried. Each test zeroes the backoff, because the waiting is not what
    is under test -- how many times the file is opened is.
    """

    @pytest.fixture(autouse=True)
    def _no_waiting(self, monkeypatch):
        monkeypatch.setattr(_sidecar_io, "_SIDECAR_RETRY_DELAY", 0.0)

    def _written(self, tmp_path):
        path = sidecar_path_for(tmp_path / "ajanis_pridemate.txt")
        write_sidecar(_sidecar(), path)
        return path

    def test_a_read_that_succeeds_on_the_third_attempt_still_loads(
        self, tmp_path, monkeypatch, caplog,
    ):
        path = self._written(tmp_path)
        good = path.read_text(encoding="utf-8")
        attempts = []

        def flaky(self, *args, **kwargs):
            attempts.append(1)
            if len(attempts) <= 2:
                raise OSError(errno.EINVAL, "Invalid argument")
            return good

        monkeypatch.setattr(Path, "read_text", flaky)

        with caplog.at_level("WARNING"):
            loaded = read_sidecar(path)

        assert len(attempts) == 3
        assert loaded.card == "Ajani's Pridemate"
        assert str(path) in caplog.text

    def test_a_missing_sidecar_is_not_retried(self, tmp_path, monkeypatch):
        """Absence is an answer: SidecarCache reports it as an unconverted
        script, and waiting out the backoff first would slow every ordinary
        run for nothing."""
        path = self._written(tmp_path)
        attempts = []

        def missing(self, *args, **kwargs):
            attempts.append(1)
            raise FileNotFoundError(errno.ENOENT, "No such file or directory")

        monkeypatch.setattr(Path, "read_text", missing)

        with pytest.raises(FileNotFoundError):
            read_sidecar(path)

        assert len(attempts) == 1

    def test_an_errno_that_is_not_transient_propagates_at_once(
        self, tmp_path, monkeypatch,
    ):
        path = self._written(tmp_path)
        attempts = []

        def broken(self, *args, **kwargs):
            attempts.append(1)
            raise OSError(errno.EISDIR, "Is a directory")

        monkeypatch.setattr(Path, "read_text", broken)

        with pytest.raises(OSError, match="Is a directory"):
            read_sidecar(path)

        assert len(attempts) == 1

    def test_a_lock_that_never_clears_raises_the_last_error(
        self, tmp_path, monkeypatch,
    ):
        path = self._written(tmp_path)
        attempts = []

        def always_locked(self, *args, **kwargs):
            attempts.append(1)
            raise PermissionError(errno.EACCES, "Permission denied")

        monkeypatch.setattr(Path, "read_text", always_locked)

        with pytest.raises(PermissionError):
            read_sidecar(path)

        assert len(attempts) == _sidecar_io._SIDECAR_READ_ATTEMPTS


class TestOptionKeys:
    """A charm mode's key carries ``option``; a root or plain key does not (FR-005a)."""

    _SCRIPT = "cardsfolder/c/cryptic_command.txt"

    def _sidecar(self) -> ProvenanceSidecar:
        root = ProvenanceKey(self._SCRIPT, 0, "spell", 0)
        return ProvenanceSidecar(
            card="cryptic command", script_file=self._SCRIPT,
            lines=(
                SidecarLine(line_index=3, line_kind="spell", provenance=(root,),
                            script_text="CharmNum$ 2 | Choices$ SV1,SV2 | SP$ Charm"),
                SidecarLine(line_index=4, line_kind="option",
                            provenance=(ProvenanceKey(self._SCRIPT, 0, "spell", 0, 0),),
                            script_text="SV1: DB$ Counter"),
                SidecarLine(line_index=5, line_kind="option",
                            provenance=(ProvenanceKey(self._SCRIPT, 0, "spell", 0, 1),),
                            script_text="SV1: DB$ Draw | NumCards$ 1"),
            ),
            dropped_keys=(ProvenanceKey(self._SCRIPT, 1, "spell", 0, 3),),
        )

    def test_round_trip_keeps_the_option_component(self):
        sidecar = self._sidecar()
        assert sidecar_from_dict(sidecar_to_dict(sidecar)) == sidecar

    def test_option_is_written_only_when_set(self):
        data = sidecar_to_dict(self._sidecar())
        assert data["lines"][0]["provenance"] == [
            {"face": 0, "trait_kind": "spell", "index_within_kind": 0}
        ]
        assert data["lines"][1]["provenance"][0]["option"] == 0
        assert data["dropped_keys"][0]["option"] == 3

    def test_a_gen1_sidecar_reads_with_no_option(self):
        data = sidecar_to_dict(self._sidecar())
        for line in data["lines"]:
            for key in line["provenance"]:
                key.pop("option", None)
        sidecar = sidecar_from_dict(data)
        assert all(key.option is None for line in sidecar.lines for key in line.provenance)


class TestSidecarRoots:
    """``--cards-folder`` values to a cache's ``tree -> folder`` (FR-063b)."""

    def test_a_folder_named_for_its_tree_is_that_tree(self):
        from effects.infrastructure.sidecar_io import tree_of_folder

        assert tree_of_folder(Path("output/cardsfolder/")) == "cardsfolder"
        assert tree_of_folder(Path("output/tokenscripts")) == "tokenscripts"

    def test_a_kept_aside_copy_keeps_its_original_tree(self):
        from effects.infrastructure.sidecar_io import sidecar_roots

        roots = sidecar_roots(
            [Path("output/gen1-cardsfolder"), Path("output/gen1-tokenscripts")],
            Path("output/effects/variant-scripts"),
        )
        assert roots == {
            "cardsfolder": Path("output/gen1-cardsfolder"),
            "tokenscripts": Path("output/gen1-tokenscripts"),
            "variant-scripts": Path("output/effects/variant-scripts"),
        }

    def test_the_script_vocabularies_read_every_segment(self, tmp_path):
        from effects.infrastructure.sidecar_io import script_vocabularies

        key = ProvenanceKey("cardsfolder/a/a.txt", 0, "spell", 0)
        write_sidecar(ProvenanceSidecar(
            card="a", script_file="cardsfolder/a/a.txt",
            lines=(SidecarLine(
                0, "spell", (key,), script_api_type="Draw",
                script_param_keys=("NumCards", "SP", "SubAbility"),
                script_text="NumCards$ 1 | SP$ Draw | SubAbility$ SV1 "
                            "[SEG] SV1: DB$ Pump | NumAtt$ +1",
            ),),
        ), tmp_path / "cardsfolder" / "a" / "a.provenance.json")
        api_types, keys = script_vocabularies([tmp_path / "cardsfolder"])
        assert api_types == ["Draw"]
        assert {"DB", "NumAtt", "NumCards", "SP", "SubAbility"} <= set(keys)


class TestLoadTreeTexts:
    """The holdout's token carriers, read straight off a converted tree."""

    def test_each_script_file_maps_to_its_lines_script_texts(self, tmp_path):
        from effects.infrastructure.sidecar_io import load_tree_texts

        key = ProvenanceKey("tokenscripts/goblin.txt", 0, "spell", 1)
        write_sidecar(
            ProvenanceSidecar(
                card="goblin", script_file="tokenscripts/goblin.txt",
                lines=(
                    SidecarLine(line_index=0, line_kind="spell", provenance=(key,),
                                script_text="SP$ DealDamage | NumDmg$ 1"),
                    SidecarLine(line_index=1, line_kind="keyword", provenance=()),
                ),
            ),
            sidecar_path_for(tmp_path / "tokenscripts" / "goblin.txt"),
        )
        assert load_tree_texts(tmp_path / "tokenscripts") == {
            "tokenscripts/goblin.txt": ["SP$ DealDamage | NumDmg$ 1"],
        }

    def test_a_missing_tree_is_empty(self, tmp_path):
        from effects.infrastructure.sidecar_io import load_tree_texts

        assert load_tree_texts(tmp_path / "absent") == {}
        assert load_tree_texts(None) == {}


class TestTheTreeWalk:
    def test_every_sidecar_is_read_recursively_in_path_order(self, tmp_path):
        from effects.infrastructure.sidecar_io import iter_sidecars

        for script in ("cardsfolder/b/bear.txt", "cardsfolder/a/ant.txt"):
            write_sidecar(
                ProvenanceSidecar(card=script, script_file=script, lines=()),
                sidecar_path_for(tmp_path / script),
            )
        (tmp_path / "cardsfolder" / "a" / "ant.txt").write_text("not a sidecar")
        found = [sidecar.script_file for _path, sidecar in iter_sidecars(tmp_path)]
        assert found == ["cardsfolder/a/ant.txt", "cardsfolder/b/bear.txt"]

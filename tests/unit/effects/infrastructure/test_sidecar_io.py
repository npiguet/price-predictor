"""The provenance join (T036).

Three cases and one alignment rule. The three cases are the whole contract: a
key resolves, a key was deduplicated away, or the sidecar does not describe the
card at all — and only the third may fail.
"""

from __future__ import annotations

import json

import pytest

from effects.domain.provenance import (
    ProvenanceKey,
    ProvenanceSidecar,
    RoleSpan,
    SidecarLine,
    SubAbilityLink,
)
from effects.infrastructure.sidecar_io import (
    SidecarCache,
    read_sidecar,
    sidecar_from_dict,
    sidecar_path_for,
    sidecar_to_dict,
    write_sidecar,
)

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


class TestTheThreeJoinCases:
    def test_a_key_in_lines_resolves(self):
        line = _sidecar().line_for(_TRIGGER)
        assert line.line_kind == "triggered"

    def test_a_dropped_key_resolves_to_no_line_and_does_not_raise(self):
        """The converter deduplicated it; the trait is still live at runtime."""
        assert _sidecar().line_for(_DROPPED) is None
        assert _sidecar().row_for(_DROPPED) is None

    def test_a_gap_inside_a_declared_range_fails_loudly(self):
        with pytest.raises(KeyError, match="neither the lines nor the dropped_keys"):
            _sidecar().line_for(_ABSENT)

    def test_the_failure_names_the_script_file_and_the_likely_cause(self):
        with pytest.raises(KeyError) as excinfo:
            _sidecar().line_for(_ABSENT)
        assert _SCRIPT in str(excinfo.value)
        assert "reconversion" in str(excinfo.value)


class TestARuntimeOnlyTrait:
    """A trait Forge attaches to a live card that the parsed script never had.

    Level up, bestow and scavenge each append a spell ability. A static that
    grants a trigger to other permanents keys that trigger to the granting
    card's script, which declares no trigger. A disguise creature's face-down
    state keys to a face the script never wrote down. All three belong to no
    line and no dropped key, and failing on them stops training on a corpus
    that is entirely correct.

    They are told apart by what the sidecar declared: nothing at all for the
    key's ``(face, trait_kind)``, or nothing this far along it.
    """

    def test_an_index_past_everything_declared_resolves_to_no_line(self):
        # The fixture declares static[0] (rendered) and static[2] (dropped).
        beyond = ProvenanceKey(_SCRIPT, 0, "static", 5)
        assert _sidecar().is_runtime_only(beyond)
        assert _sidecar().line_for(beyond) is None
        assert _sidecar().row_for(beyond) is None

    def test_a_gap_inside_the_declared_range_still_fails(self):
        """``dropped_keys`` is declared-minus-claimed, so every index the
        converter parsed is in one of the two lists. A gap between them means
        the sidecar was rebuilt against a different trait list."""
        assert not _sidecar().is_runtime_only(_ABSENT)
        with pytest.raises(KeyError):
            _sidecar().line_for(_ABSENT)

    def test_a_kind_the_face_never_declared_is_runtime_only(self):
        """Nothing was declared for the kind, so nothing about it was parsed.

        On the real corpus this is a static that grants a trigger to *other*
        permanents: the granted trigger keys to the granting card's script,
        which declares no trigger of its own. This fixture does declare a
        trigger, so ``activated`` stands in for the undeclared kind.
        """
        granted = ProvenanceKey(_SCRIPT, 0, "activated", 0)
        assert _sidecar().is_runtime_only(granted)
        assert _sidecar().line_for(granted) is None

    def test_a_dropped_key_counts_toward_what_was_declared(self):
        """It is a trait the converter saw, so it bounds the parsed range."""
        assert _sidecar().is_runtime_only(ProvenanceKey(_SCRIPT, 0, "static", 3))
        assert not _sidecar().is_runtime_only(
            ProvenanceKey(_SCRIPT, 0, "static", 2)
        )

    def test_a_face_the_script_never_declared_is_runtime_only(self):
        """Each face has its own trait list, and a disguise creature's
        face-down state keys to a face the converted script never wrote
        down."""
        assert _sidecar().is_runtime_only(
            ProvenanceKey(_SCRIPT, 1, "keyword", 0)
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

    def test_the_cache_joins_a_record_key_to_its_row(self, tmp_path):
        cache = self._write_trees(tmp_path)
        assert cache.row_for(_TRIGGER) == 1
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
        assert cache.row_for(key) == 0
        assert cache.line_for(key).script_text == "SP$ DealDamage | NumDmg$ 7"

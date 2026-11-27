"""``SourceTrees`` and ``ConvertedTrees``: which folder holds which tree, and
what is read from them."""

from __future__ import annotations

from pathlib import Path

import pytest

from effects.domain.provenance import ProvenanceKey, ProvenanceSidecar, SidecarLine, SourceTree
from effects.domain.split import HoldoutRule
from effects.infrastructure.converted_trees import SourceTrees, process_sidecars
from effects.infrastructure.sidecar_io import sidecar_path_for
from price_predictor.domain.converted_provenance import ConversionRefused
from tests.unit.conversion_fixtures import (
    DIGEST,
    FORGE_COMMIT,
    OTHER_COMMIT,
    conversion_member,
    stand_in_checkout,
    write_conversion,
    write_jar,
)
from tests.unit.effects.file_builders import write_sidecar


def _convert(root: Path, script_file: str, name: str, text: str) -> None:
    tree, relative = script_file.split("/", 1)
    txt = root / tree / relative
    txt.parent.mkdir(parents=True, exist_ok=True)
    txt.write_text(f"name: {name}\ntypes: instant\n", encoding="utf-8")
    write_sidecar(
        ProvenanceSidecar(
            card=name, script_file=script_file,
            lines=(SidecarLine(
                line_index=0, line_kind="spell",
                provenance=(ProvenanceKey(script_file, 0, "spell", 0),),
                script_text=text, prose_offset=0,
            ),),
        ),
        sidecar_path_for(txt),
    )


def test_each_folder_is_keyed_by_the_tree_its_name_names(tmp_path) -> None:
    source = SourceTrees.of(
        (tmp_path / "tokenscripts", tmp_path / "old-cardsfolder"),
        variant_scripts=tmp_path / "variants",
    )
    trees = source.converted()

    assert trees.roots == {
        SourceTree.TOKENSCRIPTS: tmp_path / "tokenscripts",
        SourceTree.CARDSFOLDER: tmp_path / "old-cardsfolder",
        SourceTree.VARIANT_SCRIPTS: tmp_path / "variants",
    }
    assert trees.cards_folder == tmp_path / "old-cardsfolder"
    assert source.sidecar_roots() == trees.roots


def test_without_a_card_tree_there_are_no_cards(tmp_path) -> None:
    """A token tree alone is no card tree: a token is not a deckable card."""
    _convert(tmp_path, "tokenscripts/goblin.txt", "Goblin", "SP$ Bespoke")
    trees = SourceTrees.of((tmp_path / "tokenscripts",)).converted()

    assert trees.cards_folder is None
    assert trees.card_files == {}
    assert trees.token_texts() == {"tokenscripts/goblin.txt": ["SP$ Bespoke"]}


def test_cards_texts_and_tokens_are_read_through_one_sidecar_cache(tmp_path) -> None:
    _convert(tmp_path, "cardsfolder/s/shock.txt", "Shock", "SP$ DealDamage | NumDmg$ 2")
    _convert(tmp_path, "tokenscripts/goblin.txt", "Goblin", "SP$ Bespoke")
    trees = SourceTrees.of((tmp_path / "cardsfolder", tmp_path / "tokenscripts")).converted()

    assert trees.card_files == {"shock": "cardsfolder/s/shock.txt"}
    assert trees.card_texts() == {"shock": ["SP$ DealDamage | NumDmg$ 2"]}
    assert trees.sidecars.get("tokenscripts/goblin.txt").card == "Goblin"


#: Every text held out, unless a second carrier prints it.
_ONE_CARRIER = HoldoutRule(permille=1000, max_carriers=1)


def test_a_token_printing_a_text_is_a_carrier_of_it(tmp_path) -> None:
    _convert(tmp_path, "cardsfolder/s/shock.txt", "Shock", "SP$ Bespoke")
    cards_only = SourceTrees.of((tmp_path / "cardsfolder",)).converted()
    assert cards_only.holdout(_ONE_CARRIER).names == {"shock"}

    _convert(tmp_path, "tokenscripts/goblin.txt", "Goblin", "SP$ Bespoke")
    with_tokens = SourceTrees.of((tmp_path / "cardsfolder", tmp_path / "tokenscripts")).converted()
    assert with_tokens.holdout(_ONE_CARRIER).names == set()


def test_the_variant_tree_is_the_last_folder_and_absent_when_unnamed(tmp_path) -> None:
    cards = (tmp_path / "cardsfolder", tmp_path / "tokenscripts")
    assert SourceTrees.of(cards).folders() == list(cards)
    assert SourceTrees.of(cards, tmp_path / "variants").folders() == [
        *cards, tmp_path / "variants",
    ]
    # An empty flag value is unset, as omitting it is.
    assert SourceTrees.of(cards, "").variant_scripts is None


def test_a_converted_reading_can_share_a_cache_already_built(tmp_path) -> None:
    source = SourceTrees.of((tmp_path / "cardsfolder",))
    shared = process_sidecars(source.sidecar_roots())
    assert source.converted(sidecars=shared).sidecars is shared
    assert source.converted().sidecars is not shared


def test_a_process_keeps_one_cache_per_roots(tmp_path) -> None:
    roots = SourceTrees.of((tmp_path / "cardsfolder",)).sidecar_roots()
    assert process_sidecars(roots) is process_sidecars(dict(roots))
    other = SourceTrees.of((tmp_path / "other-cardsfolder",)).sidecar_roots()
    assert process_sidecars(other) is not process_sidecars(roots)


# ── the conversion the card trees come from ─────


@pytest.fixture(autouse=True)
def _checkout_at_the_jars_commit(monkeypatch) -> None:
    stand_in_checkout(monkeypatch)


def _trees(root: Path) -> SourceTrees:
    return SourceTrees.of((root / "cardsfolder", root / "tokenscripts"))


def test_both_card_trees_name_their_one_conversion(tmp_path) -> None:
    root = write_conversion(tmp_path / "converted")
    jar = write_jar(tmp_path / "connector.jar")

    conversion = _trees(root).conversion(jar=jar)

    assert conversion.forge_commit == FORGE_COMMIT
    assert conversion.manifest == root / "run.json"
    assert conversion.sidecar_digest == DIGEST


def test_a_tree_converted_at_another_commit_is_refused_naming_both(tmp_path) -> None:
    root = write_conversion(tmp_path / "converted", conversion_member(OTHER_COMMIT))
    jar = write_jar(tmp_path / "connector.jar", FORGE_COMMIT)

    with pytest.raises(ConversionRefused) as refused:
        _trees(root).conversion(jar=jar)

    assert FORGE_COMMIT in str(refused.value)
    assert OTHER_COMMIT in str(refused.value)
    assert str(jar) in str(refused.value)


def test_trees_of_two_conversions_are_refused(tmp_path) -> None:
    first = write_conversion(tmp_path / "first")
    second = write_conversion(tmp_path / "second")
    trees = SourceTrees.of((first / "cardsfolder", second / "tokenscripts"))

    with pytest.raises(ConversionRefused, match="2 conversions"):
        trees.conversion(jar=write_jar(tmp_path / "connector.jar"))

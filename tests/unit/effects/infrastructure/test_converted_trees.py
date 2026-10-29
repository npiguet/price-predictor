"""``ConvertedTrees``: which folder holds which tree, and what is read from them."""

from __future__ import annotations

from pathlib import Path

from effects.domain.provenance import ProvenanceKey, ProvenanceSidecar, SidecarLine, SourceTree
from effects.domain.split import HoldoutRule
from effects.domain.text_holdout import HoldoutUnit
from effects.infrastructure.converted_trees import ConvertedTrees
from effects.infrastructure.sidecar_io import sidecar_path_for, write_sidecar


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
                script_text=text,
            ),),
        ),
        sidecar_path_for(txt),
    )


def test_each_folder_is_keyed_by_the_tree_its_name_names(tmp_path) -> None:
    trees = ConvertedTrees.from_folders(
        (tmp_path / "tokenscripts", tmp_path / "gen1-cardsfolder"),
        variant_scripts=tmp_path / "variants",
    )

    assert trees.roots == {
        SourceTree.TOKENSCRIPTS: tmp_path / "tokenscripts",
        SourceTree.CARDSFOLDER: tmp_path / "gen1-cardsfolder",
        SourceTree.VARIANT_SCRIPTS: tmp_path / "variants",
    }
    assert trees.cards_folder == tmp_path / "gen1-cardsfolder"
    assert trees.root_paths == {
        "tokenscripts": str(tmp_path / "tokenscripts"),
        "cardsfolder": str(tmp_path / "gen1-cardsfolder"),
        "variant-scripts": str(tmp_path / "variants"),
    }


def test_without_a_card_tree_there_are_no_cards(tmp_path) -> None:
    """A token tree alone is no card tree: a token is not a deckable card."""
    _convert(tmp_path, "tokenscripts/goblin.txt", "Goblin", "SP$ Bespoke")
    trees = ConvertedTrees.from_folders((tmp_path / "tokenscripts",))

    assert trees.cards_folder is None
    assert trees.card_files == {}
    assert trees.token_texts() == {"tokenscripts/goblin.txt": ["SP$ Bespoke"]}


def test_cards_texts_and_tokens_are_read_through_one_sidecar_cache(tmp_path) -> None:
    _convert(tmp_path, "cardsfolder/s/shock.txt", "Shock", "SP$ DealDamage | NumDmg$ 2")
    _convert(tmp_path, "tokenscripts/goblin.txt", "Goblin", "SP$ Bespoke")
    trees = ConvertedTrees.from_folders((tmp_path / "cardsfolder", tmp_path / "tokenscripts"))

    assert trees.card_files == {"Shock": "cardsfolder/s/shock.txt"}
    assert trees.card_texts() == {"Shock": ["SP$ DealDamage | NumDmg$ 2"]}
    assert trees.sidecars.get("tokenscripts/goblin.txt").card == "Goblin"


#: Every text held out, unless a second carrier prints it.
_ONE_CARRIER = HoldoutRule(permille=1000, max_carriers=1, unit=HoldoutUnit.TEXT)


def test_a_token_printing_a_text_is_a_carrier_of_it(tmp_path) -> None:
    _convert(tmp_path, "cardsfolder/s/shock.txt", "Shock", "SP$ Bespoke")
    cards_only = ConvertedTrees.from_folders((tmp_path / "cardsfolder",))
    assert cards_only.holdout(_ONE_CARRIER).names == {"shock"}

    _convert(tmp_path, "tokenscripts/goblin.txt", "Goblin", "SP$ Bespoke")
    with_tokens = ConvertedTrees.from_folders((tmp_path / "cardsfolder", tmp_path / "tokenscripts"))
    assert with_tokens.holdout(_ONE_CARRIER).names == set()

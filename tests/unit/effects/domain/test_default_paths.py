"""The cards-folder default names the artifact-layout's gen-3 converted tree."""

from __future__ import annotations

from pathlib import Path

from effects.domain import default_paths

GEN3_CONVERTED = (
    "output/effects/gen3/converted/cardsfolder/",
    "output/effects/gen3/converted/tokenscripts/",
)


def test_the_cards_folder_default_is_the_gen3_converted_tree():
    assert default_paths.DEFAULT_CARDS_FOLDERS == GEN3_CONVERTED


def test_the_path_form_holds_the_same_two_trees():
    assert default_paths.DEFAULT_CARDS_FOLDER_PATHS == tuple(Path(f) for f in GEN3_CONVERTED)

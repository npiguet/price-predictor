"""Taking a variant round's staged scripts back out of Forge's custom-cards
directory (FR-058b; python R8)."""

from __future__ import annotations

import re
from pathlib import Path

from effects.infrastructure.variant_staging import (
    STAGED_MANIFEST,
    custom_cards_dir,
    unstage_variants,
)

_CONNECTOR = (Path(__file__).resolve().parents[4] / "forge-connector" / "src" / "main" / "java"
              / "com" / "pricepredictor" / "connector")


def _staged(directory: Path, *names: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).write_text(f"Name:{name}\n", encoding="utf-8")
    (directory / STAGED_MANIFEST).write_text("\n".join(names) + "\n", encoding="utf-8")


def test_every_staged_script_is_removed_and_the_manifest_emptied(tmp_path):
    _staged(tmp_path, "bolt_variant_0a1b2c3d.txt", "shock_variant_99aa00ff.txt")

    assert unstage_variants(tmp_path) == 2

    assert not list(tmp_path.glob("*.txt"))
    assert (tmp_path / STAGED_MANIFEST).read_text(encoding="utf-8") == ""


def test_a_card_a_person_put_there_is_left_alone(tmp_path):
    """Only files the staging copied, named by its manifest, are ours."""
    _staged(tmp_path, "bolt_variant_0a1b2c3d.txt")
    (tmp_path / "my_cube_card.txt").write_text("Name:Mine\n", encoding="utf-8")

    unstage_variants(tmp_path)

    assert (tmp_path / "my_cube_card.txt").exists()


def test_nothing_staged_removes_nothing(tmp_path):
    assert unstage_variants(tmp_path) == 0
    assert unstage_variants(tmp_path / "missing") == 0


def test_a_listed_file_already_gone_is_not_an_error(tmp_path):
    _staged(tmp_path, "bolt_variant_0a1b2c3d.txt")
    (tmp_path / "bolt_variant_0a1b2c3d.txt").unlink()

    assert unstage_variants(tmp_path) == 0


def test_the_manifest_is_the_one_the_staging_writes():
    source = (_CONNECTOR / "ForgeEnvironmentInitializer.java").read_text(encoding="utf-8")
    declared = re.search(r'STAGED_MANIFEST\s*=\s*"([^"]+)"', source)
    assert declared is not None and declared.group(1) == STAGED_MANIFEST


def test_the_directory_is_forge_s_user_custom_cards_under_the_headless_assets_dir():
    """The worker sets Forge's assets dir to ``forge-gui/`` and a headless GUI
    reads its user dir as ``<assets>/data/``; ``USER_CUSTOM_CARDS_DIR`` is
    ``<user>/custom/cards/``."""
    assert custom_cards_dir().parts[-5:] == ("forge", "forge-gui", "data", "custom", "cards")

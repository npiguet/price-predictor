"""Variant scripts Forge can read, and only on the script surface (T139).

A variant deck is made of cards nobody printed, so a variant run that appended
to ``match-outcomes.txt`` would put synthetic cards into the scorer's training
data. Variant rounds play through the same records-only worker as coverage
rounds, so ``test_effects_coverage_isolation.py`` is the check that no sealed
file is ever constructed; this module checks what is particular to variants.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from effects.application.collect_variants import generate_variants
from effects.infrastructure.one_shot_workers import write_variant_sidecars
from effects.infrastructure.sidecar_io import read_sidecar, sidecar_path_for
from tests.integration.conftest import require_jar

pytestmark = pytest.mark.integration

_BOLT = (
    "Name:Test Bolt\nManaCost:R\nTypes:Instant\n"
    "A:SP$ DealDamage | Cost$ R | ValidTgts$ Any | NumDmg$ 3 | "
    "SpellDescription$ CARDNAME deals 3 damage to any target.\n"
    "Oracle:Test Bolt deals 3 damage to any target.\n"
)


def test_generated_variants_are_scripts_forge_can_read(tmp_path: Path) -> None:
    """A variant that Forge cannot load never reaches a game, which would make
    the whole stage silently produce nothing."""
    source = tmp_path / "cards"
    source.mkdir()
    (source / "test_bolt.txt").write_text(_BOLT, encoding="utf-8")

    variants = generate_variants(
        source, tmp_path / "variants", held_out=frozenset(), limit=3,
    )
    assert variants, "no variant was generated from a perturbable script"

    for variant in variants:
        text = variant.path.read_text(encoding="utf-8")
        # Forge's script grammar: a Name line, a Types line, and Key$ Value
        # parameters separated by pipes.
        assert text.startswith("Name:")
        assert "Types:" in text
        assert "SP$ DealDamage" in text
        assert "NumDmg$" in text


def test_a_variant_tree_holds_no_converted_text(tmp_path: Path) -> None:
    """Variants exist on the script surface only (FR-056)."""
    source = tmp_path / "cards"
    source.mkdir()
    (source / "test_bolt.txt").write_text(_BOLT, encoding="utf-8")
    output = tmp_path / "variants"

    generate_variants(source, output, held_out=frozenset(), limit=3)

    for path in output.glob("*.txt"):
        text = path.read_text(encoding="utf-8")
        # A converted file starts with "name: " (lowercased); a Forge script
        # starts with "Name:".
        assert not text.startswith("name: ")


def test_every_variant_script_gets_a_sidecar(tmp_path: Path) -> None:
    """FR-056's other half. Without a sidecar a variant's records name a key
    the reader cannot resolve, and the join fails loudly on a corpus that is in
    fact correct — so the sidecar pass runs before the first game."""
    require_jar()
    source = tmp_path / "cards"
    source.mkdir()
    (source / "test_bolt.txt").write_text(_BOLT, encoding="utf-8")
    output = tmp_path / "variants"

    variants = generate_variants(
        source, output, held_out=frozenset(), limit=3,
    )
    assert variants

    code = write_variant_sidecars(output)
    assert code == 0, "VariantSidecarMain failed"

    for variant in variants:
        sidecar_path = sidecar_path_for(variant.path)
        assert sidecar_path.exists(), f"no sidecar for {variant.name}"
        sidecar = read_sidecar(sidecar_path)
        # The key the runtime writes is the path the reader resolves.
        assert sidecar.script_file == (
            f"variant-scripts/{variant.path.name}"
        )
        assert sidecar.lines, "a perturbed ability line produced no sidecar row"

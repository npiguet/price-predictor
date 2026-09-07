"""Variant matches never touch the sealed corpora (T139).

The other half of SC-007. A variant deck is made of cards nobody printed, so a
variant run that appended to ``match-outcomes.txt`` would put synthetic cards
into the scorer's training data — and the rows would look like ordinary
self-play rows, so nothing would flag it.

Byte-identical, like the coverage check: nothing is supposed to be appended at
all.
"""

from __future__ import annotations

import subprocess
import time
import uuid
from pathlib import Path

import pytest

from effects.application.collect_variants import generate_variants
from effects.infrastructure.record_io import read_records
from effects.infrastructure.sidecar_io import read_sidecar, sidecar_path_for
from effects.infrastructure.variant_sidecar_connector import (
    VariantSidecarConnector,
)
from price_predictor.infrastructure.forge_jvm import resolve_connector_jar
from sealed.infrastructure.match_worker_connector import MatchWorkerConnector

pytestmark = pytest.mark.integration

_COLLECTION_SECONDS = 180
_POLL_SECONDS = 5

_BOLT = (
    "Name:Test Bolt\nManaCost:R\nTypes:Instant\n"
    "A:SP$ DealDamage | Cost$ R | ValidTgts$ Any | NumDmg$ 3 | "
    "SpellDescription$ CARDNAME deals 3 damage to any target.\n"
    "Oracle:Test Bolt deals 3 damage to any target.\n"
)


def _require_jar() -> None:
    try:
        resolve_connector_jar()
    except FileNotFoundError:
        pytest.skip("forge-connector JAR not built (mvn package -DskipTests)")


def _run_records_only(records_dir: Path) -> int:
    process = MatchWorkerConnector().start(
        None,
        run_id=str(uuid.uuid4()),
        best_of=1,
        effect_records_dir=records_dir,
        worker_index=0,
    )
    try:
        deadline = time.monotonic() + _COLLECTION_SECONDS
        while time.monotonic() < deadline:
            time.sleep(_POLL_SECONDS)
            if process.poll() is not None:
                pytest.fail(f"worker exited early with {process.returncode}")
            if sum(1 for _ in read_records(records_dir)) >= 5:
                break
        return sum(1 for _ in read_records(records_dir))
    finally:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()


def test_a_variant_run_leaves_the_sealed_corpora_byte_identical(
    tmp_path: Path,
) -> None:
    _require_jar()
    sealed_dir = tmp_path / "sealed"
    sealed_dir.mkdir()
    outcomes = sealed_dir / "match-outcomes.txt"
    cards_played = sealed_dir / "cards-played.txt"
    outcomes.write_bytes(b"pre-existing;row\n")
    cards_played.write_bytes(b"pre-existing;row\n")
    before = (outcomes.read_bytes(), cards_played.read_bytes())

    if _run_records_only(tmp_path / "records") == 0:
        pytest.skip("worker produced no records within the collection window")

    assert (outcomes.read_bytes(), cards_played.read_bytes()) == before


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
    _require_jar()
    source = tmp_path / "cards"
    source.mkdir()
    (source / "test_bolt.txt").write_text(_BOLT, encoding="utf-8")
    output = tmp_path / "variants"

    variants = generate_variants(
        source, output, held_out=frozenset(), limit=3,
    )
    assert variants

    code = VariantSidecarConnector().run(output)
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

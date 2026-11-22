"""The variant card tree: capped, weighted and recorded like any other source."""

from __future__ import annotations

from pathlib import Path

import pytest

from effects.domain.corpus_plan import BuildCorpusError
from effects.domain.provenance import ProvenanceKey, ProvenanceSidecar, SidecarLine
from effects.domain.script_text import script_reading
from effects.infrastructure.corpus_store import CorpusStore
from effects.infrastructure.sidecar_io import sidecar_path_for
from tests.unit.effects.application.build_corpus_support import (
    _resolution,
    _write_corpus,
)
from tests.unit.effects.builders import (
    write_ended_shard,
    write_sidecar,
)

_VARIANT_FILE = "variant-scripts/bolt_numdmg_2.txt"


_VARIANT_KEY = ProvenanceKey(_VARIANT_FILE, 0, "spell", 0)


_VARIANT_TEXT = "SP$ DealDamage | NumDmg$ 2"


def _write_variant(root: Path, script_file: str, text: str) -> None:
    """A sidecar in the flat variant tree, which carries no prose surface.

    `collect-variants` writes the perturbed *source* script and a sidecar
    beside it; nothing converts it, so `script_text` is the only surface the
    line has.
    """
    relative = script_file.split("/", 1)[1]
    path = root / "variant-scripts" / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")
    write_sidecar(
        ProvenanceSidecar(
            card="Bolt Variant",
            script_file=script_file,
            lines=(
                SidecarLine(
                    line_index=0, line_kind="spell",
                    provenance=(_VARIANT_KEY,), script_text=text, prose_offset=0,
                ),
            ),
        ),
        sidecar_path_for(path),
    )


@pytest.fixture(scope="module")
def a_corpus_with_variants(tmp_path_factory):
    """The ordinary fixture plus a variant tree and a shard that uses it."""
    root = tmp_path_factory.mktemp("variants") / "source"
    fixture = _write_corpus(root)
    _write_variant(root, _VARIANT_FILE, _VARIANT_TEXT)
    write_ended_shard(
        fixture.records / "variants" / "run.0-v.jsonl.gz",
        [
            _resolution(f"g-var.{i}", "g-var", ability=(_VARIANT_KEY,))
            for i in range(6)
        ],
    )
    return fixture, str(root / "variant-scripts")


@pytest.fixture(scope="module")
def variant_build(tmp_path_factory, a_corpus_with_variants) -> CorpusStore:
    """``a_corpus_with_variants`` built with its variant tree."""
    fixture, variant_root = a_corpus_with_variants
    output = tmp_path_factory.mktemp("variant-build") / "curated"
    return fixture.build(output, variant_scripts=variant_root)


def test_a_variant_text_is_capped_and_weighted_like_any_other(variant_build):
    """FR-138/FR-141 do not exempt the variant tree.

    Without `--variant-scripts` a variant line resolves to no text at all, so
    it lands in neither the rarity table nor the per-text cap while every real
    ability is in both — uncapped records competing against capped ones.
    """
    assert script_reading(_VARIANT_TEXT) in variant_build.load().rarity


def test_without_the_variant_tree_the_build_is_refused(
    tmp_path: Path, a_corpus_with_variants,
):
    """The state this flag exists to end is refused, naming the tree, before
    anything under ``--output`` is touched."""
    fixture, _ = a_corpus_with_variants

    with pytest.raises(BuildCorpusError, match="variant-scripts"):
        fixture.build(tmp_path / "curated")

    assert not (tmp_path / "curated").exists()


def test_the_manifest_records_the_variant_tree_it_read(a_corpus_with_variants, variant_build):
    """Which trees resolved decides which texts the rarity table names."""
    _, variant_root = a_corpus_with_variants
    assert variant_build.load().variant_scripts == Path(variant_root).resolve().as_posix()


# ── quality, the gate-one slice, per-source reporting, repacking ───────

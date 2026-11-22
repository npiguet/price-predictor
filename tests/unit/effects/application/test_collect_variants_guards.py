"""The collector's guard against variants printing a held-out text is a
refusal: no converted card tree to check against, or a variant with no
sidecar to read, stops the run (finding 14.5)."""

from __future__ import annotations

import pytest

from effects.application.collect_variants import (
    CollectorAbort,
    CollectVariantsConfig,
    GeneratedVariant,
    _keep_unheld,
    drop_held_out_variants,
)
from effects.domain.split import HoldoutRule
from tests.unit.effects.application.test_collect_variants_held_out import (
    TestNoVariantPrintsAHeldOutText as Printing,
)


def _bare_variant(tmp_path, name: str) -> GeneratedVariant:
    path = tmp_path / f"{name}.txt"
    path.write_text(f"Name:{name}\n", encoding="utf-8")
    return GeneratedVariant(name=name, source_card="x", path=path, perturbation="numeric")


def test_a_variant_with_no_sidecar_is_an_error_naming_it(tmp_path, caplog):
    held = Printing()._holdout()
    bare = _bare_variant(tmp_path, "bare_variant_9")

    with pytest.raises(CollectorAbort):
        drop_held_out_variants([bare], held)

    assert "bare_variant_9" in caplog.text


def test_no_converted_card_tree_is_an_error_not_a_warning(tmp_path, caplog):
    variants = tmp_path / "variant-scripts"
    variants.mkdir()
    config = CollectVariantsConfig(
        effect_records=tmp_path / "records", variant_scripts=variants,
        cards_folders=(tmp_path / "missing" / "cardsfolder",), holdout=HoldoutRule(),
    )

    with pytest.raises(CollectorAbort):
        _keep_unheld(config, [_bare_variant(variants, "any_variant_1")])

    assert "converted card tree" in caplog.text

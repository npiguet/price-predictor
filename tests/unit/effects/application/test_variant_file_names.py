"""One file-name rule for a variant script (16.3).

The Python writer (``write_variant``), the Python reader (the variant-tree
``ConvertedCardLocator``) and the Java collector's key
(``CardFilenames.variantScriptFile``) name a variant by ``sanitize(name)``
alone: unlike an ordinary rebalanced card, a variant of ``A-Acererak the
Archlich`` is NOT given the literal ``a-`` stem, because the writer never did.
``CardFilenamesTest.variantFileStemsMatchThePythonTable`` carries this same
table. A variant is never a split ``//`` card: a multi-faced source is not
eligible, so no ``//`` row exists.
"""

from __future__ import annotations

import pytest

from effects.application.collect_variants import write_variant
from effects.domain.script_variants import variant_digest, variant_name
from price_predictor.infrastructure.card_filenames import sanitize_card_name
from sealed.infrastructure.converted_card_locator import ConvertedCardLocator

#: (variant card name, expected file stem): literal, and the same in Java.
VARIANT_FILE_STEMS = [
    ("A-Acererak the Archlich Variant 1a2b3c4d",
     "a_acererak_the_archlich_variant_1a2b3c4d"),
    ("Ajani's Pridemate Variant 0badf00d", "ajanis_pridemate_variant_0badf00d"),
    ("Borborygmos, Enraged Variant 12345678",
     "borborygmos_enraged_variant_12345678"),
    ("Lim-Dûl's Vault Variant deadbeef", "lim_duls_vault_variant_deadbeef"),
    ("Serra Angel Variant 00ff00ff", "serra_angel_variant_00ff00ff"),
]


@pytest.mark.parametrize(("name", "stem"), VARIANT_FILE_STEMS)
def test_the_writer_names_a_variant_by_its_sanitized_name(name, stem):
    assert sanitize_card_name(name) == stem


@pytest.mark.parametrize(("name", "stem"), VARIANT_FILE_STEMS)
def test_the_reader_finds_the_file_the_writer_names(tmp_path, name, stem):
    (tmp_path / f"{stem}.txt").write_text("name:x\n", encoding="utf-8")
    locator = ConvertedCardLocator(tmp_path, tree="variant-scripts")
    assert locator.script_file(name) == f"variant-scripts/{stem}.txt"


def test_a_variant_of_a_rebalanced_card_is_written_and_read_under_one_name(tmp_path):
    lines = ["Name:A-Acererak the Archlich", "ManaCost:2 B", "Types:Legendary Creature"]
    written = write_variant(tmp_path, "A-Acererak the Archlich", lines, "changed")
    digest = variant_digest(lines)
    expected = sanitize_card_name(variant_name("A-Acererak the Archlich", digest))
    assert written.path.name == f"{expected}.txt"
    assert expected.startswith("a_acererak")
    locator = ConvertedCardLocator(tmp_path, tree="variant-scripts")
    assert locator.script_file(written.name) == f"variant-scripts/{expected}.txt"

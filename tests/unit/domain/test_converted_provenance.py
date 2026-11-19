"""The conversion record's rules (finding 11.3, task K03): reading the latest
conversion from a ``run.json`` document, the JAR manifest's commit, and the
refusal of a connector built at another Forge commit."""

from __future__ import annotations

from pathlib import Path

import pytest

from price_predictor.domain.converted_provenance import (
    Conversion,
    ConversionRefused,
    latest_conversion,
    manifest_forge_commit,
    require_same_commit,
)

COMMIT = "0123456789abcdef0123456789abcdef01234567"
MANIFEST = Path("converted/run.json")


def test_a_document_with_no_conversion_names_none():
    assert latest_conversion({"invocations": [{"command": "convert"}]}, MANIFEST) is None
    assert latest_conversion({}, MANIFEST) is None


def test_the_latest_recorded_conversion_is_read():
    document = {"invocations": [
        {"conversion": {"forge_commit": "f" * 40}},
        {"conversion": {"forge_commit": COMMIT, "forge_dirty": True, "sidecar_digest": "d"}},
        {"command": "a refused convert, which changed nothing"},
    ]}

    assert latest_conversion(document, MANIFEST) == Conversion(MANIFEST, COMMIT, True, "d")


def test_a_short_commit_is_no_commit():
    document = {"invocations": [{"conversion": {"forge_commit": COMMIT[:11]}}]}

    with pytest.raises(ConversionRefused, match="names no Forge commit"):
        latest_conversion(document, MANIFEST)


def test_the_manifest_names_the_commit_even_folded():
    text = f"Manifest-Version: 1.0\r\nForge-Commit: {COMMIT[:20]}\r\n {COMMIT[20:]}\r\n"

    assert manifest_forge_commit(text) == COMMIT


@pytest.mark.parametrize("text", [
    "Manifest-Version: 1.0\r\n",
    "Forge-Commit: \r\n",
    "Forge-Commit: ${forge.commit}\r\n",
])
def test_a_manifest_without_a_full_commit_names_none(text):
    assert manifest_forge_commit(text) is None


def test_another_commit_is_refused_naming_both_first():
    conversion = Conversion(MANIFEST, COMMIT, False, "d")

    with pytest.raises(ConversionRefused) as refused:
        require_same_commit(conversion, "f" * 40, Path("connector.jar"))

    message = str(refused.value)
    assert message.index(COMMIT) < 120
    assert message.index("f" * 40) < 200


def test_the_same_commit_passes():
    require_same_commit(Conversion(MANIFEST, COMMIT, False, "d"), COMMIT, Path("c.jar"))

"""Both canonical digests, pinned against a fixed document.

The blake2b digest is pinned by every checkpoint trained on a manifest and the
sha256 one by every scorecard of a frozen set, so neither may move.
"""

from __future__ import annotations

from effects.domain.canonical_digest import canonical_digest, canonical_json, canonical_sha256

_DOC = {"b": [1, 2], "a": {"y": "é", "x": None}}


def test_canonical_json_sorts_keys_and_drops_whitespace():
    assert canonical_json(_DOC) == '{"a":{"x":null,"y":"\\u00e9"},"b":[1,2]}'


def test_blake2b_digest_is_pinned():
    assert canonical_digest(_DOC).hex() == "7537d4df387193cb8c53d8254ee793c7"


def test_sha256_digest_is_pinned():
    assert canonical_sha256(_DOC) == (
        "13c4b603ea683dfff3569190fedf272d9fc456880388655caa27659c3492bfb4")


def test_digests_ignore_how_a_mapping_was_built():
    rebuilt = {"a": {"x": None, "y": "é"}, "b": [1, 2]}
    assert canonical_digest(rebuilt) == canonical_digest(_DOC)
    assert canonical_sha256(rebuilt) == canonical_sha256(_DOC)


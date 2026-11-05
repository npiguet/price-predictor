"""The probe scripts' hash and the selection set's digest go through the package's
shared ones (T007, T008)."""

from __future__ import annotations

from effect_knowledge_probes.common import stable_hash

from effects.domain.canonical_digest import canonical_sha256
from effects.domain.seeded_draw import id_hash
from effects.domain.selection_set import canonical_digest


def test_probe_stable_hash_is_the_shared_id_hash():
    for value, crc in (("run.0-abc.17", 3616614414), ("game-1", 4140153980)):
        assert stable_hash(value) == id_hash(value) == crc


def test_the_selection_set_digest_is_the_sha256_variant_without_its_own_digest():
    doc = {"b": [1, 2], "a": {"y": "é", "x": None}}
    assert canonical_digest({**doc, "digest": "stale"}) == canonical_sha256(doc)

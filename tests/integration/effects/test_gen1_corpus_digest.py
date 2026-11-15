"""The real gen-1 corpus still hashes to the digest its checkpoint recorded.

Every checkpoint pinned to a gen-1 corpus must stay evaluable (FR-042): a
change to ``CorpusManifest``'s digest rule that the trimmed unit fixture does
not exercise shows here, on the whole 8 MB manifest. It needs the real corpus,
so it is an integration test, and a missing corpus fails it rather than
skipping it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from effects.domain.corpus_manifest import CorpusManifest

pytestmark = pytest.mark.integration

#: The gen-1 corpus's run directory under the artifact layout.
_RUN = Path("output/effects/gen1/corpus/2026-09-17-textless")
_NAS = Path("Y:/Nicolas/mtg/mtg-models-data")
#: Where its manifest is: on this machine, on the NAS mirror of the layout,
#: or where the NAS kept it before the layout.
_CANDIDATES = (
    _RUN / "manifest.json",
    _NAS / _RUN / "manifest.json",
    _NAS / "effects" / "training-data" / "gen1" / "corpus" / "manifest.json",
)
#: The digest the gen-1 checkpoint (run 09.17f) recorded for that manifest.
_GEN1_RECORDED_DIGEST = "e3c0c9bb913b9aeda0d07eb587b5445c"


def test_the_gen1_corpus_still_matches_the_digest_its_checkpoint_recorded():
    found = next((path for path in _CANDIDATES if path.is_file()), None)
    if found is None:
        pytest.fail(
            "the gen-1 corpus manifest is in none of "
            + ", ".join(str(path) for path in _CANDIDATES)
            + ": restore the corpus from the NAS",
        )
    raw = json.loads(found.read_text(encoding="utf-8"))
    assert CorpusManifest.from_dict(raw).digest() == _GEN1_RECORDED_DIGEST

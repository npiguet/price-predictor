"""The tokenizer's files as a training run checks them, once, at startup.

A run hashes the vocabulary and the keyword-definition file here, before the
tokenizer is built, and carries the hashes on its ``RunSpec``: every save then
records what was hashed, never a re-read of a file that may since have changed.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

from effects.domain.tokenizer_files import TokenizerFiles
from effects.infrastructure.effect_model_store import tokenizer_hashes

logger = logging.getLogger(__name__)


class KeywordDefinitionsError(ValueError):
    """``--keyword-definitions`` names no file, or one other than the corpus
    was built against."""


def checked_tokenizer_hashes(files: TokenizerFiles, manifest) -> Mapping[str, str]:
    """The two files' hashes, once the keyword file is known to exist and to
    be the one the corpus's manifest digests.

    A manifest recording no digest was built with no keyword table: there is
    nothing to compare, which is logged, and the run records the hash it took.
    """
    if files.keywords is None or not files.keywords.is_file():
        raise KeywordDefinitionsError(
            f"--keyword-definitions {files.keywords}: no file at that path. Training "
            "does not read a missing table as empty. Run extract-keyword-definitions, "
            "or pass the file the corpus was built with."
        )
    hashes = tokenizer_hashes(files)
    digest = manifest.keyword_definitions_digest
    if not digest:
        logger.warning(
            "The corpus manifest records no keyword_definitions_digest (it was "
            "built without a keyword table), so %s is not compared with the "
            "corpus's.", files.keywords,
        )
    elif digest != hashes["keyword_definitions_hash"]:
        raise KeywordDefinitionsError(_mismatch(files, digest, hashes))
    return hashes


def _mismatch(files: TokenizerFiles, digest: str, hashes: Mapping[str, str]) -> str:
    return (
        f"--keyword-definitions {files.keywords} hashes to "
        f"{hashes['keyword_definitions_hash'][:12]}, but the corpus manifest's "
        f"keyword_definitions_digest is {digest[:12]}: the corpus was built "
        "against another keyword file. Pass that one, or rebuild the corpus."
    )

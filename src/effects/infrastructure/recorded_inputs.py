"""The paths a command reads that a flag does not name: those a checkpoint or a
corpus manifest records.

``run.json`` lists every path a run read (``.claude/skills/artifact-layout``,
rule 3), those taken from a checkpoint or a manifest included. These resolvers
answer for the run manifest at a command's start, before the command has
resolved them itself, so each reads what the command is about to read and
returns what it found: a file that cannot be read yields nothing here and
is refused by the command with its own message.
"""

from __future__ import annotations

import functools
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def corpus_variant_tree(corpus: Path | str | None, given: Path | None) -> list[Path]:
    """The variant-script tree a run over ``corpus`` reads: ``given``, else the
    one the corpus manifest records (``CorpusManifest.variant_scripts_or``)."""
    if given is not None:
        return [Path(given)]
    if corpus is None:
        return []
    try:
        from effects.infrastructure.corpus_store import CorpusStore

        tree = CorpusStore(Path(corpus)).load().variant_scripts_or(None)
    except Exception as failure:  # noqa: BLE001 — the command refuses it, in its own words
        logger.debug("no manifest at %s to list its variant tree: %s", corpus, failure)
        return []
    return [] if tree is None else [tree]


def _provenance(checkpoint: Path):
    """The provenance ``checkpoint`` recorded, or None where it cannot be read."""
    loaded = _checkpoint(checkpoint)
    return None if loaded is None else loaded.provenance


def _checkpoint(checkpoint: Path):
    """``checkpoint`` loaded, or None where it cannot be read. The weights are
    read once per manifest, not once per list that asks."""
    try:
        return _checkpoint_at(Path(checkpoint), Path(checkpoint).stat().st_mtime_ns)
    except OSError:
        return None


@functools.lru_cache(maxsize=2)
def _checkpoint_at(checkpoint: Path, modified_ns: int):
    """``checkpoint`` as of its modification time ``modified_ns``."""
    try:
        from effects.infrastructure.effect_model_store import load_effect_checkpoint

        return load_effect_checkpoint(checkpoint)
    except Exception as failure:  # noqa: BLE001 — the command refuses it, in its own words
        logger.debug("checkpoint %s not read to list its inputs: %s", checkpoint, failure)
        return None


def checkpoint_cards_folders(checkpoint: Path, given) -> list[Path]:
    """The converted trees a command run on ``checkpoint`` reads: ``given``
    (``--cards-folder``), else the folders the checkpoint recorded."""
    if given:
        return [Path(folder) for folder in given]
    loaded = _checkpoint(checkpoint)
    return [] if loaded is None else [Path(folder) for folder in loaded.cards_folders]


def tokenizer_paths(
    checkpoint: Path, *, vocab: Path | None = None, keywords: Path | None = None,
) -> list[Path]:
    """The vocabulary and keyword table a command run on ``checkpoint`` reads:
    the flag's, else the one the checkpoint recorded."""
    provenance = _provenance(checkpoint)
    if provenance is None:
        return [path for path in (vocab, keywords) if path is not None]
    files = provenance.tokenizer_files
    return [vocab or files.vocab, keywords or files.keywords]


def corpus_paths(
    checkpoint: Path, *, corpus: Path | None = None, variant_scripts: Path | None = None,
) -> list[Path]:
    """The corpus a command run on ``checkpoint`` reads (the flag's, else the
    recorded one) and the variant tree its manifest names."""
    provenance = _provenance(checkpoint)
    recorded = Path(provenance.corpus_path) if provenance and provenance.corpus_path else None
    chosen = corpus or recorded
    if chosen is None:
        return [] if variant_scripts is None else [variant_scripts]
    return [chosen, *corpus_variant_tree(chosen, variant_scripts)]

"""Where a curated dataset lives on disk (FR-135).

``validation/gate-one/`` and ``validation/samples/`` hold **copies** of
records also in the strata beside them, so a consumer reads one directory at
a time and never ``validation/`` or the corpus root recursively
(``iter_shards`` recurses).
"""

from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Iterator
from pathlib import Path

from effects.domain.corpus_manifest import CorpusManifest, SourceShard
from effects.domain.records import RecordKind, acting_keys
from effects.domain.strata import OUTPUT_STRATA, Stratum
from effects.infrastructure.record_codec import parse_record_line, record_from_dict
from effects.infrastructure.shard_io import (
    iter_shard_lines,
    iter_shards,
    peek_envelope,
    read_records_matching,
)

logger = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.json"
#: Where ``clear_outputs`` moves the manifest of the dataset it is clearing.
PREVIOUS_MANIFEST_NAME = "manifest.previous.json"


class CorpusStore:
    """Reads and writes one curated dataset directory."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)

    @property
    def manifest_path(self) -> Path:
        return self.directory / MANIFEST_NAME

    @property
    def previous_manifest_path(self) -> Path:
        """The cleared dataset's manifest, kept as a record and read by nothing."""
        return self.directory / PREVIOUS_MANIFEST_NAME

    def output_dir(self, stratum: Stratum) -> Path:
        """Where ``stratum``'s shards live: training at the root, the rest under
        ``validation/``."""
        if stratum is Stratum.TRAINING:
            return self.directory / stratum
        return self.directory / "validation" / stratum

    @property
    def training_dir(self) -> Path:
        return self.output_dir(Stratum.TRAINING)

    @property
    def card_disjoint_dir(self) -> Path:
        return self.output_dir(Stratum.CARD_DISJOINT)

    @property
    def game_disjoint_dir(self) -> Path:
        return self.output_dir(Stratum.GAME_DISJOINT)

    @property
    def gate_one_dir(self) -> Path:
        """Resolution records of card-disjoint games whose acting text is held
        out: copies of records still in ``card-disjoint/``."""
        return self.output_dir(Stratum.GATE_ONE)

    @property
    def samples_dir(self) -> Path:
        return self.directory / "validation" / "samples"

    def sample_path(self, stratum: str) -> Path:
        """The fixed validation sample the trainer reads for ``stratum``."""
        return self.samples_dir / f"{stratum}.jsonl.gz"

    @property
    def parts_dir(self) -> Path:
        """Per-source parts the write pass leaves for the repack step, outside
        every stratum so a recursive read never meets one."""
        return self.directory / ".parts"

    def parts_dir_for(self, stratum: str) -> Path:
        return self.parts_dir / stratum

    def clear_outputs(self) -> None:
        """Delete every output directory, moving the manifest aside first.

        A dataset is rebuilt whole (FR-144), so no leftover shard joins a
        dataset its manifest does not describe. The manifest goes first, so a
        build that then fails leaves no manifest whose digest still matches
        checkpoints trained on the old dataset.
        """
        if self.manifest_path.exists():
            self.manifest_path.replace(self.previous_manifest_path)
        for directory in (
            *(self.output_dir(stratum) for stratum in OUTPUT_STRATA),
            self.samples_dir, self.parts_dir,
        ):
            shutil.rmtree(directory, ignore_errors=True)

    def save(self, manifest: CorpusManifest) -> Path:
        """Write the manifest atomically, and drop the one ``clear_outputs`` set aside."""
        self.directory.mkdir(parents=True, exist_ok=True)
        staging = self.manifest_path.with_name(f"{MANIFEST_NAME}.tmp")
        staging.write_text(
            json.dumps(manifest.as_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        staging.replace(self.manifest_path)
        self.previous_manifest_path.unlink(missing_ok=True)
        return self.manifest_path

    def load(self) -> CorpusManifest:
        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"{self.manifest_path}: not a curated corpus (no manifest.json). "
                "Build one with `python -m effects build-corpus`."
            )
        return CorpusManifest.from_dict(
            json.loads(self.manifest_path.read_text(encoding="utf-8"))
        )


def current_sources(records_dir: Path) -> tuple[SourceShard, ...]:
    """Every raw shard under ``records_dir``, by relative path and size."""
    root = Path(records_dir)
    return tuple(
        SourceShard(name=shard.relative_to(root).as_posix(), size=shard.stat().st_size)
        for shard in iter_shards(root)
    )


def training_acting_keys(directory: Path) -> set[tuple]:
    """Every distinct ``acting_keys`` tuple among the records under ``directory``.

    ``directory`` is the curated corpus's ``training/``: what the checkpoint
    actually trained on, and so what the identity baseline could have
    memorized. A record curation wrote several times (``--reuse-cap``) is
    parsed once; the set is all gate 1 needs, and the records are never held.
    """
    shards = iter_shards(directory)
    read: set[str] = set()
    acting: set[tuple] = set()
    for done, shard in enumerate(shards, start=1):
        for line in iter_shard_lines(shard):
            head = peek_envelope(line)
            if head is not None:
                if head.record_id in read:
                    continue
                read.add(head.record_id)
            keys = acting_keys(record_from_dict(parse_record_line(line)))
            if keys:
                acting.add(keys)
        if done % 50 == 0:
            logger.info(
                "gate 1: %d of %d training shards read, %d acting lines",
                done, len(shards), len(acting),
            )
    return acting


def card_disjoint_resolutions(corpus_path: Path, games: frozenset[str]) -> Iterator:
    """The resolution records of the curated corpus's card-disjoint ``games``, streamed.

    Resolutions only, the one kind gate 1's stratum holds; the stratum's other
    kinds are never parsed.
    """
    return read_records_matching(
        CorpusStore(corpus_path).card_disjoint_dir,
        game_ids=games, kinds={RecordKind.RESOLUTION},
    )

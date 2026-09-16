"""Where a curated dataset lives on disk (FR-135).

The directory layout is the contract: ``training/``,
``validation/card-disjoint/`` and ``validation/game-disjoint/`` hold ordinary
shards, so every existing reader loads them unchanged, and ``manifest.json``
holds the decisions that produced them.
"""

from __future__ import annotations

import json
from pathlib import Path

from effects.domain.corpus_manifest import CorpusManifest, SourceShard

MANIFEST_NAME = "manifest.json"


class CorpusStore:
    """Reads and writes one curated dataset directory."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)

    @property
    def manifest_path(self) -> Path:
        return self.directory / MANIFEST_NAME

    @property
    def training_dir(self) -> Path:
        return self.directory / "training"

    @property
    def card_disjoint_dir(self) -> Path:
        return self.directory / "validation" / "card-disjoint"

    @property
    def game_disjoint_dir(self) -> Path:
        return self.directory / "validation" / "game-disjoint"

    @property
    def gate_one_dir(self) -> Path:
        """Resolution records of card-disjoint games whose acting text is held out."""
        return self.directory / "validation" / "gate-one"

    @property
    def samples_dir(self) -> Path:
        return self.directory / "validation" / "samples"

    def sample_path(self, stratum: str) -> Path:
        """The fixed validation sample the trainer reads for ``stratum``."""
        return self.samples_dir / f"{stratum}.jsonl.gz"

    @property
    def parts_dir(self) -> Path:
        """Per-source parts the write pass leaves for the repack step.

        Outside every stratum directory, because ``iter_shards`` is recursive
        and a part left under ``training/`` would be read as a shard.
        """
        return self.directory / ".parts"

    def parts_dir_for(self, stratum: str) -> Path:
        return self.parts_dir / stratum

    def clear_outputs(self) -> None:
        """Delete the three shard directories, leaving the manifest in place.

        A dataset is rebuilt whole rather than extended (FR-144): the split and
        the rarity table are corpus-wide, so a shard left over from an earlier
        build belongs to a different split. Same-named shards are
        truncate-overwritten anyway, but a source shard that has been renamed
        or removed — or a rebuild pointed at another ``--records-dir`` — leaves
        files behind that every reader loads as part of the dataset while the
        manifest and its digest describe only the newer build.

        The manifest itself is left alone so a build that fails before writing
        one does not also destroy the record of what used to be here.
        """
        import shutil

        for directory in (
            self.training_dir, self.card_disjoint_dir, self.game_disjoint_dir,
            self.gate_one_dir, self.samples_dir, self.parts_dir,
        ):
            shutil.rmtree(directory, ignore_errors=True)

    def save(self, manifest: CorpusManifest) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(
            json.dumps(manifest.as_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
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
    from effects.infrastructure.record_io import iter_shards

    root = Path(records_dir)
    return tuple(
        SourceShard(name=shard.relative_to(root).as_posix(), size=shard.stat().st_size)
        for shard in iter_shards(root)
    )

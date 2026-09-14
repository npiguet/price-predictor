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

    def __init__(self, directory) -> None:
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


def current_sources(records_dir) -> tuple[SourceShard, ...]:
    """Every raw shard under ``records_dir``, by relative path and size."""
    from effects.infrastructure.record_io import iter_shards

    root = Path(records_dir)
    return tuple(
        SourceShard(name=shard.relative_to(root).as_posix(), size=shard.stat().st_size)
        for shard in iter_shards(root)
    )

"""``encode-abilities``: compute the ability cache once, offline."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from effects.domain import default_paths
from effects.domain.ability_cache_layout import DEFAULT_CACHE_ROOT
from effects.domain.baselines import Baseline, taxonomy_matrix
from effects.domain.provenance import ProvenanceSidecar
from effects.infrastructure.ability_cache_store import AbilityCacheStore
from effects.infrastructure.effect_model_store import (
    InferenceCheckpoint,
    load_for_inference,
    model_output_for,
)
from effects.infrastructure.sidecar_io import iter_sidecars, rendered_prose

logger = logging.getLogger(__name__)



@dataclass
class EncodeAbilitiesConfig:
    variant: str = Baseline.FULL
    checkpoint: Path | None = None
    cards_folders: tuple[Path, ...] = default_paths.DEFAULT_CARDS_FOLDER_PATHS
    variant_scripts: Path | None = None
    vocab_path: Path | None = None
    keyword_definitions: Path | None = None
    output_root: Path = field(default_factory=lambda: DEFAULT_CACHE_ROOT)
    clean: bool = False

    def resolved_checkpoint(self) -> Path:
        """The checkpoint this variant reads, so a baseline is never encoded
        with another variant's weights; an explicit value overrides."""
        if self.checkpoint is not None:
            return Path(self.checkpoint)
        return model_output_for(self.variant) / "latest.pt"


@dataclass(frozen=True, slots=True)
class EncodeSummary:
    sources: int = 0
    rows: int = 0


def run(config: EncodeAbilitiesConfig) -> EncodeSummary:
    """Encode every source under the configured trees; idempotent."""
    store = AbilityCacheStore(config.output_root, variant=config.variant)
    if config.clean:
        store.clean()

    # Hash-checked before anything is written.
    loaded = load_for_inference(
        config.resolved_checkpoint(),
        vocab_path=config.vocab_path, keyword_path=config.keyword_definitions,
    )
    e_dim = loaded.checkpoint.encoder_config.e_dim
    provenance = loaded.checkpoint.provenance

    folders = [Path(f) for f in config.cards_folders]
    if config.variant_scripts:
        folders.append(Path(config.variant_scripts))
    # A relative default resolved from another directory names nothing, and
    # skipping it would write a cache missing whole trees and report success.
    missing = [str(f) for f in folders if not f.is_dir()]
    if missing:
        raise FileNotFoundError(f"no source tree at {', '.join(missing)}")

    sources, skipped = read_sources(folders)
    if config.variant == Baseline.TAXONOMY:
        matrices = (taxonomy_matrix(sidecar, e_dim) for _path, sidecar in sources)
    else:
        matrices = encode_tree(build_runner(loaded), sources)
    rows = 0
    for (_path, sidecar), matrix in zip(sources, matrices, strict=True):
        store.write(sidecar.script_file, matrix, sidecar)
        rows += matrix.shape[0]

    logger.info(
        "Encoded %d sources (%d ability rows; %d sidecars with no line skipped) "
        "into %s [variant=%s]%s",
        len(sources), rows, skipped, config.output_root, config.variant,
        f", withheld keyword {provenance.withheld_keyword}"
        if provenance.withheld_keyword else "",
    )
    return EncodeSummary(sources=len(sources), rows=rows)


def read_sources(
    folders: list[Path],
) -> tuple[list[tuple[Path, ProvenanceSidecar]], int]:
    """Every sidecar with a line to encode, beside its path; and how many had none."""
    sources, skipped = [], 0
    for folder in folders:
        for sidecar_path, sidecar in iter_sidecars(folder):
            if sidecar.lines:
                sources.append((sidecar_path, sidecar))
            else:
                skipped += 1
    return sources, skipped


def encode_tree(
    runner, sources: list[tuple[Path, ProvenanceSidecar]],
) -> list[np.ndarray]:
    """Each source's ``(n_lines, e_dim)`` matrix, from one encode of the
    tree's distinct inputs. A truncated text is reported under the first line
    that printed it."""
    index: dict[tuple[str, tuple], int] = {}
    keys: dict[str, tuple] = {}
    rows_of: list[list[int]] = []
    for sidecar_path, sidecar in sources:
        rendered = rendered_prose(sidecar_path, sidecar)
        rows = []
        for line in sidecar.lines:
            unit = runner.unit_of(line, line.prose_in(rendered))
            rows.append(index.setdefault(unit, len(index)))
            keys.setdefault(unit[0], line.provenance)
        rows_of.append(rows)
    logger.info(
        "Encoding %d distinct ability inputs for %d sources",
        len(index), len(sources),
    )
    encoded = runner.encode(list(index), keys_of=keys.__getitem__)
    return [encoded[rows] for rows in rows_of]


def build_runner(loaded: InferenceCheckpoint):
    """The checkpoint's encoder, built once for the whole run."""
    from effects.application.ability_encoder_runner import (
        AbilityEncoderRunner,
    )
    from effects.domain.ability_encoder import TruncationLog, log_truncation
    from effects.infrastructure.tokenizer_loading import tokenizer_for

    checkpoint, vocab_path, keyword_path = loaded
    return AbilityEncoderRunner.from_checkpoint(
        checkpoint, tokenizer_for(checkpoint, vocab_path, keyword_path),
        vocab_path=vocab_path,
        truncations=TruncationLog(log_truncation(logger, "encode-abilities")),
    )

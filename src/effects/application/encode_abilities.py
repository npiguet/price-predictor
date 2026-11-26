"""``encode-abilities``: compute the ability cache once, offline."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from effects.domain import default_paths
from effects.domain.baselines import Baseline, taxonomy_matrix
from effects.domain.provenance import ProvenanceSidecar
from effects.domain.tokenizer_files import TokenizerFiles
from effects.infrastructure.ability_cache_store import (
    AbilityCacheStore,
    cache_stamp,
    remove_stamp,
    write_stamp,
)
from effects.infrastructure.converted_trees import SourceTrees
from effects.infrastructure.effect_model_store import (
    EffectCheckpoint,
    inference_files,
    load_effect_checkpoint,
)
from effects.infrastructure.sidecar_io import iter_sidecars, rendered_prose

logger = logging.getLogger(__name__)

#: The variants whose model reads no ``e`` the encoder writes: ``identity``'s
#: comes from the free table its checkpoint carries, ``state-only`` zeroes it.
UNENCODED_VARIANTS = frozenset({Baseline.IDENTITY, Baseline.STATE_ONLY})


class UnencodedVariantError(ValueError):
    """``encode-abilities --variant`` naming a variant whose model reads no cache."""


@dataclass
class EncodeAbilitiesConfig:
    variant: Baseline = Baseline.FULL
    #: The checkpoint to encode with, which the command names: no default.
    checkpoint: Path = field(kw_only=True, metadata={"help": (
        "The effect-model checkpoint (.pt) to encode with; the baseline named "
        "by --variant needs its own checkpoint"
    )})
    cards_folders: tuple[Path, ...] = default_paths.DEFAULT_CARDS_FOLDER_PATHS
    variant_scripts: Path | None = None
    vocab_path: Path | None = None
    keyword_definitions: Path | None = None
    #: The cache run's directory, which the command names: no default.
    output_root: Path = field(kw_only=True, metadata={"help": (
        "Cache run directory the ability cache is written under "
        "(output/effects/<gen>/cache/<run>/)"
    )})
    clean: bool = False

    def __post_init__(self) -> None:
        if self.variant in UNENCODED_VARIANTS:
            raise UnencodedVariantError(
                f"encode-abilities --variant {self.variant} would write the encoder's "
                f"vectors, which the {self.variant} model never reads (identity reads "
                "the free table its checkpoint carries, state-only zeroes every e): "
                f"evaluate it with --variant-checkpoint {self.variant}=PATH, no cache"
            )

    @property
    def source_trees(self) -> SourceTrees:
        """``--cards-folder`` and ``--variant-scripts`` as one value."""
        return SourceTrees.of(self.cards_folders, self.variant_scripts)


@dataclass(frozen=True, slots=True)
class EncodeSummary:
    sources: int = 0
    rows: int = 0


def run(config: EncodeAbilitiesConfig) -> EncodeSummary:
    """Encode every source under the configured trees; idempotent. The stamp
    goes before the first file is written and is written after the last."""
    store = AbilityCacheStore(config.output_root, variant=config.variant)
    if config.clean:
        store.clean()

    # Hash-checked before anything is written.
    checkpoint = load_effect_checkpoint(config.checkpoint)
    files = inference_files(
        checkpoint, vocab=config.vocab_path, keywords=config.keyword_definitions,
    )
    folders = config.source_trees.folders()
    # A relative default resolved from another directory names nothing, and
    # skipping it would write a cache missing whole trees and report success.
    missing = [str(f) for f in folders if not f.is_dir()]
    if missing:
        raise FileNotFoundError(f"no source tree at {', '.join(missing)}")

    sources, skipped = read_sources(folders)
    remove_stamp(config.output_root, config.variant)
    rows = 0
    for (_path, sidecar), matrix in zip(
        sources, _matrices(config, checkpoint, files, sources), strict=True,
    ):
        store.write(sidecar.script_file, matrix, sidecar)
        rows += matrix.shape[0]
    write_stamp(config.output_root, _stamp(config, checkpoint, files), variant=config.variant)

    logger.info(
        "Encoded %d sources (%d ability rows; %d sidecars with no line skipped) "
        "into %s [variant=%s]%s",
        len(sources), rows, skipped, config.output_root, config.variant,
        f", withheld keyword {checkpoint.provenance.withheld_keyword}"
        if checkpoint.provenance.withheld_keyword else "",
    )
    return EncodeSummary(sources=len(sources), rows=rows)


def _matrices(config: EncodeAbilitiesConfig, checkpoint: EffectCheckpoint,
              files: TokenizerFiles, sources):
    """Each source's rows: the taxonomy's hash embedding, else the encoder's."""
    if config.variant == Baseline.TAXONOMY:
        e_dim = checkpoint.encoder_config.e_dim
        return (taxonomy_matrix(sidecar, e_dim) for _path, sidecar in sources)
    return encode_tree(build_runner(checkpoint, files), sources)


def _stamp(config: EncodeAbilitiesConfig, checkpoint: EffectCheckpoint, files: TokenizerFiles):
    """The stamp of this cache: the converted trees alone, the variant tree
    being no tree a gate or probe reads the cache over."""
    from effects.infrastructure.vocabulary_file import vocabulary_surface

    return cache_stamp(
        config.checkpoint, checkpoint, cards_folders=config.cards_folders,
        surface=vocabulary_surface(files.vocab),
    )


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


def build_runner(checkpoint: EffectCheckpoint, files: TokenizerFiles):
    """The checkpoint's encoder, built once for the whole run."""
    from effects.application.ability_encoder_runner import (
        AbilityEncoderRunner,
        RunnerSettings,
    )
    from effects.domain.ability_encoder import TruncationLog, log_truncation
    from effects.infrastructure.tokenizer_loading import tokenizer_for

    tokenizer = tokenizer_for(checkpoint, files)
    return AbilityEncoderRunner.from_checkpoint(
        checkpoint, tokenizer,
        RunnerSettings(
            surface=tokenizer.surface,
            truncations=TruncationLog(log_truncation(logger, "encode-abilities")),
        ),
    )

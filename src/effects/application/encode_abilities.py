"""``encode-abilities``: compute the ability cache once, offline.

Every downstream consumer reads the cache rather than the encoder. That is the
point of the bottleneck: an ability's vector is computed once for the whole
corpus and looked up thereafter, so a deck builder or a scorer pays a dictionary
hit instead of a transformer forward pass.

``--variant`` resolves the checkpoint *and* the output suffix together, so a
baseline is never encoded with another variant's weights and never lands on top
of the shipping cache. The ``taxonomy`` variant has no encoder at all and emits
its lookup into the same row layout, so every ``e``-geometry check reads one
file shape whichever variant it is looking at.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from effects.domain.ability_cache_layout import DEFAULT_CACHE_ROOT
from effects.domain.baselines import Baseline
from effects.domain.provenance import ProvenanceSidecar, SidecarLine
from effects.infrastructure.ability_cache_store import AbilityCacheStore
from effects.infrastructure.effect_model_store import (
    InferenceCheckpoint,
    load_for_inference,
    model_output_for,
)
from effects.infrastructure.sidecar_io import SIDECAR_SUFFIX, read_sidecar

logger = logging.getLogger(__name__)

DEFAULT_CARDS_FOLDERS: tuple[Path, ...] = (
    Path("output/cardsfolder/"), Path("output/tokenscripts/"),
)


@dataclass
class EncodeAbilitiesConfig:
    variant: str = Baseline.FULL
    checkpoint: Path | None = None
    cards_folders: tuple[Path, ...] = DEFAULT_CARDS_FOLDERS
    variant_scripts: Path | None = None
    vocab_path: Path | None = None
    keyword_definitions: Path | None = None
    output_root: Path = field(default_factory=lambda: DEFAULT_CACHE_ROOT)
    clean: bool = False

    def resolved_checkpoint(self) -> Path:
        """The checkpoint this variant reads; an explicit value overrides.

        Resolving it from the variant is what stops a baseline being encoded
        with the shipping model's weights — a mistake that produces a cache
        which loads, has the right shape, and answers a different question.
        """
        if self.checkpoint is not None:
            return Path(self.checkpoint)
        return model_output_for(self.variant) / "latest.pt"


@dataclass(frozen=True, slots=True)
class EncodeSummary:
    sources: int = 0
    rows: int = 0
    skipped: int = 0
    cleaned: int = 0


def iter_sidecars(cards_folder: Path) -> list[Path]:
    return sorted(Path(cards_folder).rglob(f"*{SIDECAR_SUFFIX}"))


def taxonomy_vector(line: SidecarLine, e_dim: int) -> np.ndarray:
    """The ``taxonomy`` baseline's stand-in for an encoded ``e``.

    A deterministic hash embedding of the sidecar's API type and parameter-key
    set: everything the script says about what the line *does*, and nothing
    about how it says it. If the full model cannot beat this, the encoder is
    reading no more than a taxonomy would give for free.
    """
    vector = np.zeros(e_dim, dtype=np.float32)
    features = [f"api:{line.script_api_type or ''}"]
    features += [f"param:{key}" for key in line.script_param_keys]
    for feature in features:
        digest = hashlib.sha256(feature.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % e_dim
        sign = 1.0 if digest[4] % 2 else -1.0
        vector[index] += sign
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else vector


def taxonomy_matrix(sidecar: ProvenanceSidecar, e_dim: int) -> np.ndarray:
    """The ``taxonomy`` variant's ``(n_lines, e_dim)`` matrix, row-aligned to
    its sidecar, with no encoder at all."""
    if not sidecar.lines:
        return np.zeros((0, e_dim), dtype=np.float32)
    return np.stack([taxonomy_vector(line, e_dim) for line in sidecar.lines])


def run(config: EncodeAbilitiesConfig) -> EncodeSummary:
    """Encode every source under the configured trees.

    Idempotent: re-running rewrites the same rows from the same checkpoint. The
    hash check runs before anything is written, so a vocabulary that moved since
    training stops the run rather than filling the cache with vectors that mean
    nothing.
    """
    store = AbilityCacheStore(config.output_root, variant=config.variant)
    cleaned = store.clean() if config.clean else 0

    # Loaded once and hash-checked before anything is written.
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
        "Encoded %d sources (%d ability rows) into %s [variant=%s]%s",
        len(sources), rows, config.output_root, config.variant,
        f", withheld keyword {provenance.withheld_keyword}"
        if provenance.withheld_keyword else "",
    )
    return EncodeSummary(
        sources=len(sources), rows=rows, skipped=skipped, cleaned=cleaned,
    )


def read_sources(
    folders: list[Path],
) -> tuple[list[tuple[Path, ProvenanceSidecar]], int]:
    """Every sidecar with a line to encode, beside its path; and how many had none."""
    sources, skipped = [], 0
    for folder in folders:
        for sidecar_path in iter_sidecars(folder):
            sidecar = read_sidecar(sidecar_path)
            if sidecar.lines:
                sources.append((sidecar_path, sidecar))
            else:
                skipped += 1
    return sources, skipped


def encode_tree(
    runner, sources: list[tuple[Path, ProvenanceSidecar]],
) -> list[np.ndarray]:
    """Each source's ``(n_lines, e_dim)`` matrix, from one encode of the tree.

    A line is keyed by exactly what the encoder reads, its text and role spans,
    so every card printing *Flying* shares one encoding, and the forward passes
    run over the whole tree's distinct inputs in length-sorted groups rather
    than a card at a time. A truncated text is reported under the first line
    that printed it.
    """
    index: dict[tuple[str, tuple], int] = {}
    keys: dict[str, tuple] = {}
    rows_of: list[list[int]] = []
    for sidecar_path, sidecar in sources:
        prose = prose_of(sidecar, sidecar_path)
        rows = []
        for line in sidecar.lines:
            unit = runner.unit_of(line, prose)
            rows.append(index.setdefault(unit, len(index)))
            keys.setdefault(unit[0], line.provenance)
        rows_of.append(rows)
    logger.info(
        "Encoding %d distinct ability inputs for %d sources",
        len(index), len(sources),
    )
    encoded = runner.encode(list(index), keys_of=keys.__getitem__)
    return [encoded[rows] for rows in rows_of]


def prose_of(sidecar: ProvenanceSidecar, sidecar_path: Path) -> list[str]:
    """The converted prose lines beside a sidecar, as the trainer's join reads them.

    A variant has none: the ``.txt`` beside its sidecar is the Forge source
    script, and read as prose it would encode a script line as card text where
    the trainer (``SidecarCache.prose_for``) encodes the variant's script.
    """
    from effects.domain.provenance import SourceTree
    from effects.infrastructure.sidecar_io import converted_text_path, prose_lines

    if sidecar.tree == SourceTree.VARIANT_SCRIPTS:
        return []
    return prose_lines(converted_text_path(sidecar_path))


def build_runner(loaded: InferenceCheckpoint):
    """The checkpoint's encoder, built once for the whole run.

    Moving the weights to the device is the expensive part, and doing it per
    card would dominate the encode.
    """
    from effects.domain.ability_encoder import TruncationLog, log_truncation
    from effects.infrastructure.ability_encoder_runner import (
        AbilityEncoderRunner,
    )
    from effects.infrastructure.model_runner import tokenizer_for

    checkpoint, vocab_path, keyword_path = loaded
    return AbilityEncoderRunner.from_checkpoint(
        checkpoint, tokenizer_for(checkpoint, vocab_path, keyword_path),
        vocab_path=vocab_path,
        truncations=TruncationLog(log_truncation(logger, "encode-abilities")),
    )

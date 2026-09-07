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
from effects.domain.provenance import ProvenanceSidecar, SidecarLine
from effects.infrastructure.ability_cache_store import AbilityCacheStore
from effects.infrastructure.effect_model_store import (
    VARIANT_FULL,
    EffectModelStore,
    model_output_for,
    resolve_inference_paths,
)
from effects.infrastructure.sidecar_io import SIDECAR_SUFFIX, read_sidecar

logger = logging.getLogger(__name__)

DEFAULT_CARDS_FOLDERS: tuple[Path, ...] = (
    Path("output/cardsfolder/"), Path("output/tokenscripts/"),
)
VARIANT_TAXONOMY = "taxonomy"


@dataclass
class EncodeAbilitiesConfig:
    variant: str = VARIANT_FULL
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


def tree_of(cards_folder: Path) -> str:
    """The source-tree name a converted folder holds.

    Read from the folder's own name, which mirrors the tree it was converted
    from — the same string a provenance key's ``script_file`` is prefixed with.
    """
    name = Path(cards_folder).name
    return name or "cardsfolder"


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


def encode_sidecar(
    sidecar: ProvenanceSidecar,
    *,
    variant: str,
    e_dim: int,
    encode_line=None,
) -> np.ndarray:
    """One source's ``(n_lines, e_dim)`` matrix, row-aligned to its sidecar.

    ``encode_line`` maps a :class:`SidecarLine` to its vector; the ``taxonomy``
    variant supplies its own and needs no model.
    """
    if not sidecar.lines:
        return np.zeros((0, e_dim), dtype=np.float32)
    if variant == VARIANT_TAXONOMY:
        rows = [taxonomy_vector(line, e_dim) for line in sidecar.lines]
    elif encode_line is None:
        raise ValueError(
            f"--variant {variant} needs an encoder; only {VARIANT_TAXONOMY} "
            "can be emitted without one"
        )
    else:
        rows = [encode_line(line) for line in sidecar.lines]
    return np.stack(rows).astype(np.float32)


def run(config: EncodeAbilitiesConfig, *, encode_line=None) -> EncodeSummary:
    """Encode every source under the configured trees.

    Idempotent: re-running rewrites the same rows from the same checkpoint. The
    hash check runs before anything is written, so a vocabulary that moved since
    training stops the run rather than filling the cache with vectors that mean
    nothing.
    """
    store = AbilityCacheStore(config.output_root, variant=config.variant)
    cleaned = store.clean() if config.clean else 0

    e_dim, provenance = _load_checkpoint_facts(config)

    folders = [Path(f) for f in config.cards_folders if Path(f).is_dir()]
    if config.variant_scripts and Path(config.variant_scripts).is_dir():
        folders.append(Path(config.variant_scripts))

    sources = rows = skipped = 0
    for folder in folders:
        for sidecar_path in iter_sidecars(folder):
            sidecar = read_sidecar(sidecar_path)
            if not sidecar.lines:
                skipped += 1
                continue
            matrix = encode_sidecar(
                sidecar, variant=config.variant, e_dim=e_dim,
                encode_line=encode_line,
            )
            store.write(sidecar.script_file, matrix, sidecar)
            sources += 1
            rows += matrix.shape[0]

    logger.info(
        "Encoded %d sources (%d ability rows) into %s [variant=%s]%s",
        sources, rows, config.output_root, config.variant,
        f", withheld keyword {provenance.withheld_keyword}"
        if provenance.withheld_keyword else "",
    )
    return EncodeSummary(sources=sources, rows=rows, skipped=skipped, cleaned=cleaned)


def _load_checkpoint_facts(config: EncodeAbilitiesConfig):
    """The bottleneck width and the provenance, hash-checked before any write."""
    path = config.resolved_checkpoint()
    checkpoint = EffectModelStore(path.parent).load(path)
    vocab_path, keyword_path = resolve_inference_paths(
        checkpoint.provenance,
        vocab_path=config.vocab_path,
        keyword_path=config.keyword_definitions,
    )
    checkpoint.provenance.verify_hashes(
        vocab_path=vocab_path, keyword_path=keyword_path,
    )
    return checkpoint.encoder_config.e_dim, checkpoint.provenance

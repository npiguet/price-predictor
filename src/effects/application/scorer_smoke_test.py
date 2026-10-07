"""``scorer-smoke-test``: does pooled ``e`` add anything the scorer can use?

The effect model's cache is meant to feed the sealed scorer one day. Before any
consumer is rebuilt around it, this asks the cheapest version of the question
end to end: train the scorer's Phase A on card vectors that carry pooled ``e``
beside the sealed encoder's own, and read its validation accuracy against a
scorer trained without it (FR-071). The answer is informational and gates
nothing.

Everything happens under ``--scratch-dir``, in four steps:

1. the converted ``.txt`` files are copied into ``<scratch>/cardsfolder/``;
2. ``python -m sealed encode-cards`` writes the sealed vectors of
   ``--sealed-encoder-checkpoint`` beside them;
3. each card's pooled ``e`` is spliced into its sealed vector, the effect
   checkpoint's width of zeros for a card with no ability line;
4. ``python -m sealed train-scorer`` runs Phase A on that tree, its checkpoint
   directory under the scratch directory too.

Nothing is written under ``output/cardsfolder/``: the sealed pipeline reads
that tree, and a smoke test must not change what the real scorer trains on.
The sealed steps run as subprocesses because ``effects`` imports nothing from
``sealed.application`` (FR-090) — the commands are the interface, as they are
for an operator.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

#: The converted tree the scratch copy is taken from. Read, never written.
DEFAULT_CARDS_PATH = Path("output/cardsfolder")

#: The vocabulary file a sealed encoder checkpoint is paired with: the sealed
#: pipeline writes it beside the checkpoint, which is where ``encode-cards``'s
#: own default (``models/sealed/encoder/vocab.txt``) points for the default
#: checkpoint.
SEALED_VOCAB_NAME = "vocab.txt"


@dataclass(frozen=True)
class ScorerSmokeTestConfig:
    scratch_dir: Path
    checkpoint: Path = field(
        default_factory=lambda: Path("models/effects/effect-model/latest.pt"),
    )
    sealed_encoder_checkpoint: Path = field(
        default_factory=lambda: Path("models/sealed/encoder/latest.pt"),
    )
    abilities_root: Path = field(
        default_factory=lambda: Path("output/effects/abilities"),
    )
    cards_path: Path = DEFAULT_CARDS_PATH

    @property
    def scratch_cards(self) -> Path:
        return Path(self.scratch_dir) / "cardsfolder"

    @property
    def scorer_dir(self) -> Path:
        return Path(self.scratch_dir) / "scorer"


@dataclass(frozen=True)
class ScorerSmokeTestResult:
    cards_written: int
    cards_without_e: int
    scorer_exit_code: int


def encode_cards_command(config: ScorerSmokeTestConfig) -> list[str]:
    """The ``encode-cards`` call that writes the sealed vectors into scratch."""
    checkpoint = Path(config.sealed_encoder_checkpoint)
    return [
        sys.executable, "-m", "sealed", "encode-cards",
        "--encoder-checkpoint", str(checkpoint),
        "--vocab-path", str(checkpoint.parent / SEALED_VOCAB_NAME),
        "--cards-path", str(config.scratch_cards),
    ]


def train_scorer_command(config: ScorerSmokeTestConfig) -> list[str]:
    """The Phase A ``train-scorer`` call on the scratch tree.

    ``--embedding-lr 0`` is Phase A: the encoder stays frozen and the scorer
    reads the ``.npz`` cache, which is the only phase that reads ``e`` at all.
    """
    return [
        sys.executable, "-m", "sealed", "train-scorer",
        "--cards-path", str(config.scratch_cards),
        "--checkpoint-dir", str(config.scorer_dir),
        "--embedding-lr", "0",
    ]


def copy_card_texts(source: Path, target: Path) -> int:
    """Copy every converted ``.txt`` from ``source`` into ``target``, by layout.

    Only the texts: the sealed vectors are encoded afresh from
    ``--sealed-encoder-checkpoint`` rather than taken from whatever encoder last
    wrote ``source``'s ``.npz`` files.
    """
    copied = 0
    for text in Path(source).rglob("*.txt"):
        destination = Path(target) / text.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(text, destination)
        copied += 1
    return copied


def pooled_e_by_path(abilities_root: Path, *, variant: str = "full") -> dict[str, np.ndarray]:
    """Pooled ``e`` per converted card, keyed by its path under ``cardsfolder``.

    Keyed by path (``a/ajanis_pridemate``) rather than by card name, because
    the sealed vector it is spliced into sits at the same path in the scratch
    tree, so no name resolution can pair the wrong two. Only the card tree:
    a token is not a card the scorer scores.
    """
    from effects.domain.ability_cache_layout import (
        ARRAY_KEY,
        CACHE_SUFFIX,
        pooled_card_vector,
    )

    suffix = CACHE_SUFFIX if variant == "full" else f".{variant}{CACHE_SUFFIX}"
    root = Path(abilities_root) / "cardsfolder"
    pooled: dict[str, np.ndarray] = {}
    if not root.is_dir():
        return pooled
    for path in root.rglob(f"*{suffix}"):
        stem = path.name[: -len(suffix)]
        # Another variant's file (`<name>.taxonomy.npz`) matches the full
        # variant's glob too; card stems never carry a dot.
        if "." in stem:
            continue
        with np.load(path) as data:
            matrix = data[ARRAY_KEY]
        if matrix.size:
            relative = path.parent.relative_to(root) / stem
            pooled[relative.as_posix()] = pooled_card_vector(matrix)
    return pooled


def run(config: ScorerSmokeTestConfig, *, runner=subprocess.run) -> ScorerSmokeTestResult:
    """Build the scratch tree, splice ``e`` in, and run Phase A on it."""
    from effects.application.geometry_checks import write_scorer_smoke_cache
    from effects.infrastructure.effect_model_store import EffectModelStore

    checkpoint_path = Path(config.checkpoint)
    checkpoint = EffectModelStore(checkpoint_path.parent).load(checkpoint_path)
    pooled = pooled_e_by_path(config.abilities_root, variant=checkpoint.variant)
    if not pooled:
        raise FileNotFoundError(
            f"no ability cache for {checkpoint_path} under {config.abilities_root}; "
            "run encode-abilities first"
        )
    width = 2 * checkpoint.encoder_config.e_dim
    widths = {vector.shape[0] for vector in pooled.values()}
    if widths != {width}:
        raise ValueError(
            f"the cache under {config.abilities_root} pools to widths "
            f"{sorted(widths)}, not the {width} {checkpoint_path}'s e_dim gives; "
            "it was encoded by another checkpoint"
        )

    copied = copy_card_texts(config.cards_path, config.scratch_cards)
    logger.info("copied %d converted cards into %s", copied, config.scratch_cards)
    encode = runner(encode_cards_command(config), check=False)
    if encode.returncode != 0:
        raise RuntimeError(f"encode-cards exited {encode.returncode}")

    written, missing = write_scorer_smoke_cache(pooled, config.scratch_cards, e_width=width)
    logger.info(
        "spliced pooled e into %d sealed vectors (%d cards have no ability line "
        "and carry zeros)", written, missing,
    )
    trained = runner(train_scorer_command(config), check=False)
    return ScorerSmokeTestResult(
        cards_written=written,
        cards_without_e=missing,
        scorer_exit_code=trained.returncode,
    )

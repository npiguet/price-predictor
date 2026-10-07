"""Shared loaders for the knowledge-probe suite.

Measures nothing on its own. Every other module in this directory reads a
checkpoint, its sidecars, its ability cache and the records of the probe games
through these functions, so the suite runs on any checkpoint — gen-1 included —
without assuming gen-1's paths or its width of ``e`` (FR-075).

The probe records come from the curated corpus's two validation strata, which
hold whole games: a probe game is read from the shard the frozen probe set
recorded for it, not from the raw corpus, which would mean decompressing
thousands of shards to find a few hundred games.

Nothing here does work at import time.
"""

from __future__ import annotations

import json
import re
import zlib
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CARDS_FOLDERS = (ROOT / "output" / "cardsfolder", ROOT / "output" / "tokenscripts")
DEFAULT_ABILITIES = ROOT / "output" / "effects" / "abilities"
DEFAULT_REPORTS = ROOT / "output" / "effects" / "reports"

#: The two validation strata, as the corpus names its directories.
STRATA: tuple[str, ...] = ("card-disjoint", "game-disjoint")

_GAME_ID_RE = re.compile(r'"game_id":\s*"([^"]+)"')
_RECORD_ID_RE = re.compile(r'"record_id":\s*"([^"]+)"')


def stable_hash(value: str) -> int:
    """``crc32`` of a string: the order every probe-set selection follows.

    A function of the string alone, so a rebuild that keeps a game or a record
    keeps choosing it.
    """
    return zlib.crc32(value.encode("utf-8"))


def checkpoint_stem(path: Path) -> str:
    """The name a report directory carries for a checkpoint.

    ``latest.pt`` says nothing about which run it is, so its directory names it.
    """
    path = Path(path)
    return path.parent.name if path.stem == "latest" else path.stem


# ── checkpoint, sidecars and cache ──────────────────────────────────────


def load_checkpoint(path: Path):
    from effects.infrastructure.effect_model_store import EffectModelStore

    return EffectModelStore(Path(path).parent).load(Path(path))


def sidecar_cache(cards_folders: Iterable[Path], variant_scripts: Path | None = None):
    from effects.infrastructure.sidecar_io import SidecarCache, sidecar_roots

    return SidecarCache(sidecar_roots(list(cards_folders), variant_scripts))


def sidecar_paths(cards_folders: Iterable[Path]) -> list[Path]:
    from effects.infrastructure.sidecar_io import SIDECAR_SUFFIX

    return [
        path for folder in cards_folders
        for path in sorted(Path(folder).rglob(f"*{SIDECAR_SUFFIX}"))
    ]


def cache_path(abilities_root: Path, script_file: str) -> Path:
    """The ``.npz`` of one source file: the tree path the key itself names."""
    relative = Path(script_file)
    return Path(abilities_root) / relative.parent / f"{relative.stem}.npz"


@dataclass
class LineItem:
    """One unique ability text, as the checkpoint's cache encodes it.

    Keyed by the provenance of its first carrier, because a text changes between
    generations (whole-chain encoding) while the key still names the same trait.
    """

    key: object
    text: str
    script_text: str | None
    line_kind: str
    api: str | None
    card: str
    script_file: str
    carriers: set[str] = field(default_factory=set)
    e: np.ndarray | None = None


def line_items(
    cards_folders: Iterable[Path], surface: str, abilities_root: Path | None = None,
    *, text_by_key: dict | None = None,
) -> list[LineItem]:
    """Every unique encoded text over the sidecar trees, with its cached ``e``.

    One item per text on the checkpoint's surface, since two lines that read the
    same share one ``e``. ``abilities_root`` None reads labels only.
    ``text_by_key``, when given, is filled with every keyed line's text, so a
    probe item frozen under one key finds its text whichever carrier named it.
    """
    from effects.domain.ability_encoder import encoding_text
    from effects.infrastructure.sidecar_io import (
        converted_text_path,
        prose_for,
        prose_lines,
        read_sidecar,
    )

    items: dict[str, LineItem] = {}
    for path in sidecar_paths(cards_folders):
        sidecar = read_sidecar(path)
        matrix = None
        if abilities_root is not None:
            npz = cache_path(abilities_root, sidecar.script_file)
            if not npz.exists():
                continue
            matrix = np.load(npz)["e"]
            if len(matrix) != len(sidecar.lines):
                continue
        rendered = prose_lines(converted_text_path(path)) if surface == "prose" else []
        for row, line in enumerate(sidecar.lines):
            if not line.provenance:
                continue
            text = encoding_text(line, prose_for(line, rendered), surface)
            if not text:
                continue
            item = items.get(text)
            if item is None:
                item = items[text] = LineItem(
                    key=line.provenance[0], text=text, script_text=line.script_text,
                    line_kind=line.line_kind, api=line.script_api_type,
                    card=sidecar.card, script_file=sidecar.script_file,
                    e=None if matrix is None else matrix[row].astype(np.float32),
                )
            item.carriers.add(sidecar.card)
            if text_by_key is not None:
                for key in line.provenance:
                    text_by_key[key] = text
    return list(items.values())


def card_types(converted_txt: Path) -> str:
    """The first face's type line from a converted card, lowercased."""
    if not converted_txt.exists():
        return ""
    for line in converted_txt.read_text(encoding="utf-8").splitlines():
        if line.startswith("types: "):
            return line[len("types: "):].lower()
    return ""


# ── records ─────────────────────────────────────────────────────────────


def stratum_dir(corpus: Path, stratum: str) -> Path:
    return Path(corpus) / "validation" / stratum


def shards_of(directory: Path) -> list[Path]:
    from effects.infrastructure.record_io import iter_shards

    return list(iter_shards(Path(directory))) if Path(directory).exists() else []


def read_games(
    shards: Iterable[Path], game_ids: set[str] | frozenset[str],
    record_ids: set[str] | frozenset[str] | None = None,
) -> Iterator[tuple[Path, object]]:
    """``(shard, record)`` for every record of the named games.

    A line is parsed only once its ``game_id`` matches, read with a regex: the
    strata hold whole games, so most lines of a shard belong to games nobody
    asked for, and parsing them would cost most of the read. ``record_ids``
    narrows it further to the records a probing run actually reads.
    """
    from effects.infrastructure.record_io import (
        iter_shard_lines,
        record_from_dict,
        refuse_mixed_generations,
    )

    shards = list(shards)
    refuse_mixed_generations(shards)
    for shard in shards:
        for line in iter_shard_lines(shard):
            match = _GAME_ID_RE.search(line[:4000])
            if match is None or match.group(1) not in game_ids:
                continue
            if record_ids is not None:
                found = _RECORD_ID_RE.search(line[:400])
                if found is None or found.group(1) not in record_ids:
                    continue
            yield shard, record_from_dict(json.loads(line))


def batched(sequence: list, size: int) -> Iterator[list]:
    for start in range(0, len(sequence), size):
        yield sequence[start : start + size]


# ── the model ───────────────────────────────────────────────────────────


@dataclass
class ProbeModel:
    """A checkpoint rebuilt for inference, with everything the probes read."""

    checkpoint: object
    encoder: object
    model: object
    batcher: object
    fields: tuple
    device: object
    surface: str
    e_dim: int


def load_probe_model(
    checkpoint_path: Path,
    cards_folders: list[Path],
    records: list,
    *,
    vocab_path: Path | None = None,
    keyword_path: Path | None = None,
    checkpoint=None,
) -> ProbeModel:
    """Encoder, model and batcher at the checkpoint's own size and surface.

    Through the evaluator's loader, so the probes read the model exactly as
    gate 1 does: the encoder built at its recorded size, keywords expanded at
    the inference constant, the variant masks applied.
    """
    from effects.domain.ability_encoder import surface_of
    from effects.infrastructure.effect_model_store import resolve_inference_paths
    from effects.infrastructure.model_runner import load_runnable

    checkpoint = checkpoint or load_checkpoint(checkpoint_path)
    vocab, keywords = resolve_inference_paths(
        checkpoint.provenance, vocab_path=vocab_path, keyword_path=keyword_path,
    )
    config = SimpleNamespace(cards_folders=list(cards_folders), variant_scripts=None)
    encoder, model, batcher, fields = load_runnable(
        config, checkpoint, vocab_path=vocab, keyword_path=keywords, records=records,
    )
    encoder.eval()
    model.eval()
    return ProbeModel(
        checkpoint=checkpoint, encoder=encoder, model=model, batcher=batcher,
        fields=fields, device=batcher.device, surface=surface_of(vocab),
        e_dim=checkpoint.encoder_config.e_dim,
    )


def peak_gpu_bytes() -> int:
    import torch

    return int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, sort_keys=True), encoding="utf-8")

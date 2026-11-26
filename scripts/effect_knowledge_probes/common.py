"""Shared loaders for the knowledge-probe suite.

Measures nothing on its own. Every other module in this directory reads a
checkpoint, its sidecars, its ability cache and the records of the probe games
through these functions, so the suite runs on any gen-3 checkpoint without
assuming its paths or its width of ``e`` (FR-075).

The probe records come from the curated corpus's two validation strata, which
hold whole games: a probing run reads the games of the selection set's halves
from them (``halves``), not from the raw corpus, which would mean
decompressing thousands of shards to find a few hundred games. Every reader
here takes gen-3 shards only.

Nothing here does work at import time.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from _lib.paths import REPO
from _lib.stats import auc  # noqa: F401  (re-exported: the probes score with common.auc)

# Re-exported: both probe batteries name a checkpoint's report directory and
# read their default trees one way, so the two reports of one arm read alike.
from effect_embedding_probes.common import (  # noqa: F401
    DEFAULT_CARDS_FOLDERS,
    card_facts,
    checkpoint_label,
)

from effects.domain.ability_tokenizer import Surface
from effects.domain.seeded_draw import id_hash

#: The two validation strata, as the corpus names its directories.
STRATA: tuple[str, ...] = ("card-disjoint", "game-disjoint")


def stable_hash(value: str) -> int:
    """``crc32`` of a string: the order every probe-set selection follows.

    A function of the string alone, so a rebuild that keeps a game or a record
    keeps choosing it.
    """
    return id_hash(value)


# ── checkpoint, sidecars and cache ──────────────────────────────────────


def load_checkpoint(path: Path):
    """The checkpoint at ``path``; exits on one written under another input
    layout than this code builds (``gen-3``), which no probe can read."""
    from effects.domain.checkpoint_layout import CheckpointLayoutError
    from effects.infrastructure.effect_model_store import load_effect_checkpoint

    try:
        return load_effect_checkpoint(path)
    except CheckpointLayoutError as error:
        raise SystemExit(f"{path}: {error}") from None


def sidecar_cache(cards_folders: Iterable[Path], variant_scripts: Path | None = None):
    from effects.infrastructure.converted_trees import SourceTrees

    return SourceTrees.of(cards_folders, variant_scripts).sidecar_cache()


def sidecar_paths(cards_folders: Iterable[Path]) -> list[Path]:
    from effects.infrastructure import sidecar_io

    return [path for folder in cards_folders for path in sidecar_io.sidecar_paths(folder)]


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


def _cached_matrix(cache, sidecar) -> tuple[bool, np.ndarray | None]:
    """``(usable, matrix)``: a sidecar is usable without a cache, or with a cached
    matrix that has one row per line."""
    if cache is None:
        return True, None
    matrix = cache.read(sidecar.script_file)
    if matrix is None or len(matrix) != len(sidecar.lines):
        return False, None
    return True, matrix


def _row_text(line, rendered, surface: Surface) -> str | None:
    """A line's encoding text, None for one without provenance."""
    from effects.domain.ability_encoder import encoding_text

    if not line.provenance:
        return None
    return encoding_text(line, line.prose_in(rendered), surface)


def _new_item(sidecar, line, text: str, e) -> LineItem:
    return LineItem(
        key=line.provenance[0], text=text, script_text=line.script_text,
        line_kind=line.line_kind, api=line.script_api_type,
        card=sidecar.card, script_file=sidecar.script_file,
        e=None if e is None else e.astype(np.float32),
    )


@dataclass(frozen=True)
class _SidecarRead:
    """One sidecar with its rendered prose and its cached matrix, when there is one."""

    sidecar: object
    rendered: list
    matrix: np.ndarray | None


def _add_sidecar_items(items: dict[str, LineItem], read: _SidecarRead,
                       surface: Surface) -> None:
    sidecar = read.sidecar
    for row, line in enumerate(sidecar.lines):
        text = _row_text(line, read.rendered, surface)
        if not text:
            continue
        item = items.get(text)
        if item is None:
            item = items[text] = _new_item(
                sidecar, line, text, None if read.matrix is None else read.matrix[row])
        item.carriers.add(sidecar.card)


def line_items(cards_folders: Iterable[Path], surface: Surface, cache=None) -> list[LineItem]:
    """Every unique encoded text over the sidecar trees, with its cached ``e``.

    One item per text on the checkpoint's surface, since two lines that read the
    same share one ``e``. ``cache`` is the store :func:`checked_cache` gave;
    None reads labels only.
    """
    from effects.infrastructure.sidecar_io import (
        converted_text_path,
        prose_lines,
        read_sidecar,
    )

    items: dict[str, LineItem] = {}
    for path in sidecar_paths(cards_folders):
        sidecar = read_sidecar(path)
        usable, matrix = _cached_matrix(cache, sidecar)
        if not usable:
            continue
        # Read on both surfaces: a line with no prose reads its script reading on either.
        rendered = prose_lines(converted_text_path(path))
        _add_sidecar_items(items, _SidecarRead(sidecar, rendered, matrix), surface)
    return list(items.values())


def checked_cache(context: ProbeContext, abilities_root: Path, surface: Surface):
    """The shipping cache under ``abilities_root``, once its stamp names the
    probed checkpoint and trees (finding 11.7): ``CacheStampError`` otherwise."""
    from effects.infrastructure.ability_cache_store import cache_stamp, require_cache_stamp

    return require_cache_stamp(abilities_root, lambda: cache_stamp(
        context.checkpoint_path, context.checkpoint,
        cards_folders=context.cards_folders, surface=surface,
    ))


def acting_text_function(sidecars, surface: Surface):
    """``key -> text or None``: a key's encoding text on ``surface``, the
    checkpoint-independent text the probes' fold groups and line labels key on."""
    from effects.domain.ability_encoder import encoding_text

    memo: dict = {}

    def acting_text(key):
        if key not in memo:
            line = sidecars.line_or_none(key)
            memo[key] = (None if line is None
                         else encoding_text(line, sidecars.prose_for(key), surface))
        return memo[key]

    return acting_text


def manifest_surface(manifest) -> Surface:
    """The encoding surface a corpus's manifest records (script when none)."""
    return getattr(manifest, "surface", None) or Surface.SCRIPT


# ── records ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class GameFilter:
    """The games, and optionally the records, a probing run reads."""

    game_ids: set[str] | frozenset[str]
    record_ids: set[str] | frozenset[str] | None = None

    def wants(self, record_id: str, game_id: str) -> bool:
        return game_id in self.game_ids and (
            self.record_ids is None or record_id in self.record_ids)


def _shard_records(shard: Path, wanted: GameFilter) -> Iterator[object]:
    from effects.infrastructure.record_codec import parse_record_line, record_from_dict
    from effects.infrastructure.shard_io import iter_shard_lines, peek_envelope

    for line in iter_shard_lines(shard):
        head = peek_envelope(line)
        if head is not None and not wanted.wants(head.record_id, head.game_id):
            continue
        record = record_from_dict(parse_record_line(line))
        if head is None and not wanted.wants(record.record_id, record.game_id):
            continue
        yield record


def read_games(
    shards: Iterable[Path], game_ids: set[str] | frozenset[str],
    record_ids: set[str] | frozenset[str] | None = None,
) -> Iterator[tuple[Path, object]]:
    """``(shard, record)`` for every record of the named games.

    A line is parsed only once its envelope names a wanted game: the strata
    hold whole games, so most lines of a shard belong to games nobody asked
    for, and parsing them would cost most of the read. ``record_ids`` narrows
    it further to the records a probing run actually reads.
    """
    from effects.infrastructure.shard_io import require_generation

    wanted = GameFilter(game_ids, record_ids)
    shards = list(shards)
    require_generation(shards)
    for shard in shards:
        for record in _shard_records(shard, wanted):
            yield shard, record


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
    surface: Surface
    e_dim: int


@dataclass(frozen=True)
class ModelSources:
    """Where :func:`load_probe_model` reads beyond the checkpoint file: the
    vocabulary and keyword files, an already loaded checkpoint, the variant-script tree."""

    vocab_path: Path | None = None
    keyword_path: Path | None = None
    checkpoint: object = None
    variant_scripts: Path | None = None


def load_probe_model(
    checkpoint_path: Path,
    cards_folders: list[Path],
    records: list,
    sources: ModelSources = ModelSources(),  # noqa: B008  (frozen)
) -> ProbeModel:
    """Encoder, model and batcher at the checkpoint's own size and surface.

    Through the evaluator's loader, so the probes read the model exactly as
    gate 1 does: the encoder built at its recorded size, keywords expanded at
    the inference constant, the variant masks applied, and the sidecars read
    over the same trees, the variant-script tree among them.
    """
    from effects.application.model_loading import load_runnable
    from effects.infrastructure.effect_model_store import inference_files

    checkpoint = sources.checkpoint or load_checkpoint(checkpoint_path)
    # Hash-checked like every inference command: a probe over a vocabulary
    # that moved since training reads vectors that look valid and mean nothing.
    files = inference_files(checkpoint, vocab=sources.vocab_path,
                            keywords=sources.keyword_path)
    runnable = load_runnable(
        checkpoint, files=files,
        sidecars=sidecar_cache(cards_folders, sources.variant_scripts), records=records,
    )
    return ProbeModel(
        checkpoint=checkpoint, encoder=runnable.encoder, model=runnable.model,
        batcher=runnable.batcher, fields=runnable.fields,
        device=runnable.batcher.materializer.device, surface=runnable.batcher.surface,
        e_dim=checkpoint.encoder_config.e_dim,
    )


def predict(probe_model, records: list, read) -> list:
    """One forward pass over ``records``; ``read(outputs, verdict, surface, row)`` per record.

    ``outputs`` is the per-entity head's, ``verdict`` the verdict head's, both
    for the whole batch; ``read`` picks out what its probe asks of one row.
    """
    return predict_all(probe_model, records, read, chunk=max(len(records), 1))


def predict_all(probe_model, records: list, read, *, chunk: int = 32) -> list:
    """:func:`predict` over ``records`` a chunk at a time, to bound the batch,
    through the package's one chunked forward (``record_forward``), so a probe
    reads the model at the precision the control score selects on."""
    from effects.application.record_forward import RecordForward

    forward = RecordForward(probe_model.encoder, probe_model.model,
                            probe_model.batcher.materializer)
    out: list = []
    for part in forward.records(records, probe_model.batcher, size=chunk):
        outputs = part.outputs.per_entity()
        verdict = part.outputs.verdict()
        out += [read(outputs, verdict, surface, row)
                for row, surface in enumerate(part.surfaces)]
    return out


# ── scores and tables ───────────────────────────────────────────────────


def fmt(value) -> str:
    """One table cell: blank for None, three decimals for a float."""
    if value is None:
        return ""
    if isinstance(value, float):
        return "nan" if np.isnan(value) else f"{value:.3f}"
    return str(value)


def fmt_mean(values, digits: int = 3) -> str:
    """The mean of ``values`` as a table cell, ``n/a`` for none."""
    return f"{np.mean(values):.{digits}f}" if len(values) else "n/a"


# ── what one checkpoint's probes read ───────────────────────────────────


def rooted(path) -> Path:
    """A recorded path, relative ones read from the repository root.

    Checkpoints and manifests record paths as the command that wrote them was
    given them, from the repository root.
    """
    path = Path(path)
    return path if path.is_absolute() else REPO / path


@dataclass
class ProbeContext:
    """Everything a probe of one checkpoint reads, resolved from what it recorded (FR-075).

    The curated corpus it trained on, verified against the digest the
    checkpoint pinned; the sidecar trees it resolved keys against; and the
    variant-script tree that corpus records, without which a variant card's
    ability keys resolve to no text. No probe names a corpus, a selection set or a
    tree of its own, so every one runs unchanged on any checkpoint.
    """

    checkpoint_path: Path
    checkpoint: object
    corpus: Path
    manifest: object
    corpus_digest: str
    cards_folders: tuple[Path, ...]
    variant_scripts: Path | None
    _sidecars: object = field(default=None, repr=False)
    _listed: dict | None = field(default=None, repr=False)

    def sidecars(self):
        """One sidecar cache over the checkpoint's trees, built on first use."""
        if self._sidecars is None:
            self._sidecars = sidecar_cache(self.cards_folders, self.variant_scripts)
        return self._sidecars

    def stratum_shards(self, stratum: str) -> tuple[Path, ...]:
        """The shards the manifest lists for ``stratum``, checked against the
        disk once per context (finding 11.4): a stale shard is refused, so no
        probe reads one."""
        if self._listed is None:
            from effects.infrastructure.corpus_store import CorpusStore
            from effects.infrastructure.curated_shards import require_listed_shards

            self._listed = require_listed_shards(CorpusStore(self.corpus), self.manifest)
        return self._listed[stratum]

    def validation_shards(self, limit: int | None = None) -> list[Path]:
        """The first ``limit`` listed shards of each validation stratum (all
        when None), card-disjoint first."""
        return [shard for stratum in STRATA for shard in self.stratum_shards(stratum)[:limit]]

    def model(self, records: list, *, vocab_path: Path | None = None,
              keyword_path: Path | None = None) -> ProbeModel:
        """The checkpoint ready to run over ``records``, on this context's trees."""
        return load_probe_model(
            self.checkpoint_path, list(self.cards_folders), records,
            ModelSources(vocab_path=vocab_path, keyword_path=keyword_path,
                         checkpoint=self.checkpoint, variant_scripts=self.variant_scripts),
        )


def _curated_corpus(checkpoint, corpus: Path | None) -> Path:
    """The explicit corpus, else the one the checkpoint recorded; exits on none."""
    recorded = checkpoint.provenance.corpus_path
    corpus = Path(corpus) if corpus is not None else (rooted(recorded) if recorded else None)
    if corpus is None or not (corpus / "manifest.json").exists():
        raise SystemExit("no curated corpus: pass --corpus, or probe a checkpoint "
                         "trained with --corpus")
    return corpus


def _verified_manifest(corpus: Path, checkpoint_path: Path, provenance):
    """``(manifest, digest)`` of the corpus; exits on one rebuilt since training."""
    from effects.infrastructure.corpus_store import CorpusStore

    manifest = CorpusStore(corpus).load()
    digest = manifest.digest()
    if provenance.corpus_digest and provenance.corpus_digest != digest:
        raise SystemExit(
            f"{corpus} has been rebuilt since {checkpoint_path} trained on it "
            f"(recorded {provenance.corpus_digest}, found {digest})")
    return manifest, digest


def _probe_folders(checkpoint, cards_folders: Iterable[Path] | None) -> Iterable[Path]:
    if cards_folders is not None:
        return cards_folders
    recorded_folders = getattr(checkpoint, "cards_folders", ())
    return tuple(rooted(f) for f in recorded_folders) or DEFAULT_CARDS_FOLDERS


def probe_context(
    checkpoint_path: Path,
    *,
    corpus: Path | None = None,
    cards_folders: Iterable[Path] | None = None,
    variant_scripts: Path | None = None,
) -> ProbeContext:
    """Resolve one checkpoint's probing inputs; an explicit argument overrides.

    Refuses a checkpoint that names no curated corpus, and a corpus rebuilt
    since the checkpoint trained on it: a selection set frozen against the rebuild
    describes different games.
    """
    checkpoint = load_checkpoint(checkpoint_path)
    corpus = _curated_corpus(checkpoint, corpus)
    manifest, digest = _verified_manifest(corpus, checkpoint_path, checkpoint.provenance)
    folders = _probe_folders(checkpoint, cards_folders)
    if variant_scripts is None and manifest.variant_scripts:
        variant_scripts = rooted(manifest.variant_scripts)
    return ProbeContext(
        checkpoint_path=Path(checkpoint_path), checkpoint=checkpoint, corpus=corpus,
        manifest=manifest, corpus_digest=digest,
        cards_folders=tuple(Path(f) for f in folders),
        variant_scripts=variant_scripts,
    )


def grid_run(doc: str, *, records: int, per_game: int | None = None,
             options=None) -> tuple[argparse.Namespace, ProbeContext]:
    """A one-off probe's command line, parsed, and its checkpoint's context.

    Every grid reads ``--records`` qualifying records from at most ``--shards``
    listed shards per stratum, ``--per-game`` from one game where it takes more
    than one; ``options(parser)`` adds the grid's own arguments.
    """
    parser = argparse.ArgumentParser(description=doc.splitlines()[0])
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--records", type=int, default=records)
    parser.add_argument("--shards", type=int, default=200)
    if per_game is not None:
        parser.add_argument("--per-game", type=int, default=per_game)
    if options is not None:
        options(parser)
    args = parser.parse_args()
    return args, probe_context(args.checkpoint)


def heading(args, context: ProbeContext, what: str) -> None:
    """A grid's first line: the checkpoint's run and epoch, and what it read."""
    print(f"{args.checkpoint.parent.name} ({context.checkpoint.epoch}), {what}", flush=True)


def peak_gpu_bytes() -> int:
    import torch

    return int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, sort_keys=True), encoding="utf-8")

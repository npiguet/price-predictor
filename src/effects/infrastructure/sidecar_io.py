"""Reading and writing ``<name>.provenance.json`` beside a converted card.

Written by the Java parser (``convert``, or ``VariantSidecarMain`` for the
variant tree); the join rule itself lives in :mod:`effects.domain.provenance`.
"""

from __future__ import annotations

import errno
import json
import logging
import time
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from effects.domain.provenance import (
    JOIN_GAP_KINDS,
    Joined,
    KeyResolution,
    ProvenanceKey,
    ProvenanceSidecar,
    Role,
    RoleSpan,
    SidecarError,
    SidecarLine,
    SourceTree,
    SubAbilityLink,
    split_script_file,
)
from effects.domain.text_holdout import normalize_script_text
from price_predictor.infrastructure.converted_provenance import SIDECAR_SUFFIX, sidecar_paths

if TYPE_CHECKING:
    from effects.domain.ability_tokenizer import Surface

logger = logging.getLogger(__name__)


def tree_of_folder(folder: Path) -> str:
    """The source tree a converted folder holds, read from its name.

    A name ending in ``-<tree>`` counts too: a kept-aside copy such as
    ``output/old-cardsfolder`` still keys as ``cardsfolder/…`` (FR-063b).
    """
    name = Path(folder).name
    for tree in SourceTree:
        if name == tree or name.endswith(f"-{tree}"):
            return tree
    return name or SourceTree.CARDSFOLDER


def sidecar_roots(
    folders, variant_scripts: Path | None = None,
) -> dict[str, Path]:
    """``tree -> folder`` for a ``SidecarCache``, from ``--cards-folder`` values.

    Two folders of one tree are refused: keyed alike, one would shadow the
    other's every card.
    """
    roots: dict[str, Path] = {}
    for folder in folders:
        _add_root(roots, tree_of_folder(folder), Path(folder))
    if variant_scripts:
        _add_root(roots, SourceTree.VARIANT_SCRIPTS, Path(variant_scripts))
    return roots


def _add_root(roots: dict[str, Path], tree: str, folder: Path) -> None:
    if tree in roots:
        raise ValueError(
            f"--cards-folder {roots[tree]} and {folder} both hold source tree "
            f"{tree!r}; give one folder per tree"
        )
    roots[tree] = folder


def script_vocabularies(folders) -> tuple[list[str], list[str]]:
    """Every ``script_api_type`` and script parameter key the sidecars carry.

    The script-API head's two vocabularies (FR-060b), fixed before any record
    is read. Parameter keys come from every segment of ``script_text``.
    """
    from concurrent.futures import ThreadPoolExecutor

    from effects.domain.value_targets import params, segments

    paths = [path for folder in folders for path in sidecar_paths(folder)]
    api_types: set[str] = set()
    param_keys: set[str] = set()
    # Threads: the cost is file opens, which release the interpreter lock.
    with ThreadPoolExecutor(max_workers=_SCAN_THREADS) as pool:
        for raw in pool.map(lambda p: p.read_bytes(), paths, chunksize=64):
            for line in json.loads(raw).get("lines", ()):
                if line.get("script_api_type"):
                    api_types.add(line["script_api_type"])
                param_keys.update(line.get("script_param_keys") or ())
                for segment in segments(line.get("script_text") or ""):
                    param_keys.update(params(segment))
    return sorted(api_types), sorted(param_keys)


#: Threads reading sidecars for :func:`script_vocabularies`.
_SCAN_THREADS = 16


def sidecar_path_for(converted_txt: Path) -> Path:
    """The sidecar beside a converted ``.txt``, the same pairing ``.npz`` uses."""
    converted_txt = Path(converted_txt)
    return converted_txt.with_suffix(SIDECAR_SUFFIX)


def sidecar_to_dict(sidecar: ProvenanceSidecar) -> dict:
    return {
        "card": sidecar.card,
        "script_file": sidecar.script_file,
        "lines": [_line_to_dict(line) for line in sidecar.lines],
        "dropped_keys": [key.sidecar_dict() for key in sidecar.dropped_keys],
    }


def _line_to_dict(line: SidecarLine) -> dict:
    """One line as the converter writes it."""
    entry = {
        "line_index": line.line_index, "line_kind": line.line_kind,
        "prose_offset": line.prose_offset,
    }
    entry.update({
        "provenance": [key.sidecar_dict() for key in line.provenance],
        "sub_ability_links": [
            {"path": list(link.path), "label": link.label}
            for link in line.sub_ability_links
        ],
        "script_api_type": line.script_api_type,
        "script_param_keys": list(line.script_param_keys),
        "script_text": line.script_text,
        "role_spans": [
            {"start": span.start, "end": span.end, "role": span.role}
            for span in line.role_spans
        ],
    })
    return entry


def sidecar_from_dict(data: dict) -> ProvenanceSidecar:
    script_file = data["script_file"]
    return ProvenanceSidecar(
        card=data["card"],
        script_file=script_file,
        lines=tuple(
            _line_from_dict(line, script_file) for line in data.get("lines", ())
        ),
        dropped_keys=_keys_from_dict(data.get("dropped_keys", ()), script_file),
    )


def _line_from_dict(data: dict, script_file: str) -> SidecarLine:
    """One line as :func:`_line_to_dict` writes it; a key's script file
    defaults to the sidecar's own."""
    if "prose_offset" not in data:
        raise SidecarError(
            f"line {data['line_index']} records no prose_offset: reconvert the "
            "cards (`python -m price_predictor convert`) with the current converter"
        )
    return SidecarLine(
        line_index=data["line_index"],
        line_kind=data["line_kind"],
        provenance=_keys_from_dict(data.get("provenance", ()), script_file),
        sub_ability_links=tuple(
            SubAbilityLink(path=tuple(link["path"]), label=link.get("label", ""))
            for link in data.get("sub_ability_links", ())
        ),
        script_api_type=data.get("script_api_type"),
        script_param_keys=tuple(data.get("script_param_keys", ())),
        script_text=data.get("script_text"),
        role_spans=tuple(
            RoleSpan(start=span["start"], end=span["end"], role=Role(span["role"]))
            for span in data.get("role_spans", ())
        ),
        prose_offset=data["prose_offset"],
    )


def _keys_from_dict(data, script_file: str) -> tuple[ProvenanceKey, ...]:
    return tuple(ProvenanceKey.from_dict(key, script_file=script_file) for key in data)


#: Errnos Windows reports while an indexer or virus scanner briefly holds a
#: good file open; both come back clean on a retry.
_TRANSIENT_ERRNOS = frozenset({errno.EINVAL, errno.EACCES})

#: Attempts, and the wait before the second; doubling, five span under a
#: second.
_SIDECAR_READ_ATTEMPTS = 5
_SIDECAR_RETRY_DELAY = 0.05


def iter_sidecars(folder: Path) -> Iterator[tuple[Path, ProvenanceSidecar]]:
    """Every sidecar under ``folder``, read, beside its path."""
    for path in sidecar_paths(folder):
        yield path, read_sidecar(path)


def read_sidecar(path: Path) -> ProvenanceSidecar:
    """Load one sidecar. Raises ``FileNotFoundError`` if it is missing.

    A transient open failure (:data:`_TRANSIENT_ERRNOS`) is retried with
    backoff, so one scanner hold cannot kill a two-hour corpus build. Absence
    is not retried: it is an answer, not a hiccup.
    """
    path = Path(path)
    for attempt in range(1, _SIDECAR_READ_ATTEMPTS + 1):
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            raise
        except OSError as exc:
            if exc.errno not in _TRANSIENT_ERRNOS and not isinstance(
                exc, PermissionError
            ):
                raise
            if attempt == _SIDECAR_READ_ATTEMPTS:
                raise
            if attempt == 1:
                logger.warning(
                    "%s could not be opened (%s); retrying up to %d times",
                    path, exc, _SIDECAR_READ_ATTEMPTS - 1,
                )
            time.sleep(_SIDECAR_RETRY_DELAY * 2 ** (attempt - 1))
        else:
            return _parsed_sidecar(path, text)
    raise AssertionError("unreachable: the last attempt either returns or raises")


def _parsed_sidecar(path: Path, text: str) -> ProvenanceSidecar:
    """One sidecar's JSON, read; a contract breach names the file."""
    try:
        return sidecar_from_dict(json.loads(text))
    except SidecarError as exc:
        raise SidecarError(f"{path}: {exc}") from exc


def load_tree_texts(root: Path | None) -> dict[str, list[str]]:
    """``script file -> its sidecar's script texts`` over one converted tree.

    Keyed by the script file as provenance keys spell it. An absent or
    missing root is an empty tree.
    """
    if root is None or not Path(root).is_dir():
        return {}
    out: dict[str, list[str]] = {}
    for _path, sidecar in iter_sidecars(root):
        texts = sidecar.script_texts()
        if texts:
            out[sidecar.script_file] = texts
    return out


def converted_text_path(sidecar_path: Path) -> Path:
    """The converted ``.txt`` beside a sidecar."""
    sidecar_path = Path(sidecar_path)
    stem = sidecar_path.name[: -len(SIDECAR_SUFFIX)]
    return sidecar_path.with_name(f"{stem}.txt")


def prose_lines(converted_txt: Path) -> list[str]:
    """The converted file's rendered lines, indexable by ``line_index``.

    The prose surface encodes these; the sidecar carries only the script.
    """
    path = Path(converted_txt)
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8").splitlines()


def rendered_prose(sidecar_path: Path, sidecar: ProvenanceSidecar) -> list[str]:
    """The converted prose lines beside a sidecar, as :meth:`SidecarCache.prose_for`
    reads them; none for a variant, whose ``.txt`` is the Forge source script."""
    if sidecar.tree == SourceTree.VARIANT_SCRIPTS:
        return []
    return prose_lines(converted_text_path(sidecar_path))


class LineText(NamedTuple):
    """One sidecar line, its converted prose, and the text it is encoded from."""

    line: SidecarLine
    prose: str | None
    #: Empty for a line with neither surface (``encoding_text``).
    text: str


def encoded_lines(
    sidecar_path: Path, sidecar: ProvenanceSidecar, surface: Surface,
) -> list[LineText]:
    """Every line of one sidecar as ``surface`` encodes it, row-aligned with
    the ability cache: the one spelling of read the prose, cut the line's
    out, fall back to the other surface (``encoding_text``)."""
    # Here rather than at the top: the encoder module imports torch, and the
    # survey workers that read sidecars must not.
    from effects.domain.ability_encoder import encoding_text

    rendered = rendered_prose(sidecar_path, sidecar)
    out = []
    for line in sidecar.lines:
        prose = line.prose_in(rendered)
        out.append(LineText(line, prose, encoding_text(line, prose, surface)))
    return out


class UnconvertedScript(KeyError):
    """A provenance key names a script the converted corpus does not hold.

    A ``KeyError`` so callers' handlers for an unresolvable key catch it.
    """


class UnconfiguredTree(KeyError):
    """A key names a source tree this run configured no converted root for.

    The expected case is a variant key on a run without ``--variant-scripts``.
    """


@dataclass
class JoinGaps:
    """Keys :meth:`SidecarCache.join` read as no line, by why, one field per
    such resolution: a script never converted (``unresolved``,
    ``UNCONVERTED``, by script file), a source tree the run has no root for
    (``unconfigured``, ``UNCONFIGURED_TREE``, by tree), a keyword's bare key
    naming several lines (``ambiguous``, ``AMBIGUOUS``, by script file), a
    trait Forge attached to a live card that no sidecar line describes
    (``runtime_only``, ``RUNTIME_ONLY``, by script file), a trait the sidecar
    drops by design (``dropped``, ``DROPPED``, by script file), and a key its
    sidecar does not describe (``undescribed``, the join's raising case, by
    script file), each with how often it was asked. Every answer that yields
    no text is counted; ``runtime_only`` and ``dropped`` are expected, the
    others are damage.

    A cache's are cumulative (:meth:`SidecarCache.join_gaps`); a pool worker
    hands back what each shard added (:meth:`SidecarCache.take_join_gaps`),
    and its caller sums them (:meth:`merge`).
    """

    unresolved: Counter = field(default_factory=Counter)
    undescribed: Counter = field(default_factory=Counter)
    unconfigured: Counter = field(default_factory=Counter)
    ambiguous: Counter = field(default_factory=Counter)
    runtime_only: Counter = field(default_factory=Counter)
    dropped: Counter = field(default_factory=Counter)

    def copy(self) -> JoinGaps:
        return JoinGaps(*(Counter(gaps) for gaps in self._kinds()))

    def merge(self, other: JoinGaps) -> None:
        for mine, theirs in zip(self._kinds(), other._kinds(), strict=True):
            mine.update(theirs)

    def __bool__(self) -> bool:
        return any(self._kinds())

    def _kinds(self) -> tuple[Counter, ...]:
        return tuple(getattr(self, name) for name in JOIN_GAP_KINDS)

    def __sub__(self, earlier: JoinGaps) -> JoinGaps:
        return JoinGaps(*(now - then for now, then in zip(
            self._kinds(), earlier._kinds(), strict=True)))

    def report_lines(self, *, examples: int = 5) -> list[str]:
        """The gaps as printed lines, the most asked first; none when empty."""
        lines = []
        for name, what, gaps in (
            ("unresolved", "scripts the converted corpus does not hold", self.unresolved),
            ("undescribed", "scripts whose sidecar does not describe a key", self.undescribed),
            ("unconfigured", "source trees this run has no root for", self.unconfigured),
            ("ambiguous", "scripts with a keyword's bare key naming several lines",
             self.ambiguous),
            ("runtime_only", "scripts with a trait no sidecar line describes",
             self.runtime_only),
            ("dropped", "scripts with a trait the sidecar drops by design", self.dropped),
        ):
            if gaps:
                most = ", ".join(f"{key} ({n})" for key, n in gaps.most_common(examples))
                lines.append(f"keys read as no line, {name}: {len(gaps)} {what}, "
                             f"{sum(gaps.values())} lookups; most asked: {most}")
        return lines


class SidecarCache:
    """Sidecars held by script file, read once per card.

    Per run, holding only what the corpus names. :meth:`join` is the one
    key-to-line lookup; every other one reads through it.
    """

    def __init__(self, roots: dict[str, Path]) -> None:
        """``roots`` maps a source tree name to the directory it was written to."""
        self._roots = {str(name): Path(root) for name, root in roots.items()}
        self._cache: dict[str, ProvenanceSidecar] = {}
        #: Script files found to have no sidecar, so a miss stats the disk once.
        self._missing: set[str] = set()
        self._prose: dict[str, list[str]] = {}
        self._script_texts: dict[ProvenanceKey, str | None] = {}
        #: Every key :meth:`join` read as no line, by why, counted so a broken
        #: conversion or a missing tree shows.
        self._gaps = JoinGaps()
        #: What :meth:`take_join_gaps` last handed out.
        self._taken = JoinGaps()

    def path_for(self, script_file: str) -> Path:
        tree, relative = split_script_file(script_file)
        if tree not in self._roots:
            raise UnconfiguredTree(
                f"no converted root configured for source tree {tree!r}; "
                f"known trees: {sorted(self._roots)}"
            )
        return sidecar_path_for(self._roots[tree] / relative)

    def get(self, script_file: str) -> ProvenanceSidecar:
        """One script's sidecar; ``UnconfiguredTree`` or ``UnconvertedScript``
        where there is none. Not a lookup of a key, so it counts nothing."""
        cached = self._cache.get(script_file)
        if cached is not None:
            return cached
        path = self.path_for(script_file)
        if script_file in self._missing or not path.exists():
            self._missing.add(script_file)
            raise UnconvertedScript(
                f"{script_file} was never converted: no sidecar at {path}"
            )
        cached = self._cache[script_file] = read_sidecar(path)
        return cached

    def join(self, key: ProvenanceKey) -> Joined:
        """The key's resolution and the line it names: the one lookup.

        A script with no sidecar is ``UNCONVERTED``, one of a source tree this
        run has no root for ``UNCONFIGURED_TREE`` (the first of each tree
        logged); both, and every other answer that yields no line, are counted
        in :meth:`join_gaps`. A key inside a declared range in neither list is counted as
        ``undescribed`` and raises ``KeyError``: the sidecar does not describe
        the card the record was collected against.
        """
        sidecar = self._sidecar_of(key)
        if isinstance(sidecar, KeyResolution):
            return Joined(sidecar, None)
        try:
            resolution = sidecar.resolution_of(key)
        except KeyError:
            self._gaps.undescribed[key.script_file] += 1
            raise
        counter = self._textless_counter(resolution)
        if counter is not None:
            counter[key.script_file] += 1
        line = sidecar.line_for(key) if resolution is KeyResolution.LINE else None
        return Joined(resolution, line)

    def _textless_counter(self, resolution: KeyResolution) -> Counter | None:
        """The gap counter of a sidecar answer that names no line, else None."""
        return {
            KeyResolution.AMBIGUOUS: self._gaps.ambiguous,
            KeyResolution.RUNTIME_ONLY: self._gaps.runtime_only,
            KeyResolution.DROPPED: self._gaps.dropped,
        }.get(resolution)

    def _sidecar_of(self, key: ProvenanceKey) -> ProvenanceSidecar | KeyResolution:
        """The sidecar a key's script names, else the resolution, counted,
        that says why there is none."""
        try:
            return self.get(key.script_file)
        except UnconfiguredTree:
            self._note_unconfigured(key.tree)
            return KeyResolution.UNCONFIGURED_TREE
        except UnconvertedScript:
            self._gaps.unresolved[key.script_file] += 1
            return KeyResolution.UNCONVERTED

    def _note_unconfigured(self, tree: str) -> None:
        if tree not in self._gaps.unconfigured:
            logger.warning(
                "keys under source tree %r read as no line: this run configured "
                "no root for it (known trees: %s); later ones only counted",
                tree, sorted(self._roots))
        self._gaps.unconfigured[tree] += 1

    def join_gaps(self) -> JoinGaps:
        """Every key :meth:`join` read as no line so far, by why."""
        return self._gaps.copy()

    def take_join_gaps(self) -> JoinGaps:
        """What :meth:`join_gaps` gained since the last take: a pool worker,
        whose cache outlives a shard, hands this back with each shard's result."""
        now = self.join_gaps()
        gained, self._taken = now - self._taken, now
        return gained

    def line_for(self, key: ProvenanceKey) -> SidecarLine | None:
        """:meth:`join`'s line: None where the key names none, ``KeyError``
        on a key the sidecar does not describe."""
        return self.join(key).line

    def line_or_none(self, key: ProvenanceKey) -> SidecarLine | None:
        """:meth:`line_for`, with a key the sidecar does not describe read as
        no line too (still counted), for readers that read keys off a board
        rather than train on them."""
        try:
            return self.line_for(key)
        except KeyError:
            return None

    def first_line(self, keys) -> SidecarLine | None:
        """The first line any of ``keys`` names, under :meth:`line_or_none`."""
        for key in keys:
            line = self.line_or_none(key)
            if line is not None:
                return line
        return None

    def script_text(self, key: ProvenanceKey) -> str | None:
        """The normalized script text of the line a key names, under
        :meth:`line_or_none`; resolved once per key, since the holdout tests
        and the coverage counts ask for the same keys shard after shard."""
        if key not in self._script_texts:
            line = self.line_or_none(key)
            self._script_texts[key] = (
                normalize_script_text(line.script_text)
                if line is not None and line.script_text else None
            )
        return self._script_texts[key]

    def prose_for(self, key: ProvenanceKey) -> str | None:
        """The converted prose of the line a key names, under :meth:`line_for`.

        None for a variant: the ``.txt`` beside its sidecar is the Forge
        source script, so the prose surface falls back to the script.
        """
        if key.tree == SourceTree.VARIANT_SCRIPTS:
            return None
        line = self.line_for(key)
        if line is None:
            return None
        rendered = self._prose.get(key.script_file)
        if rendered is None:
            rendered = prose_lines(
                converted_text_path(self.path_for(key.script_file))
            )
            self._prose[key.script_file] = rendered
        return line.prose_in(rendered)

    def resolution_of(self, key: ProvenanceKey) -> KeyResolution:
        """:meth:`join`'s resolution; a mismatch inside an existing sidecar
        still raises."""
        return self.join(key).resolution

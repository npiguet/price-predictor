"""Reading and writing ``<name>.provenance.json`` beside a converted card.

Written by the Java parser (``convert``, or ``VariantSidecarMain`` for the
variant tree); the join rule itself lives in :mod:`effects.domain.provenance`.
"""

from __future__ import annotations

import errno
import json
import logging
import time
from collections.abc import Iterator
from pathlib import Path

from effects.domain.provenance import (
    KeyResolution,
    ProvenanceKey,
    ProvenanceSidecar,
    Role,
    RoleSpan,
    SidecarLine,
    SourceTree,
    SubAbilityLink,
    split_script_file,
)
from effects.domain.text_holdout import normalize_script_text

logger = logging.getLogger(__name__)

SIDECAR_SUFFIX = ".provenance.json"

_JSON_SEPARATORS = (",", ":")


def tree_of_folder(folder: Path) -> str:
    """The source tree a converted folder holds, read from its name.

    A name ending in ``-<tree>`` counts too: a kept-aside copy such as
    ``output/gen1-cardsfolder`` still keys as ``cardsfolder/…`` (FR-063b).
    """
    name = Path(folder).name
    for tree in SourceTree:
        if name == tree or name.endswith(f"-{tree}"):
            return tree
    return name or SourceTree.CARDSFOLDER


def sidecar_roots(
    folders, variant_scripts: Path | None = None,
) -> dict[str, Path]:
    """``tree -> folder`` for a ``SidecarCache``, from ``--cards-folder`` values."""
    roots = {tree_of_folder(folder): Path(folder) for folder in folders}
    if variant_scripts:
        roots[SourceTree.VARIANT_SCRIPTS] = Path(variant_scripts)
    return roots


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
    """One line as the converter writes it; ``prose_offset`` only where known."""
    entry = {"line_index": line.line_index, "line_kind": line.line_kind}
    if line.prose_offset is not None:
        entry["prose_offset"] = line.prose_offset
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
            SidecarLine(
                line_index=line["line_index"],
                line_kind=line["line_kind"],
                provenance=tuple(
                    ProvenanceKey.from_dict(key, script_file=script_file)
                    for key in line.get("provenance", ())
                ),
                sub_ability_links=tuple(
                    SubAbilityLink(
                        path=tuple(link["path"]), label=link.get("label", ""),
                    )
                    for link in line.get("sub_ability_links", ())
                ),
                script_api_type=line.get("script_api_type"),
                script_param_keys=tuple(line.get("script_param_keys", ())),
                script_text=line.get("script_text"),
                role_spans=tuple(
                    RoleSpan(
                        start=span["start"], end=span["end"], role=Role(span["role"]),
                    )
                    for span in line.get("role_spans", ())
                ),
                prose_offset=line.get("prose_offset"),
            )
            for line in data.get("lines", ())
        ),
        dropped_keys=tuple(
            ProvenanceKey.from_dict(key, script_file=script_file)
            for key in data.get("dropped_keys", ())
        ),
    )


#: Errnos Windows reports while an indexer or virus scanner briefly holds a
#: good file open; both come back clean on a retry.
_TRANSIENT_ERRNOS = frozenset({errno.EINVAL, errno.EACCES})

#: Attempts, and the wait before the second; doubling, five span under a
#: second.
_SIDECAR_READ_ATTEMPTS = 5
_SIDECAR_RETRY_DELAY = 0.05


def sidecar_paths(folder: Path) -> list[Path]:
    """Every sidecar under ``folder``, recursively, in path order so every
    walk over a tree visits it the same way."""
    return sorted(Path(folder).rglob(f"*{SIDECAR_SUFFIX}"))


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
            return sidecar_from_dict(json.loads(text))
    raise AssertionError("unreachable: the last attempt either returns or raises")


def write_sidecar(sidecar: ProvenanceSidecar, path: Path) -> None:
    """Write one sidecar beside its converted text.

    The inverse of :func:`read_sidecar`, for producing one without Forge.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(sidecar_to_dict(sidecar), separators=_JSON_SEPARATORS),
        encoding="utf-8",
    )


def load_tree_texts(root: Path | None) -> dict[str, list[str]]:
    """``script file -> its sidecar's script texts`` over one converted tree.

    Keyed by the script file as provenance keys spell it. An absent or
    missing root is an empty tree.
    """
    if root is None or not Path(root).is_dir():
        return {}
    out: dict[str, list[str]] = {}
    for _path, sidecar in iter_sidecars(root):
        texts = [line.script_text for line in sidecar.lines if line.script_text]
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


def prose_for(line, lines: list[str]) -> str | None:
    """One line's converted prose, without its ``kind[n]: `` prefix.

    Role spans are offsets into the prose without the prefix. The split is
    the sidecar's ``prose_offset``, since a continuation line has no prefix
    but may hold a ``{T}: ``; an older sidecar without it is cut at the first
    ``": "``, right for every line such sidecars hold.
    """
    if not lines or not (0 <= line.line_index < len(lines)):
        return None
    rendered = lines[line.line_index]
    if line.prose_offset is not None:
        return rendered[line.prose_offset:]
    separator = rendered.find(": ")
    return rendered if separator < 0 else rendered[separator + 2:]


class UnconvertedScript(KeyError):
    """A provenance key names a script the converted corpus does not hold.

    A ``KeyError`` so callers' handlers for an unresolvable key catch it.
    """


class UnconfiguredTree(KeyError):
    """A key names a source tree this run configured no converted root for.

    The expected case is a variant key on a run without ``--variant-scripts``.
    """


class SidecarCache:
    """Sidecars held by script file, read once per card.

    Per run, holding only what the corpus names.
    """

    def __init__(self, roots: dict[str, Path]) -> None:
        """``roots`` maps a source tree name to the directory it was written to."""
        self._roots = {str(name): Path(root) for name, root in roots.items()}
        self._cache: dict[str, ProvenanceSidecar] = {}
        #: Script files found to have no sidecar, so a miss stats the disk once.
        self._missing: set[str] = set()
        self._prose: dict[str, list[str]] = {}
        self._script_texts: dict[ProvenanceKey, str | None] = {}
        #: ``script_file -> times asked`` for keys naming an unconverted script,
        #: counted so a broken conversion shows.
        self.unresolved: dict[str, int] = {}
        #: ``script_file -> times asked`` for keys :meth:`line_or_none` read as
        #: no line although the sidecar exists and does not describe them.
        self.undescribed: dict[str, int] = {}

    def path_for(self, script_file: str) -> Path:
        tree, relative = split_script_file(script_file)
        if tree not in self._roots:
            raise UnconfiguredTree(
                f"no converted root configured for source tree {tree!r}; "
                f"known trees: {sorted(self._roots)}"
            )
        return sidecar_path_for(self._roots[tree] / relative)

    def get(self, script_file: str) -> ProvenanceSidecar:
        cached = self._cache.get(script_file)
        if cached is not None:
            return cached
        path = self.path_for(script_file)
        if script_file in self._missing or not path.exists():
            self._missing.add(script_file)
            self.unresolved[script_file] = self.unresolved.get(script_file, 0) + 1
            raise UnconvertedScript(
                f"{script_file} was never converted: no sidecar at {path}"
            )
        cached = self._cache[script_file] = read_sidecar(path)
        return cached

    def line_for(self, key: ProvenanceKey) -> SidecarLine | None:
        """The rendered line a record's key names, or None where there is none.

        None also for a script never converted; an unconfigured tree still
        raises, being a misconfigured run.
        """
        try:
            return self.get(key.script_file).line_for(key)
        except UnconvertedScript:
            return None

    def line_or_none(self, key: ProvenanceKey) -> SidecarLine | None:
        """The line a key names, or None wherever the join gives none.

        For readers that read keys off a board rather than train on them. A
        key the sidecar does not describe, which :meth:`line_for` raises on, is
        tallied in :attr:`undescribed`.
        """
        try:
            return self.line_for(key)
        except UnconfiguredTree:
            return None
        except KeyError:
            self.undescribed[key.script_file] = (
                self.undescribed.get(key.script_file, 0) + 1
            )
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
        """The converted prose of the line a key names.

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
        return prose_for(line, rendered)

    def row_for(self, key: ProvenanceKey) -> int | None:
        """The ability-cache row a record's key names, or None where it was dropped."""
        return self.get(key.script_file).row_for(key)

    def resolution_of(self, key: ProvenanceKey) -> KeyResolution:
        """The join's answer for a key, with the two no-sidecar cases folded in.

        An unconverted script and an unconfigured tree are both
        ``UNCONVERTED``; a mismatch inside an existing sidecar still raises.
        """
        try:
            return self.get(key.script_file).resolution_of(key)
        except (UnconvertedScript, UnconfiguredTree):
            return KeyResolution.UNCONVERTED

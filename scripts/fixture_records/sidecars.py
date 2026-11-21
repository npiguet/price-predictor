"""Raw sidecar JSON by script file, and copying the sidecars a fixture's
records name beside it under the same tree layout, so a ``SidecarCache``
rooted there resolves every key."""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterable, Iterator
from pathlib import Path

_KEY_FIELDS = frozenset({"script_file", "face", "trait_kind", "index_within_kind"})
_SIDECAR_SUFFIX = ".provenance.json"


def keys_in(obj) -> Iterator[dict]:
    """Every provenance-key dict anywhere below ``obj``."""
    if isinstance(obj, dict):
        yield from _keys_in_dict(obj)
    elif isinstance(obj, list):
        for value in obj:
            yield from keys_in(value)


def _keys_in_dict(obj: dict) -> Iterator[dict]:
    if _KEY_FIELDS <= obj.keys():
        yield obj
        return
    for value in obj.values():
        yield from keys_in(value)


def _position(key: dict) -> tuple:
    """Where a key sits on its card, ``built`` included: a saga's chapters
    share their keyword's key and differ only there."""
    built = key.get("built") or {}
    return (key["face"], key["trait_kind"], key["index_within_kind"],
            built.get("kind"), built.get("index"))


def _sidecar_path(root: Path, relative: str) -> Path:
    return (root / relative).with_suffix(_SIDECAR_SUFFIX)


class Sidecars:
    """Raw sidecar JSON by script file, read on demand from the source roots."""

    def __init__(self, roots: dict[str, Path]) -> None:
        self.roots = roots
        self._cache: dict[str, dict | None] = {}

    def path(self, script_file: str) -> Path | None:
        tree, _, relative = script_file.partition("/")
        root = self.roots.get(tree)
        return _sidecar_path(root, relative) if root is not None else None

    def get(self, script_file: str) -> dict | None:
        if script_file not in self._cache:
            self._cache[script_file] = self._read(script_file)
        return self._cache[script_file]

    def _read(self, script_file: str) -> dict | None:
        path = self.path(script_file)
        if path is None or not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def line(self, key: dict) -> dict | None:
        """The raw sidecar line a key names, or None."""
        sidecar = self.get(key["script_file"])
        if sidecar is None:
            return None
        wanted = _position(key)
        return next((line for line in sidecar["lines"]
                     if any(_position(found) == wanted for found in line["provenance"])),
                    None)

    def api_type(self, key: dict) -> str | None:
        line = self.line(key)
        return line.get("script_api_type") if line is not None else None

    def script(self, key: dict) -> str:
        line = self.line(key)
        return (line.get("script_text") or "") if line is not None else ""

    def prints_keyword(self, script_file: str) -> bool:
        """Whether the card has a line a keyword produced."""
        sidecar = self.get(script_file)
        return sidecar is not None and any(
            line.get("script_api_type") == "Keyword" for line in sidecar["lines"])


def named_script_files(records: Iterable[dict]) -> list[str]:
    """Every script file a key anywhere in ``records`` names, sorted."""
    return sorted({key["script_file"] for record in records for key in keys_in(record)})


def copy_sidecars(records: dict[str, dict], sidecars: Sidecars, sidecars_out: Path,
                  *, keep_existing: bool) -> tuple[int, list[str]]:
    """Copy every sidecar and converted text the records name; with
    ``keep_existing``, a file already in the fixture is left as it is. Returns
    the count copied and the script files no given tree holds."""
    copied = 0
    unresolved: list[str] = []
    for script_file in named_script_files(records.values()):
        source = sidecars.path(script_file)
        if source is None or not source.exists():
            unresolved.append(script_file)
        elif _copy_one(source, sidecars_out, script_file, keep_existing):
            copied += 1
    return copied, unresolved


def _copy_one(source: Path, sidecars_out: Path, script_file: str, keep_existing: bool) -> bool:
    """Copy one sidecar and its converted text, if any; False when kept as it was."""
    tree, _, relative = script_file.partition("/")
    target = _sidecar_path(sidecars_out / tree, relative)
    if keep_existing and target.exists():
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    text = source.with_name(source.name[: -len(_SIDECAR_SUFFIX)] + ".txt")
    if text.exists():
        shutil.copyfile(text, target.with_name(text.name))
    return True

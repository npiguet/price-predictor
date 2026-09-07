"""Reading and writing ``<name>.provenance.json`` beside a converted card.

The sidecar is written by the Java parser — `price_predictor convert` for the
two converted trees, and `VariantSidecarMain` (spawned by `effects
collect-variants`) for the perturbed one, where it is the only output. It is
read here into the pure dataclasses of :mod:`effects.domain.provenance`.

The join rule it implements is the one thing this module exists for:

- a key that appears in ``lines`` resolves to that line;
- a key in ``dropped_keys`` resolves to **no** line and is kept — the converter
  deduplicated that trait away, but it is still live at runtime and still
  produces a key, so the record supervises through its state and payload with
  nothing to join to;
- a key in neither **fails loudly**, because the only way that happens is a
  reconversion between collection and training, and a corpus scored against the
  wrong sidecar is worse than one that stops.

Per-line provenance entries omit ``script_file`` — it is the same for every key
in one file and lives in the header.
"""

from __future__ import annotations

import json
from pathlib import Path

from effects.domain.provenance import (
    ProvenanceKey,
    ProvenanceSidecar,
    RoleSpan,
    SidecarLine,
    SubAbilityLink,
)

SIDECAR_SUFFIX = ".provenance.json"

#: The one tree whose sidecars sit beside a *source* script rather than a
#: converted one, and which therefore has no prose surface at all.
VARIANT_SCRIPTS_TREE = "variant-scripts"

_JSON_SEPARATORS = (",", ":")


def sidecar_path_for(converted_txt: Path) -> Path:
    """The sidecar beside a converted ``.txt``, the same pairing ``.npz`` uses."""
    converted_txt = Path(converted_txt)
    return converted_txt.with_suffix(SIDECAR_SUFFIX)


def sidecar_to_dict(sidecar: ProvenanceSidecar) -> dict:
    return {
        "card": sidecar.card,
        "script_file": sidecar.script_file,
        "lines": [
            {
                "line_index": line.line_index,
                "line_kind": line.line_kind,
                "provenance": [
                    {
                        "face": key.face,
                        "trait_kind": key.trait_kind,
                        "index_within_kind": key.index_within_kind,
                    }
                    for key in line.provenance
                ],
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
            }
            for line in sidecar.lines
        ],
        "dropped_keys": [
            {
                "face": key.face,
                "trait_kind": key.trait_kind,
                "index_within_kind": key.index_within_kind,
            }
            for key in sidecar.dropped_keys
        ],
    }


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
                        start=span["start"], end=span["end"], role=span["role"],
                    )
                    for span in line.get("role_spans", ())
                ),
            )
            for line in data.get("lines", ())
        ),
        dropped_keys=tuple(
            ProvenanceKey.from_dict(key, script_file=script_file)
            for key in data.get("dropped_keys", ())
        ),
    )


def read_sidecar(path: Path) -> ProvenanceSidecar:
    """Load one sidecar. Raises ``FileNotFoundError`` if it is missing.

    A missing sidecar is not tolerated the way a partial shard line is: a record
    that names a card with no sidecar cannot be joined at all, and silently
    skipping it would drop training signal without saying so.
    """
    path = Path(path)
    return sidecar_from_dict(json.loads(path.read_text(encoding="utf-8")))


def write_sidecar(sidecar: ProvenanceSidecar, path: Path) -> None:
    """Write one sidecar beside its converted text.

    The inverse of :func:`read_sidecar`, used by the tests and by any tool that
    needs to produce a sidecar without a Forge runtime.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(sidecar_to_dict(sidecar), separators=_JSON_SEPARATORS),
        encoding="utf-8",
    )


def converted_text_path(sidecar_path: Path) -> Path:
    """The converted ``.txt`` beside a sidecar."""
    sidecar_path = Path(sidecar_path)
    stem = sidecar_path.name[: -len(SIDECAR_SUFFIX)]
    return sidecar_path.with_name(f"{stem}.txt")


def prose_lines(converted_txt: Path) -> list[str]:
    """The converted file's rendered lines, indexable by ``line_index``.

    The prose surface encodes these; the sidecar carries only the script. A
    reader that used ``script_text`` for both would silently encode the wrong
    surface, and the result would look like a working run.
    """
    path = Path(converted_txt)
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8").splitlines()


def prose_for(line, lines: list[str]) -> str | None:
    """One line's converted prose, without its ``kind[n]: `` prefix.

    The prefix is structure rather than card text, and the role spans the
    tokenizer applies are offsets into the prose without it.
    """
    if not lines or not (0 <= line.line_index < len(lines)):
        return None
    rendered = lines[line.line_index]
    separator = rendered.find(": ")
    return rendered if separator < 0 else rendered[separator + 2:]


class SidecarCache:
    """Sidecars held by script file, read once per card.

    Training joins millions of records against a few tens of thousands of cards,
    so the read has to happen once per card rather than once per record. The
    cache is per run and holds only what the corpus actually names.
    """

    def __init__(self, roots: dict[str, Path]) -> None:
        """``roots`` maps a source tree name to the directory it was written to.

        e.g. ``{"cardsfolder": Path("output/cardsfolder"), "tokenscripts":
        Path("output/tokenscripts")}``. A key's ``script_file`` names its tree,
        which is what lets one filename resolve differently in two trees.
        """
        self._roots = {name: Path(root) for name, root in roots.items()}
        self._cache: dict[str, ProvenanceSidecar] = {}
        self._prose: dict[str, list[str]] = {}

    def path_for(self, script_file: str) -> Path:
        tree, _, relative = script_file.partition("/")
        if tree not in self._roots:
            raise KeyError(
                f"no converted root configured for source tree {tree!r}; "
                f"known trees: {sorted(self._roots)}"
            )
        return sidecar_path_for(self._roots[tree] / relative)

    def get(self, script_file: str) -> ProvenanceSidecar:
        cached = self._cache.get(script_file)
        if cached is None:
            cached = read_sidecar(self.path_for(script_file))
            self._cache[script_file] = cached
        return cached

    def line_for(self, key: ProvenanceKey) -> SidecarLine | None:
        """The rendered line a record's key names, or None where it was dropped."""
        return self.get(key.script_file).line_for(key)

    def prose_for(self, key: ProvenanceKey) -> str | None:
        """The converted prose of the line a key names.

        Read from the ``.txt`` beside the sidecar and held per card, because the
        prose surface encodes it and the sidecar carries only the script.

        A variant has none: nobody printed the card, so there is no oracle text
        to convert, and the ``.txt`` beside its sidecar is the Forge source
        script rather than converted prose. Returning None is what makes the
        prose surface fall back to the script for a variant instead of encoding
        a source-script line as if it were card text.
        """
        if key.script_file.startswith(f"{VARIANT_SCRIPTS_TREE}/"):
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

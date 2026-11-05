"""Write ``src/effects/domain/output_vocabularies.py``: the head's closed output lists.

Root spec section 11.2 and section 11.3 draw ``subtypes_gained``/``subtypes_lost``,
the created-objects groups' subtypes and ``keywords_gained``/``keywords_lost`` over
the subtypes and keywords the corpus's effects grant or remove, a closed list the
corpus build records (FR-052, FR-053). Until ``build-corpus`` writes those lists to
its manifest, this survey of record directories writes them: every subtype and
keyword ``effects.domain.closed_lists`` reads off a record, keywords limited to
Forge's own (the keyword-definitions file) plus the overlay keywords, which always
lead the keyword list. ``--manifest`` instead copies the lists a gen-3 ``build-corpus``
recorded in its manifest (``subtypes``, ``keywords``), already in the head's column order.

A checkpoint records the lists it trained on and a loader refuses another, so
rewriting the module means retraining.

Usage: python scripts/scan_output_vocabularies.py --records-dir DIR [--records-dir DIR ...]
       [--keyword-definitions FILE] [--output FILE]
       python scripts/scan_output_vocabularies.py --manifest CORPUS/manifest.json [--output FILE]
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

from _lib.paths import REPO

from effects.domain.closed_lists import (
    keywords_named,
    ordered_keywords,
    ordered_subtypes,
    subtypes_named,
)
from effects.domain.keyword_names import normalize_keyword
from effects.domain.records import EffectRecord
from effects.infrastructure.shard_io import read_records, read_shard

OUTPUT = REPO / "src" / "effects" / "domain" / "output_vocabularies.py"
KEYWORD_DEFINITIONS = REPO / "output" / "effects" / "keyword-definitions.json"


def forge_keywords(definitions: Path) -> frozenset[str]:
    """Forge's keywords, each display name in :func:`normalize_keyword`'s spelling."""
    names = json.loads(definitions.read_text(encoding="utf-8"))
    return frozenset(normalize_keyword(name) for name in names)


def survey(
    records: Iterable[EffectRecord], known: frozenset[str],
) -> dict[str, tuple[str, ...]]:
    """The two closed lists over ``records``, in the head's column order."""
    subtypes: set[str] = set()
    keywords: set[str] = set()
    for record in records:
        subtypes |= subtypes_named(record)
        keywords |= keywords_named(record, known)
    return {
        "SUBTYPES_CHANGED": ordered_subtypes(subtypes),
        "KEYWORDS_CHANGED": ordered_keywords(keywords),
    }


_COMMENTS = {
    "SUBTYPES_CHANGED": "Subtypes effects grant, remove or give a token, in Forge's order.",
    "KEYWORDS_CHANGED": "Keywords effects grant, remove or give a token: overlay keywords first.",
}


def render(lists: dict[str, tuple[str, ...]], sources: list[str]) -> str:
    """The checked-in module's text."""
    lines = [
        '"""The head\'s closed output lists (FR-052, FR-053): generated, do not edit.',
        "",
        "Written by ``python scripts/scan_output_vocabularies.py`` from the records under",
        ", ".join(f"``{source}``" for source in sources) + ".",
        "A checkpoint records both lists and a loader refuses another, so rewriting this",
        "module means retraining.",
        '"""',
    ]
    for name, values in lists.items():
        lines += ["", f"#: {_COMMENTS[name]}", f"{name}: tuple[str, ...] = ("]
        lines += [f"    {json.dumps(value, ensure_ascii=False)}," for value in values]
        lines.append(")")
    return "\n".join(lines) + "\n"


def from_manifest(path: Path) -> dict[str, tuple[str, ...]]:
    """The two lists a gen-3 corpus manifest recorded; a manifest without them
    (built before gen-3) is refused."""
    from effects.domain.corpus_manifest import CorpusManifest

    manifest = CorpusManifest.from_dict(json.loads(path.read_text(encoding="utf-8")))
    if not manifest.subtypes or not manifest.keywords:
        raise SystemExit(f"{path} records no subtype or keyword list: not a gen-3 manifest")
    return {"SUBTYPES_CHANGED": manifest.subtypes, "KEYWORDS_CHANGED": manifest.keywords}


def _read(source: Path) -> Iterable[EffectRecord]:
    return read_shard(source) if source.is_file() else read_records(source)


def _source_name(directory: Path) -> str:
    try:
        return directory.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return directory.as_posix()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--records-dir", type=Path, action="append",
                        help="a records directory, or one shard, to survey (repeatable)")
    source.add_argument("--manifest", type=Path,
                        help="a gen-3 corpus manifest whose recorded lists to copy")
    parser.add_argument("--keyword-definitions", type=Path, default=KEYWORD_DEFINITIONS,
                        help="Forge's keyword table (default: %(default)s)")
    parser.add_argument("--output", type=Path, default=OUTPUT,
                        help="the module to write (default: %(default)s)")
    args = parser.parse_args(argv)
    if args.manifest is not None:
        lists = from_manifest(args.manifest)
        sources = [_source_name(args.manifest)]
    else:
        known = forge_keywords(args.keyword_definitions)
        lists = survey(
            (record for source in args.records_dir for record in _read(source)),
            known,
        )
        sources = [_source_name(directory) for directory in args.records_dir]
    with open(args.output, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(render(lists, sources))
    print(", ".join(f"{len(values)} {name}" for name, values in lists.items())
          + f"; wrote {args.output}")


if __name__ == "__main__":
    main()

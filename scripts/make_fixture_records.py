"""Cut a small, kind-balanced sample of real gen-1 records into a test fixture.

Tests of the gen-2 readers, families, value targets and probes need records
shaped exactly like the collector's output, so the fixture is cut from real
shards rather than built by hand. Every category below is filled with a few
records; a chosen cost half brings its effect half, a chosen probe brings the
combat record it mirrors, and a chosen fired trigger brings the resolution
records that follow it in the same game, so the joins the probes make (an
interaction, a link) have both ends present.

The sidecars and converted texts every chosen record names, anywhere in its
envelope, state or payload, are copied beside it under the same tree layout,
so a ``SidecarCache`` rooted there resolves every key.

    python scripts/make_fixture_records.py
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import sys
from collections import defaultdict
from collections.abc import Callable, Iterator
from pathlib import Path

from effects.infrastructure.record_io import (
    SHARD_GLOBS,
    iter_shard_lines,
    record_from_dict,
)

_KEY_FIELDS = {"script_file", "face", "trait_kind", "index_within_kind"}

#: Records that follow a fired trigger in its game, kept so an interaction join
#: (trigger -> the resolution it put on the stack) has its second half.
_FOLLOWERS_PER_TRIGGER = 6


def _keys_in(obj) -> Iterator[dict]:
    """Every provenance-key dict anywhere below ``obj``."""
    if isinstance(obj, dict):
        if _KEY_FIELDS <= obj.keys():
            yield obj
            return
        for value in obj.values():
            yield from _keys_in(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _keys_in(value)


class _Sidecars:
    """Raw sidecar JSON by script file, read on demand from the source roots."""

    def __init__(self, roots: dict[str, Path]) -> None:
        self.roots = roots
        self._cache: dict[str, dict | None] = {}

    def path(self, script_file: str) -> Path | None:
        tree, _, relative = script_file.partition("/")
        root = self.roots.get(tree)
        if root is None:
            return None
        return (root / relative).with_suffix(".provenance.json")

    def get(self, script_file: str) -> dict | None:
        if script_file not in self._cache:
            path = self.path(script_file)
            self._cache[script_file] = (
                json.loads(path.read_text(encoding="utf-8"))
                if path is not None and path.exists() else None
            )
        return self._cache[script_file]

    def api_type(self, key: dict) -> str | None:
        sidecar = self.get(key["script_file"])
        if sidecar is None:
            return None
        for line in sidecar["lines"]:
            for candidate in line["provenance"]:
                if (candidate["face"], candidate["trait_kind"],
                        candidate["index_within_kind"]) == (
                        key["face"], key["trait_kind"], key["index_within_kind"]):
                    return line.get("script_api_type")
        return None


def _categories(sidecars: _Sidecars) -> dict[str, Callable[[dict], bool]]:
    def acting_api(record: dict) -> str | None:
        keys = record.get("ability") or ()
        return sidecars.api_type(keys[0]) if keys else None

    def is_kind(kind: str, discriminator: str | None = None):
        def test(record: dict) -> bool:
            if record["kind"] != kind:
                return False
            return discriminator is None or (
                record.get("moment") or record.get("subkind")) == discriminator
        return test

    observed_effect = is_kind("resolution", "resolution")
    # Most specific first: a record lands in the first category it fits.
    return {
        "modal": lambda r: is_kind("resolution")(r) and acting_api(r) == "Charm",
        "interventional": lambda r: observed_effect(r) and r["interventional"],
        "combat-probe": lambda r: r["kind"] == "combat" and bool(
            r["payload"].get("probed_keyword")),
        "blockers-forbidden": lambda r: is_kind("playability", "blockers")(r) and bool(
            r["payload"].get("forbidden")),
        "activation": is_kind("resolution", "activation"),
        "effect": lambda r: observed_effect(r) and not r["fork"] and bool(r.get("link_id")),
        "trigger-fired": lambda r: r["kind"] == "trigger" and r["payload"].get("fired"),
        "trigger-unfired": lambda r: r["kind"] == "trigger" and not r["payload"].get("fired"),
        "continuous": is_kind("continuous"),
        "rewrite": is_kind("rewrite"),
        "combat": lambda r: r["kind"] == "combat" and not r["fork"],
        "decision": is_kind("playability", "decision"),
        "attackers": is_kind("playability", "attackers"),
        "blockers": is_kind("playability", "blockers"),
    }


def _shards(records_dir: Path) -> list[Path]:
    found: set[Path] = set()
    for pattern in SHARD_GLOBS:
        found.update(records_dir.rglob(pattern))
    return sorted(found)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--records-dir", type=Path, default=Path("output/effects/records"))
    parser.add_argument("--cards-folder", action="append", type=Path,
                        help="converted roots, named by tree (repeatable); "
                             "default output/cardsfolder and output/tokenscripts")
    parser.add_argument("--output", type=Path,
                        default=Path("tests/fixtures/effects/gen1-records.jsonl.gz"))
    parser.add_argument("--sidecars-out", type=Path,
                        default=Path("tests/fixtures/effects/gen1-sidecars"))
    parser.add_argument("--per-category", type=int, default=3)
    parser.add_argument("--max-shards", type=int, default=400)
    args = parser.parse_args()

    folders = args.cards_folder or [Path("output/cardsfolder"), Path("output/tokenscripts")]
    sidecars = _Sidecars({folder.name: folder for folder in folders})
    categories = _categories(sidecars)
    chosen: dict[str, list[dict]] = defaultdict(list)
    wanted_links: set[str] = set()
    wanted_mirrors: set[str] = set()
    extras: dict[str, dict] = {}

    shards = _shards(args.records_dir)
    # Interleave the collection runs rather than reading one run's shards first,
    # so a category only another run reaches (probes, variants) is found early.
    shards = shards[::7] + [s for i, s in enumerate(shards) if i % 7]
    for shard_no, shard in enumerate(shards[: args.max_shards]):
        followers: dict[str, int] = {}
        for line in iter_shard_lines(shard):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            record_id = record["record_id"]
            game = record["game_id"]
            if followers.get(game) and record["kind"] == "resolution":
                extras[record_id] = record
                followers[game] -= 1
            if record.get("link_id") in wanted_links and record_id not in extras:
                extras[record_id] = record
            if record_id in wanted_mirrors:
                extras[record_id] = record
            for name, test in categories.items():
                if len(chosen[name]) >= args.per_category or not test(record):
                    continue
                chosen[name].append(record)
                if record.get("link_id"):
                    wanted_links.add(record["link_id"])
                if record.get("mirror_of"):
                    wanted_mirrors.add(record["mirror_of"])
                if name == "trigger-fired":
                    followers[game] = _FOLLOWERS_PER_TRIGGER
                break
        if all(len(chosen[name]) >= args.per_category for name in categories):
            break
        if shard_no % 25 == 0:
            filled = sum(len(chosen[name]) >= args.per_category for name in categories)
            print(f"{shard_no} shards read, {filled}/{len(categories)} categories filled",
                  file=sys.stderr)

    records: dict[str, dict] = {}
    for name in categories:
        for record in chosen[name]:
            records[record["record_id"]] = record
    records.update(extras)
    missing = [name for name in categories if not chosen[name]]

    # Every record must parse with the current reader before it is a fixture.
    for record in records.values():
        record_from_dict(record)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output, "wt", encoding="utf-8") as out:
        for record in sorted(records.values(), key=lambda r: (r["game_id"], r["timestamp"])):
            out.write(json.dumps(record, separators=(",", ":")) + "\n")

    copied = 0
    for script_file in sorted({key["script_file"]
                               for record in records.values()
                               for key in _keys_in(record)}):
        source = sidecars.path(script_file)
        if source is None or not source.exists():
            continue
        tree, _, relative = script_file.partition("/")
        target = (args.sidecars_out / tree / relative).with_suffix(".provenance.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        text = source.with_name(source.name[: -len(".provenance.json")] + ".txt")
        if text.exists():
            shutil.copyfile(text, target.with_name(text.name))
        copied += 1

    for name in categories:
        print(f"{name:20s} {len(chosen[name])}", file=sys.stderr)
    print(f"{len(records)} records ({len(extras)} joined) -> {args.output}; "
          f"{copied} sidecars -> {args.sidecars_out}", file=sys.stderr)
    if missing:
        print(f"no record found for: {', '.join(missing)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

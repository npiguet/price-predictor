"""Repair the first-strike-step combat records of a built gen-2 corpus.

    python -u scripts/repair_first_strike_steps.py --corpus output/effects/corpus-gen2/ [--dry-run]

Gen-2's collector decided "this is combat damage" by the phase name containing
``COMBAT_DAMAGE``, which Forge's first-strike step, ``COMBAT_FIRST_STRIKE_DAMAGE``,
does not. In that step only damage to a player opened a combat record, through
a path that reads the event's own combat flag; damage to creatures, the life
changes and the deaths went to the ordinary bracket and reached no record. A
first-strike-step record therefore holds its player damage and nothing it
caused: the defending player's life never moves, and a first striker that
kills its blocker does so in no record at all.

The raw shards are left alone — gen-2 is not collected again — and this script
repairs the built corpus instead, in place. Per first-strike-step combat record
that is not a probe fork:

- **A blocked fight in which either side strikes first drops the record.** Its
  creature damage and deaths are gone, and the snapshot may already carry part
  of them, so no repair is exact.
- **Otherwise the missing player outcomes are added**, as the collector writes
  them in the regular step: one ``life_change`` per player for the damage it
  took, a positive ``life_change`` for the controller of each lifelinker, and
  a ``poison_change`` instead of life loss for damage from an infect creature.
  Keywords are resolved as gate 2 resolves them, printed and granted alike.

A record that already carries a life or poison change is left as it is, so a
second run changes nothing. The counts go to ``repairs.json`` beside the
manifest; the manifest and its digest are untouched, because the build's
decisions are.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

REPAIR_KEY = "first_strike_step_outcomes"
_STRIKES_FIRST = frozenset({"first_strike", "double_strike"})
_JSON_SEPARATORS = (",", ":")


def _event(kind: str, subject: str, delta: int, cause: str | None = None) -> dict:
    """An outcome event in the collector's own shape and key order."""
    params: dict = {"delta": delta}
    if cause is not None:
        params["cause"] = cause
    return {
        "type": kind, "subjects": [subject], "params": params,
        "duration": "instant", "attributed_to": "unresolved",
    }


def is_first_strike_step(raw: dict) -> bool:
    """A real combat record of the first-strike damage step."""
    return (
        raw.get("kind") == "combat"
        and (raw.get("state", {}).get("global") or {}).get("combat_substep")
        == "first_strike"
        and not (raw.get("payload") or {}).get("probed_keyword")
    )


def repair(raw: dict, keywords_of) -> tuple[str, dict | None]:
    """``(outcome, record)`` for one raw record dict.

    ``outcome`` is ``kept`` (not a first-strike-step record, or already
    carrying its outcomes), ``repaired`` (outcomes added; ``record`` is the
    new dict) or ``dropped`` (``record`` is None). ``keywords_of`` maps a
    record dict and an entity id to that entity's keywords.
    """
    if not is_first_strike_step(raw):
        return "kept", raw
    payload = raw.get("payload") or {}
    events = payload.get("events") or []
    if any(e.get("type") in ("life_change", "poison_change") for e in events):
        return "kept", raw

    for attacker, blockers in (payload.get("blocks") or {}).items():
        if blockers and any(
            _STRIKES_FIRST & keywords_of(raw, entity)
            for entity in (attacker, *blockers)
        ):
            return "dropped", None

    state = raw.get("state") or {}
    players = {p["id"] for p in state.get("players", [])}
    controller = {e["id"]: e.get("controller") for e in state.get("entities", [])}
    loss: dict[str, int] = defaultdict(int)
    gain: dict[str, int] = defaultdict(int)
    poison: dict[str, int] = defaultdict(int)
    poison_cause: dict[str, str | None] = {}
    for event in events:
        subjects = event.get("subjects") or []
        params = event.get("params") or {}
        if (
            event.get("type") != "damage_dealt" or not subjects
            or subjects[0] not in players or not params.get("combat")
        ):
            continue
        player, amount, cause = subjects[0], int(params.get("amount", 0)), params.get("cause")
        keywords = keywords_of(raw, cause) if cause else frozenset()
        if "infect" in keywords:
            poison[player] += amount
            poison_cause[player] = controller.get(cause)
        else:
            loss[player] += amount
        if "lifelink" in keywords and controller.get(cause):
            gain[controller[cause]] += amount

    added = [_event("life_change", p, gain[p]) for p in sorted(gain) if gain[p]]
    added += [_event("life_change", p, -loss[p]) for p in sorted(loss) if loss[p]]
    added += [
        _event("poison_change", p, poison[p], poison_cause.get(p))
        for p in sorted(poison) if poison[p]
    ]
    if not added:
        return "kept", raw
    repaired = dict(raw)
    repaired["payload"] = {**payload, "events": [*events, *added]}
    return "repaired", repaired


class ResolvedKeywords:
    """``(raw record, entity id) -> keywords``, through gate 2's resolver.

    The record is parsed once per record id, and the resolver memoizes each
    provenance key, so a shard costs a parse per first-strike record.
    """

    def __init__(self, resolver) -> None:
        from effects.infrastructure.record_codec import record_from_dict

        self._resolver = resolver
        self._parse = record_from_dict
        self._last: tuple[str | None, dict] = (None, {})

    def __call__(self, raw: dict, entity_id: str) -> frozenset[str]:
        record_id = raw.get("record_id")
        if self._last[0] != record_id:
            record = self._parse(raw)
            self._last = (record_id, {e.id: e for e in record.state.entities})
        entity = self._last[1].get(entity_id)
        if entity is None:
            return frozenset()
        return frozenset(self._resolver.keywords_of(entity))


def repair_shard(path: Path, keywords_of, *, write: bool) -> Counter:
    """Repair one shard; rewrite it only when something changed."""
    from effects.infrastructure.record_codec import parse_record_line
    from effects.infrastructure.shard_io import iter_shard_lines

    counts: Counter = Counter()
    lines: list[str] = []
    for text in iter_shard_lines(path):
        raw = parse_record_line(text)
        outcome, record = repair(raw, keywords_of)
        counts[outcome] += is_first_strike_step(raw)
        if outcome == "kept":
            lines.append(text)
        elif outcome == "repaired":
            lines.append(json.dumps(record, separators=_JSON_SEPARATORS))
    counts["records"] = len(lines) + counts["dropped"]
    if write and (counts["repaired"] or counts["dropped"]):
        temporary = path.with_name(path.name + ".repairing")
        with gzip.open(temporary, "wt", encoding="utf-8") as handle:
            for text in lines:
                handle.write(text)
                handle.write("\n")
        os.replace(temporary, path)
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--cards-folder", type=Path, action="append",
                        default=None)
    parser.add_argument("--variant-scripts", type=Path,
                        default=Path("output/effects/variant-scripts"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    from effects.domain.damage_step_keywords import KeywordResolver
    from effects.infrastructure.sidecar_io import SidecarCache

    folders = args.cards_folder or [
        Path("output/cardsfolder"), Path("output/tokenscripts"),
    ]
    keywords_of = ResolvedKeywords(KeywordResolver(
        SidecarCache.from_folders(folders, args.variant_scripts),
    ))
    shards = sorted(args.corpus.rglob("*.jsonl.gz"))
    print(f"{len(shards)} shards under {args.corpus}", flush=True)
    total: Counter = Counter()
    by_part: dict[str, Counter] = defaultdict(Counter)
    for index, shard in enumerate(shards, 1):
        counts = repair_shard(shard, keywords_of, write=not args.dry_run)
        total.update(counts)
        # training, validation/card-disjoint, validation/samples, ...
        by_part[shard.parent.relative_to(args.corpus).as_posix()].update(counts)
        if index % 100 == 0 or index == len(shards):
            print(f"{index}/{len(shards)} shards: {dict(total)}", flush=True)

    if not args.dry_run:
        path = args.corpus / "repairs.json"
        repairs = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        runs = repairs.setdefault(REPAIR_KEY, [])
        runs.append({
            "date": datetime.now(UTC).isoformat(timespec="seconds"),
            "records_read": total["records"],
            "repaired": total["repaired"],
            "dropped": total["dropped"],
            "already_complete": total["kept"],
            "by_part": {
                part: {"records_read": c["records"], "repaired": c["repaired"],
                       "dropped": c["dropped"]}
                for part, c in sorted(by_part.items())
            },
        })
        path.write_text(json.dumps(repairs, indent=2) + "\n", encoding="utf-8")
        print(f"recorded in {path}", flush=True)
    for part, counts in sorted(by_part.items()):
        print(f"  {part}: {dict(counts)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

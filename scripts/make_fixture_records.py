"""Cut a small, kind-balanced sample of real records into a test fixture.

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

    python scripts/make_fixture_records.py                 # gen-1
    python scripts/make_fixture_records.py --generation 2  # gen-2

``--generation 2`` adds the categories the rule detectors are tested on: a
resolution whose effect half kills a creature at toughness 0 or less in the
state-based check (704.5f fires), one that leaves a creature at toughness 1
(704.5f does not fire), a creature killed by damage (704.5g), the legend rule
(704.5j), a planeswalker at loyalty 0 (704.5i) and an Aura whose host left
(704.5m), a damage-step probe stripping deathtouch, triggers a keyword
generates (fired and not), a predefined token
kind made (111.10) and an ability whose source is one, and records from a
random-seat game and from a game the Forge AI played on both seats, and the
shapes the detector families' parameters read: a resolution and a decision
candidate acting through a keyword's own line, a candidate whose card prints a
keyword, a candidate whose cost has a part (``Sac<...>``), a candidate listing
targets, a damage rewrite a prevention shield replaced, a creature leaving the
battlefield in combat, a token's static and a counter-conditioned static.
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

from effects.domain.script_text import params as params_of
from effects.domain.script_text import root as root_of
from effects.infrastructure.record_codec import record_from_dict
from effects.infrastructure.shard_io import iter_shard_lines, iter_shards

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

    def line(self, key: dict) -> dict | None:
        """The raw sidecar line a key names, or None."""
        sidecar = self.get(key["script_file"])
        if sidecar is None:
            return None
        for line in sidecar["lines"]:
            for candidate in line["provenance"]:
                if (candidate["face"], candidate["trait_kind"],
                        candidate["index_within_kind"]) == (
                        key["face"], key["trait_kind"], key["index_within_kind"]):
                    return line
        return None

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


def _toughness(entity: dict) -> int | None:
    pt = entity.get("pt")
    if not pt:
        return None
    return sum(part[1] for part in (pt["base"], pt["boosts"], pt["counters"]))


def _toughness_after(record: dict) -> dict[str, tuple[int, bool, bool]]:
    """``entity id -> (toughness after the record's own P/T changes, damaged,
    died in the state-based check)`` for each creature the record names."""
    events = record["payload"].get("events") or []
    entities = {e["id"]: e for e in record["state"]["entities"]}
    delta: dict[str, int] = {}
    damaged: set[str] = set()
    died: set[str] = set()
    uncreatured: set[str] = set()
    for event in events:
        params = event.get("params") or {}
        for subject in event.get("subjects") or ():
            if event["type"] == "type_change" and "creature" in (params.get("types_removed") or ()):
                # No longer a creature, so a death is not 704.5f's (an Aura
                # with nothing to enchant is 704.5m).
                uncreatured.add(subject)
            elif event["type"] == "pt_change":
                delta[subject] = delta.get(subject, 0) + int(params.get("toughness_delta", 0))
            elif event["type"] == "damage_dealt":
                damaged.add(subject)
            elif (event["type"] == "zone_change" and event.get("attributed_to") == "unresolved"
                  and params.get("from_zone") == "battlefield"
                  and params.get("to_zone") == "graveyard"):
                died.add(subject)
    out = {}
    for entity_id in set(delta) | died:
        entity = entities.get(entity_id)
        if (entity is None or entity_id in uncreatured
                or "creature" not in (entity.get("types") or ())):
            continue
        toughness = _toughness(entity)
        if toughness is not None:
            out[entity_id] = (toughness + delta.get(entity_id, 0),
                              entity_id in damaged, entity_id in died)
    return out


def _effect_resolution(record: dict) -> bool:
    return (record["kind"] == "resolution" and record.get("moment") == "resolution"
            and not record["fork"])


def _toughness_zero_death(record: dict) -> bool:
    """704.5f fires: a creature dies in the check at toughness 0 or less, undamaged."""
    return _effect_resolution(record) and any(
        toughness <= 0 and died and not damaged
        for toughness, damaged, died in _toughness_after(record).values())


def _left_at_toughness_one(record: dict) -> bool:
    """704.5f does not fire: a creature's toughness is lowered to 1 and it stays."""
    if not _effect_resolution(record):
        return False
    lowered = {
        subject for event in record["payload"].get("events") or ()
        if event["type"] == "pt_change"
        and int((event.get("params") or {}).get("toughness_delta", 0)) < 0
        for subject in event.get("subjects") or ()}
    return any(entity_id in lowered and toughness == 1 and not died
               for entity_id, (toughness, _, died) in _toughness_after(record).items())


def _lethal_damage_death(record: dict) -> bool:
    """704.5g fires: a creature dealt damage by the resolution dies in the check."""
    return _effect_resolution(record) and any(
        damaged and died and toughness > 0
        for toughness, damaged, died in _toughness_after(record).values())


def _sba_fires(check: str) -> Callable[[dict], bool]:
    """A death the ``sba_zone_move`` detector puts on the rule's fires side and
    sees happen: the detector itself picks the record, so the fixture holds
    what the family test reads. Only effect halves with an unresolved
    battlefield death are parsed, which keeps the scan cheap."""
    from effects.domain.rule_catalogue import Polarity
    from effects.domain.rule_detectors.sba_zone_move import FAMILY

    def test(record: dict) -> bool:
        if not _effect_resolution(record) or not any(
                e["type"] == "zone_change" and e.get("attributed_to") == "unresolved"
                for e in record["payload"].get("events") or ()):
            return False
        found = FAMILY.match(record_from_dict(record), None, {"check": check})
        return found is not None and found.polarity is Polarity.FIRES and found.observed
    return test


#: Rule 111.10's predefined token kinds the fixture looks for.
_PREDEFINED_TOKENS = frozenset({"treasure", "food", "clue", "blood", "map", "gold",
                                "powerstone", "shard", "incubator", "junk"})


def _keyword_trigger(fired: bool) -> Callable[[dict], bool]:
    """A trigger a keyword generates: its acting key is the keyword's trait."""
    def test(record: dict) -> bool:
        keys = record.get("ability") or ()
        return (record["kind"] == "trigger" and bool(record["payload"].get("fired")) == fired
                and bool(keys) and keys[0]["trait_kind"] == "keyword")
    return test


def _subtypes(type_line: str) -> set[str]:
    _, dash, rest = type_line.partition(" - ")
    return {word.lower() for word in rest.split()} if dash else set()


def _predefined_token_made(record: dict) -> bool:
    return _effect_resolution(record) and any(
        e["type"] == "token_created" and _subtypes(str(
            ((e.get("params") or {}).get("characteristics") or {}).get("types") or ""))
        & _PREDEFINED_TOKENS
        for e in record["payload"].get("events") or ())


def _predefined_token_source(record: dict) -> bool:
    if not _effect_resolution(record):
        return False
    source = (record["state"].get("refs") or {}).get("source")
    return any(e["id"] == source and e.get("token_script_id")
               and {s.lower() for s in e.get("subtypes") or ()} & _PREDEFINED_TOKENS
               for e in record["state"]["entities"])


def _candidate_key(record: dict) -> dict | None:
    """A decision record's one candidate's first key."""
    if record["kind"] != "playability" or record.get("subkind") != "decision":
        return None
    candidates = record["payload"].get("candidates") or ()
    keys = (candidates[0].get("ability") or ()) if candidates else ()
    return keys[0] if keys else None


def _combatant_leaves(record: dict) -> bool:
    """An effect half moving a creature that is attacking or blocking off the battlefield."""
    if not _effect_resolution(record):
        return False
    entities = {e["id"]: e for e in record["state"]["entities"]}
    for event in record["payload"].get("events") or ():
        if event["type"] != "zone_change" or (
                (event.get("params") or {}).get("from_zone") != "battlefield"):
            continue
        for subject in event.get("subjects") or ():
            combat = (entities.get(subject) or {}).get("combat") or {}
            if combat.get("attacking") or combat.get("blocking"):
                return True
    return False


def _detector_shapes(sidecars: _Sidecars) -> dict[str, Callable[[dict], bool]]:
    """Records the detector families' parameters read: a keyword's own line
    acting, a candidate's card printing a keyword, a cost part, targets, a
    prevention shield, a combatant leaving, a token's or a counter-conditioned
    static."""
    from effects.domain.rule_detectors.continuous_contribution import (
        condition_counter_types,
    )
    from effects.domain.rule_detectors.rewrite_event import PREVENTS

    def first_key(record: dict) -> dict | None:
        keys = record.get("ability") or ()
        return keys[0] if keys else None

    def keyword_resolution(record: dict) -> bool:
        key = first_key(record)
        return (_effect_resolution(record) and key is not None
                and key["trait_kind"] == "keyword"
                and bool(record["payload"].get("events")))

    def keyword_decision(record: dict) -> bool:
        key = _candidate_key(record)
        return key is not None and key["trait_kind"] == "keyword"

    def keyword_card_decision(record: dict) -> bool:
        key = _candidate_key(record)
        return (key is not None and key["trait_kind"] == "spell"
                and sidecars.prints_keyword(key["script_file"]))

    def cost_part_decision(record: dict) -> bool:
        key = _candidate_key(record)
        return key is not None and "<" in (
            params_of(root_of(sidecars.script(key))).get("Cost") or "")

    def targeting_decision(record: dict) -> bool:
        if _candidate_key(record) is None:
            return False
        targets = record["payload"]["candidates"][0].get("legal_targets") or ()
        return 0 < len(targets) < 32

    def prevention_rewrite(record: dict) -> bool:
        key = first_key(record)
        return (record["kind"] == "rewrite" and key is not None
                and record["payload"]["incoming"]["type"] == "damage_dealt"
                and record["payload"].get("result") in ("replaced", "updated")
                and bool(PREVENTS.search(sidecars.script(key))))

    def token_static(record: dict) -> bool:
        key = first_key(record)
        return (record["kind"] == "continuous" and key is not None
                and key["script_file"].startswith("tokenscripts/"))

    def counter_conditioned_static(record: dict) -> bool:
        key = first_key(record)
        return (record["kind"] == "continuous" and key is not None
                and bool(condition_counter_types(sidecars.script(key))))

    return {
        "keyword-resolution": keyword_resolution,
        "keyword-decision": keyword_decision,
        "keyword-card-decision": keyword_card_decision,
        "cost-part-decision": cost_part_decision,
        "targeting-decision": targeting_decision,
        "prevention-rewrite": prevention_rewrite,
        "combatant-leaves": _combatant_leaves,
        "token-static": token_static,
        "counter-conditioned-static": counter_conditioned_static,
    }


def _gen2_categories(sidecars: _Sidecars) -> dict[str, Callable[[dict], bool]]:
    """Gen-2's own categories, matched before the shared ones."""
    return {
        **_detector_shapes(sidecars),
        "704.5j-fires": _sba_fires("legend_rule"),
        "704.5m-fires": _sba_fires("aura_illegal"),
        "704.5i-fires": _sba_fires("loyalty_zero"),
        "704.5f-fires": _toughness_zero_death,
        "704.5g-fires": _lethal_damage_death,
        "704.5f-does-not-fire": _left_at_toughness_one,
        "keyword-trigger-fired": _keyword_trigger(True),
        "keyword-trigger-unfired": _keyword_trigger(False),
        "predefined-token-source": _predefined_token_source,
        "predefined-token-made": _predefined_token_made,
        "probe-deathtouch": lambda r: r["kind"] == "combat" and (
            r["payload"].get("probed_keyword") == "deathtouch"),
        "random-seat": lambda r: r.get("random_seat") is True,
    }


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--generation", type=int, choices=(1, 2), default=1)
    parser.add_argument("--records-dir", type=Path,
                        help="default output/effects/records, or records-gen2 for --generation 2")
    parser.add_argument("--cards-folder", action="append", type=Path,
                        help="converted roots, named by tree (repeatable); "
                             "default output/cardsfolder and output/tokenscripts")
    parser.add_argument("--output", type=Path,
                        help="default tests/fixtures/effects/gen<N>-records.jsonl.gz")
    parser.add_argument("--sidecars-out", type=Path,
                        help="default tests/fixtures/effects/gen<N>-sidecars")
    parser.add_argument("--per-category", type=int, default=3)
    parser.add_argument("--max-shards", type=int, default=400)
    args = parser.parse_args()
    gen = args.generation
    args.records_dir = args.records_dir or Path(
        "output/effects/records" if gen == 1 else "output/effects/records-gen2")
    args.output = args.output or Path(f"tests/fixtures/effects/gen{gen}-records.jsonl.gz")
    args.sidecars_out = args.sidecars_out or Path(f"tests/fixtures/effects/gen{gen}-sidecars")

    folders = args.cards_folder or [Path("output/cardsfolder"), Path("output/tokenscripts")]
    sidecars = _Sidecars({folder.name: folder for folder in folders})
    categories = _categories(sidecars)
    if gen == 2:
        categories = {**_gen2_categories(sidecars), **categories}
    chosen: dict[str, list[dict]] = defaultdict(list)
    wanted_links: set[str] = set()
    wanted_mirrors: set[str] = set()
    extras: dict[str, dict] = {}

    shards = iter_shards(args.records_dir)
    # Interleave the collection runs rather than reading one run's shards first,
    # so a category only another run reaches (probes, variants) is found early.
    shards = shards[::7] + [s for i, s in enumerate(shards) if i % 7]
    forge_ai_wanted = gen == 2
    for shard_no, shard in enumerate(shards[: args.max_shards]):
        followers: dict[str, int] = {}
        seat_games: set[str] = set()
        plain: dict[str, list[dict]] = defaultdict(list)
        for line in iter_shard_lines(shard):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            record_id = record["record_id"]
            game = record["game_id"]
            if record.get("random_seat"):
                seat_games.add(game)
            elif forge_ai_wanted and not record["fork"] and len(plain[game]) < 2:
                plain[game].append(record)
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
        if forge_ai_wanted:
            # Games never span shards, so a game is known to be Forge AI on
            # both seats once its shard has been read through.
            for game, kept in plain.items():
                if game not in seat_games and len(chosen["forge-ai"]) < args.per_category:
                    chosen["forge-ai"].append(kept[0])
        if all(len(chosen[name]) >= args.per_category for name in categories) and (
                not forge_ai_wanted or len(chosen["forge-ai"]) >= args.per_category):
            break
        if shard_no % 25 == 0:
            filled = sum(len(chosen[name]) >= args.per_category for name in categories)
            print(f"{shard_no} shards read, {filled}/{len(categories)} categories filled",
                  file=sys.stderr)

    records: dict[str, dict] = {}
    for name in chosen:
        for record in chosen[name]:
            records[record["record_id"]] = record
    records.update(extras)
    missing = [name for name in [*categories, *(["forge-ai"] if forge_ai_wanted else [])]
               if not chosen[name]]

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

    for name in chosen:
        print(f"{name:22s} {len(chosen[name])}", file=sys.stderr)
    print(f"{len(records)} records ({len(extras)} joined) -> {args.output}; "
          f"{copied} sidecars -> {args.sidecars_out}", file=sys.stderr)
    if missing:
        print(f"no record found for: {', '.join(missing)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

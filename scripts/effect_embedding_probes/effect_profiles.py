"""Aggregate what each ability text was observed to do in the training corpus.

Streams every shard of the checkpoint's curated training corpus (its
manifest's ``training/``) and accumulates, per
acting ability text, an **effect profile**: how often the text resolved, and
across its effect-half resolutions the share that had each kind of outcome
(a creature died, a card was drawn, a token was made, ...), the mean amounts
(life, damage, counters, power), and which side of the table the outcomes
landed on. Cost halves contribute the observed cost (mana, tap, sacrifice,
life), static abilities their continuous contributions, triggers their fired
share.

The acting text follows ``effects.application.gate_one.acting_text``: the
first key of ``record.ability`` that resolves to a sidecar line, encoded on
the checkpoint's surface. The key → text map comes from ``cache/keymap.pkl``
(written by ``build_texts.py``), so no worker builds a ``SidecarCache``.
Records are parsed as raw JSON and never held: each worker folds one shard
into one row of sums per record and returns them.

Each record is profiled **once**. Curation writes a training record up to
``--reuse-cap`` times, as separate lines with one ``record_id``, so the
profile, the probes' ``n/(n+5)`` weights and the kind shares would otherwise
count a repeated record as several observations. The copies can land in
different shards, so the rows are de-duplicated where they are merged.

"Own side" means the acting player or a permanent they control in the
record's pre-resolution snapshot; "opponent side" the other player or their
permanents. A subject the snapshot does not hold (a token created by the
resolution, a card in a hidden zone) counts toward neither.

Run: ``python scripts/effect_embedding_probes/effect_profiles.py --checkpoint PATH
--abilities-root DIR [--workers 8]``
(CPU, a few minutes). Writes ``effect_profiles.csv`` to the report directory.
"""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import add_probe_arguments, resolve_paths  # noqa: E402

_KEYMAP: dict | None = None

#: Event types whose per-record presence becomes a ``has_<type>`` share.
PRESENCE_TYPES = (
    "zone_change", "destroyed", "sacrificed", "token_created",
    "permanent_copied", "life_change", "damage_dealt", "damage_prevented",
    "energy_change", "poison_change", "mana_produced", "card_drawn",
    "card_discarded", "card_milled", "card_revealed", "card_looked_at",
    "library_shuffled", "counter_change", "pt_change", "keyword_change",
    "type_change", "color_change", "ability_change", "control_change",
    "attached", "tapped", "untapped", "spell_countered",
    "continuous_effect_created", "delayed_trigger_created",
)

#: Zone transitions, ``(from, to)``, and the name of the share they feed.
TRANSITIONS = {
    ("battlefield", "graveyard"): "bf_to_gy",
    ("battlefield", "exile"): "bf_to_exile",
    ("battlefield", "hand"): "bf_to_hand",
    ("battlefield", "library"): "bf_to_library",
    ("library", "hand"): "lib_to_hand",
    ("library", "battlefield"): "lib_to_bf",
    ("graveyard", "hand"): "gy_to_hand",
    ("graveyard", "battlefield"): "gy_to_bf",
    ("hand", "battlefield"): "hand_to_bf",
    ("stack", "graveyard"): "stack_to_gy",
    ("exile", "battlefield"): "exile_to_bf",
}


def _init(keymap_path: str) -> None:
    global _KEYMAP
    with open(keymap_path, "rb") as handle:
        _KEYMAP = pickle.load(handle)


def acting_text(record: dict) -> str | None:
    for key in record.get("ability") or ():
        text = _KEYMAP.get((key["script_file"], key["face"], key["trait_kind"],
                            key["index_within_kind"], key.get("option")))
        if text is not None:
            return text
    return None


def _side(subject: str, actor: str, entities: dict) -> str | None:
    if subject.startswith("P"):
        return "own" if subject == actor else "opp"
    entity = entities.get(subject)
    if entity is None or entity.get("controller") is None:
        return None
    return "own" if entity["controller"] == actor else "opp"


def _resolution(acc: dict, record: dict) -> None:
    actor = record.get("actor_player")
    entities = {e["id"]: e for e in record["state"].get("entities", ())}
    events = record["payload"].get("events") or ()
    acc["res_n"] += 1
    if not events:
        acc["res_empty"] += 1
    present: set[str] = set()
    sides = {"own": 0, "opp": 0}
    subjects: set[str] = set()
    sums = defaultdict(float)
    for event in events:
        kind = event["type"]
        present.add(kind)
        params = event.get("params") or {}
        for subject in event.get("subjects") or ():
            subjects.add(subject)
            side = _side(subject, actor, entities)
            if side is not None:
                sides[side] += 1
        subs = event.get("subjects") or ()
        first = subs[0] if subs else ""
        side = _side(first, actor, entities) if first else None
        entity = entities.get(first, {})
        is_creature = "creature" in (entity.get("types") or ())
        if kind == "zone_change":
            origin = (params.get("from_zone") or entity.get("zone") or "").lower()
            dest = (params.get("to_zone") or "").lower()
            name = TRANSITIONS.get((origin, dest))
            if name:
                present.add(name)
            if origin == "battlefield" and dest in ("graveyard", "exile") and is_creature:
                present.add("creature_removed")
                present.add(f"creature_removed_{side}" if side else "creature_removed")
            if origin == "battlefield" and dest == "hand" and is_creature:
                present.add("creature_bounced")
        elif kind == "destroyed" and is_creature:
            present.add("creature_removed")
            if side:
                present.add(f"creature_removed_{side}")
        elif kind == "life_change" and side:
            sums[f"life_{side}"] += float(params.get("delta") or 0)
        elif kind == "damage_dealt":
            amount = float(params.get("amount") or 0)
            target = "player" if first.startswith("P") else "permanent"
            sums["damage_total"] += amount
            sums[f"damage_to_{target}"] += amount
            if side:
                sums[f"damage_{side}"] += amount
        elif kind in ("card_drawn", "card_discarded", "card_milled") and side:
            sums[f"{kind}_{side}"] += float(params.get("count") or 1)
        elif kind == "counter_change":
            counter = str(params.get("counter_type") or "")
            family = {"P1P1": "p1p1", "M1M1": "m1m1", "LOYALTY": "loyalty"}.get(
                counter, "other")
            delta = float(params.get("delta") or 0)
            sums[f"counters_{family}"] += delta
            if side and family == "p1p1":
                sums[f"counters_p1p1_{side}"] += delta
        elif kind == "pt_change":
            power = float(params.get("power_delta") or 0)
            sums["power_delta"] += power
            sums["toughness_delta"] += float(params.get("toughness_delta") or 0)
            if side:
                sums[f"power_delta_{side}"] += power
        elif kind == "token_created":
            sums["tokens"] += float(params.get("count") or 1)
            chars = params.get("characteristics") or {}
            if "creature" in str(chars.get("types", "")).lower():
                sums["creature_tokens"] += float(params.get("count") or 1)
        elif kind == "mana_produced":
            sums["mana"] += float(sum((params.get("mana_by_color") or {}).values()))
        elif kind == "energy_change":
            sums["energy"] += float(params.get("delta") or 0)
        elif kind in ("keyword_change", "tapped", "untapped", "control_change") and side:
            present.add(f"{kind}_{side}")
    for name in present:
        acc[f"has_{name}"] += 1
    for name, value in sums.items():
        acc[f"sum_{name}"] += value
    acc["sum_events"] += len(events)
    acc["sum_subjects"] += len(subjects)
    acc["sum_side_own"] += sides["own"]
    acc["sum_side_opp"] += sides["opp"]


def _activation(acc: dict, record: dict) -> None:
    costs = record["payload"].get("costs") or {}
    acc["cost_n"] += 1
    acc["sum_cost_mana"] += float(sum((costs.get("mana_by_color") or {}).values()))
    acc["sum_cost_life"] += float(costs.get("life") or 0)
    for field in ("tapped", "sacrificed", "discarded", "exiled"):
        if costs.get(field):
            acc[f"has_cost_{field}"] += 1


def _continuous(acc: dict, record: dict) -> None:
    actor = record.get("actor_player")
    entities = {e["id"]: e for e in record["state"].get("entities", ())}
    contributions = record["payload"].get("contributions") or ()
    acc["cont_n"] += 1
    acc["sum_cont_entities"] += len(contributions)
    for contribution in contributions:
        side = _side(contribution.get("entity") or "", actor, entities)
        boost = contribution.get("pt_boost") or [0, 0]
        acc["sum_cont_power"] += float(boost[0] or 0)
        acc["sum_cont_toughness"] += float(boost[1] or 0)
        if side:
            acc[f"sum_cont_{side}"] += 1
        if contribution.get("keywords"):
            acc["sum_cont_keywords"] += 1
        if contribution.get("types") or contribution.get("colors"):
            acc["sum_cont_typecolour"] += 1


def fold_record(acc: dict, record: dict) -> None:
    """One record's contribution to its acting text's sums."""
    kind = record["kind"]
    acc["n_records"] += 1
    if kind == "resolution":
        if record.get("moment") == "activation":
            acc["n_cost"] += 1
            _activation(acc, record)
        else:
            acc["n_effect"] += 1
            _resolution(acc, record)
    elif kind == "continuous":
        acc["n_continuous"] += 1
        _continuous(acc, record)
    elif kind == "trigger":
        acc["n_trigger"] += 1
        if (record["payload"] or {}).get("fired"):
            acc["trigger_fired"] += 1
    else:
        acc[f"n_{kind}"] += 1


def fold_shard(path: str) -> list[tuple[str, str, str, dict]]:
    """``(record_id, acting text, game_id, sums)`` for each record of one shard."""
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for raw in handle:
            try:
                record = json.loads(raw)
            except json.JSONDecodeError:
                break
            text = acting_text(record)
            if text is None:
                continue
            acc: dict = defaultdict(float)
            fold_record(acc, record)
            rows.append((record["record_id"], text, record["game_id"], dict(acc)))
    return rows


class ProfileTotals:
    """Each acting text's sums over its distinct records."""

    def __init__(self) -> None:
        self.sums: dict[str, dict] = defaultdict(lambda: defaultdict(float))
        self.games: dict[str, set] = defaultdict(set)
        self._seen: set[str] = set()

    def add(self, record_id: str, text: str, game_id: str, acc: dict) -> None:
        """Fold one record's sums in; a copy of one already folded adds nothing."""
        if record_id in self._seen:
            return
        self._seen.add(record_id)
        target = self.sums[text]
        for key, value in acc.items():
            target[key] += value
        self.games[text].add(game_id)

    def frame(self) -> pd.DataFrame:
        """One row per text: its game count, its sums, and the shares :func:`derive` reads."""
        rows = []
        for text, acc in self.sums.items():
            row = {"text": text, "n_games": len(self.games[text])}
            row.update(acc)
            rows.append(row)
        return derive(pd.DataFrame(rows).fillna(0.0))


def main() -> None:
    parser = add_probe_arguments(
        argparse.ArgumentParser(description=__doc__.splitlines()[0]),
    )
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    paths = resolve_paths(args)
    if paths.training is None:
        raise SystemExit(
            f"{paths.checkpoint} names no curated corpus, so there are no "
            "training shards to profile"
        )
    shards = sorted(str(p) for p in paths.training.glob("*.jsonl*"))
    keymap = paths.reusable("cache/keymap.pkl")
    totals = ProfileTotals()
    with Pool(args.workers, initializer=_init,
              initargs=(str(keymap),)) as pool:
        for done, rows in enumerate(pool.imap_unordered(fold_shard, shards), 1):
            for row in rows:
                totals.add(*row)
            if done % 50 == 0:
                print(f"{done}/{len(shards)} shards, {len(totals.sums)} texts", flush=True)
    frame = totals.frame()
    paths.profile_table.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(paths.profile_table, index=False)
    print(f"{len(frame)} acting texts, {int(frame.n_records.sum())} records "
          f"-> {paths.profile_table}")


def derive(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-record shares and means from the sums (NaN where undefined)."""
    res = frame["res_n"].replace(0, np.nan) if "res_n" in frame else np.nan
    for column in [c for c in frame.columns if c.startswith("has_") and
                   not c.startswith("has_cost_")]:
        frame[f"p_{column[4:]}"] = frame[column] / res
    for column in [c for c in frame.columns if c.startswith("sum_") and
                   not c.startswith(("sum_cost_", "sum_cont_"))]:
        frame[f"mean_{column[4:]}"] = frame[column] / res
    side_total = (frame.get("sum_side_own", 0) + frame.get("sum_side_opp", 0)).replace(0, np.nan)
    frame["share_opp_side"] = frame.get("sum_side_opp", 0) / side_total
    cost = frame["cost_n"].replace(0, np.nan) if "cost_n" in frame else np.nan
    for column in [c for c in frame.columns if c.startswith("has_cost_")]:
        frame[f"p_{column[4:]}"] = frame[column] / cost
    for column in ("sum_cost_mana", "sum_cost_life"):
        if column in frame:
            frame[f"mean_{column[4:]}"] = frame[column] / cost
    cont = frame["cont_n"].replace(0, np.nan) if "cont_n" in frame else np.nan
    for column in [c for c in frame.columns if c.startswith("sum_cont_")]:
        frame[f"mean_{column[4:]}"] = frame[column] / cont
    trig = frame["n_trigger"].replace(0, np.nan) if "n_trigger" in frame else np.nan
    if "trigger_fired" in frame:
        frame["p_trigger_fired"] = frame["trigger_fired"] / trig
    total = frame["n_records"]
    for kind in ("effect", "cost", "continuous", "trigger", "rewrite", "combat",
                 "playability"):
        column = f"n_{kind}"
        if column in frame:
            frame[f"kind_share_{kind}"] = frame[column] / total
    return frame


if __name__ == "__main__":
    main()

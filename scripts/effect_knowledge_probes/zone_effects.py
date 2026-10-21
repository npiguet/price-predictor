"""Predicted effects on zones other than the battlefield: draw, discard, mill, exile,
return to hand, return to play.

    python scripts/effect_knowledge_probes/zone_effects.py CHECKPOINT [--records 2000]
        [--shards 200] [--per-game 3] [--per-sweep 40]

The other probes read what an ability does to a permanent on the battlefield. This
one reads what it does to a player's hand, library and graveyard, and to cards that
sit in them. It takes real resolution records from the validation games whose
acting chain names a zone-moving API (``Draw``, ``Discard``, ``Mill``, ``ChangeZone``,
``ChangeZoneAll``, ``Dig``, ``Destroy``, ``DestroyAll``, ``Sacrifice``, ``SacrificeAll``;
the destroy and sacrifice APIs are there so exile has something to be told apart
from) and reports four things:

1. **Card counts at each player.** Every player slot of every chosen record is an
   item, labelled by the cards that player actually drew, discarded or milled. The
   model's answer is the affected gate times the field's predicted rate. AUC says
   whether the model knows a player is affected at all; on records where exactly
   one player was, the share where it predicts more for that player than for the
   other says whether it knows which.
2. **Where a moved card goes, by the zone it starts in.** Every card the
   resolution moved is an item, grouped by its zone before the move (battlefield,
   graveyard, hand, exile, stack) and the outcome it got. The model's answer is
   its zone-outcome distribution at that card's slot, times the affected gate as
   every count here is; the table gives the mean
   probability of the right outcome and how often it is the most probable outcome
   other than ``stayed``. The outcome vocabulary is the head's: a card put into a
   graveyard from anywhere is ``died``, and one put onto the battlefield is
   ``blinked``.
3. **The amount.** For ``Draw``, ``Discard`` and ``Mill`` chains with one literal
   ``NumCards$``, the acting text's number is set to 1 through 6 and re-encoded by
   the checkpoint's own encoder, and the predicted count at the affected player is
   read, as the damage sweep does for ``NumDmg$``.
4. **Who.** For those chains with one player target, where that player is the one
   affected, the target is handed to the other player. The rule moves the effect
   with it; the probe reads the predicted count at both players before and after.
"""
from __future__ import annotations

import argparse
import importlib.util
import re
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "effect_knowledge_probes.run", Path(__file__).resolve().parent / "run.py")
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)
common = run.sibling("common")
sweeps = run.sibling("sweeps")
auc = run.sibling("target_restrictions").auc

CORPUS = Path("output/effects/corpus-gen2")
ZONE_APIS = frozenset({
    "Draw", "Discard", "Mill", "ChangeZone", "ChangeZoneAll", "Dig",
    "Destroy", "DestroyAll", "Sacrifice", "SacrificeAll",
})
#: The player count field each card-count API writes.
COUNT_FIELDS = {"Draw": "cards_drawn", "Discard": "cards_discarded", "Mill": "cards_milled"}
AMOUNTS = tuple(range(1, 7))
ORIGIN_ZONES = ("battlefield", "graveyard", "hand", "exile", "stack", "library")
_API_RE = re.compile(r"(?:SP|AB|DB)\$\s*(\w+)")
_NUMCARDS_RE = re.compile(r"(NumCards\$\s*)(\d+)")


# ── the chain ───────────────────────────────────────────────────────────


def chain_apis(script_text: str | None) -> set[str]:
    """Every API a chain's segments name, ``SP$``, ``AB$`` and ``DB$`` alike."""
    return set(_API_RE.findall(script_text or ""))


def set_amount(text: str, amount: int) -> str:
    """The acting text with its literal ``NumCards$`` set to ``amount``."""
    return _NUMCARDS_RE.sub(lambda m: f"{m.group(1)}{amount}", text)


def player_targets(record) -> list[str]:
    players = {p.id for p in record.state.players}
    return [t for t in record.state.refs.targets if t in players]


def retargeted(record, old: str, new: str):
    """``record`` with its player target ``old`` replaced by ``new``."""
    refs = record.state.refs
    targets = tuple(new if t == old else t for t in refs.targets)
    return replace(record, state=replace(record.state, refs=replace(refs, targets=targets)))


# ── which records qualify ───────────────────────────────────────────────


def chosen_record(record, sidecars):
    """``(record, apis, script text, targets)`` for a qualifying resolution, or None."""
    from effects.domain.effect_targets import derive_targets
    from effects.domain.records import Moment, RecordKind

    if (record.kind is not RecordKind.RESOLUTION or record.moment is not Moment.RESOLUTION
            or record.fork or not record.ability):
        return None
    try:
        line = sidecars.line_for(record.ability[0])
    except Exception:  # an unjoinable or unconverted key carries no text
        return None
    text = line.script_text if line is not None else None
    apis = chain_apis(text) & ZONE_APIS
    if not apis:
        return None
    return record, apis, text, derive_targets(record)


def qualifying_records(directories, count: int, shards: int, per_game: int, sidecars):
    from effects.infrastructure.record_io import iter_shards, read_shard

    chosen, per = [], defaultdict(int)
    for directory in directories:
        for shard in iter_shards(directory)[:shards]:
            for record in read_shard(shard):
                if per[record.game_id] >= per_game:
                    continue
                found = chosen_record(record, sidecars)
                if found is not None:
                    per[record.game_id] += 1
                    chosen.append(found)
                    if len(chosen) >= count:
                        return chosen
    return chosen


# ── reading the model ───────────────────────────────────────────────────


def _read(outputs, surface, row) -> dict:
    """``{"players": {id: {field: count}}, "cards": {id: outcome probabilities}}``.

    Both gated (:class:`HeadReading`): a count is the gate times the predicted
    rate, and each outcome's probability the gate times the class's, so the
    ``stayed``-or-unaffected mass is ``1 - sum`` of the moved outcomes.
    """
    from effects.domain.head_decoding import HeadReading

    players, cards = {}, {}
    for index, slot in enumerate(surface.slots):
        reading = HeadReading(outputs[row, index])
        if slot.kind.name == "PLAYER":
            players[slot.player_id] = {
                field: float(reading.expected(field)) for field in COUNT_FIELDS.values()
            }
        elif slot.kind.name == "CARD" and slot.entity_id is not None:
            cards[slot.entity_id] = reading.zone_probabilities().cpu().numpy()
    return {"players": players, "cards": cards}


def predict_all(probe_model, records, chunk: int = 32) -> list[dict]:
    out = []
    for i in range(0, len(records), chunk):
        out += sweeps._predict(probe_model, records[i:i + chunk],
                               lambda o, v, s, r: _read(o, s, r))
    return out


def _fmt(values, digits: int = 2) -> str:
    return f"{np.mean(values):.{digits}f}" if len(values) else "n/a"


# ── the four reports ────────────────────────────────────────────────────


def report_counts(chosen, readings) -> None:
    print("\ncard counts at each player: every player slot of every chosen record")
    print("| field | player items | affected | AUC | records naming the API | "
          "one player affected | more predicted for that player | actual, mean | "
          "predicted, mean |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for api, field in COUNT_FIELDS.items():
        labels, scores = [], []
        naming = single = right = 0
        actual, predicted = [], []
        for (record, apis, _, targets), reading in zip(chosen, readings):
            counts = {}
            for player in record.state.players:
                entry = targets.get(player.id)
                value = (entry.fields.get(field) or 0) if entry is not None else 0
                counts[player.id] = value
                if player.id in reading["players"]:
                    labels.append(value > 0)
                    scores.append(reading["players"][player.id][field])
            if api not in apis:
                continue
            naming += 1
            hit = [p for p, v in counts.items() if v > 0]
            if len(hit) == 1 and len(counts) == 2 and all(
                    p in reading["players"] for p in counts):
                single += 1
                other = next(p for p in counts if p != hit[0])
                right += (reading["players"][hit[0]][field]
                          > reading["players"][other][field])
                actual.append(counts[hit[0]])
                predicted.append(reading["players"][hit[0]][field])
        share = f"{right / single:.2f}" if single else "n/a"
        print(f"| {field} | {len(labels)} | {sum(labels)} | {auc(labels, scores):.3f} | "
              f"{naming} | {single} | {share} | {_fmt(actual)} | {_fmt(predicted)} |",
              flush=True)


def report_zones(chosen, readings) -> None:
    from effects.domain.effect_model import ZONE_OUTCOMES

    stayed = ZONE_OUTCOMES.index("stayed")
    rows = defaultdict(lambda: {"right": [], "top": [], "instead": defaultdict(int)})
    moved = defaultdict(lambda: ([], []))
    for (record, _, _, targets), reading in zip(chosen, readings):
        for entity in record.state.entities:
            if entity.id not in reading["cards"]:
                continue
            probs = reading["cards"][entity.id]
            entry = targets.get(entity.id)
            outcome = entry.fields.get("zone_outcome") if entry is not None else None
            went = outcome is not None and outcome != stayed
            moved[entity.zone][0].append(went)
            moved[entity.zone][1].append(float(probs.sum() - probs[stayed]))
            if not went:
                continue
            others = probs.copy()
            others[stayed] = -1.0
            top = int(np.argmax(others))
            row = rows[(entity.zone, ZONE_OUTCOMES[outcome])]
            row["right"].append(float(probs[outcome]))
            row["top"].append(top == outcome)
            if top != outcome:
                row["instead"][ZONE_OUTCOMES[top]] += 1

    print("\nwhether a card moves, by the zone it starts in: every card slot of every "
          "chosen record")
    print("| zone before | cards | moved | AUC |")
    print("|---|---:|---:|---:|")
    for zone in ORIGIN_ZONES:
        if zone in moved:
            labels, scores = moved[zone]
            print(f"| {zone} | {len(labels)} | {sum(labels)} | {auc(labels, scores):.3f} |")

    print("\nwhere a moved card goes, by the zone it starts in")
    print("| zone before | outcome | cards | P(outcome), mean | top outcome other than "
          "stayed is right | most often predicted instead |")
    print("|---|---|---:|---:|---:|---|")
    order = {z: i for i, z in enumerate(ORIGIN_ZONES)}
    for (zone, outcome), row in sorted(rows.items(),
                                       key=lambda kv: (order.get(kv[0][0], 99),
                                                       -len(kv[1]["right"]))):
        instead = max(row["instead"].items(), key=lambda kv: kv[1])[0] \
            if row["instead"] else ""
        print(f"| {zone} | {outcome} | {len(row['right'])} | {_fmt(row['right'])} | "
              f"{_fmt(row['top'])} | {instead} |", flush=True)


def _affected_player(record, targets, field):
    hit = [p.id for p in record.state.players
           if targets.get(p.id) is not None and (targets[p.id].fields.get(field) or 0) > 0]
    return hit[0] if len(hit) == 1 else None


def report_amounts(chosen, probe_model, per_sweep: int) -> None:
    print("\nthe amount: NumCards$ set to each value, predicted count at the affected "
          "player, mean over records")
    print("| API | records | " + " | ".join(str(a) for a in AMOUNTS) + " |")
    print("|---|---:|" + "---:|" * len(AMOUNTS))
    batcher = probe_model.batcher
    for api, field in COUNT_FIELDS.items():
        rows = []
        for record, apis, text, targets in chosen:
            if api not in apis or len(_NUMCARDS_RE.findall(text or "")) != 1:
                continue
            player = _affected_player(record, targets, field)
            if player is None:
                continue
            key = record.ability[0]
            original = batcher.text_of(key)
            if original is None or not _NUMCARDS_RE.search(original):
                continue
            curve = []
            for amount in AMOUNTS:
                with batcher.reading_as(key, set_amount(original, amount)):
                    reading = sweeps._predict(probe_model, [record],
                                              lambda o, v, s, r: _read(o, s, r))[0]
                curve.append(reading["players"].get(player, {}).get(field, float("nan")))
            rows.append(curve)
            if len(rows) >= per_sweep:
                break
        if rows:
            means = np.nanmean(np.asarray(rows), axis=0)
            print(f"| {api} | {len(rows)} | " + " | ".join(f"{m:.2f}" for m in means)
                  + " |", flush=True)
        else:
            print(f"| {api} | 0 |" + " n/a |" * len(AMOUNTS), flush=True)


def report_who(chosen, readings, probe_model) -> None:
    print("\nwho: the player target handed to the other player (rule: the count moves "
          "with it)")
    print("| API | records | targeted player, before | after | other player, before | "
          "after |")
    print("|---|---:|---:|---:|---:|---:|")
    for api, field in COUNT_FIELDS.items():
        edited, index = [], []
        for (record, apis, _, targets), reading in zip(chosen, readings):
            if api not in apis or len(record.state.players) != 2:
                continue
            aimed = player_targets(record)
            player = _affected_player(record, targets, field)
            if len(aimed) != 1 or player != aimed[0]:
                continue
            other = next(p.id for p in record.state.players if p.id != player)
            edited.append(retargeted(record, player, other))
            index.append((reading, player, other))
        if not edited:
            print(f"| {api} | 0 | n/a | n/a | n/a | n/a |", flush=True)
            continue
        after = predict_all(probe_model, edited)
        cells = defaultdict(list)
        for (before, player, other), reading in zip(index, after):
            cells["tb"].append(before["players"][player][field])
            cells["ta"].append(reading["players"][player][field])
            cells["ob"].append(before["players"][other][field])
            cells["oa"].append(reading["players"][other][field])
        print(f"| {api} | {len(edited)} | {_fmt(cells['tb'])} | {_fmt(cells['ta'])} | "
              f"{_fmt(cells['ob'])} | {_fmt(cells['oa'])} |", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--records", type=int, default=2000)
    parser.add_argument("--shards", type=int, default=200)
    parser.add_argument("--per-game", type=int, default=3)
    parser.add_argument("--per-sweep", type=int, default=40)
    args = parser.parse_args()

    sidecars = common.sidecar_cache(common.DEFAULT_CARDS_FOLDERS)
    directories = [CORPUS / "validation" / "card-disjoint",
                   CORPUS / "validation" / "game-disjoint"]
    chosen = qualifying_records(directories, args.records, args.shards, args.per_game,
                                sidecars)
    checkpoint = common.load_checkpoint(args.checkpoint)
    probe_model = common.load_probe_model(
        args.checkpoint, list(common.DEFAULT_CARDS_FOLDERS), [c[0] for c in chosen],
        checkpoint=checkpoint)
    print(f"{args.checkpoint.parent.name} ({checkpoint.epoch}), {len(chosen)} validation "
          f"resolutions naming a zone-moving API", flush=True)

    readings = predict_all(probe_model, [c[0] for c in chosen])
    report_counts(chosen, readings)
    report_zones(chosen, readings)
    report_amounts(chosen, probe_model, args.per_sweep)
    report_who(chosen, readings, probe_model)


if __name__ == "__main__":
    main()

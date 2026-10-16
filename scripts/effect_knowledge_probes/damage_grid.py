"""Predicted death of a damage spell's target, damage 1..8 x toughness 1..8, averaged over records.

    python scripts/effect_knowledge_probes/damage_grid.py CHECKPOINT [host]

Uses the frozen probe set's damage-sweep records (host "spell" by default), edits the
acting text's NumDmg$ through the batcher's text memo, as the damage sweep does, and the
target's toughness as the toughness sweep does.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "effect_knowledge_probes.run", Path(__file__).resolve().parent / "run.py")
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)
common = run.sibling("common")
sweeps = run.sibling("sweeps")

checkpoint_path = Path(sys.argv[1])
host = sys.argv[2] if len(sys.argv) > 2 else "spell"
corpus = Path("output/effects/corpus-gen2")
probe_set = json.load(open("output/effects/reports/knowledge-probes-set-b3adb69a2950.json",
                           encoding="utf-8"))
rows = [r for r in probe_set["sweeps"]["damage"] if r["host"] == host]
ids = {r["record_id"] for r in rows}
found = {}
for stratum, games in probe_set["games"].items():
    shards = [corpus / n for n in probe_set["sources"][stratum]]
    for _, rec in common.read_games(shards, set(games), ids):
        found[rec.record_id] = rec
records = [found[r["record_id"]] for r in rows]
checkpoint = common.load_checkpoint(checkpoint_path)
probe_model = common.load_probe_model(
    checkpoint_path, list(common.DEFAULT_CARDS_FOLDERS), records, checkpoint=checkpoint)
values = range(1, 9)
grid = np.zeros((8, 8))
batcher = probe_model.batcher
for row, record in zip(rows, records):
    key = record.ability[0]
    original = batcher._resolve(key)
    edited = [sweeps.set_toughness(record, row["entity"], t) for t in values]
    try:
        for i, amount in enumerate(values):
            batcher._text_by_key[key] = (sweeps.set_damage(original[0], amount), original[1])
            deaths = sweeps._predict(probe_model, edited,
                                     lambda o, v, s, r: sweeps._death(o, s, r, row["entity"]))
            grid[i] += deaths
    finally:
        batcher._text_by_key[key] = original
grid /= len(records)
print(f"{checkpoint_path.parent.name}, {len(records)} {host} records")
print("| damage \\ toughness | " + " | ".join(str(t) for t in values) + " |")
print("|---|" + "---:|" * 8)
for i, amount in enumerate(values):
    print(f"| {amount} | " + " | ".join(f"{grid[i, j]:.2f}" for j in range(8)) + " |")

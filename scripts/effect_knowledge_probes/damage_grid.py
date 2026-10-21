"""Predicted death of a damage spell's target, damage 1..8 x toughness 1..8, averaged over records.

    python scripts/effect_knowledge_probes/damage_grid.py CHECKPOINT [--source SOURCE]

Each cell edits the acting text's ``NumDmg$`` through the batcher's text memo, as the
damage sweep does, and the target's toughness, as the toughness sweep does, and reads
the target's predicted death.

``--source probe`` (the default) uses the frozen probe set's damage-sweep records.
``--source training`` uses training records that qualify for lethality copies (FR-063e)
with a literal amount: the grid then reads the model on the very records its copies were
drawn from, so a soft diagonal there is a fault of fit rather than of generalization.
``--source card-disjoint`` takes qualifying records from the held-out stratum, the same
selection on texts no training card carries.
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "effect_knowledge_probes.run", Path(__file__).resolve().parent / "run.py")
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)
common = run.sibling("common")
sweeps = run.sibling("sweeps")

CORPUS = Path("output/effects/corpus-gen2")
PROBE_SET = Path("output/effects/reports/knowledge-probes-set-b3adb69a2950.json")
VALUES = range(1, 9)


def probe_records(host: str) -> list[tuple[object, str]]:
    probe_set = json.loads(PROBE_SET.read_text(encoding="utf-8"))
    rows = [r for r in probe_set["sweeps"]["damage"] if r["host"] == host]
    ids = {r["record_id"] for r in rows}
    found = {}
    for stratum, games in probe_set["games"].items():
        shards = [CORPUS / n for n in probe_set["sources"][stratum]]
        for _, rec in common.read_games(shards, set(games), ids):
            found[rec.record_id] = rec
    return [(found[r["record_id"]], r["entity"]) for r in rows]


def qualifying_records(directory: Path, count: int, shards: int) -> list[tuple[object, str]]:
    """The first ``count`` single-target damage records with a literal amount that qualify."""
    from effects.domain.damage_step_keywords import KeywordResolver
    from effects.domain.effect_targets import derive_targets
    from effects.domain.lethality_counterfactuals import qualify
    from effects.infrastructure.record_io import iter_shards, read_shard

    sidecars = common.sidecar_cache(common.DEFAULT_CARDS_FOLDERS)
    resolver = KeywordResolver(sidecars)
    chosen = []
    for shard in iter_shards(directory)[:shards]:
        for record in read_shard(shard):
            if not record.ability:
                continue
            try:
                line = sidecars.line_for(record.ability[0])
            except Exception:
                continue
            qualified = qualify(record, line, derive_targets(record), resolver.keywords_of)
            if (qualified is None or qualified.kind != "damage" or qualified.stated != "text"
                    or len(qualified.hit) != 1):
                continue
            chosen.append((record, qualified.hit[0]))
            if len(chosen) >= count:
                return chosen
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--source", choices=("probe", "training", "card-disjoint"),
                        default="probe")
    parser.add_argument("--host", default="spell", help="probe source: spell or triggered")
    parser.add_argument("--records", type=int, default=40)
    parser.add_argument("--shards", type=int, default=200)
    args = parser.parse_args()
    if args.source == "probe":
        pairs = probe_records(args.host)
    else:
        directory = CORPUS / ("training" if args.source == "training"
                              else "validation/card-disjoint")
        pairs = qualifying_records(directory, args.records, args.shards)
    records = [record for record, _ in pairs]
    checkpoint = common.load_checkpoint(args.checkpoint)
    probe_model = common.load_probe_model(
        args.checkpoint, list(common.DEFAULT_CARDS_FOLDERS), records, checkpoint=checkpoint)
    grid = np.zeros((8, 8))
    batcher = probe_model.batcher
    for record, entity in pairs:
        key = record.ability[0]
        original = batcher.text_of(key)
        edited = [sweeps.set_toughness(record, entity, t) for t in VALUES]
        for i, amount in enumerate(VALUES):
            with batcher.reading_as(key, sweeps.set_damage(original, amount)):
                grid[i] += sweeps._predict(probe_model, edited,
                                           lambda o, v, s, r: sweeps._death(o, s, r, entity))
    grid /= max(len(pairs), 1)
    print(f"{args.checkpoint.parent.name}, {len(pairs)} {args.source} records")
    print("| damage \\ toughness | " + " | ".join(str(t) for t in VALUES) + " |")
    print("|---|" + "---:|" * 8)
    for i, amount in enumerate(VALUES):
        print(f"| {amount} | " + " | ".join(f"{grid[i, j]:.2f}" for j in range(8)) + " |")


if __name__ == "__main__":
    main()

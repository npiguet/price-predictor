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
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    # Run by path: put scripts/ on the path so the suite imports as a package.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from effect_knowledge_probes import common, record_walk, sweeps  # noqa: E402
from effect_knowledge_probes.board_edits import BoardEdit  # noqa: E402

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

    sidecars = common.sidecar_cache(common.DEFAULT_CARDS_FOLDERS)
    resolver = KeywordResolver(sidecars)

    def choose(record):
        if not record.ability:
            return None
        line = sidecars.line_or_none(record.ability[0])
        if line is None:
            return None
        qualified = qualify(record, line, derive_targets(record), resolver.keywords_of)
        if (qualified is None or qualified.kind != "damage" or qualified.stated != "text"
                or len(qualified.hit) != 1):
            return None
        return record, qualified.hit[0]

    return record_walk.qualifying([directory], choose, count, per_game=None, shards=shards)


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
        edited = [BoardEdit(record).set_toughness(entity, t).record for t in VALUES]
        for i, amount in enumerate(VALUES):
            with batcher.reading_as(key, sweeps.set_damage_amount(original, amount)):
                grid[i] += common.predict(probe_model, edited,
                                          lambda o, v, s, r: sweeps.death(o, s, r, entity))
    grid /= max(len(pairs), 1)
    print(f"{args.checkpoint.parent.name}, {len(pairs)} {args.source} records")
    print("| damage \\ toughness | " + " | ".join(str(t) for t in VALUES) + " |")
    print("|---|" + "---:|" * 8)
    for i, amount in enumerate(VALUES):
        print(f"| {amount} | " + " | ".join(f"{grid[i, j]:.2f}" for j in range(8)) + " |")


if __name__ == "__main__":
    main()

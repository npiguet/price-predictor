"""Predicted affordability when the mana is the right amount in the wrong colours.

    python scripts/effect_knowledge_probes/colour_pips.py CHECKPOINT [--records 300]
        [--shards 200] [--per-game 2]

The affordability sweep gives the acting player the cost's own colours and varies
how much mana there is. This probe holds the amount and varies the colours. A real
playability decision record names a candidate whose mana cost basic lands pay one
pip each (``sweeps.affordability_ready``); the acting player's lands are replaced by
untapped basics in six configurations, through ``sweeps.set_untapped_lands``, so the
lands and the production count written beside them agree:

==================  ===============================================================
configuration       the acting player's untapped basics
==================  ===============================================================
exact               one land of each coloured pip's colour, and one land of an
                    off colour per point of generic mana
surplus             exact, plus two off-colour lands
one short           exact, less one land: an off-colour one where the cost has
                    generic mana, else the last coloured pip's
pip missing         exact, with the last coloured pip's land turned off-colour
pip missing, +2     pip missing, plus two off-colour lands: more mana than the
                    cost, still missing a colour
all off colour      as many lands as the cost, every one off-colour
==================  ===============================================================

The off colour is the first of WUBRG the cost does not name. The rule's answer is
computed for every configuration by matching pips to lands, so a cost whose
missing pip another land still pays (``{U}{U}`` short one Island but with
``{1}`` paid by a second Island) is scored as the rule has it. The model's answer
is the predicted ``affordable`` verdict. Its absolute level sits low on every
checkpoint, so the comparisons that matter are within one record: how far the
verdict falls from ``exact`` when a colour goes missing, against how far it falls
when one land goes missing.

``pip missing`` is split by whether the missing pip's colour is still among the
lands (``{1}{U}{U}`` given one Island: the model has to count pips) or gone
altogether (``{1}{R}`` given no Mountain: it only has to see the colour).
"""
from __future__ import annotations

import argparse
import importlib.util
from collections import defaultdict
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "effect_knowledge_probes.run", Path(__file__).resolve().parent / "run.py")
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)
common = run.sibling("common")
sweeps = run.sibling("sweeps")

CORPUS = Path("output/effects/corpus-gen2")
WUBRG = ("W", "U", "B", "R", "G")
CONFIGURATIONS = ("exact", "surplus", "one short", "pip missing", "pip missing, +2",
                  "all off colour")


# ── the cost and the rule ───────────────────────────────────────────────


def parse_cost(cost: str) -> tuple[list[tuple[str, ...]], int]:
    """``(coloured pips, generic)``: each coloured pip as the colours that pay it.

    ``{2}{R}{W/U}`` is ``([("R",), ("W", "U")], 2)``.
    """
    coloured: list[tuple[str, ...]] = []
    generic = 0
    for pip in sweeps._PIP_RE.findall(cost or ""):
        if pip.isdigit():
            generic += int(pip)
        else:
            coloured.append(tuple(pip.split("/")))
    return coloured, generic


def payable(coloured: list[tuple[str, ...]], generic: int, lands: list[str]) -> bool:
    """Whether basic lands of ``lands``' colours pay the cost, one pip per land."""
    def match(i: int, free: list[str]) -> bool:
        if i == len(coloured):
            return len(free) >= generic
        for colour in set(coloured[i]):
            if colour in free:
                rest = list(free)
                rest.remove(colour)
                if match(i + 1, rest):
                    return True
        return False

    return match(0, list(lands))


def shape(coloured: list[tuple[str, ...]]) -> str:
    """The kind of coloured requirement: one pip, a repeated pip, several colours,
    or hybrid."""
    if any(len(pip) > 1 for pip in coloured):
        return "hybrid"
    if len(coloured) == 1:
        return "one pip"
    return "repeated pip" if len(set(coloured)) == 1 else "several colours"


def configurations(cost: str) -> dict[str, tuple[list[str], bool]]:
    """``configuration -> (land colours, the rule's answer)`` for one cost."""
    coloured, generic = parse_cost(cost)
    named = {colour for pip in coloured for colour in pip}
    off = next(colour for colour in WUBRG if colour not in named)
    exact = [pip[0] for pip in coloured] + [off] * generic
    missing = [pip[0] for pip in coloured[:-1]] + [off] + [off] * generic
    lands = {
        "exact": exact,
        "surplus": exact + [off, off],
        # The last land is an off-colour one where the cost has generic mana.
        "one short": exact[:-1],
        "pip missing": missing,
        "pip missing, +2": missing + [off, off],
        "all off colour": [off] * len(exact),
    }
    return {name: (colours, payable(coloured, generic, colours))
            for name, colours in lands.items()}


def missing_colour_present(cost: str) -> bool:
    """Whether ``pip missing`` still leaves a land of the missing pip's colour."""
    coloured, _ = parse_cost(cost)
    return coloured[-1][0] in [pip[0] for pip in coloured[:-1]]


# ── which records qualify ───────────────────────────────────────────────


def qualifying_records(directories, count: int, shards: int, per_game: int, sidecars):
    from effects.domain.records import PlayabilityDecisionPayload
    from effects.infrastructure.record_io import iter_shards, read_shard

    chosen, per = [], defaultdict(int)
    for directory in directories:
        for shard in iter_shards(directory)[:shards]:
            for record in read_shard(shard):
                if (not isinstance(record.payload, PlayabilityDecisionPayload)
                        or record.fork or per[record.game_id] >= per_game):
                    continue
                if not sweeps.affordability_ready(record, sidecars):
                    continue
                coloured, _ = parse_cost(sweeps.candidate_mana(record))
                if len({c for pip in coloured for c in pip}) >= len(WUBRG):
                    continue
                per[record.game_id] += 1
                chosen.append(record)
                if len(chosen) >= count:
                    return chosen
    return chosen


# ── reading the model ───────────────────────────────────────────────────


def _affordable(outputs, verdict, surface, row) -> float:
    import torch

    from effects.domain.effect_model import VERDICT_BITS

    return float(torch.sigmoid(verdict[row, VERDICT_BITS.index("affordable")]))


def _fmt(values) -> str:
    return f"{np.mean(values):.3f}" if len(values) else "n/a"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--records", type=int, default=300)
    parser.add_argument("--shards", type=int, default=200)
    parser.add_argument("--per-game", type=int, default=2)
    args = parser.parse_args()

    sidecars = common.sidecar_cache(common.DEFAULT_CARDS_FOLDERS)
    directories = [CORPUS / "validation" / "card-disjoint",
                   CORPUS / "validation" / "game-disjoint"]
    records = qualifying_records(directories, args.records, args.shards, args.per_game,
                                 sidecars)
    checkpoint = common.load_checkpoint(args.checkpoint)
    probe_model = common.load_probe_model(
        args.checkpoint, list(common.DEFAULT_CARDS_FOLDERS), records, checkpoint=checkpoint)
    print(f"{args.checkpoint.parent.name} ({checkpoint.epoch}), {len(records)} validation "
          f"decisions with a land-payable cost", flush=True)

    # predicted[name][i], rule[name][i] for record i
    predicted: dict[str, list[float]] = {name: [] for name in CONFIGURATIONS}
    rule: dict[str, list[bool]] = {name: [] for name in CONFIGURATIONS}
    shapes, present = [], []
    for start in range(0, len(records), 16):
        chunk = records[start:start + 16]
        edited, answers = [], []
        for record in chunk:
            cost = sweeps.candidate_mana(record)
            shapes.append(shape(parse_cost(cost)[0]))
            present.append(missing_colour_present(cost))
            for name, (lands, answer) in configurations(cost).items():
                edited.append(sweeps.set_untapped_lands(
                    record, record.actor_player, list(lands), len(lands)))
                answers.append((name, answer))
        for (name, answer), p in zip(answers, sweeps._predict(probe_model, edited,
                                                              _affordable)):
            predicted[name].append(p)
            rule[name].append(answer)
    predicted = {k: np.asarray(v) for k, v in predicted.items()}
    rule = {k: np.asarray(v, bool) for k, v in rule.items()}
    shapes, present = np.asarray(shapes), np.asarray(present, bool)

    print("\nby configuration")
    print("| configuration | records | rule: affordable | predicted, mean | "
          "below exact, mean | below exact, share |")
    print("|---|---:|---:|---:|---:|---:|")
    exact = predicted["exact"]
    for name in CONFIGURATIONS:
        drop = exact - predicted[name]
        print(f"| {name} | {len(drop)} | {rule[name].mean():.2f} | "
              f"{_fmt(predicted[name])} | {_fmt(drop)} | {(drop > 0).mean():.2f} |")

    auc = run.sibling("target_restrictions").auc
    print("\nranking: rule-affordable configurations against unaffordable ones")
    print("| items | AUC |")
    print("|---|---:|")
    for title, names in (("every configuration", CONFIGURATIONS),
                         ("colours only (exact, surplus against pip missing, +2 and "
                          "all off colour)",
                          ("exact", "surplus", "pip missing, +2", "all off colour")),
                         ("amount only (exact against one short)", ("exact", "one short"))):
        labels = np.concatenate([rule[n] for n in names])
        scores = np.concatenate([predicted[n] for n in names])
        print(f"| {title} | {auc(labels, scores):.3f} |")

    print("\npip missing against one short, by the cost's coloured requirement "
          "(drop below exact, mean)")
    print("| cost | records | pip missing | one short | pip missing, +2 | "
          "all off colour |")
    print("|---|---:|---:|---:|---:|---:|")
    groups = [(s, shapes == s) for s in ("one pip", "repeated pip", "several colours",
                                         "hybrid")]
    groups += [("missing colour still among the lands", present),
               ("missing colour gone from the lands", ~present)]
    for title, mask in groups:
        if not mask.any():
            continue
        cells = [_fmt((exact - predicted[n])[mask & ~rule[n]])
                 for n in ("pip missing", "one short", "pip missing, +2", "all off colour")]
        print(f"| {title} | {int(mask.sum())} | " + " | ".join(cells) + " |", flush=True)


if __name__ == "__main__":
    main()

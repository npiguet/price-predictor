"""D3 — state exchange: run every policy on every other policy's states.

Two confounds ruin a naive corpus comparison of drafting agents. **Supply**: an
agent with unusual colour taste is passed different packs, so its picks differ
without its judgement differing. **Self-selection**: the states an agent reaches
were produced by its own earlier picks, so "it looks good on its own drafts" can
be an artefact of the trajectory rather than of the pick. State exchange removes
both at once — evaluate policy A on the states policy B actually faced, paired
state by state.

Three parts, all inference-only over frozen checkpoints:

Part 1 — agreement matrix.
    For each seat label (``gen4``, ``gen1``, ``forge-full``) take that label's
    recorded pick states and run all three checkpoints on them. Report argmax
    agreement of each policy with the recorded pick and with each other policy,
    plus mean Jensen–Shannon distance between policies at temperature 1, broken
    down by pack number and by early (picks 1-8) vs late (9-15).
    Sanity: gen-4 on gen-4's own states must reproduce ~100 % of the recorded
    picks, and likewise gen-1 on gen-1's.

Part 2 — does gen-4 pick *better* on foreign states?
    On the states the ``forge-full`` and ``gen1`` seats actually faced, compare
    the card each policy would take against the card actually taken, graded by
    two independent card-value graders:
      (a) ``shrunk_score_play`` from the sealed win-rate labels. This is the
          encoder's own training target and the scorer's card values track it at
          Spearman 0.68, so agreement with it is partly circular.
      (b) Forge's bundled human draft rank (``draft_rank`` in
          ``output/scorer-probes/forge_hints.csv``), normalized position in a
          set's human pick order, **lower is better** — so the probe grades with
          ``-draft_rank``, making higher better for both graders. This one is
          exogenous to all three models and is the better grader.
    Deltas are paired per state; standard errors are clustered on ``draft_id``
    (a pod shares its card pool, so its picks move together).

Part 3 — leverage profile.
    Gen-3/gen-4 RL gave all 45 picks the same advantage and the same weight in
    the batch mean, but a pick's leverage on the final 23-card deck is
    concentrated early in each pack. On one fixed state set (gen-4's own), the
    per-(pack, pick) profile of KL(gen1 || gen4), KL(gen3 || gen4), the gen1/gen4
    argmax disagreement rate, and the mean pack size — the mechanical control,
    since a shrinking pack raises agreement on its own. Disagreement is reported
    raw and divided by the two-independent-uniform-draws baseline ``1 - 1/k``.

Usage
-----
    python scripts/draft_probes/d3_exchange.py --limit-drafts 200

``--batch 64`` is the default; drop it (16 works) when another job holds most of
the card's VRAM, or the driver's system-memory fallback turns the second forward
pass into a crawl without ever raising an OOM.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _lib.corpora import card_column
from _lib.markdown import banner, num, print_table
from _lib.stats import cluster_mean_se, js_distance, kl_divergence, softmax, spearman
from draft_corpus_common import score_play
from draft_probes.probe_lib import (
    CORPUS,
    HINTS,
    MODELS,
    OUT,
    WIN_RATES,
    CardTable,
    argmaxes,
    iter_corpus_states,
    iter_runners,
    write_csv,
)

POLICIES = {gen: MODELS[gen] for gen in ("gen1", "gen3", "gen4")}
SOURCES = ["gen4", "gen1", "forge-full"]
FOREIGN = ["forge-full", "gen1"]  # state sources gen-4 never produced


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    return kl_divergence(p, q, floor=1e-12)


def load_win_rate_grader() -> dict[str, float]:
    """name -> shrunk_score_play, dropping blank cells (no signal)."""
    if not WIN_RATES.exists():
        print(f"WARNING: missing {WIN_RATES}")
        return {}
    return score_play(WIN_RATES)


def load_rank_grader() -> dict[str, float]:
    """name -> -draft_rank (Forge's human pick order; lower rank is better)."""
    if not HINTS.exists():
        print(f"WARNING: missing {HINTS}")
    return card_column(HINTS, "draft_rank", -1.0)


# ------------------------------------------------------------------ the probe

def collect(limit: int) -> tuple[CardTable, dict[str, list]]:
    table = CardTable()
    buckets: dict[str, list] = {s: [] for s in SOURCES}
    for s in iter_corpus_states(CORPUS, table, labels=SOURCES,
                                limit_drafts=limit):
        buckets[s.label].append(s)
    return table, buckets


def run_policies(table: CardTable, buckets: dict[str, list],
                 batch: int) -> dict[str, dict[str, dict]]:
    """``runs[source][policy]``: per-sample softmax and argmax of every
    checkpoint on every source's states, each checkpoint loaded once."""
    missing = [str(p) for p in POLICIES.values() if not p.exists()]
    if missing:
        raise SystemExit(f"missing checkpoint {', '.join(missing)}")
    runs: dict[str, dict[str, dict]] = {src: {} for src in SOURCES if buckets[src]}
    for name, runner in iter_runners(table, batch, POLICIES):
        for src in runs:
            logits = runner.logits(buckets[src])
            runs[src][name] = {"probs": [softmax(ln) for ln in logits],
                               "argmax": argmaxes(logits)}
            print(f"  {name} on '{src}' states done", flush=True)
    return runs


def slices(samples: list) -> dict[str, np.ndarray]:
    pack = np.array([s.state.pack_number for s in samples])
    pick = np.array([s.state.pick_number for s in samples])
    return {
        "all": np.ones(len(samples), dtype=bool),
        "pack1": pack == 1, "pack2": pack == 2, "pack3": pack == 3,
        "early(<=8)": pick <= 8, "late(>=9)": pick >= 9,
    }


def part1(buckets: dict[str, list], runs: dict[str, dict[str, dict]]) -> list[dict]:
    rows: list[dict] = []
    for src in SOURCES:
        samples = buckets[src]
        if not samples:
            continue
        sl = slices(samples)
        target = np.array([s.target for s in samples])
        valid = target >= 0
        names = list(POLICIES)
        # policy vs recorded
        for p in names:
            arg = runs[src][p]["argmax"]
            for sname, mask in sl.items():
                m = mask & valid
                rows.append({
                    "state_source": src, "a": p, "b": "recorded", "slice": sname,
                    "n": int(m.sum()),
                    "agreement": float((arg[m] == target[m]).mean()) if m.any()
                    else float("nan"),
                    "js": float("nan"),
                })
        # policy vs policy
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                aa, ab = runs[src][a]["argmax"], runs[src][b]["argmax"]
                pa, pb = runs[src][a]["probs"], runs[src][b]["probs"]
                js = np.array([js_distance(x, y) for x, y in zip(pa, pb)])
                for sname, mask in sl.items():
                    rows.append({
                        "state_source": src, "a": a, "b": b, "slice": sname,
                        "n": int(mask.sum()),
                        "agreement": float((aa[mask] == ab[mask]).mean()),
                        "js": float(js[mask].mean()),
                    })
    return rows


def part2(buckets: dict[str, list], runs: dict[str, dict[str, dict]],
          graders: dict[str, dict[str, float]]) -> list[dict]:
    rows: list[dict] = []
    names = list(POLICIES)
    for src in SOURCES:
        samples = buckets[src]
        if not samples:
            continue
        for gname, grade in graders.items():
            if not grade:
                continue
            # common subset: every policy's choice AND the recorded card graded
            keep: list[int] = []
            for i, s in enumerate(samples):
                if s.target < 0 or s.taken_name not in grade:
                    continue
                if all(samples[i].pack_names[runs[src][p]["argmax"][i]] in grade
                       for p in names):
                    keep.append(i)
            if not keep:
                continue
            clusters = [samples[i].draft_id for i in keep]
            rec = np.array([grade[samples[i].taken_name] for i in keep])
            for p in names:
                arg = runs[src][p]["argmax"]
                pol = np.array([
                    grade[samples[i].pack_names[arg[i]]] for i in keep
                ])
                d = pol - rec
                same = np.array([arg[i] == samples[i].target for i in keep])
                _, se = cluster_mean_se(d, clusters)
                rows.append({
                    "state_source": src, "grader": gname, "policy": p,
                    "n": len(keep),
                    "mean_delta": float(d.mean()),
                    "se_clustered": se,
                    "t": float(d.mean() / se) if se > 0 else float("nan"),
                    "mean_grade_policy": float(pol.mean()),
                    "mean_grade_recorded": float(rec.mean()),
                    "agree_with_recorded": float(same.mean()),
                    "frac_better": float((d > 0).mean()),
                    "frac_worse": float((d < 0).mean()),
                    "n_clusters": len(set(clusters)),
                })
    return rows


def part3(samples: list, runs: dict[str, dict]) -> list[dict]:
    """Per-(pack, pick) policy movement on ONE fixed state set."""
    pack = np.array([s.state.pack_number for s in samples])
    pick = np.array([s.state.pick_number for s in samples])
    size = np.array([len(s.pack_names) for s in samples], dtype=float)
    p1, p3, p4 = runs["gen1"]["probs"], runs["gen3"]["probs"], runs["gen4"]["probs"]
    kl14 = np.array([_kl(a, b) for a, b in zip(p1, p4)])
    kl34 = np.array([_kl(a, b) for a, b in zip(p3, p4)])
    dis = (runs["gen1"]["argmax"] != runs["gen4"]["argmax"]).astype(float)
    dis34 = (runs["gen3"]["argmax"] != runs["gen4"]["argmax"]).astype(float)

    rows: list[dict] = []
    for pk in sorted(set(pack.tolist())):
        for pi in sorted(set(pick.tolist())):
            m = (pack == pk) & (pick == pi)
            if not m.any():
                continue
            k = float(size[m].mean())
            base = 1.0 - 1.0 / k if k > 1 else float("nan")
            rows.append({
                "pack": int(pk), "pick": int(pi), "n": int(m.sum()),
                "pack_size": k,
                "kl_gen1_gen4": float(kl14[m].mean()),
                "kl_gen3_gen4": float(kl34[m].mean()),
                "disagree_gen1_gen4": float(dis[m].mean()),
                "disagree_gen3_gen4": float(dis34[m].mean()),
                "random_baseline": base,
                "disagree_over_baseline": float(dis[m].mean() / base)
                if base and base == base and base > 0 else float("nan"),
                "disagree_over_logk": float(dis[m].mean() / math.log(k))
                if k > 1 else float("nan"),
            })
    return rows


# --------------------------------------------------------------------- report

def report(agree, grades, lev, meta) -> None:
    banner("PART 1 - agreement matrix (state exchange)")
    cols = ("all", "pack1", "pack2", "pack3", "early(<=8)", "late(>=9)")
    for src in SOURCES:
        rs = [r for r in agree if r["state_source"] == src]
        if not rs:
            continue
        n = next((r["n"] for r in rs if r["slice"] == "all"), 0)
        print(f"\nstates from seat label '{src}'   n={n}")
        rows = []
        for key in dict.fromkeys((r["a"], r["b"]) for r in rs):
            by = {x["slice"]: x for x in rs if (x["a"], x["b"]) == key}
            rows.append([f"{key[0]} vs {key[1]}"]
                        + [num(by[c]["agreement"]) for c in cols] + [num(by["all"]["js"])])
        print_table(["pair", "all", "pack1", "pack2", "pack3", "early", "late", "JS(all)"],
                    rows, "l" + "r" * 7)

    banner("PART 2 - grade of the policy's card minus grade of the card taken\n"
           "        (paired per state; SE clustered on draft_id; higher = better)")
    for gname in ("shrunk_score_play", "-draft_rank"):
        rs = [r for r in grades if r["grader"] == gname]
        if not rs:
            continue
        print(f"\ngrader: {gname}")
        print_table(["states", "policy", "n", "dmean", "SE", "t", "agree", "better", "worse"],
                    [[r["state_source"], r["policy"], r["n"], num(r["mean_delta"], 5),
                      num(r["se_clustered"], 5), num(r["t"], 2),
                      num(r["agree_with_recorded"], 3), num(r["frac_better"], 3),
                      num(r["frac_worse"], 3)] for r in rs], "llrrrrrrr")

    banner("PART 3 - leverage profile on gen-4's own states")
    print_table(["pack", "pick", "n", "|pack|", "KL(1||4)", "KL(3||4)", "dis14", "dis34",
                 "rand", "dis/rand"],
                [[r["pack"], r["pick"], r["n"], num(r["pack_size"], 2),
                  num(r["kl_gen1_gen4"]), num(r["kl_gen3_gen4"]),
                  num(r["disagree_gen1_gen4"], 3), num(r["disagree_gen3_gen4"], 3),
                  num(r["random_baseline"]), num(r["disagree_over_baseline"])]
                 for r in lev], "r" * 10)
    print("\ntrend across pick index (mean over packs, weighted by n):")
    trend = []
    for key in ("kl_gen1_gen4", "kl_gen3_gen4", "disagree_gen1_gen4",
                "disagree_over_baseline"):
        by_pick: dict[int, list] = {}
        for r in lev:
            by_pick.setdefault(r["pick"], []).append((r["n"], r[key]))
        ok = []
        for pick in sorted(by_pick):
            ws = [(n, v) for n, v in by_pick[pick] if v == v]
            if ws:
                ok.append((pick, sum(n * v for n, v in ws) / sum(n for n, _ in ws)))
        rho = (spearman(np.array([p for p, _ in ok]), np.array([v for _, v in ok]))
               if len(ok) > 2 else float("nan"))
        trend.append([key, num(np.mean([v for p, v in ok if p <= 8])),
                      num(np.mean([v for p, v in ok if p >= 9])), f"{rho:+.3f}"])
    print_table(["metric", "picks 1-8", "picks 9-15", "spearman(pick, value)"], trend, "lrrr")
    print(f"\ngrader coverage: win-rate {meta['n_win_rate_cards']} cards, "
          f"draft_rank {meta['n_rank_cards']} cards")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit-drafts", type=int, default=200)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"corpus {CORPUS}")
    table, buckets = collect(args.limit_drafts)
    for s in SOURCES:
        print(f"  states[{s}] = {len(buckets[s])}")

    runs = run_policies(table, buckets, args.batch)

    graders = {
        "shrunk_score_play": load_win_rate_grader(),
        "-draft_rank": load_rank_grader(),
    }
    meta = {
        "corpus": str(CORPUS),
        "limit_drafts": args.limit_drafts,
        "checkpoints": {n: str(p) for n, p in POLICIES.items()},
        "n_states": {s: len(buckets[s]) for s in SOURCES},
        "n_win_rate_cards": len(graders["shrunk_score_play"]),
        "n_rank_cards": len(graders["-draft_rank"]),
        "foreign_sources": FOREIGN,
    }

    agree = part1(buckets, runs)
    grades = part2(buckets, runs, graders)
    lev = part3(buckets["gen4"], runs["gen4"])

    (args.out / "d3_exchange.json").write_text(
        json.dumps({"meta": meta, "agreement": agree, "grades": grades,
                    "leverage": lev}, indent=2), encoding="utf-8")
    for name, rows in (("d3_agreement", agree), ("d3_grades", grades),
                       ("d3_leverage", lev)):
        if rows:
            write_csv(args.out / f"{name}.csv", rows)

    report(agree, grades, lev, meta)
    print(f"\nwrote {args.out / 'd3_exchange.json'} and d3_*.csv")


if __name__ == "__main__":
    main()

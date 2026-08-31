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
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from probe_lib import (  # noqa: E402
    CardTable,
    PolicyRunner,
    iter_corpus_states,
    load_agent,
)

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "output" / "draft-probes"
G4 = REPO / "models/draft/agent/gen4"

CORPUS = G4 / "lr1e-5_t2all_decay0.3-yardstick-v-forge-drafts.jsonl"
POLICIES = [
    ("gen1", REPO / "models/draft/agent/gen1/l6_lr3e-4_decay_20260604_080249.pt"),
    ("gen3", REPO / "models/draft/agent/gen3/temperature-on-all-agents"
     "/lr1e-5_t2_20260805_221050.pt"),
    ("gen4", G4 / "lr1e-5_t2all_decay0.3.pt"),
]
SOURCES = ["gen4", "gen1", "forge-full"]
FOREIGN = ["forge-full", "gen1"]  # state sources gen-4 never produced

WIN_RATES = Path(
    "Y:/Nicolas/mtg/mtg-models-data/sealed/training-data/matches-bo1"
    "/cards-win-rates.txt"
)
HINTS_CSV = REPO / "output" / "scorer-probes" / "forge_hints.csv"


# ---------------------------------------------------------------- small maths

def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    m = p > 0
    return float(np.sum(p[m] * np.log(p[m] / np.maximum(q[m], 1e-12))))


def _js(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)
    return float(np.sqrt(max(0.0, 0.5 * _kl(p, m) + 0.5 * _kl(q, m))))


def cluster_se(values: np.ndarray, clusters: list[str]) -> float:
    """SE of the mean of ``values`` with errors clustered on ``clusters``."""
    n = len(values)
    if n == 0:
        return float("nan")
    mean = values.mean()
    sums: dict[str, float] = defaultdict(float)
    for v, c in zip(values, clusters):
        sums[c] += v - mean
    var = sum(s * s for s in sums.values()) / (n * n)
    return float(math.sqrt(max(var, 0.0)))


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    def rank(a):
        order = np.argsort(a)
        r = np.empty(len(a), dtype=float)
        r[order] = np.arange(len(a), dtype=float)
        return r
    rx, ry = rank(np.asarray(x, float)), rank(np.asarray(y, float))
    rx -= rx.mean()
    ry -= ry.mean()
    d = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / d) if d > 0 else float("nan")


# ------------------------------------------------------------------- graders

def load_win_rate_grader() -> dict[str, float]:
    """name -> shrunk_score_play, dropping blank cells (no signal)."""
    out: dict[str, float] = {}
    if not WIN_RATES.exists():
        print(f"WARNING: missing {WIN_RATES}")
        return out
    with WIN_RATES.open(encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n").split(";")
        col = header.index("shrunk_score_play")
        for line in fh:
            parts = line.rstrip("\n").split(";")
            if len(parts) != len(header) or parts[col] == "":
                continue
            try:
                out[parts[0]] = float(parts[col])
            except ValueError:
                continue
    return out


def load_rank_grader() -> dict[str, float]:
    """name -> -draft_rank (Forge's human pick order; lower rank is better)."""
    out: dict[str, float] = {}
    if not HINTS_CSV.exists():
        print(f"WARNING: missing {HINTS_CSV}")
        return out
    with HINTS_CSV.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            v = (row.get("draft_rank") or "").strip()
            if not v:
                continue
            try:
                out[row["name"]] = -float(v)
            except ValueError:
                continue
    return out


# ------------------------------------------------------------------ the probe

def collect(limit: int) -> tuple[CardTable, dict[str, list]]:
    table = CardTable()
    buckets: dict[str, list] = {s: [] for s in SOURCES}
    for s in iter_corpus_states(CORPUS, table, labels=SOURCES,
                                limit_drafts=limit):
        buckets[s.label].append(s)
    return table, buckets


def run_policies(table: CardTable, samples: list, batch: int) -> dict[str, dict]:
    """For each checkpoint: per-sample softmax, argmax and chosen card name.

    Batches are formed over a length-sorted view of the states and the results
    unsorted afterwards. A state grows from ~15 tokens at pack 1 pick 1 to a few
    hundred by pack 3, and :class:`PolicyRunner` pads each batch to its longest
    member, so sorting cuts the padding a mixed batch would carry. Padding is
    masked out, so this changes nothing but the wall clock.
    """
    order = sorted(range(len(samples)),
                   key=lambda i: int(samples[i].state.card_idx.shape[0]))
    ordered = [samples[i] for i in order]
    res: dict[str, dict] = {}
    for name, ckpt in POLICIES:
        if not ckpt.exists():
            raise SystemExit(f"missing checkpoint {ckpt}")
        t0 = time.time()
        model, _cfg = load_agent(ckpt)
        runner = PolicyRunner(model, table, batch_size=batch)
        got: list = []
        step = max(batch * 8, 1)
        for start in range(0, len(ordered), step):
            got.extend(runner.logits(ordered[start:start + step]))
            print(f"    {name}: {len(got)}/{len(ordered)} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        logits: list = [None] * len(samples)
        for pos, i in enumerate(order):
            logits[i] = got[pos]
        probs = [_softmax(l) for l in logits]
        arg = np.array([int(np.argmax(l)) for l in logits])
        res[name] = {"probs": probs, "argmax": arg}
        del model, runner
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        print(f"  forward pass done: {name}")
    return res


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
        names = [p for p, _ in POLICIES]
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
                js = np.array([_js(x, y) for x, y in zip(pa, pb)])
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
    names = [p for p, _ in POLICIES]
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
                rows.append({
                    "state_source": src, "grader": gname, "policy": p,
                    "n": len(keep),
                    "mean_delta": float(d.mean()),
                    "se_clustered": cluster_se(d, clusters),
                    "t": float(d.mean() / cluster_se(d, clusters))
                    if cluster_se(d, clusters) > 0 else float("nan"),
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

def _fmt(v: float, w: int = 7, p: int = 4) -> str:
    return f"{v:{w}.{p}f}" if v == v else " " * (w - 3) + "n/a"


def report(agree, grades, lev, meta) -> None:
    print("\n" + "=" * 78)
    print("PART 1 - agreement matrix (state exchange)")
    print("=" * 78)
    for src in SOURCES:
        rs = [r for r in agree if r["state_source"] == src]
        if not rs:
            continue
        n = next((r["n"] for r in rs if r["slice"] == "all"), 0)
        print(f"\nstates from seat label '{src}'   n={n}")
        print(f"{'pair':22s} {'all':>7s} {'pack1':>7s} {'pack2':>7s} "
              f"{'pack3':>7s} {'early':>7s} {'late':>7s}   {'JS(all)':>8s}")
        seen = []
        for r in rs:
            key = (r["a"], r["b"])
            if key in seen:
                continue
            seen.append(key)
            by = {x["slice"]: x for x in rs if (x["a"], x["b"]) == key}
            js = by["all"]["js"]
            print(f"{r['a']+' vs '+r['b']:22s} "
                  + " ".join(_fmt(by[s]["agreement"]) for s in
                             ("all", "pack1", "pack2", "pack3",
                              "early(<=8)", "late(>=9)"))
                  + f"   {_fmt(js, 8):>8s}")

    print("\n" + "=" * 78)
    print("PART 2 - grade of the policy's card minus grade of the card taken")
    print("        (paired per state; SE clustered on draft_id; higher = better)")
    print("=" * 78)
    for gname in ("shrunk_score_play", "-draft_rank"):
        rs = [r for r in grades if r["grader"] == gname]
        if not rs:
            continue
        print(f"\ngrader: {gname}")
        print(f"{'states':12s} {'policy':7s} {'n':>6s} {'dmean':>9s} "
              f"{'SE':>8s} {'t':>7s} {'agree':>7s} {'better':>7s} {'worse':>7s}")
        for r in rs:
            print(f"{r['state_source']:12s} {r['policy']:7s} {r['n']:6d} "
                  f"{r['mean_delta']:9.5f} {r['se_clustered']:8.5f} "
                  f"{r['t']:7.2f} {r['agree_with_recorded']:7.3f} "
                  f"{r['frac_better']:7.3f} {r['frac_worse']:7.3f}")

    print("\n" + "=" * 78)
    print("PART 3 - leverage profile on gen-4's own states")
    print("=" * 78)
    print(f"{'pack':>4s} {'pick':>4s} {'n':>6s} {'|pack|':>7s} "
          f"{'KL(1||4)':>9s} {'KL(3||4)':>9s} {'dis14':>7s} {'dis34':>7s} "
          f"{'rand':>7s} {'dis/rand':>9s}")
    for r in lev:
        print(f"{r['pack']:4d} {r['pick']:4d} {r['n']:6d} {r['pack_size']:7.2f} "
              f"{r['kl_gen1_gen4']:9.4f} {r['kl_gen3_gen4']:9.4f} "
              f"{r['disagree_gen1_gen4']:7.3f} {r['disagree_gen3_gen4']:7.3f} "
              f"{_fmt(r['random_baseline'])} "
              f"{_fmt(r['disagree_over_baseline'], 9)}")
    print("\ntrend across pick index (mean over packs, weighted by n):")
    for key in ("kl_gen1_gen4", "kl_gen3_gen4", "disagree_gen1_gen4",
                "disagree_over_baseline"):
        by_pick = {}
        for r in lev:
            by_pick.setdefault(r["pick"], []).append((r["n"], r[key]))
        picks = sorted(by_pick)
        vals = []
        for p in picks:
            ws = [(n, v) for n, v in by_pick[p] if v == v]
            vals.append(sum(n * v for n, v in ws) / sum(n for n, _ in ws)
                        if ws else float("nan"))
        ok = [(p, v) for p, v in zip(picks, vals) if v == v]
        rho = spearman(np.array([p for p, _ in ok]),
                       np.array([v for _, v in ok])) if len(ok) > 2 else float("nan")
        early = [v for p, v in ok if p <= 8]
        late = [v for p, v in ok if p >= 9]
        print(f"  {key:24s} picks1-8 {np.mean(early):8.4f}  "
              f"picks9-15 {np.mean(late):8.4f}  spearman(pick, value) {rho:+.3f}")
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

    runs: dict[str, dict[str, dict]] = {}
    for src in SOURCES:
        if not buckets[src]:
            continue
        print(f"running 3 checkpoints on '{src}' states")
        runs[src] = run_policies(table, buckets[src], args.batch)

    graders = {
        "shrunk_score_play": load_win_rate_grader(),
        "-draft_rank": load_rank_grader(),
    }
    meta = {
        "corpus": str(CORPUS),
        "limit_drafts": args.limit_drafts,
        "checkpoints": {n: str(p) for n, p in POLICIES},
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
        if not rows:
            continue
        with (args.out / f"{name}.csv").open("w", newline="",
                                             encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    report(agree, grades, lev, meta)
    print(f"\nwrote {args.out / 'd3_exchange.json'} and d3_*.csv")


if __name__ == "__main__":
    main()

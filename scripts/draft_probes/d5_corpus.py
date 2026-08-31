"""D5 — three corpus-only probes of what the gen-4 margin actually is.

Everything here is arithmetic over the four gen-4 yardstick corpora (500 drafts
each, 8-seat pods, 3x15, argmax policy) and over the four gen-4 online-GRPO
training logs. No model is loaded; no GPU is touched.

A1 — the geometry the model cannot see
    The typed-token state a draft agent consumes carries no seat index and no
    pass direction: the policy literally cannot know who sits upstream of it.
    But the corpus records the seating exactly, so we can ask whether the gen-4
    margin nevertheless moves with it. If it does, the "advantage" is partly a
    property of where gen-4 seats landed in the pod rather than of how they
    pick. Outcome is the leave-one-out pod-relative ``deck_score`` the agent was
    actually trained on. The known confound is crowding — more strong seats in a
    pod means fewer good cards each, worth roughly -0.16..-0.21 per added seat
    in prior work — and it is correlated with neighbour labels by construction,
    so every neighbour effect is reported both as an OLS coefficient net of pod
    gen-4 count and as a within-fixed-count contrast.

    The mirror question runs the same regression over *all* seats: is a seat
    that sits downstream of a gen-4 seat starved relative to one downstream of a
    ``forge-full`` seat?

A2 — build-around traps
    Forge's card scripts carry a hand-written ``AI:RemoveDeck:Random`` line
    (``CardRules.getAiHints().getRemRandomDecks()``) marking cards Forge's own
    deck builders refuse to play in a random/limited deck unless their partners
    showed up. ``LimitedDeckBuilder`` reads it; Forge's *drafter* never does, so
    Forge takes such cards on raw power and then cannot play them. The flag is
    already staged as the ``ai_remove_deck_kind == "Random"`` column of
    ``output/scorer-probes/forge_hints.csv`` (scripts/scorer_probes/forge_hints.py),
    so nothing is re-extracted here.

    The question is whether the RL taught gen-4 to stop taking cards its reward
    pays nothing for. A gen-4 number alone cannot answer that: gen-1 is a
    distillation of Forge and inherits Forge's blindness, so the gen-4-minus-gen-1
    difference is the part attributable to reinforcement learning, and the
    gen-1-minus-forge-full difference is what distillation alone already cost.

A3 — did the training signal drive the training steps?
    The gradient was norm-clipped at 1.0 while pre-clip norms ran 6-12, so every
    optimizer step was effectively direction-normalised to a fixed length. If
    that is the whole story, the per-round policy displacement KL(pi_k||pi_k+1)
    should be uncorrelated with whether the round carried any learning signal
    (reward std, advantage spread, near-zero-advantage fraction) — the run
    drifts at a constant rate regardless, which is a mechanical explanation for
    every run peaking early and then declining.

Usage
-----
    python scripts/draft_probes/d5_corpus.py
    python scripts/draft_probes/d5_corpus.py --out-dir output/draft-probes
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from draft.domain.draft_geometry import DraftGeometry, DraftRecord  # noqa: E402
from draft.infrastructure.draft_record_io import read_records  # noqa: E402

G4 = REPO / "models" / "draft" / "agent" / "gen4"
RUNS = (
    "lr1e-5_t2all_decay0.3",
    "lr1e-5_t2all_nodecay",
    "lr1e-5_t3all_decay0.3",
    "lr1e-5_t3learner_t2field_decay0.3",
)
CORPUS_SUFFIX = "-yardstick-v-forge-drafts.jsonl"
LOG_SUFFIX = "-training.log"
HINTS_CSV = REPO / "output" / "scorer-probes" / "forge_hints.csv"
DEFAULT_OUT = REPO / "output" / "draft-probes"

LABELS = ("gen4", "gen1", "forge-full")


# --------------------------------------------------------------- OLS + cluster

def ols_cluster(
    y: np.ndarray, X: np.ndarray, clusters: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """OLS coefficients and cluster-robust (CR1) standard errors.

    ``X`` must already contain its intercept column. Clusters are an integer
    label per row; the sandwich uses the usual G/(G-1) * (N-1)/(N-K) correction.
    """
    n, k = X.shape
    xtx_inv = np.linalg.pinv(X.T @ X)
    beta = xtx_inv @ (X.T @ y)
    resid = y - X @ beta
    meat = np.zeros((k, k))
    for g in np.unique(clusters):
        m = clusters == g
        xg_u = X[m].T @ resid[m]
        meat += np.outer(xg_u, xg_u)
    n_g = len(np.unique(clusters))
    correction = (n_g / max(n_g - 1, 1)) * ((n - 1) / max(n - k, 1))
    cov = xtx_inv @ meat @ xtx_inv * correction
    return beta, np.sqrt(np.maximum(np.diag(cov), 0.0))


def cluster_mean_se(
    values: np.ndarray, clusters: np.ndarray,
) -> tuple[float, float]:
    """Mean of ``values`` with a standard error clustered on ``clusters``."""
    if len(values) == 0:
        return float("nan"), float("nan")
    X = np.ones((len(values), 1))
    beta, se = ols_cluster(values, X, clusters)
    return float(beta[0]), float(se[0])


def fmt_pm(mean: float, se: float, digits: int = 3) -> str:
    if math.isnan(mean):
        return "     n/a"
    return f"{mean:+.{digits}f} +- {se:.{digits}f}"


def stars(beta: float, se: float) -> str:
    if se <= 0 or math.isnan(se):
        return ""
    t = abs(beta / se)
    return "***" if t > 2.58 else "**" if t > 1.96 else "*" if t > 1.64 else ""


def pearson(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Pearson r and a large-sample SE (Fisher z), NaN when degenerate."""
    if len(a) < 4 or a.std() == 0 or b.std() == 0:
        return float("nan"), float("nan")
    r = float(np.corrcoef(a, b)[0, 1])
    se_z = 1.0 / math.sqrt(len(a) - 3)
    # delta-method back-transform of the Fisher-z SE onto the r scale
    return r, float((1 - r * r) * se_z)


def spearman(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    def rank(v: np.ndarray) -> np.ndarray:
        order = v.argsort()
        r = np.empty(len(v), dtype=float)
        r[order] = np.arange(len(v), dtype=float)
        # average ties
        _, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
        sums = np.zeros(len(cnt))
        np.add.at(sums, inv, r)
        return (sums / cnt)[inv]

    return pearson(rank(a), rank(b))


# ------------------------------------------------------------------ seat table

class SeatRow:
    """One (draft, seat) observation, everything three analyses need."""

    __slots__ = (
        "corpus", "draft_id", "seat", "agent", "score", "loo",
        "up_agent", "down_agent", "pod_gen4", "n_flagged", "deck",
    )

    def __init__(self, **kw) -> None:
        for key, value in kw.items():
            setattr(self, key, value)


def pass_neighbours(geo: DraftGeometry) -> tuple[list[int], list[int]]:
    """(downstream, upstream) seat index per seat, for **pack 1**.

    Derived from ``DraftGeometry`` rather than hardcoded: in pack 1 the booster
    seat ``s`` opened is passed to whoever authors its offset-1 pick, which is
    exactly ``seat_of_pick(k, 1)``.
    """
    down = [0] * geo.pod_size
    for seat in range(geo.pod_size):
        k, _ = geo.booster_for_pick(seat, 1, 1)
        down[seat] = geo.seat_of_pick(k, 1)
    up = [0] * geo.pod_size
    for seat, d in enumerate(down):
        up[d] = seat
    return down, up


def loo_rewards(scores: list[float | None]) -> list[float | None]:
    """deck_score minus the pod mean of the *other* seats (the RL reward)."""
    ok = [s for s in scores if s is not None]
    total = sum(ok)
    out: list[float | None] = []
    for s in scores:
        if s is None or len(ok) < 2:
            out.append(None)
        else:
            out.append(s - (total - s) / (len(ok) - 1))
    return out


def load_flagged() -> set[str]:
    """Card names carrying ``AI:RemoveDeck:Random`` (``RemRandomDecks``)."""
    if not HINTS_CSV.exists():
        raise SystemExit(
            f"missing {HINTS_CSV}; run scripts/scorer_probes/forge_hints.py first"
        )
    with HINTS_CSV.open(encoding="utf-8", newline="") as handle:
        return {
            row["name"]
            for row in csv.DictReader(handle)
            if row["ai_remove_deck_kind"] == "Random"
        }


class PickStats:
    """Per-agent pick-level counters for A2."""

    __slots__ = (
        "picks", "avail", "took", "avail_choice", "took_choice",
        "avail_alt", "took_alt", "taken_total", "taken_in_deck",
    )

    def __init__(self) -> None:
        for slot in self.__slots__:
            setattr(self, slot, 0)


def build_tables(flagged: set[str]) -> tuple[list[SeatRow], dict[str, PickStats]]:
    rows: list[SeatRow] = []
    stats: dict[str, PickStats] = defaultdict(PickStats)
    for run in RUNS:
        path = G4 / f"{run}{CORPUS_SUFFIX}"
        if not path.exists():
            raise SystemExit(f"missing corpus {path}")
        for record in read_records(path):
            geo = DraftGeometry.from_record(record)
            down, up = pass_neighbours(geo)
            agents = [s.agent for s in record.seats]
            loo = loo_rewards([s.deck_score for s in record.seats])
            pod_gen4 = sum(1 for a in agents if a == "gen4")

            taken_flagged: list[Counter] = [Counter() for _ in agents]
            for k, booster in enumerate(record.boosters):
                picks = booster.picks
                n = len(picks)
                for j in range(n):
                    seat = geo.seat_of_pick(k, j)
                    agent = agents[seat]
                    st = stats[agent]
                    st.picks += 1
                    card = picks[j]
                    legal = picks[j:]
                    n_flag = sum(1 for c in legal if c in flagged)
                    if n_flag == 0:
                        continue
                    is_flag = card in flagged
                    st.avail += 1
                    st.took += is_flag
                    if len(legal) > 1:                    # not the last card
                        st.avail_choice += 1
                        st.took_choice += is_flag
                    if n_flag < len(legal):               # an unflagged alt existed
                        st.avail_alt += 1
                        st.took_alt += is_flag
                    if is_flag:
                        taken_flagged[seat][card] += 1

            for seat, agent in enumerate(agents):
                deck = Counter(record.seats[seat].deck)
                tf = taken_flagged[seat]
                st = stats[agent]
                st.taken_total += sum(tf.values())
                st.taken_in_deck += sum(min(c, deck[name]) for name, c in tf.items())
                rows.append(SeatRow(
                    corpus=run,
                    draft_id=record.draft_id,
                    seat=seat,
                    agent=agent,
                    score=record.seats[seat].deck_score,
                    loo=loo[seat],
                    up_agent=agents[up[seat]],
                    down_agent=agents[down[seat]],
                    pod_gen4=pod_gen4,
                    n_flagged=sum(tf.values()),
                    deck=None,
                ))
    return rows, dict(stats)


# ------------------------------------------------------------------ analysis 1

def dummies(values: list[str], base: str, levels: tuple[str, ...]) -> np.ndarray:
    cols = [l for l in levels if l != base]
    return np.array(
        [[1.0 if v == c else 0.0 for c in cols] for v in values], dtype=float
    ), cols


def regress(
    rows: list[SeatRow],
    neigh: str | tuple[str, ...],
    base: str = "forge-full",
    with_own: bool = False,
    control_crowding: bool = True,
) -> dict:
    y = np.array([r.loo for r in rows], dtype=float)
    neighs = (neigh,) if isinstance(neigh, str) else neigh
    blocks = [np.ones((len(rows), 1))]
    names: list[str] = []
    for nb in neighs:
        D, cols = dummies([getattr(r, nb) for r in rows], base, LABELS)
        blocks.append(D)
        names += [f"{nb}={c}" for c in cols]
    if with_own:
        O, ocols = dummies([r.agent for r in rows], "forge-full", LABELS)
        blocks.append(O)
        names += [f"own={c}" for c in ocols]
    if control_crowding:
        blocks.append(np.array([[float(r.pod_gen4)] for r in rows]))
        names.append("pod_gen4_count")
    X = np.hstack(blocks)
    _, cl = np.unique([f"{r.corpus}/{r.draft_id}" for r in rows], return_inverse=True)
    beta, se = ols_cluster(y, X, cl)
    return {
        "n": len(rows),
        "n_clusters": int(cl.max() + 1),
        "base": "+".join(neighs) + f"={base}",
        "terms": [
            {"name": nm, "beta": float(b), "se": float(s), "t": float(b / s) if s else None}
            for nm, b, s in zip(["intercept"] + names, beta, se)
        ],
    }


def within_count_table(rows: list[SeatRow], neigh: str) -> list[dict]:
    """Neighbour contrast held inside each fixed pod gen-4 count."""
    out = []
    by_count: dict[int, list[SeatRow]] = defaultdict(list)
    for r in rows:
        by_count[r.pod_gen4].append(r)
    for count in sorted(by_count):
        sub = by_count[count]
        cell = {"pod_gen4_count": count, "n": len(sub)}
        for label in LABELS:
            grp = [r for r in sub if getattr(r, neigh) == label]
            if not grp:
                cell[label] = None
                continue
            v = np.array([r.loo for r in grp], dtype=float)
            _, cl = np.unique([r.draft_id for r in grp], return_inverse=True)
            m, s = cluster_mean_se(v, cl)
            cell[label] = {"n": len(grp), "mean": m, "se": s}
        out.append(cell)
    return out


def analysis1(rows: list[SeatRow]) -> dict:
    g4 = [r for r in rows if r.agent == "gen4" and r.loo is not None]
    allr = [r for r in rows if r.loo is not None]

    crowd_y = np.array([r.loo for r in g4], dtype=float)
    crowd_X = np.column_stack([np.ones(len(g4)), [r.pod_gen4 for r in g4]])
    _, cl = np.unique([f"{r.corpus}/{r.draft_id}" for r in g4], return_inverse=True)
    cb, cse = ols_cluster(crowd_y, crowd_X, cl)

    return {
        "n_gen4_seats": len(g4),
        "n_all_seats": len(allr),
        "gen4_mean_loo": dict(zip(
            ("mean", "se"),
            cluster_mean_se(crowd_y, cl))),
        "crowding_only": {"slope": float(cb[1]), "se": float(cse[1])},
        "upstream_gen4seats": regress(g4, "up_agent"),
        "downstream_gen4seats": regress(g4, "down_agent"),
        "joint_gen4seats": regress(g4, ("up_agent", "down_agent")),
        "upstream_gen4seats_nocontrol": regress(g4, "up_agent", control_crowding=False),
        "downstream_gen4seats_nocontrol": regress(
            g4, "down_agent", control_crowding=False),
        "upstream_within_count": within_count_table(g4, "up_agent"),
        "downstream_within_count": within_count_table(g4, "down_agent"),
        "mirror_all_seats_upstream": regress(allr, "up_agent", with_own=True),
        "mirror_all_seats_upstream_nocontrol": regress(
            allr, "up_agent", with_own=True, control_crowding=False),
    }


# ------------------------------------------------------------------ analysis 2

def rate(num: int, den: int) -> dict:
    if den == 0:
        return {"n": 0, "k": 0, "rate": None, "se": None}
    p = num / den
    return {"n": den, "k": num, "rate": p, "se": math.sqrt(p * (1 - p) / den)}


def analysis2(rows: list[SeatRow], stats: dict[str, PickStats]) -> dict:
    per_agent = {}
    for label in LABELS:
        st = stats.get(label)
        if st is None:
            continue
        per_agent[label] = {
            "picks": st.picks,
            "take_rate_when_available": rate(st.took, st.avail),
            "take_rate_real_choice": rate(st.took_choice, st.avail_choice),
            "take_rate_unflagged_alt": rate(st.took_alt, st.avail_alt),
            "flagged_taken": st.taken_total,
            "flagged_taken_played": rate(st.taken_in_deck, st.taken_total),
        }

    # mean pod-relative score by how many flagged cards the seat drafted
    by_bucket: dict[str, list[dict]] = {}
    for label in LABELS:
        sub = [r for r in rows if r.agent == label and r.loo is not None]
        buckets: dict[str, list[SeatRow]] = defaultdict(list)
        for r in sub:
            n = r.n_flagged
            key = str(n) if n <= 3 else "4+"
            buckets[key].append(r)
        cells = []
        for key in sorted(buckets, key=lambda k: (k == "4+", k)):
            grp = buckets[key]
            v = np.array([r.loo for r in grp], dtype=float)
            _, cl = np.unique(
                [f"{r.corpus}/{r.draft_id}" for r in grp], return_inverse=True)
            m, s = cluster_mean_se(v, cl)
            cells.append({"flagged_drafted": key, "n": len(grp), "mean": m, "se": s})
        # within-agent slope of loo on flagged count, clustered
        v = np.array([r.loo for r in sub], dtype=float)
        X = np.column_stack([
            np.ones(len(sub)),
            [r.n_flagged for r in sub],
            [r.pod_gen4 for r in sub],
        ])
        _, cl = np.unique(
            [f"{r.corpus}/{r.draft_id}" for r in sub], return_inverse=True)
        b, se = ols_cluster(v, X, cl)
        by_bucket[label] = {
            "cells": cells,
            "mean_flagged_drafted": float(np.mean([r.n_flagged for r in sub])),
            "slope_per_flagged_card": {"beta": float(b[1]), "se": float(se[1])},
        }

    diffs = {}
    for key in ("take_rate_when_available", "take_rate_real_choice",
                "take_rate_unflagged_alt", "flagged_taken_played"):
        for a, b in (("gen4", "gen1"), ("gen1", "forge-full"), ("gen4", "forge-full")):
            ra, rb = per_agent[a][key], per_agent[b][key]
            if ra["rate"] is None or rb["rate"] is None:
                continue
            d = ra["rate"] - rb["rate"]
            se = math.sqrt(ra["se"] ** 2 + rb["se"] ** 2)
            diffs[f"{key}:{a}-{b}"] = {"diff": d, "se": se, "z": d / se if se else None}
    return {"per_agent": per_agent, "by_flagged_count": by_bucket, "contrasts": diffs}


# ------------------------------------------------------------------ analysis 3

_ROUND = re.compile(r"^\[[^\]]+\]\s+round (\d+) \|")
_REWARD = re.compile(
    r"reward\s+: learner seats=(\d+) R mean=([-+][\d.]+) std=([\d.]+) \| "
    r"A std=([\d.]+) \|A\|<0\.1=([\d.]+)% \|A\|>0\.5=([\d.]+)% max\|A\|=([\d.]+)")
_EXPLORE = re.compile(
    r"explore\s+: H=([\d.]+) ppl=([\d.]+) off-argmax=([\d.]+)%")
_MOVE = re.compile(
    r"movement\s+: mean logpi=([-+][\d.]+) policy_loss=([-+][\d.]+) "
    r"grad_norm=([\d.]+) KL\(prev\|\|new\)=([\d.]+) KL\(init\|\|new\)=([\d.]+) "
    r"lr=([\d.eE+-]+)")
_PROGRESS = re.compile(r"progress\s+: anchor margin=([-+][\d.]+).*window=(\d+) drafts")
_BEST = re.compile(r"best anchor margin: ([-+][\d.]+) at round (\d+)")


def parse_log(path: Path) -> tuple[list[dict], dict]:
    """Parse one online-GRPO training log into per-round dicts.

    Rounds are keyed by round number and the **first** occurrence wins: at least
    one of these logs has a hand-written annotation at the bottom that re-quotes
    two earlier rounds verbatim, and those must not become duplicate rows.
    """
    seen: dict[int, dict] = {}
    cur: dict | None = None
    meta: dict = {"path": str(path)}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = _ROUND.search(line)
        if m:
            rnd = int(m.group(1))
            cur = None if rnd in seen else {"round": rnd}
            if cur is not None:
                seen[rnd] = cur
            continue
        if (m := _BEST.search(line)):
            meta.setdefault("best_margin", float(m.group(1)))
            meta.setdefault("best_round", int(m.group(2)))
            continue
        if cur is None:
            continue
        if (m := _REWARD.search(line)):
            cur.update(seats=int(m.group(1)), r_mean=float(m.group(2)),
                       r_std=float(m.group(3)), a_std=float(m.group(4)),
                       a_near0=float(m.group(5)) / 100, a_big=float(m.group(6)) / 100,
                       a_max=float(m.group(7)))
        elif (m := _EXPLORE.search(line)):
            cur.update(entropy=float(m.group(1)), ppl=float(m.group(2)),
                       off_argmax=float(m.group(3)) / 100)
        elif (m := _MOVE.search(line)):
            cur.update(mean_logpi=float(m.group(1)), policy_loss=float(m.group(2)),
                       grad_norm=float(m.group(3)), kl_step=float(m.group(4)),
                       kl_init=float(m.group(5)), lr=float(m.group(6)))
        elif (m := _PROGRESS.search(line)):
            cur.update(margin=float(m.group(1)), window=int(m.group(2)))
    rounds = [seen[r] for r in sorted(seen) if "kl_step" in seen[r]]
    return rounds, meta


def analysis3() -> dict:
    out = {"runs": {}, "pooled": {}}
    pooled: dict[str, list[float]] = defaultdict(list)
    for run in RUNS:
        path = G4 / f"{run}{LOG_SUFFIX}"
        if not path.exists():
            raise SystemExit(f"missing log {path}")
        rounds, meta = parse_log(path)
        arr = {k: np.array([r[k] for r in rounds], dtype=float)
               for k in ("round", "kl_step", "kl_init", "r_std", "a_near0",
                         "a_big", "a_max", "grad_norm", "lr", "margin",
                         "entropy", "off_argmax", "a_std", "r_mean")
               if all(k in r for r in rounds)}
        n = len(rounds)
        log_kl = np.log(np.maximum(arr["kl_step"], 1e-12))
        corrs = {}
        for name, x in (("reward_std", arr["r_std"]),
                        ("adv_near_zero_frac", arr["a_near0"]),
                        ("adv_frac_gt_0.5", arr["a_big"]),
                        ("adv_max_abs", arr["a_max"]),
                        ("adv_std", arr.get("a_std", np.zeros(n))),
                        ("grad_norm_preclip", arr["grad_norm"]),
                        ("lr", arr["lr"])):
            r_p, se_p = pearson(arr["kl_step"], x)
            r_s, _ = spearman(arr["kl_step"], x)
            r_l, se_l = pearson(log_kl, x)
            corrs[name] = {
                "pearson_r": r_p, "pearson_se": se_p,
                "spearman_rho": r_s,
                "pearson_r_logKL": r_l, "pearson_se_logKL": se_l,
                "x_mean": float(x.mean()), "x_std": float(x.std()),
            }
        # KL(pi_0||pi_k) accumulation: is it ~linear (constant drift) in rounds?
        acc = {}
        if n > 5:
            lr_change = arr["lr"] != arr["lr"][0]
            acc["kl_init_final"] = float(arr["kl_init"][-1])
            acc["kl_init_at_best"] = (
                float(arr["kl_init"][np.argmin(np.abs(arr["round"] - meta.get(
                    "best_round", arr["round"][-1])))])
                if "best_round" in meta else None)
            acc["r_vs_round_linear"] = pearson(arr["round"], arr["kl_init"])[0]
            acc["r_vs_sqrt_round"] = pearson(
                np.sqrt(arr["round"] + 1), arr["kl_init"])[0]
            acc["mean_kl_step"] = float(arr["kl_step"].mean())
            acc["sum_kl_step"] = float(arr["kl_step"].sum())
            acc["kl_init_over_sum_kl_step"] = (
                acc["kl_init_final"] / acc["sum_kl_step"]
                if acc["sum_kl_step"] else None)
            acc["frac_rounds_after_lr_drop"] = float(lr_change.mean())
        best_round = meta.get("best_round")
        frac = (best_round / arr["round"][-1]) if best_round is not None else None
        out["runs"][run] = {
            "n_rounds_parsed": n,
            "first_round": int(arr["round"][0]),
            "last_round": int(arr["round"][-1]),
            "grad_norm_mean": float(arr["grad_norm"].mean()),
            "grad_norm_p10_p90": [float(np.percentile(arr["grad_norm"], 10)),
                                  float(np.percentile(arr["grad_norm"], 90))],
            "frac_rounds_clipped_at_1.0": float((arr["grad_norm"] > 1.0).mean()),
            "kl_step_mean": float(arr["kl_step"].mean()),
            "kl_step_cv": float(arr["kl_step"].std() / arr["kl_step"].mean()),
            "best_margin": meta.get("best_margin"),
            "best_round": best_round,
            "best_round_frac_of_run": frac,
            "margin_final": float(arr["margin"][-1]),
            "correlations_kl_step_vs": corrs,
            "kl_init_accumulation": acc,
        }
        for key in ("kl_step", "r_std", "a_near0", "a_big", "a_max", "grad_norm"):
            # z-score within run before pooling: runs differ in LR schedule
            v = arr[key]
            pooled[key].extend(((v - v.mean()) / (v.std() or 1.0)).tolist())

    pk = {k: np.array(v) for k, v in pooled.items()}
    out["pooled"] = {
        "n_rounds": len(pk["kl_step"]),
        "note": "each series z-scored within run before pooling",
        "correlations_kl_step_vs": {
            name: dict(zip(("pearson_r", "pearson_se"),
                           pearson(pk["kl_step"], pk[name])))
            for name in ("r_std", "a_near0", "a_big", "a_max", "grad_norm")
        },
    }
    return out


# ---------------------------------------------------------------------- report

def print_terms(res: dict, title: str) -> None:
    print(f"  {title}  (n={res['n']} seats, {res['n_clusters']} draft clusters; "
          f"base {res['base']})")
    for t in res["terms"]:
        if t["name"] == "intercept":
            continue
        print(f"    {t['name']:<24} {t['beta']:+.4f} +- {t['se']:.4f}"
              f"  t={t['t']:+.2f} {stars(t['beta'], t['se'])}")


def report(a1: dict, a2: dict, a3: dict) -> None:
    line = "=" * 78
    print(line)
    print("D5 - corpus probes: seating geometry, build-around traps, step size")
    print(line)

    print("\n[A1] THE GEOMETRY THE MODEL CANNOT SEE")
    print(f"  pooled over 4 yardstick corpora: {a1['n_all_seats']} seats, "
          f"{a1['n_gen4_seats']} of them gen-4")
    m = a1["gen4_mean_loo"]
    print(f"  gen-4 mean pod-relative deck_score: {fmt_pm(m['mean'], m['se'])}")
    c = a1["crowding_only"]
    print(f"  crowding slope (gen-4 seats, loo ~ pod gen-4 count): "
          f"{c['slope']:+.4f} +- {c['se']:.4f} per added gen-4 seat")
    print("\n  -- gen-4 seats only, neighbour label WITHOUT crowding control --")
    print_terms(a1["upstream_gen4seats_nocontrol"], "(a) pack-1 upstream neighbour")
    print_terms(a1["downstream_gen4seats_nocontrol"], "(b) pack-1 downstream neighbour")
    print("\n  -- same, NET of pod gen-4 count (the crowding control) --")
    print_terms(a1["upstream_gen4seats"], "(a) pack-1 upstream neighbour")
    print_terms(a1["downstream_gen4seats"], "(b) pack-1 downstream neighbour")
    print_terms(a1["joint_gen4seats"], "(a+b) both neighbours in one model")

    for which, key in (("upstream", "upstream_within_count"),
                       ("downstream", "downstream_within_count")):
        print(f"\n  -- (c) within fixed pod gen-4 count: gen-4 seat mean loo by "
              f"{which} label --")
        print(f"    {'pod g4':>6} {'n':>5} | " +
              " | ".join(f"{l:>22}" for l in LABELS))
        for cell in a1[key]:
            if cell["n"] < 40:
                continue
            parts = []
            for l in LABELS:
                v = cell[l]
                parts.append("                  n/a" if not v else
                             f"{v['mean']:+.3f}+-{v['se']:.3f} (n={v['n']:>4})")
            print(f"    {cell['pod_gen4_count']:>6} {cell['n']:>5} | " +
                  " | ".join(f"{p:>22}" for p in parts))

    print("\n  -- mirror: ALL seats, own label + pod gen-4 count controlled --")
    print_terms(a1["mirror_all_seats_upstream"],
                "outcome loo ~ upstream neighbour label")

    print("\n[A2] BUILD-AROUND TRAPS (AI:RemoveDeck:Random / RemRandomDecks)")
    pa = a2["per_agent"]
    print(f"    {'agent':<12} {'picks':>9} {'flagged avail':>14} {'take%':>8} "
          f"{'real-choice take%':>19} {'unflagged-alt take%':>21}")
    for l in LABELS:
        d = pa[l]
        a, b, cc = (d["take_rate_when_available"], d["take_rate_real_choice"],
                    d["take_rate_unflagged_alt"])
        print(f"    {l:<12} {d['picks']:>9} {a['n']:>14} "
              f"{100*a['rate']:>7.2f}% "
              f"{100*b['rate']:>10.2f}% (n={b['n']:>6}) "
              f"{100*cc['rate']:>11.2f}% (n={cc['n']:>6})")
    print()
    print(f"    {'agent':<12} {'flagged taken':>14} {'ended in 40-card deck':>23} "
          f"{'mean flagged/seat':>19}")
    for l in LABELS:
        d, bb = pa[l], a2["by_flagged_count"][l]
        p = d["flagged_taken_played"]
        print(f"    {l:<12} {d['flagged_taken']:>14} "
              f"{100*p['rate']:>16.2f}% +-{100*p['se']:.2f} "
              f"{bb['mean_flagged_drafted']:>19.3f}")
    print("\n  -- contrasts (gen-1 is the distillation-of-Forge control) --")
    for k, v in a2["contrasts"].items():
        print(f"    {k:<48} {100*v['diff']:+7.2f} pp  +-{100*v['se']:.2f}  "
              f"z={v['z']:+.2f} {stars(v['diff'], v['se'])}")
    print("\n  -- mean pod-relative deck_score by flagged cards drafted --")
    for l in LABELS:
        bb = a2["by_flagged_count"][l]
        cells = " | ".join(
            f"{c['flagged_drafted']}: {c['mean']:+.3f}+-{c['se']:.3f} (n={c['n']})"
            for c in bb["cells"])
        s = bb["slope_per_flagged_card"]
        print(f"    {l:<12} {cells}")
        print(f"    {'':<12} slope per flagged card (net of crowding): "
              f"{s['beta']:+.4f} +- {s['se']:.4f} {stars(s['beta'], s['se'])}")

    print("\n[A3] DID THE SIGNAL DRIVE THE STEP?")
    for run, d in a3["runs"].items():
        print(f"\n  {run}")
        print(f"    rounds parsed {d['n_rounds_parsed']} "
              f"({d['first_round']}..{d['last_round']}); "
              f"pre-clip grad norm mean {d['grad_norm_mean']:.2f} "
              f"(p10-p90 {d['grad_norm_p10_p90'][0]:.2f}-"
              f"{d['grad_norm_p10_p90'][1]:.2f}); "
              f"{100*d['frac_rounds_clipped_at_1.0']:.1f}% of rounds clipped")
        print(f"    KL(pi_k||pi_k+1) mean {d['kl_step_mean']:.5f} "
              f"(CV {d['kl_step_cv']:.2f})")
        for name, c in d["correlations_kl_step_vs"].items():
            if math.isnan(c["pearson_r"]):
                print(f"      vs {name:<20} degenerate: constant at "
                      f"{c['x_mean']:.3f} (round-standardised by construction)")
                continue
            print(f"      vs {name:<20} r={c['pearson_r']:+.3f}"
                  f"+-{c['pearson_se']:.3f}  rho={c['spearman_rho']:+.3f}  "
                  f"r(logKL)={c['pearson_r_logKL']:+.3f}")
        acc = d["kl_init_accumulation"]
        print(f"    KL(pi_0||pi_k): final {acc['kl_init_final']:.3f}; "
              f"r vs round {acc['r_vs_round_linear']:+.3f}, "
              f"vs sqrt(round) {acc['r_vs_sqrt_round']:+.3f}; "
              f"sum of per-round KL {acc['sum_kl_step']:.3f} "
              f"(ratio {acc['kl_init_over_sum_kl_step']:.2f})")
        print(f"    best margin {d['best_margin']:+.3f} at round {d['best_round']}"
              f" = {100*d['best_round_frac_of_run']:.0f}% through the run; "
              f"KL(pi_0||pi_k) there {acc['kl_init_at_best']:.3f}; "
              f"final margin {d['margin_final']:+.3f}")
    p = a3["pooled"]
    print(f"\n  pooled ({p['n_rounds']} rounds, {p['note']}):")
    for name, c in p["correlations_kl_step_vs"].items():
        print(f"    KL(pi_k||pi_k+1) vs {name:<12} r={c['pearson_r']:+.3f}"
              f" +- {c['pearson_se']:.3f}")
    print("\n" + line)


# ------------------------------------------------------------------------ main

def write_csvs(out_dir: Path, a1: dict, a2: dict, a3: dict) -> list[Path]:
    written = []
    p = out_dir / "d5_a1_neighbour_terms.csv"
    with p.open("w", encoding="utf-8", newline="") as h:
        w = csv.writer(h)
        w.writerow(["model", "base", "n", "n_clusters", "term", "beta", "se", "t"])
        for key in ("upstream_gen4seats", "downstream_gen4seats", "joint_gen4seats",
                    "upstream_gen4seats_nocontrol", "downstream_gen4seats_nocontrol",
                    "mirror_all_seats_upstream", "mirror_all_seats_upstream_nocontrol"):
            res = a1[key]
            for t in res["terms"]:
                w.writerow([key, res["base"], res["n"], res["n_clusters"],
                            t["name"], f"{t['beta']:.6f}", f"{t['se']:.6f}",
                            "" if t["t"] is None else f"{t['t']:.4f}"])
    written.append(p)

    p = out_dir / "d5_a2_flagged_rates.csv"
    with p.open("w", encoding="utf-8", newline="") as h:
        w = csv.writer(h)
        w.writerow(["agent", "metric", "k", "n", "rate", "se"])
        for label, d in a2["per_agent"].items():
            for key in ("take_rate_when_available", "take_rate_real_choice",
                        "take_rate_unflagged_alt", "flagged_taken_played"):
                r = d[key]
                w.writerow([label, key, r["k"], r["n"],
                            "" if r["rate"] is None else f"{r['rate']:.6f}",
                            "" if r["se"] is None else f"{r['se']:.6f}"])
    written.append(p)

    p = out_dir / "d5_a3_run_summary.csv"
    with p.open("w", encoding="utf-8", newline="") as h:
        w = csv.writer(h)
        w.writerow(["run", "n_rounds", "grad_norm_mean", "kl_step_mean",
                    "corr_kl_step_reward_std", "corr_kl_step_adv_near0",
                    "corr_kl_step_adv_max", "corr_kl_step_grad_norm",
                    "kl_init_final", "best_margin", "best_round",
                    "best_round_frac", "margin_final"])
        for run, d in a3["runs"].items():
            c = d["correlations_kl_step_vs"]
            w.writerow([
                run, d["n_rounds_parsed"], f"{d['grad_norm_mean']:.4f}",
                f"{d['kl_step_mean']:.6f}",
                f"{c['reward_std']['pearson_r']:.4f}",
                f"{c['adv_near_zero_frac']['pearson_r']:.4f}",
                f"{c['adv_max_abs']['pearson_r']:.4f}",
                f"{c['grad_norm_preclip']['pearson_r']:.4f}",
                f"{d['kl_init_accumulation']['kl_init_final']:.4f}",
                d["best_margin"], d["best_round"],
                f"{d['best_round_frac_of_run']:.4f}", f"{d['margin_final']:.4f}"])
    written.append(p)
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT,
                    help="where the d5_*.json / d5_*.csv staging files go")
    ap.add_argument("--skip", nargs="*", default=[], choices=["a1", "a2", "a3"],
                    help="analyses to skip")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    flagged = load_flagged()
    print(f"flagged (RemRandomDecks) card names: {len(flagged)}", file=sys.stderr)
    rows, stats = build_tables(flagged)
    print(f"seat rows: {len(rows)}", file=sys.stderr)

    a1 = analysis1(rows) if "a1" not in args.skip else {}
    a2 = analysis2(rows, stats) if "a2" not in args.skip else {}
    a3 = analysis3() if "a3" not in args.skip else {}

    blob = {
        "corpora": [str(G4 / f"{r}{CORPUS_SUFFIX}") for r in RUNS],
        "logs": [str(G4 / f"{r}{LOG_SUFFIX}") for r in RUNS],
        "n_flagged_names": len(flagged),
        "a1_seating_geometry": a1,
        "a2_build_around_traps": a2,
        "a3_step_vs_signal": a3,
    }
    out_json = args.out_dir / "d5_corpus.json"
    out_json.write_text(json.dumps(blob, indent=2), encoding="utf-8")
    written = write_csvs(args.out_dir, a1, a2, a3)

    report(a1, a2, a3)
    print(f"wrote {out_json}")
    for p in written:
        print(f"wrote {p}")


if __name__ == "__main__":
    main()

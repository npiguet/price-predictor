"""T4: does the gen-4 sealed scorer detect SYNERGY?

The scorer is permutation-invariant and behaves like a mean over per-card
representations (T6-P2b: PMA attention entropy 0.999993; T5: shuffling the
text<->det binding inside a deck barely moves held-out accuracy). Under a
strict mean, a card's contribution cannot depend on its deckmates, so synergy
-- if the model has any -- must live in the SAB layers shifting a *payoff's*
contextual value as the density of its *enablers* rises.

Three probes, all fixed-deck-size swap designs (every deck in a comparison has
exactly the same number of cards, so the 23-spell size prior found in T1 can
never leak into a contrast):

P-A  dose-response Delta-Delta (the headline).
     For a curated payoff/enabler/control triple (synergy_pairs.json) and real
     same-set context decks, swap k enablers into the deck's k worst slots,
     then measure what the payoff is worth in that deck vs what a same-set,
     non-synergistic control card is worth in the SAME slot:
         m_pay(k) = s(D_k + payoff) - s(D_k)
         m_ctl(k) = s(D_k + control) - s(D_k)
         d(k)     = m_pay(k) - m_ctl(k)
         Delta-dose = d(kmax) - d(0)      [payoff-vs-control gap, dosed]
     d(k) removes the same filler card in both arms, so Delta-dose is a clean
     double difference: it cancels both the filler's own value and the
     payoff/control standalone-quality gap.
     MISMATCHED-ENABLER ARM: the same dose is repeated with the enablers of a
     DIFFERENT entry (different mechanism family, same count, preferably same
     set). If the payoff's marginal rises just as much with mismatched
     enablers, the "synergy" is generic deck-quality / color drift.

P-B  duplicates ladder.
     Swap k copies of one card A into the k worst slots (k = 1..3) and compare
     the per-copy marginal against k DISTINCT cards whose single-swap marginal
     in that same context was matched to A's (measured in a pre-pass). Flat,
     equal marginals = the model neither penalises nor rewards duplicates
     beyond what mean-pooling implies.

P-C  removal-share ladder (role balance).
     Push a deck's removal count from -2 to +3 by q-matched, on-color, same-set
     swaps and ask whether the score has an interior optimum in removal share.
     Every rung is a swap, and T3 established that ANY swap out of a built deck
     costs about -0.40, so each rung is also paired against a NEUTRAL control
     that swaps the same number of cards without changing the removal count.
     Net-of-neutral therefore reads as:
       r > 0: same slots vacated, removal swapped in instead of a q-matched
              non-removal card -> "is a removal card worth more in that slot?"
       r < 0: a q-matched non-removal card replaces the deck's worst removal
              spell instead of its worst non-removal spell -> "does losing a
              removal spell cost more than losing an equally-labelled other?"

Read-only w.r.t. the repo and the Y: drive. `--smoke` runs everything tiny on
CPU. Results land in t4_results.json plus a printed markdown summary.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _lib.jsonio import write_json
from _lib.markdown import print_table
from _lib.stats import mean_se
from scorer_probes import probe_lib as pl
from scorer_probes.probe_lib import WUBRG, CardBook, DeckBank, Preset

SEED = 42

PAIRS_JSON = pl.DATA_DIR / "synergy_pairs.json"
OUT_JSON = pl.SCRATCH / "t4_results.json"

PREFERRED_METHODS = ("forge-best", "forge-3sub", "gen4-512", "gen4-256")
RUNGS = (-2, -1, 1, 2, 3)  # P-C removal-count steps away from the deck as built

# loose removal regex, applied to the whole converted text (P-C, creatures)
REMOVAL_TEXT_RE = re.compile(
    r"destroy target|exile target|deals? \d+ damage to", re.IGNORECASE
)

# mechanism -> family, first keyword hit wins (order matters)
MECH_FAMILIES: list[tuple[str, tuple[str, ...]]] = [
    ("gates", ("gate-count",)),
    ("tribal", ("tribal", "tribe", "shared-creature-type", "warriors-matter")),
    ("party", ("party",)),
    ("sacrifice", ("sacrifice", "food", "bargain", "morbid",
                   "enchantment-hits-graveyard")),
    ("graveyard", ("graveyard", "self-mill", "undergrowth", "surveil")),
    ("counters", ("counter",)),
    ("auras_equipment", ("voltron", "aura", "equipment", "equipped", "modified")),
    ("artifacts", ("metalcraft", "artifact-count", "treasure")),
    ("lifegain", ("lifegain",)),
    ("energy", ("energy",)),
    ("cycling", ("cycling", "discard")),
    ("landfall", ("landfall",)),
    ("devotion", ("devotion",)),
    ("colorless", ("processor", "ingest", "colorless-creature", "devoid")),
    ("targeting", ("heroic", "valiant", "target your own", "targets it")),
    ("spells_matter", ("spells-matter", "magecraft", "prowess", "noncreature-spell",
                       "spectacle", "spells-in-graveyard", "instants")),
]


def mechanism_family(mech: str) -> str:
    m = mech.lower()
    for fam, keys in MECH_FAMILIES:
        if any(k in m for k in keys):
            return fam
    return "other"


def summarize(xs) -> dict:
    """mean / SE / n / t / two-sided p of a one-sample (paired) mean."""
    m, se, n = mean_se(xs)
    t = m / se if se and np.isfinite(se) and se > 0 else float("nan")
    p = float(2 * stats.t.sf(abs(t), n - 1)) if n > 1 and np.isfinite(t) else float("nan")
    return {"mean": m, "se": se, "n": n, "t": t, "p": p}


def summary_cell(s: dict) -> str:
    """A ``summarize`` result as one table cell."""
    return (f"{s['mean']:+.4f} +/- {s['se']:.4f}  (n={s['n']:>3}, "
            f"t={s['t']:+.2f}, p={s['p']:.3g})")


def mean_pm(s: dict) -> str:
    return f"{s['mean']:+.4f} +/- {s['se']:.4f}"


class SynergyBook(CardBook):
    """``CardBook`` plus the removal classification probe C counts."""

    def _build(self, name: str) -> dict | None:
        info = super()._build(name)
        if info is None:
            return None
        regex_hit = bool(REMOVAL_TEXT_RE.search(info["text"].lower()))
        is_creature = info["is_creature"]
        info["is_removal"] = regex_hit if is_creature else (
            bool(info["feats"]["is_removal"]) or regex_hit)
        info["removal_creature"] = is_creature and regex_hit
        return info


def make_deck_record(book: CardBook, set_code: str, method: str,
                     names: list[str], did: int) -> dict | None:
    nonbasic = pl.nonbasic(names)
    infos = [book.get(c) for c in nonbasic]
    if any(i is None for i in infos):
        return None
    spell_idx = [i for i, inf in enumerate(infos) if not inf["is_land"]]
    if len(spell_idx) < 8:
        return None
    pips = np.zeros(5)
    for i in spell_idx:
        pips += infos[i]["pips"]
    colors = frozenset(c for c, p in zip(WUBRG, pips) if p > 0)
    # filler slots: the deck's nonland spells ranked by shrunk_score_play
    # (corpus-median fallback for unlabeled cards); bottom-k are the fillers.
    fillers = sorted(spell_idx, key=lambda i: (infos[i]["qv"], nonbasic[i], i))
    return {
        "did": did,
        "set": set_code,
        "method": method,
        "names": tuple(nonbasic),
        "infos": infos,
        "nameset": frozenset(nonbasic),
        "colors": colors,
        "spell_idx": spell_idx,
        "fillers": fillers,
        "n_removal": sum(1 for i in spell_idx if infos[i]["is_removal"]),
        "preferred": method in PREFERRED_METHODS,
    }


def collect_context_decks(book: CardBook, sets_needed: set[str],
                          cap_pref: int, cap_other: int) -> dict[str, list[dict]]:
    """Distinct decks of the wanted sets, capped per (set, preferred?)."""
    out: dict[str, list[dict]] = defaultdict(list)
    kept = unresolvable = 0

    def make(d: pl.MatchDeck) -> dict | None:
        nonlocal unresolvable
        rec = make_deck_record(book, d.set_code, d.method, d.cards, kept)
        unresolvable += rec is None
        return rec

    def group(d: pl.MatchDeck):
        return (d.set_code, d.method in PREFERRED_METHODS) if d.set_code in sets_needed else None

    for rec in pl.sample_match_decks(pl.MATCH_PATHS, lambda g: cap_pref if g[1] else cap_other,
                                     group=group, make=make):
        out[rec["set"]].append(rec)
        kept += 1
    for s in out:
        out[s].sort(key=lambda d: (not d["preferred"], d["did"]))
    print(f"  context decks: {kept} kept, {unresolvable} "
          f"unresolvable, over {len(out)} sets", flush=True)
    return out


def build_set_pools(decks_by_set: dict[str, list[dict]]) -> dict[str, Counter]:
    """Per-set candidate pool: card name -> how many collected decks use it."""
    pools: dict[str, Counter] = {}
    for s, decks in decks_by_set.items():
        c: Counter = Counter()
        for d in decks:
            for n in set(d["names"]):
                c[n] += 1
        pools[s] = c
    return pools


def swapped(names: tuple, repl: dict[int, str]) -> tuple:
    lst = list(names)
    for i, n in repl.items():
        lst[i] = n
    return tuple(lst)


def prepare_entries(raw: list[dict], book: CardBook) -> tuple[list[dict], list[dict]]:
    """Resolve entries, drop the ones with LAND enablers (Guildgates)."""
    entries, dropped = [], []
    for i, e in enumerate(raw):
        cards = {}
        bad = None
        for role, nm in [("payoff", e["payoff"]), ("control", e["control"])]:
            info = book.get(nm)
            if info is None:
                bad = f"unresolvable {role} {nm!r}"
            cards[role] = info
        enablers, land_enablers = [], []
        for nm in e["enablers"]:
            info = book.get(nm)
            if info is None:
                bad = bad or f"unresolvable enabler {nm!r}"
                continue
            (land_enablers if info["is_land"] else enablers).append(nm)
        if land_enablers:
            bad = f"land enablers {land_enablers}"
        if not bad and len(enablers) < 1:
            bad = "no usable enablers"
        rec = {
            "idx": i, "set": e["set"], "payoff": e["payoff"], "control": e["control"],
            "enablers": enablers, "mechanism": e["mechanism"],
            "family": mechanism_family(e["mechanism"]), "strength": e["strength"],
            "payoff_colors": cards["payoff"]["colors"] if cards["payoff"] else frozenset(),
            "control_colors": cards["control"]["colors"] if cards["control"] else frozenset(),
            "enabler_q": float(np.mean([book.get(n)["qv"] for n in enablers]))
                         if enablers else float("nan"),
        }
        if bad:
            dropped.append({**{k: rec[k] for k in ("idx", "set", "payoff", "mechanism")},
                            "reason": bad})
        else:
            entries.append(rec)
    return entries, dropped


def pick_mismatch(entries: list[dict], i: int, kmax: int) -> dict | None:
    """Enablers of another entry: different mechanism family, same count,
    same set if possible (else the closest-quality entry from another set)."""
    e = entries[i]
    block = {e["payoff"], e["control"]} | set(e["enablers"])
    cands = []
    for j, o in enumerate(entries):
        if j == i or o["family"] == e["family"] or len(o["enablers"]) < kmax:
            continue
        mis = o["enablers"][:kmax]
        if block & set(mis):
            continue
        cands.append((o["set"] != e["set"],                    # same set first
                      abs(o["enabler_q"] - e["enabler_q"]),    # closest quality
                      o["idx"], mis, o))
    if not cands:
        return None
    cands.sort(key=lambda c: (c[0], c[1], c[2]))
    same_set, _, _, mis, o = cands[0]
    return {"enablers": mis, "family": o["family"], "set": o["set"],
            "source_payoff": o["payoff"], "same_set": not same_set}


def probe_a(entries: list[dict], decks_by_set: dict[str, list[dict]],
            book: CardBook, bank: DeckBank, n_ctx: int) -> tuple[list[dict], list[dict]]:
    """Build every P-A deck. Returns (cells, skipped)."""
    cells, skipped = [], []
    for ei, e in enumerate(entries):
        kmax = min(3, len(e["enablers"]))
        enablers = e["enablers"][:kmax]
        mis = pick_mismatch(entries, ei, kmax)
        block = {e["payoff"], e["control"]} | set(enablers)
        enab_colors = frozenset().union(*[book.get(n)["colors"] for n in enablers])

        pool = decks_by_set.get(e["set"], [])
        why: Counter = Counter()
        ok = []
        for d in pool:
            if not (e["payoff_colors"] <= d["colors"]):
                why["colors_missing"] += 1
            elif d["nameset"] & block:
                why["already_has_a_card"] += 1
            elif len(d["fillers"]) < kmax + 1:
                why["too_few_filler_slots"] += 1
            else:
                ok.append(d)
        # preferred methods first, then contexts that also cover the enabler /
        # control colors; "relax to any method" is what the stable sort does
        # automatically once fewer than n_ctx preferred contexts qualify.
        ok.sort(key=lambda d: (not d["preferred"],
                               not (enab_colors <= d["colors"]),
                               not (e["control_colors"] <= d["colors"]),
                               d["did"]))
        min_needed = min(3, n_ctx)
        if len(ok) < min_needed:
            skipped.append({"payoff": e["payoff"], "set": e["set"],
                            "n_available": len(ok), "n_same_set_decks": len(pool),
                            "reason": f"only {len(ok)} usable contexts "
                                      f"({len(pool)} same-set decks; {dict(why)})"})
            continue
        chosen = ok[:n_ctx]

        for ctx in chosen:
            f = ctx["fillers"]
            base_names = ctx["names"]
            cell = {
                "entry": ei, "payoff": e["payoff"], "control": e["control"],
                "set": e["set"], "family": e["family"], "strength": e["strength"],
                "ctx": ctx["did"], "method": ctx["method"], "kmax": kmax,
                "preferred_ctx": ctx["preferred"], "n_available": len(ok),
                "enab_oncolor": bool(enab_colors <= ctx["colors"]),
                "ctl_oncolor": bool(e["control_colors"] <= ctx["colors"]),
                "dose": {}, "mis": None,
                "mis_meta": None if mis is None else
                            {k: mis[k] for k in ("family", "set", "same_set",
                                                 "source_payoff")},
            }
            for k in range(kmax + 1):
                repl = {f[j]: enablers[j] for j in range(k)}
                cell["dose"][k] = {
                    "base": bank.add(swapped(base_names, repl)),
                    "pay": bank.add(swapped(base_names, {**repl, f[k]: e["payoff"]})),
                    "ctl": bank.add(swapped(base_names, {**repl, f[k]: e["control"]})),
                }
            if mis is not None and not (ctx["nameset"] & set(mis["enablers"])):
                repl = {f[j]: mis["enablers"][j] for j in range(kmax)}
                cell["mis"] = {
                    "base": bank.add(swapped(base_names, repl)),
                    "pay": bank.add(swapped(base_names, {**repl, f[kmax]: e["payoff"]})),
                    "ctl": bank.add(swapped(base_names, {**repl, f[kmax]: e["control"]})),
                }
            cells.append(cell)
    return cells, skipped


def cell_row(c: dict, bank: DeckBank) -> dict:
    """One entry-context cell's dose curve and double differences."""
    kmax = c["kmax"]
    m_pay, m_ctl, d = {}, {}, {}
    for k, keys in c["dose"].items():
        b = bank.score(keys["base"])
        m_pay[k] = bank.score(keys["pay"]) - b
        m_ctl[k] = bank.score(keys["ctl"]) - b
        d[k] = m_pay[k] - m_ctl[k]
    ks = sorted(d)
    row = {
        "entry": c["entry"], "payoff": c["payoff"], "set": c["set"],
        "family": c["family"], "strength": c["strength"], "ctx": c["ctx"],
        "method": c["method"], "kmax": kmax,
        "preferred_ctx": c["preferred_ctx"], "n_available": c["n_available"],
        "enab_oncolor": c["enab_oncolor"], "ctl_oncolor": c["ctl_oncolor"],
        "m_pay": {k: m_pay[k] for k in ks},
        "m_ctl": {k: m_ctl[k] for k in ks},
        "d0": d[0], "d_max": d[kmax],
        "ddose": d[kmax] - d[0],
        "pay_dose": m_pay[kmax] - m_pay[0],
        "ctl_dose": m_ctl[kmax] - m_ctl[0],
        "slope": float(np.polyfit(ks, [d[k] for k in ks], 1)[0]) if len(ks) > 1
                 else float("nan"),
    }
    if c["mis"]:
        b = bank.score(c["mis"]["base"])
        mp = bank.score(c["mis"]["pay"]) - b
        mc = bank.score(c["mis"]["ctl"]) - b
        row["mis_d"] = mp - mc
        row["mis_ddose"] = (mp - mc) - d[0]
        row["mis_pay_dose"] = mp - m_pay[0]
        row["mis_family"] = c["mis_meta"]["family"]
        row["mis_same_set"] = c["mis_meta"]["same_set"]
    return row


def entry_row(e: dict, ei: int, rows: list[dict]) -> dict:
    """An entry's cells pooled: per-field means with their SE and count."""
    agg = {"entry": ei, "payoff": e["payoff"], "control": e["control"],
           "set": e["set"], "family": e["family"], "strength": e["strength"],
           "mechanism": e["mechanism"], "n_ctx": len(rows),
           "kmax": rows[0]["kmax"],
           "n_available_ctx": rows[0]["n_available"],
           "n_preferred_ctx": sum(r["preferred_ctx"] for r in rows),
           "n_enab_oncolor": sum(r["enab_oncolor"] for r in rows),
           "n_ctl_oncolor": sum(r["ctl_oncolor"] for r in rows),
           "mis_family": next((r["mis_family"] for r in rows
                               if "mis_family" in r), None),
           "n_mis_same_set": sum(1 for r in rows if r.get("mis_same_set"))}
    for fld in ("ddose", "slope", "d0", "d_max", "pay_dose", "ctl_dose",
                "mis_ddose", "mis_pay_dose"):
        m, se, n = mean_se([r.get(fld) for r in rows])
        agg[fld], agg[fld + "_se"], agg[fld + "_n"] = m, se, n
    agg["matched_minus_mismatched"], _, _ = mean_se(
        [r["ddose"] - r["mis_ddose"] for r in rows if "mis_ddose" in r])
    return agg


def analyze_a(cells: list[dict], bank: DeckBank, entries: list[dict],
              skipped: list[dict]) -> dict:
    per_cell = [cell_row(c, bank) for c in cells]
    by_entry: dict[int, list[dict]] = defaultdict(list)
    for r in per_cell:
        by_entry[r["entry"]].append(r)
    per_entry = [entry_row(entries[ei], ei, rows) for ei, rows in sorted(by_entry.items())]

    def col(rows, fld):
        return [r[fld] for r in rows if r.get(fld) is not None and np.isfinite(r.get(fld, np.nan))]

    res = {
        "n_entries": len(per_entry),
        "n_cells": len(per_cell),
        "skipped_entries": skipped,
        "coverage": {
            "cells_preferred_method": sum(r["preferred_ctx"] for r in per_cell),
            "cells_enabler_oncolor": sum(r["enab_oncolor"] for r in per_cell),
            "cells_control_oncolor": sum(r["ctl_oncolor"] for r in per_cell),
            "cells_with_mismatch_arm": sum(1 for r in per_cell if "mis_ddose" in r),
            "cells_mismatch_same_set": sum(1 for r in per_cell
                                           if r.get("mis_same_set")),
            "methods": dict(Counter(r["method"] for r in per_cell)),
        },
        "overall": {
            "ddose": summarize(col(per_entry, "ddose")),
            "slope": summarize(col(per_entry, "slope")),
            "d0_raw_quality_gap": summarize(col(per_entry, "d0")),
            "d_max": summarize(col(per_entry, "d_max")),
            "payoff_dose": summarize(col(per_entry, "pay_dose")),
            "control_dose": summarize(col(per_entry, "ctl_dose")),
        },
        "cell_level": {
            "ddose": summarize([r["ddose"] for r in per_cell]),
            "d0_raw_quality_gap": summarize([r["d0"] for r in per_cell]),
        },
        "by_strength": {},
        "by_family": {},
        "mismatched": {},
        "dose_curve": {},
        "per_entry": per_entry,
        "per_cell": per_cell,
    }

    for st in sorted({r["strength"] for r in per_entry}):
        rows = [r for r in per_entry if r["strength"] == st]
        res["by_strength"][st] = {"ddose": summarize(col(rows, "ddose")),
                                  "d0": summarize(col(rows, "d0"))}
    for fam in sorted({r["family"] for r in per_entry}):
        rows = [r for r in per_entry if r["family"] == fam]
        res["by_family"][fam] = {"n_entries": len(rows),
                                 "ddose": summarize(col(rows, "ddose")),
                                 "d0": summarize(col(rows, "d0"))}

    # dose curve: mean m_pay(k), m_ctl(k), d(k) at each k (cell level)
    for k in range(4):
        mp = [r["m_pay"][k] for r in per_cell if k in r["m_pay"]]
        mc = [r["m_ctl"][k] for r in per_cell if k in r["m_ctl"]]
        if not mp:
            continue
        res["dose_curve"][k] = {"m_pay": summarize(mp), "m_ctl": summarize(mc),
                                "d": summarize([a - b for a, b in zip(mp, mc)])}

    have_mis = [r for r in per_entry if r.get("mis_ddose_n", 0) > 0
                and np.isfinite(r.get("mis_ddose", np.nan))]
    res["mismatched"] = {
        "n_entries": len(have_mis),
        "matched_ddose": summarize([r["ddose"] for r in have_mis]),
        "mismatched_ddose": summarize([r["mis_ddose"] for r in have_mis]),
        "paired_diff": summarize([r["ddose"] - r["mis_ddose"] for r in have_mis]),
        "matched_payoff_dose": summarize([r["pay_dose"] for r in have_mis]),
        "mismatched_payoff_dose": summarize([r["mis_pay_dose"] for r in have_mis]),
        "payoff_dose_paired_diff": summarize(
            [r["pay_dose"] - r["mis_pay_dose"] for r in have_mis]),
        "cell_level_paired_diff": summarize(
            [r["ddose"] - r["mis_ddose"] for r in per_cell if "mis_ddose" in r]),
    }
    return res


def pick_b_cards(pools: dict[str, Counter], book: CardBook,
                 n_cards: int) -> list[dict]:
    """Per set: the 2 most-played creatures with label q in [0.03, 0.10]."""
    per_set: dict[str, list[str]] = {}
    for s in sorted(pools):
        cands = []
        for name, freq in pools[s].items():
            info = book.get(name)
            if info is None or info["is_land"] or info["hybrid"]:
                continue
            if not info["is_creature"] or info["q"] is None:
                continue
            if 0.03 <= info["q"] <= 0.10:
                cands.append((-freq, name))
        cands.sort()
        per_set[s] = [n for _, n in cands[:2]]
    out = []
    for rank in (0, 1):
        for s in sorted(per_set):
            if rank < len(per_set[s]) and len(out) < n_cards:
                out.append({"card": per_set[s][rank], "set": s})
    return out[:n_cards]


def probe_b_prepass(cards: list[dict], decks_by_set: dict[str, list[dict]],
                    pools: dict[str, Counter], book: CardBook, bank: DeckBank,
                    n_ctx: int, n_cand: int) -> list[dict]:
    units = []
    for c in cards:
        A = c["card"]
        ainfo = book.get(A)
        decks = [d for d in decks_by_set.get(c["set"], [])
                 if A not in d["nameset"] and len(d["fillers"]) >= 3
                 and ainfo["colors"] <= d["colors"]]
        decks.sort(key=lambda d: (not d["preferred"], d["did"]))
        for ctx in decks[:n_ctx]:
            cands = []
            for name, freq in pools[c["set"]].items():
                if name == A or name in ctx["nameset"]:
                    continue
                info = book.get(name)
                if info is None or info["is_land"] or info["hybrid"]:
                    continue
                if not (info["colors"] <= ctx["colors"]):
                    continue
                cands.append((-freq, name))
            cands.sort()
            cand_names = [n for _, n in cands[:n_cand]]
            if len(cand_names) < 4:
                continue
            f = ctx["fillers"]
            base = bank.add(ctx["names"])
            single = {n: bank.add(swapped(ctx["names"], {f[0]: n}))
                      for n in [A] + cand_names}
            units.append({"card": A, "set": c["set"], "ctx": ctx["did"],
                          "names": ctx["names"], "fillers": f[:3],
                          "cands": cand_names, "base": base, "single": single})
    return units


def probe_b_ladders(units: list[dict], bank: DeckBank) -> None:
    for u in units:
        d_a = bank.score(u["single"][u["card"]]) - bank.score(u["base"])
        deltas = [(abs((bank.score(u["single"][n]) - bank.score(u["base"])) - d_a), n)
                  for n in u["cands"]]
        deltas.sort()
        matched = [n for _, n in deltas[:3]]
        u["d_a"] = d_a
        u["matched"] = matched
        u["matched_d"] = [bank.score(u["single"][n]) - bank.score(u["base"]) for n in matched]
        f = u["fillers"]
        u["copies"] = [bank.add(swapped(u["names"], {f[j]: u["card"]
                                                     for j in range(k)}))
                       for k in range(1, 4)]
        u["distinct"] = [bank.add(swapped(u["names"], {f[j]: matched[j]
                                                       for j in range(k)}))
                         for k in range(1, 4)]


def analyze_b(units: list[dict], bank: DeckBank) -> dict:
    rows = []
    for u in units:
        s0 = bank.score(u["base"])
        cop = [s0] + [bank.score(k) for k in u["copies"]]
        dis = [s0] + [bank.score(k) for k in u["distinct"]]
        rows.append({
            "card": u["card"], "set": u["set"], "ctx": u["ctx"],
            "single_delta": u["d_a"],
            "match_err": float(np.mean([abs(d - u["d_a"]) for d in u["matched_d"]])),
            "copy_marg": [cop[k] - cop[k - 1] for k in (1, 2, 3)],
            "dist_marg": [dis[k] - dis[k - 1] for k in (1, 2, 3)],
            "copy_total": cop[3] - cop[0],
            "dist_total": dis[3] - dis[0],
        })
    res = {"n_units": len(rows),
           "n_cards": len({r["card"] for r in rows}),
           "match_err": summarize([r["match_err"] for r in rows]),
           "by_k": {}, "totals": {}, "per_unit": rows}
    for k in (1, 2, 3):
        cm = [r["copy_marg"][k - 1] for r in rows]
        dm = [r["dist_marg"][k - 1] for r in rows]
        res["by_k"][k] = {"copy": summarize(cm), "distinct": summarize(dm),
                          "copy_minus_distinct": summarize(
                              [a - b for a, b in zip(cm, dm)])}
    res["totals"] = {
        "copy": summarize([r["copy_total"] for r in rows]),
        "distinct": summarize([r["dist_total"] for r in rows]),
        "copy_minus_distinct": summarize(
            [r["copy_total"] - r["dist_total"] for r in rows]),
    }
    return res


def q_match(out_q: float, pool: list[str], used: set[str], book: CardBook) -> str | None:
    best, bd = None, None
    for n in pool:
        if n in used:
            continue
        d = abs(book.get(n)["qv"] - out_q)
        if bd is None or d < bd:
            best, bd = n, d
    return best


def probe_c(decks: list[dict], pools: dict[str, Counter], book: CardBook,
            bank: DeckBank, n_decks: int, rng: random.Random) -> list[dict]:
    units = []
    order = list(decks)
    order.sort(key=lambda d: (not d["preferred"], d["did"]))
    rng.shuffle(order)
    order.sort(key=lambda d: not d["preferred"])  # stable: preferred first
    for ctx in order:
        if len(units) >= n_decks:
            break
        infos, names = ctx["infos"], ctx["names"]
        rem_slots = sorted([i for i in ctx["spell_idx"] if infos[i]["is_removal"]],
                           key=lambda i: (infos[i]["qv"], names[i], i))
        non_slots = sorted([i for i in ctx["spell_idx"] if not infos[i]["is_removal"]],
                          key=lambda i: (infos[i]["qv"], names[i], i))
        if len(rem_slots) < 2 or len(non_slots) < 3:
            continue
        r_pool, n_pool = [], []
        for name in pools[ctx["set"]]:
            if name in ctx["nameset"]:
                continue
            info = book.get(name)
            if info is None or info["is_land"] or info["hybrid"]:
                continue
            if not (info["colors"] <= ctx["colors"]):
                continue
            (r_pool if info["is_removal"] else n_pool).append(name)
        r_pool.sort()
        n_pool.sort()
        if len(r_pool) < 3 or len(n_pool) < 6:
            continue

        rungs: dict[int, tuple] = {}
        # r > 0: worst non-removal slots -> q-matched removal from the same set
        used: set[str] = set()
        repl: dict[int, str] = {}
        for j in range(3):
            pick = q_match(infos[non_slots[j]]["qv"], r_pool, used, book)
            if pick is None:
                break
            used.add(pick)
            repl[non_slots[j]] = pick
            rungs[j + 1] = bank.add(swapped(names, dict(repl)))
        # r < 0: worst removal slots -> q-matched non-removal from the same set
        used, repl = set(), {}
        for j in range(2):
            pick = q_match(infos[rem_slots[j]]["qv"], n_pool, used, book)
            if pick is None:
                break
            used.add(pick)
            repl[rem_slots[j]] = pick
            rungs[-(j + 1)] = bank.add(swapped(names, dict(repl)))
        # neutral control: same number of swaps, removal count unchanged
        neutral: dict[int, tuple] = {}
        used, repl = set(), {}
        for j in range(3):
            pick = q_match(infos[non_slots[j]]["qv"], n_pool, used, book)
            if pick is None:
                break
            used.add(pick)
            repl[non_slots[j]] = pick
            neutral[j + 1] = bank.add(swapped(names, dict(repl)))
        if len(rungs) < 5 or len(neutral) < 3:
            continue
        units.append({"ctx": ctx["did"], "set": ctx["set"], "method": ctx["method"],
                      "base": bank.add(names), "base_removal": ctx["n_removal"],
                      "n_spells": len(ctx["spell_idx"]),
                      "rungs": rungs, "neutral": neutral})
    return units


def count_removal(key: tuple, book: CardBook) -> int:
    return sum(1 for n in key
               if not book.get(n)["is_land"] and book.get(n)["is_removal"])


def analyze_c(units: list[dict], bank: DeckBank, book: CardBook) -> dict:
    rows = []
    for u in units:
        s0 = bank.score(u["base"])
        row = {"ctx": u["ctx"], "set": u["set"], "base_removal": u["base_removal"],
               "n_spells": u["n_spells"], "rung_delta": {}, "rung_removal": {},
               "neutral_delta": {}, "net": {}}
        for r, key in u["rungs"].items():
            row["rung_delta"][r] = bank.score(key) - s0
            row["rung_removal"][r] = count_removal(bank.keys[key], book)
        for n, key in u["neutral"].items():
            row["neutral_delta"][n] = bank.score(key) - s0
        for r in row["rung_delta"]:
            nswap = abs(r)
            if nswap in row["neutral_delta"]:
                row["net"][r] = row["rung_delta"][r] - row["neutral_delta"][nswap]
        rows.append(row)

    res = {"n_decks": len(rows), "by_rung": {}, "neutral": {},
           "base_removal": summarize([r["base_removal"] for r in rows]),
           "removal_share": summarize([r["base_removal"] / r["n_spells"]
                                       for r in rows]),
           "per_deck": rows}
    for r in RUNGS:
        d = [row["rung_delta"][r] for row in rows if r in row["rung_delta"]]
        net = [row["net"][r] for row in rows if r in row["net"]]
        real = [row["rung_removal"][r] for row in rows if r in row["rung_removal"]]
        if not d:
            continue
        res["by_rung"][r] = {
            "delta": summarize(d), "net_of_neutral": summarize(net),
            "realized_removal": summarize(real),
            "realized_minus_base": summarize(
                [row["rung_removal"][r] - row["base_removal"]
                 for row in rows if r in row["rung_removal"]]),
        }
    for n in (1, 2, 3):
        d = [row["neutral_delta"][n] for row in rows if n in row["neutral_delta"]]
        if d:
            res["neutral"][n] = summarize(d)

    # net delta binned by the deck's own base removal count (is the optimum
    # interior and does it move with where the deck starts?)
    by_base: dict[int, dict] = {}
    for b in sorted({r["base_removal"] for r in rows}):
        sel = [r for r in rows if r["base_removal"] == b]
        if len(sel) < 5:
            continue
        by_base[b] = {"n": len(sel),
                      "net": {r: summarize([s["net"][r] for s in sel if r in s["net"]])
                              for r in RUNGS}}
    res["by_base_removal"] = by_base
    return res


def report_a(res: dict) -> None:
    print("\n## P-A -- dose-response Delta-Delta (payoff vs control, "
          "enabler dose 0..kmax)\n")
    print(f"entries used: {res['n_entries']}  entry-context cells: {res['n_cells']}")
    cov = res["coverage"]
    print(f"contexts: {cov['cells_preferred_method']}/{res['n_cells']} from "
          f"preferred methods, {cov['cells_enabler_oncolor']} with all enabler "
          f"colors on-color, {cov['cells_control_oncolor']} with the control "
          f"on-color; mismatch arm on {cov['cells_with_mismatch_arm']} cells "
          f"({cov['cells_mismatch_same_set']} same-set)")
    if res["skipped_entries"]:
        print("skipped entries:")
        for s in res["skipped_entries"]:
            print(f"  - {s['payoff']} ({s['set']}): {s['reason']}")
    o = res["overall"]
    print()
    print_table(["statistic (mean over entries)", "value"], [
        ["Delta-dose = d(kmax) - d(0)", summary_cell(o["ddose"])],
        ["slope of d(k) over k", summary_cell(o["slope"])],
        ["d(0) = m_pay(0) - m_ctl(0) (standalone gap)", summary_cell(o["d0_raw_quality_gap"])],
        ["d(kmax)", summary_cell(o["d_max"])],
        ["payoff dose m_pay(kmax) - m_pay(0)", summary_cell(o["payoff_dose"])],
        ["control dose m_ctl(kmax) - m_ctl(0)", summary_cell(o["control_dose"])],
        ["Delta-dose (cell level, unpooled)", summary_cell(res["cell_level"]["ddose"])],
    ])

    print()
    print_table(["dose k", "m_pay(k)", "m_ctl(k)", "d(k)"], [
        [k, mean_pm(row["m_pay"]), mean_pm(row["m_ctl"]),
         f"{mean_pm(row['d'])} (n={row['d']['n']})"]
        for k, row in sorted(res["dose_curve"].items(), key=lambda kv: int(kv[0]))])

    print()
    split_rows = [[f"strength={st}", summary_cell(row["ddose"]), f"{row['d0']['mean']:+.4f}"]
                  for st, row in res["by_strength"].items()]
    split_rows += [[f"family={fam} ({row['n_entries']})", summary_cell(row["ddose"]),
                    f"{row['d0']['mean']:+.4f}"]
                   for fam, row in sorted(res["by_family"].items(),
                                          key=lambda kv: -kv[1]["ddose"]["mean"])]
    print_table(["split", "Delta-dose", "d(0)"], split_rows)

    m = res["mismatched"]
    print("\n### matched vs MISMATCHED enablers (the leakage control)\n")
    print(f"entries with a mismatched arm: {m['n_entries']}")
    print_table(["arm", "Delta-dose", "payoff dose"], [
        ["matched enablers", summary_cell(m["matched_ddose"]),
         summary_cell(m["matched_payoff_dose"])],
        ["mismatched enablers", summary_cell(m["mismatched_ddose"]),
         summary_cell(m["mismatched_payoff_dose"])],
        ["paired difference", summary_cell(m["paired_diff"]),
         summary_cell(m["payoff_dose_paired_diff"])],
        ["paired diff (cell level)", summary_cell(m["cell_level_paired_diff"]), ""],
    ])
    print("  [matched ~ mismatched => generic deck-quality / color drift, "
          "not synergy]")
    print("  [the 'payoff dose' column alone is NOT a synergy measure: it moves "
          "with anything the enablers do to the deck. Only the Delta-dose "
          "column (payoff minus its same-set control in the same slot) is.]")

    top = sorted(res["per_entry"], key=lambda r: -r["ddose"])
    for title, rows in (("top-5 entries by Delta-dose", top[:5]), ("bottom-5 entries", top[-5:])):
        print()
        print_table([title, "set", "family", "Delta-dose +/- SE", "n"], [
            [r["payoff"], r["set"], r["family"], f"{r['ddose']:+.4f} +/- {r['ddose_se']:.4f}",
             r["n_ctx"]] for r in rows])


def report_b(res: dict) -> None:
    print("\n## P-B -- duplicates ladder (k copies of A vs k marginal-matched "
          "distinct cards)\n")
    print(f"units (card x context): {res['n_units']}  distinct cards: {res['n_cards']}")
    print(f"single-swap matching error |dX - dA|: {summary_cell(res['match_err'])}")
    print()
    rows = [[k, summary_cell(row["copy"]), summary_cell(row["distinct"]),
             summary_cell(row["copy_minus_distinct"])]
            for k, row in ((k, res["by_k"][k]) for k in (1, 2, 3))]
    t = res["totals"]
    rows.append(["total (k=3)", summary_cell(t["copy"]), summary_cell(t["distinct"]),
                 summary_cell(t["copy_minus_distinct"])])
    print_table(["k", "copy marginal", "distinct marginal", "copy - distinct"], rows)
    print("  [k=1 is matched by construction; flat copy marginals across k=2,3 "
          "and copy ~ distinct => no duplicate penalty/bonus]")


def report_c(res: dict) -> None:
    print("\n## P-C -- removal-share ladder (fixed size, on-color, q-matched)\n")
    print(f"decks: {res['n_decks']}  base removal count: "
          f"{res['base_removal']['mean']:.2f} +/- {res['base_removal']['se']:.2f} "
          f"(share {res['removal_share']['mean']:.3f})")
    print()
    print_table(["rung r", "realized removal", "delta vs base", "net of neutral swap"], [
        [f"{r:+d}", f"{row['realized_removal']['mean']:.2f} "
                    f"({row['realized_minus_base']['mean']:+.2f})",
         summary_cell(row["delta"]), summary_cell(row["net_of_neutral"])]
        for r, row in ((r, res["by_rung"][r]) for r in RUNGS if r in res["by_rung"])])
    print()
    print_table(["neutral swaps", "delta vs base"],
                [[n, summary_cell(row)]
                 for n, row in sorted(res["neutral"].items(), key=lambda kv: int(kv[0]))])
    if res["by_base_removal"]:
        def net_cell(s: dict | None) -> str:
            s = s or {}
            return f"{s.get('mean', float('nan')):+.3f}" if s.get("n") else "-"

        print()
        print_table(["base removal", "n"] + [f"net({r:+d})" for r in RUNGS], [
            [b, row["n"]] + [net_cell(row["net"].get(r)) for r in RUNGS]
            for b, row in sorted(res["by_base_removal"].items(), key=lambda kv: int(kv[0]))])
    print("  [net > 0 on one side and < 0 on the other => interior optimum; "
          "all net ~ 0 => the scorer has no removal-share opinion]")
    print("  [net(r>0): removal vs q-matched non-removal INTO the same slots; "
          "net(r<0): losing the worst removal spell vs the worst q-matched "
          "non-removal spell]")


SIZES = {
    "entries": Preset(0, 6),
    "contexts": Preset(8, 2),
    "b_cards": Preset(30, 4),
    "b_contexts": Preset(4, 2),
    "c_decks": Preset(150, 20),
    "cap_pref": Preset(220, 40),
    "cap_other": Preset(90, 20),
    "device": Preset(None, "cpu"),
    "batch": Preset(1024, 128),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny CPU run: 6 entries x 2 contexts, 4 P-B cards, "
                         "20 P-C decks")
    ap.add_argument("--only", default="ABC", help="subset of probes to run")
    ap.add_argument("--entries", type=int, default=None)
    ap.add_argument("--contexts", type=int, default=None)
    ap.add_argument("--b-cards", type=int, default=None)
    ap.add_argument("--b-contexts", type=int, default=None)
    ap.add_argument("--b-cands", type=int, default=20,
                    help="single-swap candidates measured per P-B unit")
    ap.add_argument("--c-decks", type=int, default=None)
    ap.add_argument("--cap-pref", type=int, default=None,
                    help="context decks per set from preferred methods")
    ap.add_argument("--cap-other", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", type=Path, default=OUT_JSON)
    args = ap.parse_args()

    pl.apply_presets(args, SIZES)
    smoke = args.smoke
    sections = set(args.only.upper())
    n_entries, n_ctx = args.entries, args.contexts
    b_cards, b_ctx, c_decks = args.b_cards, args.b_contexts, args.c_decks
    cap_pref, cap_other, batch = args.cap_pref, args.cap_other, args.batch

    t_start = time.time()
    rng = random.Random(SEED)
    raw = json.load(open(PAIRS_JSON, encoding="utf-8"))

    print(f"t4_synergy: smoke={smoke} sections={''.join(sorted(sections))} "
          f"seed={SEED} entries={n_entries or 'all'} contexts={n_ctx}")
    probe = pl.Probe(device=args.device)
    print(f"device={probe.device} d_model={probe.d_model}", flush=True)
    book = SynergyBook(probe, pl.load_win_rates(), q_fallback="median")
    print(f"label fallback (corpus median shrunk_score_play): {book.q_fallback:+.5f}")

    entries, dropped = prepare_entries(raw, book)
    print(f"entries: {len(raw)} in file, {len(entries)} usable, "
          f"{len(dropped)} dropped")
    for d in dropped:
        print(f"  dropped {d['payoff']} ({d['set']}): {d['reason']}")
    if n_entries and n_entries < len(entries):
        stride = len(entries) / n_entries          # spread over sets/families
        entries = [entries[int(i * stride)] for i in range(n_entries)]
        print(f"  smoke subset: {[e['payoff'] for e in entries]}")

    sets_needed = sorted({e["set"] for e in entries})
    print(f"collecting context decks for {len(sets_needed)} sets "
          f"(cap {cap_pref} preferred + {cap_other} other per set) ...", flush=True)
    decks_by_set = collect_context_decks(book, set(sets_needed), cap_pref, cap_other)
    pools = build_set_pools(decks_by_set)
    all_decks = [d for s in sorted(decks_by_set) for d in decks_by_set[s]]

    bank = DeckBank(lambda n: book.get(n)["emb"])
    results: dict = {"meta": {
        "seed": SEED, "smoke": smoke, "device": probe.device,
        "sections": sorted(sections), "n_entries_file": len(raw),
        "n_entries_usable": len(entries), "dropped_entries": dropped,
        "n_contexts_per_entry": n_ctx, "b_cards": b_cards, "b_contexts": b_ctx,
        "c_decks": c_decks, "cap_pref": cap_pref, "cap_other": cap_other,
        "sets": sets_needed, "n_context_decks": len(all_decks),
        "checkpoint": str(pl.SCORER_CKPT), "pairs_json": str(PAIRS_JSON),
        "match_files": pl.MATCH_FILES, "preferred_methods": list(PREFERRED_METHODS),
    }}

    a_cells = a_skipped = None
    b_units = None
    c_units = None

    if "A" in sections:
        a_cells, a_skipped = probe_a(entries, decks_by_set, book, bank, n_ctx)
        print(f"P-A: {len(a_cells)} entry-context cells "
              f"({len(a_skipped)} entries skipped)", flush=True)
    if "B" in sections:
        b_cards_sel = pick_b_cards(pools, book, b_cards)
        print(f"P-B: {len(b_cards_sel)} cards "
              f"{[c['card'] for c in b_cards_sel][:6]}{' ...' if len(b_cards_sel) > 6 else ''}",
              flush=True)
        b_units = probe_b_prepass(b_cards_sel, decks_by_set, pools, book, bank,
                                  b_ctx, args.b_cands)
        print(f"P-B: {len(b_units)} card x context units (pre-pass)", flush=True)
    if "C" in sections:
        c_units = probe_c(all_decks, pools, book, bank, c_decks, rng)
        print(f"P-C: {len(c_units)} deck units", flush=True)

    print(f"round 1: {len(bank.keys)} distinct decks to score", flush=True)
    bank.score_pending(probe, batch=batch, label="round1")

    if b_units:
        probe_b_ladders(b_units, bank)
        print(f"round 2: {len(bank.keys)} distinct decks total "
              f"(+{len(bank.keys) - len(bank.scores)} new)", flush=True)
        bank.score_pending(probe, batch=batch, label="round2")

    if a_cells is not None:
        results["A_dose_response"] = analyze_a(a_cells, bank, entries, a_skipped)
        report_a(results["A_dose_response"])
    if b_units is not None:
        results["B_duplicates"] = analyze_b(b_units, bank)
        report_b(results["B_duplicates"])
    if c_units is not None:
        results["C_removal_share"] = analyze_c(c_units, bank, book)
        report_c(results["C_removal_share"])

    results["meta"]["n_distinct_decks_scored"] = len(bank.keys)
    results["meta"]["runtime_s"] = round(time.time() - t_start, 1)
    write_json(args.out, results, digits=6)
    print(f"\nwrote {args.out}  ({len(bank.keys)} decks scored, "
          f"{time.time() - t_start:.1f}s)")


if __name__ == "__main__":
    main()

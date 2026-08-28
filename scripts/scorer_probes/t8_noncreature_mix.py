"""T8: which noncreature archetypes reach the deck, overall and per color.

A gen4-512 deck holds 23 spells and about 19 creatures, so the noncreature
slots are scarce. This probe asks what fills them. It reads the 10,000 aligned
(pool, deck) pairs of the gen-4 corpus, classifies every noncreature nonland
card in each pool into one archetype, and compares what was available with what
was taken.

Three numbers per archetype:

  share      of the noncreature slots the builder filled
  take rate  chosen / available, over pool cards the deck could legally cast
             (colors subset of the deck's colors; colorless always eligible)
  lift       take rate divided by the take rate expected from the card's
             winnability label, mana value, and color count alone

The take rate answers "does the builder pick these". The lift answers the
harder question: does it pick them for being that archetype, or only for being
good cards that happen to be that archetype. Expected take rates come from
strata of (label quintile x MV bucket x mono/multicolor) measured over every
eligible noncreature card, so lift 1.0 means "taken exactly as often as any
other noncreature card of the same quality and cost".

Availability is counted in the same on-color way as inclusion, so a color's
numbers describe decks that actually play that color.

Writes t8_noncreature_mix.csv and t8_report.md. Read-only w.r.t. the repo and
the Y: drive.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict

import numpy as np

import probe_lib as pl

POOLS = pl.YDATA / "pools" / "pools-gen4-512.txt"
DECKS = pl.YDATA / "decks" / "generated-decks-gen4-512.txt"
OUT_CSV = pl.SCRATCH / "t8_noncreature_mix.csv"
OUT_MD = pl.SCRATCH / "t8_report.md"

WUBRG = "WUBRG"

# ---------------------------------------------------------------------------
# archetype classification
# ---------------------------------------------------------------------------
# Applied to noncreature nonland cards only, first match wins. Ordering matters:
# a pump aura that also pacifies is removal, an instant that draws and deals
# damage is removal. The families group the detail rows for the color tables.

DETAIL_TO_FAMILY = {
    "sweeper": "removal",
    "removal (destroy/exile)": "removal",
    "removal (damage)": "removal",
    "removal (fight)": "removal",
    "removal (shrink)": "removal",
    "removal (lockdown)": "removal",
    "removal (edict)": "removal",
    "utility removal (artifact/ench/land)": "utility removal",
    "fog": "fog",
    "mill": "mill",
    "bounce": "bounce",
    "counterspell": "counterspell",
    "combat trick": "combat trick",
    "buff (aura/equipment)": "buff",
    "card draw": "card draw",
    "discard": "discard",
    "token maker": "token maker",
    "ramp / fixing": "ramp / fixing",
    "lifegain": "lifegain",
    "recursion": "recursion",
    "planeswalker": "planeswalker",
    "other": "other",
}

# A creature-targeting clause can carry a long qualifier ("target nonartifact,
# nonblack creature"), so the patterns allow filler between verb and noun. The
# order below is the classification order and encodes the tie-breaks: mass
# before spot, kill before shrink, shrink before pump.
CREATURE = r"[^.]{0,70}(creature|permanent)"
TARGET = r"(?:another |up to \w+ |\w+ )?target "
RETURN = r"return " + TARGET
TRICK_GRANTS = (r"gets \+\d+/[+-]\d+|gains? (indestructible|hexproof|protection|flying|"
                r"first strike|double strike|deathtouch|trample|lifelink|vigilance|menace)|"
                r"gains? protection|can't be blocked|indestructible until|"
                r"has (indestructible|hexproof|protection|flying|first strike|deathtouch|trample)")


def classify(text: str, types: str, mana_cost: str) -> str:
    """One archetype per noncreature nonland card, first match wins."""
    t = text
    is_instant = "instant" in types
    aura = "aura" in types
    equip = "equipment" in types
    permanent = aura or equip or "artifact" in types or "enchantment" in types

    if "planeswalker" in types:
        return "planeswalker"

    # --- mass effects first: "destroy all creatures" also matches "destroy" ---
    if re.search(r"(destroy|exile) all|destroy each|exile each creature", t) \
            or re.search(r"(each|all) creatures? gets? -", t) \
            or re.search(r"all creatures get -", t) \
            or re.search(r"deals? [^.]{0,40}damage to each (creature|other creature)", t) \
            or re.search(r"tap all|all creatures? [^.]{0,20}don't untap", t):
        return "sweeper"

    # --- spot removal ---
    if re.search(r"(destroy|exile|gain control of) " + TARGET + CREATURE, t):
        return "removal (destroy/exile)"
    if re.search(r"fights?\b", t):
        return "removal (fight)"
    if re.search(r"deals? [^.]{0,40}damage", t) and \
            re.search(r"any target|target" + CREATURE + r"|divided|among", t):
        return "removal (damage)"
    if re.search(r"(target|enchanted) creature gets -\d|"
                 r"(target|enchanted) creature gets [+-]\d+/-\d", t):
        return "removal (shrink)"
    if re.search(r"sacrifices? a creature", t):
        return "removal (edict)"
    if re.search(r"can't attack or block|can't attack, block|can't block|"
                 r"can't attack|doesn't untap|loses all abilities", t) and \
            (permanent or re.search(r"tap target creature", t)):
        return "removal (lockdown)"
    if re.search(r"(destroy|exile) " + TARGET + r"[^.]{0,40}(artifact|enchantment|land)", t):
        return "utility removal (artifact/ench/land)"

    if re.search(r"counter " + TARGET + r"[^.]{0,40}spell", t):
        return "counterspell"
    if re.search(RETURN + r"[^.]{0,120}to (its|their) owner.{0,3} hands?", t) \
            or re.search(RETURN + r"[^.]{0,60}to the top of", t) \
            or re.search(r"put " + TARGET + r"[^.]{0,50}on top of", t):
        return "bounce"
    if re.search(RETURN + r"[^.]{0,70}from (your|a) graveyard|"
                 r"return [^.]{0,50}cards? from your graveyard", t):
        return "recursion"
    if re.search(r"prevent all combat damage", t):
        return "fog"
    if re.search(r"mills? \w+ cards?|puts? the top \w+ cards? [^.]{0,30}graveyard", t):
        return "mill"

    counters = re.search(r"put .{0,40}\+1/\+1 counters? on " + TARGET, t)
    if is_instant and (re.search(TRICK_GRANTS, t) or counters):
        return "combat trick"
    if re.search(r"creatures you control (get \+\d|have )", t):
        return "combat trick"                       # mass pump (Inspired Charge)
    if not permanent and re.search(r"can't block this turn", t):
        return "combat trick"                       # Falter-style evasion enabler
    if (aura and "enchant creature" in t) or equip or counters:
        return "buff (aura/equipment)"              # any creature aura left is a buff

    if re.search(r"draws? [^.]{0,20}cards?|draw a card", t) \
            or re.search(r"look at the top .{0,140}into your hand", t):
        return "card draw"
    if re.search(r"discards? (a|two|three|that|their|his|\d+) cards?", t):
        return "discard"
    if re.search(r"create[s]? [^.]{0,90}token", t):
        return "token maker"
    if re.search(r"\badd (\{|one mana|two mana|\w+ mana)|"
                 r"search your library for [^.]{0,60}land card", t):
        return "ramp / fixing"
    if re.search(r"gains? \d+ life|gain \d+ life|gains? life equal|gains x life", t):
        return "lifegain"
    if is_instant and re.search(r"target creature", t):
        return "combat trick"   # residual instants that target one creature
    return "other"


# ---------------------------------------------------------------------------
# card facts
# ---------------------------------------------------------------------------

class Book:
    """Memoized per-name facts: kind, colors, MV, archetype, winnability label."""

    def __init__(self, probe: pl.Probe, win_rates: dict):
        self.probe = probe
        self.wr = win_rates
        self._cache: dict[str, dict | None] = {}

    def get(self, name: str) -> dict | None:
        if name in self._cache:
            return self._cache[name]
        self._cache[name] = self._build(name)
        return self._cache[name]

    def _build(self, name: str) -> dict | None:
        if name.lower() in pl.BASIC_LAND_NAMES:
            return None
        rec = self.probe.locator.load_text(name)
        emb = self.probe.locator.load_embedding(name)
        if rec is None or emb is None:
            return None
        raw = rec.text.lower()
        types = mana_cost = ""
        for ln in raw.splitlines():
            if ln.startswith("types:"):
                types = ln.split(":", 1)[1].strip()
            elif ln.startswith("mana cost:"):
                mana_cost = ln.split(":", 1)[1].strip()
        body = "\n".join(ln for ln in raw.splitlines()
                         if not ln.startswith(("name:", "mana cost:", "types:")))
        det = emb[-pl.layout.FEATURE_COUNT:]
        colors = frozenset(c for i, c in enumerate(WUBRG) if det[1 + i] > 0)
        is_creature = "creature" in types
        is_land = "land" in types
        label = (self.wr.get(name) or {}).get("shrunk_score_play")
        return {
            "name": name,
            "is_creature": is_creature,
            "is_land": is_land,
            "colors": colors,
            "mv": float(det[pl.layout.MANA_VALUE]),
            "label": label,
            "detail": ("creature" if is_creature else
                       "land" if is_land else classify(body, types, mana_cost)),
        }


# ---------------------------------------------------------------------------
# corpus pass
# ---------------------------------------------------------------------------

def collect(book: Book, limit: int | None):
    """One row per (deck, eligible noncreature pool card): taken or not."""
    pools = list(pl.read_pools(POOLS, limit=limit))
    decks = list(pl.read_generated_decks(DECKS, limit=limit))
    if len(pools) != len(decks):
        raise SystemExit(f"corpus misalignment: {len(pools)} pools vs {len(decks)} decks")

    rows = []
    slots = Counter()          # noncreature deck slots by detail archetype
    slots_by_color = defaultdict(Counter)
    deck_shape = []
    creatures = [0, 0]         # on-color creature availability / inclusions
    for i, ((pset, pool), (_lbl, dset, deck)) in enumerate(zip(pools, decks)):
        if pset != dset:
            raise SystemExit(f"line {i}: pool set {pset} != deck set {dset}")

        deck_counts = Counter(c for c in deck if c.lower() not in pl.BASIC_LAND_NAMES)
        deck_colors: set[str] = set()
        n_creature = n_noncreature = 0
        for name, k in deck_counts.items():
            f = book.get(name)
            if f is None:
                continue
            deck_colors |= f["colors"]
            if f["is_land"]:
                continue
            if f["is_creature"]:
                n_creature += k
            else:
                n_noncreature += k
                slots[f["detail"]] += k
                # per-color tables use mono-colored cards only, so that a color's
                # slot shares and its take rates are counted over the same cards
                if len(f["colors"]) <= 1:
                    slots_by_color["".join(f["colors"]) or "C"][f["detail"]] += k
        deck_shape.append((n_creature, n_noncreature))

        for name, avail in Counter(pool).items():
            f = book.get(name)
            if f is None or f["is_land"]:
                continue
            if not f["colors"] <= deck_colors:
                continue            # uncastable in this deck: not a real choice
            taken = min(deck_counts.get(name, 0), avail)
            if f["is_creature"]:
                creatures[0] += avail
                creatures[1] += taken
                continue
            rows.append({
                "deck": i, "set": pset, "name": name, "detail": f["detail"],
                "family": DETAIL_TO_FAMILY[f["detail"]],
                "colors": "".join(sorted(f["colors"])) or "C",
                "n_colors": len(f["colors"]), "mv": f["mv"], "label": f["label"],
                "avail": avail, "taken": taken,
                "deck_colors": "".join(sorted(deck_colors)),
            })
    return rows, slots, slots_by_color, deck_shape, creatures


# ---------------------------------------------------------------------------
# quality-controlled expectation
# ---------------------------------------------------------------------------

def add_expected(rows):
    """Expected take rate per card from (label quintile x MV bucket x mono/multi).

    Cards with no winnability label get their own quintile bucket rather than
    being dropped, so shares and take rates cover the whole corpus.
    """
    labels = np.array([r["label"] for r in rows if r["label"] is not None], dtype=float)
    cuts = np.quantile(labels, [0.2, 0.4, 0.6, 0.8])

    def stratum(r):
        q = -1 if r["label"] is None else int(np.searchsorted(cuts, r["label"]))
        mv = r["mv"]
        mvb = 0 if mv <= 2 else 1 if mv == 3 else 2 if mv == 4 else 3
        return (q, mvb, min(r["n_colors"], 2))

    num = Counter()
    den = Counter()
    for r in rows:
        s = stratum(r)
        num[s] += r["taken"]
        den[s] += r["avail"]
    for r in rows:
        s = stratum(r)
        r["expected"] = r["avail"] * num[s] / den[s]
    return rows


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def table(rows, key, order=None, min_avail=200):
    agg = defaultdict(lambda: [0, 0, 0.0])
    for r in rows:
        a = agg[r[key]]
        a[0] += r["avail"]
        a[1] += r["taken"]
        a[2] += r["expected"]
    out = []
    for k, (av, tk, ex) in agg.items():
        if av < min_avail:
            continue
        out.append((k, av, tk, tk / av, (tk / ex) if ex else float("nan")))
    if order:
        out.sort(key=lambda r: order.index(r[0]) if r[0] in order else 99)
    else:
        out.sort(key=lambda r: -r[3])
    return out


def by_label_rank(rows, depth=6):
    """Take rate by within-deck label rank, removal against everything else.

    Separates two readings of removal's lift. If the preference were front-loaded
    — the deck wants its first two answers and is indifferent after — lift would
    decay with rank. If it is a flat per-card edge, lift holds or grows.
    """
    per_deck = defaultdict(list)
    for r in rows:
        if r["label"] is not None:
            per_deck[r["deck"]].append(r)
    agg = defaultdict(lambda: defaultdict(lambda: [0, 0, 0.0]))
    for rs in per_deck.values():
        for key in ("removal", "rest"):
            sub = [r for r in rs if (r["family"] == "removal") == (key == "removal")]
            sub.sort(key=lambda r: -r["label"])
            for i, r in enumerate(sub[:depth], 1):
                a = agg[key][i]
                a[0] += r["avail"]
                a[1] += r["taken"]
                a[2] += r["expected"]
    lines = ["| label rank in pool | removal take | removal lift | other take | other lift |",
             "|---|---|---|---|---|"]
    for i in range(1, depth + 1):
        a, b = agg["removal"][i], agg["rest"][i]
        lines.append(f"| {i} | {100 * a[1] / a[0]:.1f}% | {a[1] / a[2]:.2f} | "
                     f"{100 * b[1] / b[0]:.1f}% | {b[1] / b[2]:.2f} |")
    return "\n".join(lines)


def fmt(out, slots=None, total_slots=None):
    lines = ["| archetype | available | taken | take rate | lift | share of slots |",
             "|---|---|---|---|---|---|"]
    for k, av, tk, rate, lift in out:
        share = ""
        if slots is not None and total_slots:
            share = f"{100 * slots.get(k, 0) / total_slots:.1f}%"
        lines.append(f"| {k} | {av} | {tk} | {100 * rate:.1f}% | {lift:.2f} | {share} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="pool/deck pairs to read")
    ap.add_argument("--smoke", action="store_true", help="300 pairs, CPU")
    ap.add_argument("--audit", type=int, default=0,
                    help="print the N most-available cards still classed 'other'")
    args = ap.parse_args()
    limit = 300 if args.smoke else args.limit

    probe = pl.Probe(device="cpu")      # no scoring here, only the locator
    book = Book(probe, pl.load_win_rates())
    rows, slots, slots_by_color, shape, creatures = collect(book, limit)
    add_expected(rows)

    if args.audit:
        seen = Counter()
        for r in rows:
            if r["detail"] == "other":
                seen[r["name"]] += r["avail"]
        print(f"=== {len(seen)} distinct cards still 'other' ===")
        for name, n in seen.most_common(args.audit):
            rec = probe.locator.load_text(name)
            body = " / ".join(ln.split(":", 1)[1].strip()
                              for ln in rec.text.lower().splitlines()
                              if ln.startswith(("spell", "static", "triggered", "activated")))
            print(f"{n:5d} {name:<34} {body[:120]}")
        return

    n_decks = len(shape)
    mean_cr = np.mean([c for c, _ in shape])
    mean_nc = np.mean([n for _, n in shape])
    total_slots = sum(slots.values())

    fams = sorted({r["family"] for r in rows})
    fam_slots = Counter()
    for d, k in slots.items():
        fam_slots[DETAIL_TO_FAMILY[d]] += k

    nc_av = sum(r["avail"] for r in rows)
    nc_tk = sum(r["taken"] for r in rows)
    md = [f"# T8: what fills the noncreature slots\n",
          f"{n_decks} gen4-512 decks, {len(rows)} (deck, eligible card) choices.",
          f"Mean deck: {mean_cr:.1f} creatures, {mean_nc:.1f} noncreature spells.",
          f"On-color take rate: creatures {100 * creatures[1] / creatures[0]:.1f}% "
          f"({creatures[1]}/{creatures[0]}), noncreature spells "
          f"{100 * nc_tk / nc_av:.1f}% ({nc_tk}/{nc_av}).\n",
          "## By family\n",
          fmt(table(rows, "family"), fam_slots, total_slots),
          "\n## By archetype\n",
          fmt(table(rows, "detail"), slots, total_slots),
          "\n## Removal's edge by depth (does the preference front-load?)\n",
          by_label_rank(rows)]

    # per-color: mono-colored cards only, so a color's row is that color's cards
    md.append("\n## By color (mono-colored noncreature cards, and colorless)\n")
    for col in list(WUBRG) + ["C"]:
        sub = [r for r in rows if r["colors"] == (col if col != "C" else "C")]
        if not sub:
            continue
        cs = Counter({d: n for d, n in slots_by_color[col].items()})
        cfam = Counter()
        for d, k in cs.items():
            cfam[DETAIL_TO_FAMILY[d]] += k
        md.append(f"\n### {col} (n={len(sub)} choices, {sum(cs.values())} slots filled)\n")
        md.append(fmt(table(sub, "family", min_avail=100), cfam, sum(cs.values())))

    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {OUT_CSV} ({len(rows)} rows) and {OUT_MD}")


if __name__ == "__main__":
    main()

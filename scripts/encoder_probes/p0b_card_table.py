"""P0b — the per-card nameable-feature table: labels x parsed text features.

One row per card in ``cards-win-rates.txt``, joining the label columns with
features parsed from the card's converted text: mana-cost breakdown (MV,
per-color pips, generic, {X}, hybrid/Phyrexian), type-class flags, statline,
keyword flags (one flag for the keyword on the card's own static lines and
one for a mention anywhere in the text), phrase/archetype regex flags,
text-shape counts, and the big-tribe flag. This is the feature source behind
the 135-feature comparison in ``s_r18.py`` and the flag columns the shared
loaders (``c_common``, ``s_common``, ``q_common``, ``l_*``) merge into the
join table.

Reads the Y: label file and ``output/cardsfolder-512``; writes
``output/encoder-probes/card_table.pkl``.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
CARDS = REPO / "output" / "cardsfolder-512"
WIN_RATES = Path(r"Y:\Nicolas\mtg\mtg-models-data\sealed\training-data\matches-bo1\cards-win-rates.txt")
OUT = REPO / "output" / "encoder-probes"

sys.path.insert(0, str(REPO / "src"))
from sealed.infrastructure.converted_card_locator import ConvertedCardLocator  # noqa: E402

# ---------------- mana cost parsing ----------------

SYM = re.compile(r"\{([^}]*)\}")


def parse_cost(cost: str) -> dict:
    """MV, pip counts, generic, X, hybrid/phyrexian flags from a mana cost line."""
    out = dict(mv=0.0, generic=0.0, has_x=0, n_sym=0,
               pip_w=0, pip_u=0, pip_b=0, pip_r=0, pip_g=0, pip_c=0,
               hybrid=0, phyrexian=0)
    if cost is None:
        return out
    for raw in SYM.findall(cost.lower()):
        out["n_sym"] += 1
        s = raw.strip()
        if s.isdigit():
            out["generic"] += int(s)
            out["mv"] += int(s)
            continue
        if s == "x" or s == "y" or s == "z":
            out["has_x"] = 1
            continue  # X counts 0 toward MV
        # non-generic symbol: contributes 1 to MV
        out["mv"] += 1
        parts = s.split("/")
        if len(parts) > 1:
            if "p" in parts:
                out["phyrexian"] = 1
            else:
                out["hybrid"] = 1
        for p in parts:
            if p in "wubrg" and len(p) == 1:
                out[f"pip_{p}"] += 1
            elif p == "c":
                out["pip_c"] += 1
            elif p.isdigit():
                # 2/W style hybrid: generic side, don't add generic to mv again
                pass
    return out


# ---------------- text features ----------------

KEYWORDS = [
    "flying", "lifelink", "vigilance", "first strike", "double strike",
    "deathtouch", "trample", "haste", "reach", "defender", "menace",
    "hexproof", "indestructible", "ward", "protection from", "flash",
    "shroud", "prowess", "fear", "intimidate", "shadow", "skulk",
    "horsemanship", "can't be blocked",
]
KW_COL = {k: "kw_" + re.sub(r"[^a-z]+", "_", k).strip("_") for k in KEYWORDS}

# canonical combat keywords used for the count feature
COMBAT_KW = ["flying", "lifelink", "vigilance", "first strike", "double strike",
             "deathtouch", "trample", "haste", "reach", "menace"]

PHRASES = {
    # ---- removal / interaction ----
    "ph_destroy_target_creature": [r"destroy target (?:\w+ )*creature"],
    "ph_exile_target_creature": [r"exile target (?:\w+ )*creature"],
    "ph_uncond_removal": [r"destroy target creature\b", r"exile target creature\b"],
    "ph_cond_removal": [r"destroy target (?:attacking|blocking|tapped|nonblack|nonwhite|artifact|enchantment) creature",
                        r"target creature with (?:flying|power|toughness)",
                        r"destroy target creature with"],
    "ph_damage_any_target": [r"damage to any target"],
    "ph_damage_target_creature": [r"damage to target creature"],
    "ph_fight": [r"\bfights?\b"],
    "ph_edict": [r"sacrifices? a creature"],
    "ph_minus_pt": [r"gets -\d"],
    "ph_tap_target": [r"tap target creature"],
    "ph_sweeper": [r"destroy all creatures", r"each creature gets -\d", r"exile all creatures",
                   r"destroy all (?:\w+ )*creatures"],
    "ph_lockdown_aura": [r"enchanted creature can't attack", r"enchanted creature doesn't untap",
                         r"enchanted creature can't block"],
    "ph_counterspell": [r"counter target"],
    # ---- card flow ----
    "ph_draw_a_card": [r"draw a card"],
    "ph_draw_multi": [r"draw (?:two|three|four|five|x) cards"],
    "ph_scry": [r"\bscry\b", r"\bsurveil\b", r"look at the top"],
    "ph_discard": [r"discards? (?:a|two|three|\d) cards?"],
    "ph_mill": [r"\bmills?\b", r"puts? the top \w+ cards? of (?:their|his or her) library into"],
    # ---- lifegain ----
    "ph_gain_life": [r"you gain \d+ life", r"gain \d+ life"],
    # ---- tricks / pumps ----
    "ph_pump_eot": [r"gets \+\d+/\+\d+ until end of turn"],
    "ph_aura_pump": [r"enchanted creature gets \+\d"],
    "ph_counters_plus": [r"\+1/\+1 counter"],
    # ---- tokens ----
    "ph_create_token": [r"create (?:a|an|one|two|three|x|\d+) .*token"],
    # ---- mana ----
    "ph_add_mana": [r":\s*add \{"],
    "ph_search_basic_land": [r"search your library for a (?:basic )?land"],
    # ---- triggers ----
    "ph_etb": [r"when cardname enters"],
    "ph_dies_trigger": [r"when cardname dies"],
    "ph_attack_trigger": [r"whenever cardname attacks"],
    # ---- mechanics ----
    "ph_cycling": [r"\bcycling\b", r"landcycling"],
    "ph_kicker": [r"\bkicker\b", r"multikicker"],
    "ph_morph": [r"\bmorph\b", r"megamorph"],
    "ph_equip": [r"\bequip\b"],
    "ph_modal": [r"choose one"],
    "ph_sacrifice_self": [r"sacrifice cardname"],
    "ph_damage_to_you": [r"deals \d+ damage to you", r"you lose \d+ life"],
    "ph_echo_upkeep": [r"\becho\b", r"cumulative upkeep", r"at the beginning of your upkeep, sacrifice"],
    "ph_enters_tapped": [r"enters tapped"],
    "ph_loyalty": [r"loyalty"],
}
PHRASE_RE = {k: [re.compile(p) for p in v] for k, v in PHRASES.items()}

BIG_TRIBES = ("dragon", "angel", "demon", "sphinx", "hydra", "giant", "wurm", "titan")


def parse_card(text: str) -> dict:
    t = text.lower()
    lines = [ln.rstrip() for ln in t.splitlines() if ln.strip()]
    fields = {}
    body_lines = []
    for ln in lines:
        if ":" not in ln:
            body_lines.append(ln)
            continue
        head, val = ln.split(":", 1)
        head = head.strip()
        val = val.strip()
        base = head.split("[")[0]
        fields.setdefault(base, []).append(val)
        if base not in ("name", "mana cost", "types", "power toughness"):
            body_lines.append(val)

    types = (fields.get("types", [""])[0]).split()
    cost_line = fields.get("mana cost", [None])[0]
    f = parse_cost(cost_line)
    f["has_cost_line"] = int(cost_line is not None)

    pt = fields.get("power toughness", [None])[0]
    power = tough = np.nan
    if pt and "/" in pt:
        p, tt = pt.split("/", 1)
        def num(s):
            s = s.strip()
            m = re.match(r"^-?\d+", s)
            return float(m.group(0)) if m else 0.0
        power, tough = num(p), num(tt)

    statics = " ".join(fields.get("static", []))
    body = " ".join(body_lines)
    full = " ".join(lines)

    is_creature = "creature" in types
    f.update(
        n_types=len(types),
        is_creature=int(is_creature),
        is_land=int("land" in types),
        is_instant=int("instant" in types),
        is_sorcery=int("sorcery" in types),
        is_artifact=int("artifact" in types),
        is_enchantment=int("enchantment" in types),
        is_planeswalker=int("planeswalker" in types),
        is_aura=int("aura" in types),
        is_equipment=int("equipment" in types),
        is_vehicle=int("vehicle" in types),
        is_legendary=int("legendary" in types),
        is_basic=int("basic" in types),
        power=power, toughness=tough,
        big_tribe=int(any(tr in types for tr in BIG_TRIBES)),
        n_lines=len(lines),
        n_body_lines=len(body_lines),
        n_words=len(full.split()),
        n_body_words=len(body.split()),
        n_abilities=len(fields.get("spell", [])) + len(fields.get("triggered", []))
                    + len(fields.get("activated", [])) + len(fields.get("static", [])),
    )
    # keyword flags: match in static lines (the keyword as the card's own
    # ability) plus an anywhere-in-text version (catches grants and conditionals)
    for k, col in KW_COL.items():
        f[col] = int(k in statics)
        f[col + "_anywhere"] = int(k in full)
    f["kw_count"] = sum(1 for k in COMBAT_KW if k in statics)
    for col, pats in PHRASE_RE.items():
        f[col] = int(any(p.search(full) for p in pats))
    # colors
    f["n_colors"] = sum(1 for c in "wubrg" if f[f"pip_{c}"] > 0)
    f["pips_total"] = sum(f[f"pip_{c}"] for c in "wubrg")
    f["is_gold"] = int(f["n_colors"] >= 2)
    f["is_colorless_cost"] = int(f["n_colors"] == 0)
    return f


# ---------------- labels ----------------

def load_labels() -> pd.DataFrame:
    df = pd.read_csv(WIN_RATES, sep=";", dtype={"card_name": str})
    return df


def main() -> None:
    labels = load_labels()
    loc = ConvertedCardLocator(CARDS)
    rows = []
    miss = 0
    for name in labels["card_name"]:
        p = loc.text_path(name)
        if p is None:
            miss += 1
            rows.append(None)
            continue
        rows.append(parse_card(p.read_text(encoding="utf-8", errors="replace")))
    keys = sorted({k for r in rows if r for k in r})
    feat = pd.DataFrame(
        [{k: (r.get(k, np.nan) if r else np.nan) for k in keys} for r in rows],
        index=labels.index,
    )
    df = pd.concat([labels, feat], axis=1)
    df["found"] = [r is not None for r in rows]
    df["n_in_deck"] = df["wins_when_in_deck"] + df["losses_when_in_deck"]
    df["n_played"] = df["wins_when_played"] + df["losses_when_played"]
    print(f"cards in label file: {len(df)}   text not found: {miss}")
    df.to_pickle(OUT / "card_table.pkl")
    print("wrote", OUT / "card_table.pkl")
    # prevalence sanity check
    flags = [c for c in df.columns if c.startswith(("kw_", "ph_", "is_")) and not c.endswith("_anywhere")]
    prev = df.loc[df.found, flags].mean().sort_values(ascending=False)
    print("\nflag prevalence (found cards, n=%d):" % df.found.sum())
    print(prev.to_string())


if __name__ == "__main__":
    main()

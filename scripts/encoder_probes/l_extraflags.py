"""Extra per-card text flags the pre-built card table does not carry.

Only the handful the label-side artifact analysis needs on top of
``grounding/card_table.pkl``: the fog class (damage prevention), a
mana-rock refinement, and a "combat trick" shell test.  Cached to
``output/encoder-probes/l_extra_flags.pkl``.
"""

from __future__ import annotations

import pickle
import re
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from sealed.infrastructure.converted_card_locator import (  # noqa: E402
    ConvertedCardLocator,
)

CARDS = REPO / "output" / "cardsfolder-512"
OUT = REPO / "output" / "encoder-probes"
TABLE = OUT / "card_table.pkl"  # written by p0b_card_table.py

EXTRA = {
    "ph_fog": [r"prevent all combat damage", r"prevent all damage that would be dealt"],
    "ph_prevent_damage": [r"\bprevent\b.*\bdamage\b"],
    "ph_tap_for_mana": [r"\{t\}[^\n]*:\s*add \{"],
    "ph_flying_grant": [r"gains? flying", r"has flying", r"have flying"],
}
EXTRA_RE = {k: [re.compile(p) for p in v] for k, v in EXTRA.items()}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tbl = pickle.load(open(TABLE, "rb"))
    loc = ConvertedCardLocator(CARDS)
    rows = []
    miss = 0
    for name in tbl["card_name"]:
        p = loc.text_path(name)
        if p is None:
            miss += 1
            rows.append({k: 0 for k in EXTRA})
            continue
        txt = p.read_text(encoding="utf-8", errors="replace").lower()
        body = "\n".join(
            ln for ln in txt.splitlines() if not ln.startswith("name:")
        )
        rows.append(
            {k: int(any(r.search(body) for r in rs)) for k, rs in EXTRA_RE.items()}
        )
    df = pd.DataFrame(rows)
    df.insert(0, "card_name", list(tbl["card_name"]))
    df.to_pickle(OUT / "l_extra_flags.pkl")
    print(f"wrote {OUT / 'l_extra_flags.pkl'}  missing text: {miss}")
    print(df.drop(columns=["card_name"]).mean().to_string())


if __name__ == "__main__":
    main()

"""Shared harness for scorer-behavior probes.

Loads the gen-4 production scorer once, provides batched deck scoring by card
name or by raw embedding matrix, and builds a per-card metadata table (types,
mana cost, keywords, empirical win-rate labels, rarity).

All probe scripts import from this module. Read-only with respect to the repo
and Y: drive.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
SCORER_CKPT = REPO / "models/sealed/scorer/512-best_l6_h4_s4_ff2176_mlp512_lr1e-05_mwlog.pt"
CARDS_PATH = REPO / "output/cardsfolder-512"
YDATA = Path(r"Y:\Nicolas\mtg\mtg-models-data\sealed\training-data")
WIN_RATES = YDATA / "matches-bo1" / "cards-win-rates.txt"
DATA_DIR = Path(__file__).resolve().parent
SCRATCH = REPO / "output" / "scorer-probes"  # generated artifacts (gitignored)
SCRATCH.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO / "src"))

from sealed.domain.scorer_model import SetTransformerScorer  # noqa: E402
from sealed.infrastructure.scorer_store import ScorerStore  # noqa: E402
from sealed.infrastructure.converted_card_locator import (  # noqa: E402
    BASIC_LAND_NAMES,
    ConvertedCardLocator,
)
from sealed.domain import card_embedding_layout as layout  # noqa: E402


class Probe:
    """Production scorer + embedding locator, batched scoring."""

    def __init__(self, device: str | None = None, checkpoint: Path = SCORER_CKPT):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ck = ScorerStore().load_checkpoint(checkpoint)
        self.model = SetTransformerScorer(ck.config)
        self.model.load_state_dict(ck.model_state_dict)
        self.model.to(self.device).eval()
        self.locator = ConvertedCardLocator(CARDS_PATH)
        self.d_model = ck.config.d_model
        self._miss: set[str] = set()

    # ---------- embedding lookup ----------

    def embedding(self, name: str) -> np.ndarray | None:
        """544-dim embedding for a card name; None for basics/missing."""
        if name.lower() in BASIC_LAND_NAMES:
            return None
        emb = self.locator.load_embedding(name)
        if emb is None:
            self._miss.add(name)
        return emb

    def deck_matrix(self, deck: list[str]) -> np.ndarray:
        """Stack the deck's non-basic embeddings; silently drops basics,
        raises if any non-basic card is missing (probes need exact inputs)."""
        rows = []
        for n in deck:
            if n.lower() in BASIC_LAND_NAMES:
                continue
            e = self.embedding(n)
            if e is None:
                raise KeyError(f"no embedding for card: {n!r}")
            rows.append(e)
        return np.stack(rows).astype(np.float32)

    # ---------- scoring ----------

    @torch.no_grad()
    def score_matrices(self, mats: list[np.ndarray], batch_size: int = 512) -> np.ndarray:
        """Score decks given as (n_cards, 544) float32 matrices."""
        out = np.empty(len(mats), dtype=np.float64)
        for lo in range(0, len(mats), batch_size):
            chunk = mats[lo:lo + batch_size]
            lens = [m.shape[0] for m in chunk]
            mx = max(lens)
            batch = torch.zeros(len(chunk), mx, self.d_model)
            mask = torch.zeros(len(chunk), mx, dtype=torch.bool)
            for i, m in enumerate(chunk):
                batch[i, : m.shape[0]] = torch.from_numpy(m)
                mask[i, : m.shape[0]] = True
            s = self.model(batch.to(self.device), mask.to(self.device))
            out[lo:lo + len(chunk)] = s.squeeze(-1).float().cpu().numpy()
        return out

    def score_decks(self, decks: list[list[str]], batch_size: int = 512) -> np.ndarray:
        return self.score_matrices([self.deck_matrix(d) for d in decks], batch_size)


# ---------- corpora ----------

def read_pools(path: Path, limit: int | None = None):
    """Yield (set_code, [card names]) from a pools file."""
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ";" not in line:
                continue
            set_code, cards = line.split(";", 1)
            yield set_code, cards.split("|")
            n += 1
            if limit and n >= limit:
                return


def read_generated_decks(path: Path, limit: int | None = None):
    """Yield (label, set_code, [40 card names]) from a generated-decks file."""
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(";")
            if len(parts) != 3:
                continue
            yield parts[0], parts[1], parts[2].split("|")
            n += 1
            if limit and n >= limit:
                return


def read_match_decks(path: Path, limit: int | None = None):
    """Yield (set_code, method_A, deck_A, method_B, deck_B) from match-outcomes."""
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split(";")
            if len(parts) < 10:
                continue
            yield parts[2], parts[3], parts[5].split("|"), parts[4], parts[6].split("|")
            n += 1
            if limit and n >= limit:
                return


# ---------- per-card metadata ----------

EVASION_KEYWORDS = ("flying", "menace", "fear", "intimidate", "shadow",
                    "horsemanship", "skulk", "unblockable", "can't be blocked")
COMBAT_KEYWORDS = ("first strike", "double strike", "deathtouch", "lifelink",
                   "trample", "vigilance", "haste", "reach")
REMOVAL_PATTERNS = ("destroy target creature", "exile target creature",
                    "destroy target attacking", "destroy target blocking",
                    "damage to any target", "damage to target creature",
                    "gets -", "fights target", "deals damage equal to its power to target",
                    "exile target permanent", "destroy target permanent",
                    "tap target creature", "pacifism")


def card_features(text: str) -> dict:
    """Heuristic per-card features from converted card text (lowercase)."""
    t = text.lower()
    lines = t.splitlines()

    def line_val(prefix):
        for ln in lines:
            if ln.startswith(prefix):
                return ln.split(":", 1)[1].strip()
        return None

    types = (line_val("types") or "").split()
    pt = line_val("power toughness")
    power = tough = None
    if pt and "/" in pt:
        p, tt = pt.split("/", 1)
        try:
            power = float(p)
        except ValueError:
            power = 0.0
        try:
            tough = float(tt)
        except ValueError:
            tough = 0.0

    statics = " ".join(ln.split(":", 1)[1] for ln in lines if ln.startswith("static:"))
    body = " ".join(ln for ln in lines if not ln.startswith(("name:", "mana cost:", "types:")))

    is_creature = "creature" in types
    return {
        "is_creature": is_creature,
        "is_land": "land" in types,
        "is_instant": "instant" in types,
        "is_sorcery": "sorcery" in types,
        "is_artifact": "artifact" in types,
        "is_enchantment": "enchantment" in types,
        "is_planeswalker": "planeswalker" in types,
        "is_aura": "aura" in types,
        "is_equipment": "equipment" in types,
        "power": power,
        "toughness": tough,
        "has_evasion": is_creature and any(k in statics or k in body for k in EVASION_KEYWORDS),
        "has_flying": is_creature and "flying" in statics,
        "combat_kw_count": sum(1 for k in COMBAT_KEYWORDS if k in statics) if is_creature else 0,
        "is_removal": (not is_creature) and any(p in body for p in REMOVAL_PATTERNS),
        "draws_cards": "draw" in body and "card" in body,
        "vanilla": is_creature and not statics.strip()
                   and not any(ln.startswith(("spell", "triggered", "activated")) for ln in lines),
    }


def load_win_rates() -> dict[str, dict]:
    """cards-win-rates.txt -> name -> label dict (floats; None for empty cells)."""
    rows: dict[str, dict] = {}
    with open(WIN_RATES, encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split(";")
        for line in f:
            parts = line.rstrip("\n").split(";")
            if len(parts) != len(header):
                continue
            rec = {}
            for k, v in zip(header[1:], parts[1:]):
                rec[k] = float(v) if v not in ("", None) else None
            rows[parts[0]] = rec
    return rows


def det_features(emb: np.ndarray) -> dict:
    """Named deterministic features from the trailing 32 dims of an embedding."""
    d = emb[-layout.FEATURE_COUNT:]
    return {
        "is_land": d[layout.IS_LAND],
        "pips_w": d[1], "pips_u": d[2], "pips_b": d[3], "pips_r": d[4], "pips_g": d[5],
        "generic": d[layout.GENERIC],
        "mv": d[layout.MANA_VALUE],
        "colors": int(sum(d[layout.COLOR_FLAGS])),
        "power": d[layout.POWER],
        "toughness": d[layout.TOUGHNESS],
        "produces_any": float(d[layout.PRODUCES_COLORS].sum() + d[layout.PRODUCES_COLORLESS]) > 0,
    }

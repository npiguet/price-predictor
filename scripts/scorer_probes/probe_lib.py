"""Shared harness for scorer-behavior probes.

Loads the gen-4 production scorer once, provides batched deck scoring by raw
embedding matrix, a deduplicating bank of decks to score, memoized per-card
facts (types, mana cost, keywords, empirical win-rate labels), and the readers
of the match, pool and generated-deck corpora the probes sample from.

All probe scripts import from this module. Read-only with respect to the repo
and Y: drive.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Callable, Iterable
from itertools import islice
from pathlib import Path

import numpy as np
import torch
from _lib import corpora
from _lib.paths import MODELS, OUTPUT, SEALED_TRAINING_DATA

from sealed.domain import card_embedding_layout as layout
from sealed.domain.scorer_model import SetTransformerScorer
from sealed.infrastructure.converted_card_locator import (
    BASIC_LAND_NAMES,
    ConvertedCardLocator,
)
from sealed.infrastructure.match_data_loader import MatchOutcome, parse_match_outcome
from sealed.infrastructure.pool_file_reader import parse_generated_decks, parse_pools
from sealed.infrastructure.scorer_store import ScorerStore

SCORER_CKPT = MODELS / "sealed/scorer/512-best_l6_h4_s4_ff2176_mlp512_lr1e-05_mwlog.pt"
CARDS_PATH = OUTPUT / "cardsfolder-512"
YDATA = SEALED_TRAINING_DATA
WIN_RATES = YDATA / "matches-bo1" / "cards-win-rates.txt"
DATA_DIR = Path(__file__).resolve().parent
SCRATCH = OUTPUT / "scorer-probes"  # generated artifacts (gitignored)
SCRATCH.mkdir(parents=True, exist_ok=True)

WUBRG = "WUBRG"
FC = layout.FEATURE_COUNT

# The Bo7 self-play corpora, oldest first, relative to ``YDATA``.
MATCH_FILES = [
    "matches-b07/match-outcomes-gen0.txt.gz",
    "matches-b07/match-outcomes-gen1-vs-0.txt.gz",
    "matches-b07/match-outcomes-gen2-vs-0-1.txt.gz",
    "matches-b07/match-outcomes-gen3-vs-0-2.txt",
    "matches-b07/match-outcomes-gen3-vs-forge-best.txt",
    "matches-b07/match-outcomes-gen4-vs-forge-best-gen3.txt",
    "matches-b07/match-outcomes-gen5-vs-gen4-forge.txt",
]


def is_basic(name: str) -> bool:
    return name.lower() in BASIC_LAND_NAMES


def nonbasic(deck: Iterable[str]) -> list[str]:
    return [c for c in deck if not is_basic(c)]


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

    # ---------- embedding lookup ----------

    def embedding(self, name: str) -> np.ndarray | None:
        """Embedding for a card name; None for basics/missing."""
        if is_basic(name):
            return None
        return self.locator.load_embedding(name)

    def deck_matrix(self, deck: list[str]) -> np.ndarray:
        """Stack the deck's non-basic embeddings; silently drops basics,
        raises if any non-basic card is missing (probes need exact inputs)."""
        rows = []
        for n in nonbasic(deck):
            e = self.embedding(n)
            if e is None:
                raise KeyError(f"no embedding for card: {n!r}")
            rows.append(e)
        return np.stack(rows).astype(np.float32)

    # ---------- scoring ----------

    @torch.no_grad()
    def score_matrices(self, mats: list[np.ndarray], batch_size: int = 512,
                       on_batch: Callable | None = None) -> np.ndarray:
        """Score decks given as (n_cards, d_model) float32 matrices.

        ``on_batch(cards, mask)`` runs after each batch's forward pass with the
        padded inputs on the model's device, so a caller's forward hooks can
        align what they captured with the batch.
        """
        out = np.empty(len(mats), dtype=np.float64)
        for lo in range(0, len(mats), batch_size):
            chunk = mats[lo:lo + batch_size]
            mx = max(m.shape[0] for m in chunk)
            batch = torch.zeros(len(chunk), mx, self.d_model)
            mask = torch.zeros(len(chunk), mx, dtype=torch.bool)
            for i, m in enumerate(chunk):
                batch[i, : m.shape[0]] = torch.from_numpy(np.ascontiguousarray(m))
                mask[i, : m.shape[0]] = True
            batch, mask = batch.to(self.device), mask.to(self.device)
            s = self.model(batch, mask)
            out[lo:lo + len(chunk)] = s.squeeze(-1).float().cpu().numpy()
            if on_batch is not None:
                on_batch(batch, mask)
        return out


def score_streaming(probe: Probe, n: int, build: Callable[[int], np.ndarray],
                    batch_size: int = 1024, label: str = "") -> np.ndarray:
    """Score ``n`` decks whose matrices ``build(i)`` materializes one batch at a time.

    Stacking every matrix first would hold a whole corpus resident (gigabytes);
    building per batch is numerically identical and stays at a few MB.
    """
    out = np.empty(n, dtype=np.float64)
    for lo in range(0, n, batch_size):
        hi = min(n, lo + batch_size)
        out[lo:hi] = probe.score_matrices([build(i) for i in range(lo, hi)],
                                          batch_size=batch_size)
        if label:
            print(f"  {label} scored {hi}/{n}", flush=True)
    return out


class DeckBank:
    """Distinct decks, keyed by their sorted card multiset, each scored once.

    The scorer is permutation-invariant over the card set, so equal multisets
    collapse to one forward pass. ``embed(name)`` supplies a card's embedding,
    or None for a card no deck may hold; basics are dropped from every deck.
    """

    def __init__(self, embed: Callable[[str], np.ndarray | None]):
        self._embed = embed
        self.emb: dict[str, np.ndarray] = {}
        self.missing: set[str] = set()
        self.index: dict[tuple[str, ...], int] = {}
        self.keys: list[tuple[str, ...]] = []
        self.scores = np.empty(0, dtype=np.float64)

    def card_ok(self, name: str) -> bool:
        if name in self.emb:
            return True
        if name in self.missing:
            return False
        e = self._embed(name)
        if e is None:
            self.missing.add(name)
            return False
        self.emb[name] = np.asarray(e, dtype=np.float32)
        return True

    def add(self, deck: Iterable[str]) -> int | None:
        """The deck's index; None if it is empty or holds an unusable card."""
        key = tuple(sorted(nonbasic(deck)))
        if not key:
            return None
        hit = self.index.get(key)
        if hit is not None:
            return hit
        if not all(self.card_ok(c) for c in key):
            return None
        self.index[key] = len(self.keys)
        self.keys.append(key)
        return self.index[key]

    def matrix(self, i: int) -> np.ndarray:
        """A fresh (n_cards, d_model) float32 matrix for distinct deck ``i``."""
        return np.stack([self.emb[c] for c in self.keys[i]])

    def score_all(self, probe: Probe, which: list[int], batch: int = 1024,
                  transform: Callable[[np.ndarray, int], np.ndarray] | None = None,
                  label: str = "") -> np.ndarray:
        """Score decks ``which``; ``transform(matrix, i)`` edits a deck's fresh
        matrix before it is scored."""
        def build(j: int) -> np.ndarray:
            mat = self.matrix(which[j])
            return mat if transform is None else transform(mat, which[j])

        return score_streaming(probe, len(which), build, batch, label)

    def score_pending(self, probe: Probe, batch: int = 1024, label: str = "") -> None:
        """Score every deck added since the last call."""
        done = len(self.scores)
        new = self.score_all(probe, list(range(done, len(self.keys))), batch, label=label)
        self.scores = np.concatenate([self.scores, new])

    def score(self, i: int) -> float:
        return float(self.scores[i])


# ---------- corpora ----------

def read_pools(path: Path, limit: int | None = None) -> list:
    """The first ``limit`` (all when falsy) ``SealedPool``s of a pools file."""
    return parse_pools(path)[:limit or None]


def read_generated_decks(path: Path, limit: int | None = None) -> list:
    """The first ``limit`` (all when falsy) ``GeneratedDeck``s of a generated-decks file."""
    return parse_generated_decks(path)[:limit or None]


def decided_matches(path: Path, limit: int | None = None):
    """``(matches, undecided, malformed)`` over the head of a match-outcome file.

    ``matches`` holds ``(match, winner)`` for the first ``limit`` (all when falsy)
    matches one side won more games of; ``undecided`` counts drawn ones and
    ``malformed`` the lines that do not parse.
    """
    rows: list[tuple[MatchOutcome, str]] = []
    undecided = malformed = 0
    with corpora.open_text(path) as handle:
        for line in handle:
            try:
                match = parse_match_outcome(line)
            except ValueError:
                malformed += 1
                continue
            winner = corpora.decided_winner(match)
            if winner is None:
                undecided += 1
                continue
            rows.append((match, winner.value))
            if limit and len(rows) >= limit:
                break
    return rows, undecided, malformed


def iter_match_decks(path: Path, limit: int | None = None):
    """``(set_code, method_A, deck_A, method_B, deck_B)`` per match line."""
    for m in islice(corpora.iter_matches(path), limit or None):
        yield m.set_code, m.method_a, list(m.deck_a.cards), m.method_b, list(m.deck_b.cards)


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
    """cards-win-rates.txt -> name -> label dict (None for empty cells)."""
    return corpora.load_win_rates(WIN_RATES)


def det_features(emb: np.ndarray) -> dict:
    """Named deterministic features from the trailing block of an embedding."""
    d = emb[-FC:]
    pips = d[layout.COLOR_PIPS]
    return {
        "is_land": d[layout.IS_LAND],
        "pips_w": pips[0], "pips_u": pips[1], "pips_b": pips[2], "pips_r": pips[3],
        "pips_g": pips[4],
        "generic": d[layout.GENERIC],
        "mv": d[layout.MANA_VALUE],
        "colors": int(sum(d[layout.COLOR_FLAGS])),
        "power": d[layout.POWER],
        "toughness": d[layout.TOUGHNESS],
        "produces_any": float(d[layout.PRODUCES_COLORS].sum() + d[layout.PRODUCES_COLORLESS]) > 0,
    }


class CardBook:
    """Memoized per-name card facts: embedding + classification + label.

    ``get`` returns None for an unusable card (basic land, missing ``.npz`` or
    missing ``.txt``). ``q`` is the card's ``shrunk_score_play`` label; with a
    ``q_fallback`` (a number, or ``"median"`` for the median label) each entry
    also carries ``qv``, the label or the fallback where it is blank, for the
    probes that order cards by it. ``colors`` reads the colour flags, or with
    ``colors_from="pips"`` the colours that carry a cost pip.
    """

    def __init__(self, probe: Probe, win_rates: dict, q_fallback=None,
                 colors_from: str = "flags"):
        self.probe = probe
        self.wr = win_rates
        if q_fallback == "median":
            vals = [r["shrunk_score_play"] for r in win_rates.values()
                    if r.get("shrunk_score_play") is not None]
            q_fallback = float(np.median(vals)) if vals else 0.0
        self.q_fallback = q_fallback
        self.colors_from = colors_from
        self._cache: dict[str, dict | None] = {}

    def get(self, name: str) -> dict | None:
        if name not in self._cache:
            self._cache[name] = self._build(name)
        return self._cache[name]

    def _build(self, name: str) -> dict | None:
        if is_basic(name):
            return None
        emb = self.probe.embedding(name)
        if emb is None:
            return None
        text = self.probe.locator.load_text(name)
        if text is None:
            return None
        d = emb[-FC:]
        mana_cost = text.mana_cost_line()
        feats = card_features(text.text)
        q = (self.wr.get(name) or {}).get("shrunk_score_play")
        color_slots = d[layout.COLOR_PIPS] if self.colors_from == "pips" else d[layout.COLOR_FLAGS]
        threshold = 0.0 if self.colors_from == "pips" else 0.5
        info = {
            "name": name,
            "text": text.text,
            "feats": feats,
            "emb": np.ascontiguousarray(emb, dtype=np.float32),
            "is_land": float(d[layout.IS_LAND]) > 0.5,
            "mv": float(d[layout.MANA_VALUE]),
            "pips": np.asarray(d[layout.COLOR_PIPS], dtype=np.float64),
            "colors": frozenset(c for c, f in zip(WUBRG, color_slots) if float(f) > threshold),
            "produces": frozenset(
                c for c, f in zip(WUBRG, d[layout.PRODUCES_COLORS]) if float(f) > 0.5
            ),
            # '/' in the mana cost line == hybrid or phyrexian; those contaminate
            # pip counting, so every probe excludes them as swap candidates.
            "hybrid": bool(mana_cost and "/" in mana_cost),
            "is_creature": bool(feats["is_creature"]),
            # split / adventure / transform / meld: the converted text marks the
            # second face with a bare 'ALTERNATE' line ('alternate cost:' is a
            # different thing -- madness and friends -- and must not match).
            "is_multiface": " // " in name or any(
                ln.strip() == "ALTERNATE" for ln in text.text.splitlines()
            ),
            "q": q,
        }
        if self.q_fallback is not None:
            info["qv"] = self.q_fallback if q is None else float(q)
        return info


def deck_pips(book: CardBook, deck: list[str]) -> np.ndarray:
    pips = np.zeros(5)
    for n in deck:
        info = book.get(n)
        if info is not None:
            pips += info["pips"]
    return pips


def main_colors(book: CardBook, deck: list[str], k: int = 2) -> tuple[str, ...]:
    """The deck's top-k pip colors (only colors actually present)."""
    pips = deck_pips(book, deck)
    order = np.argsort(-pips)
    return tuple(WUBRG[i] for i in order[:k] if pips[i] > 0)


def context_queue(pools_path: Path, decks_path: Path, rng: random.Random):
    """``(pools, decks, queue)``: the line-aligned pool and deck corpora, and the
    indices of their same-set lines in a seeded round-robin over sets, so a
    sample drawn from the head of the queue spreads over sets instead of
    clumping in one."""
    pools = read_pools(pools_path)
    decks = read_generated_decks(decks_path)
    if len(pools) != len(decks):
        raise SystemExit(f"corpus misalignment: {len(pools)} pools vs {len(decks)} decks")

    by_set: dict[str, list[int]] = {}
    for i, (pool, deck) in enumerate(zip(pools, decks)):
        if pool.set_code == deck.set_code:
            by_set.setdefault(pool.set_code, []).append(i)
    for v in by_set.values():
        rng.shuffle(v)
    set_order = sorted(by_set)
    rng.shuffle(set_order)

    queue = [by_set[s][depth]
             for depth in range(max((len(v) for v in by_set.values()), default=0))
             for s in set_order if depth < len(by_set[s])]
    return pools, decks, queue


def resolve_context(book: CardBook, pools, decks, idx: int) -> dict | None:
    """The (pool, deck) context at line ``idx``; None unless every card resolves
    and the pool holds a card the deck did not take."""
    pool = pools[idx].cards
    deck = nonbasic(decks[idx].cards)
    if any(book.get(n) is None for n in pool) or any(book.get(n) is None for n in deck):
        return None
    avail = Counter(pool) - Counter(deck)
    if not avail:
        return None
    return {
        "cid": idx,
        "set_code": pools[idx].set_code,
        "deck": deck,
        "avail": avail,
        "main": main_colors(book, deck),
    }

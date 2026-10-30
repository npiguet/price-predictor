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
import re
from collections import Counter
from collections.abc import Callable, Hashable, Iterable, Iterator
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np
import torch
from _lib import corpora
from _lib.paths import MODELS, OUTPUT, SEALED_TRAINING_DATA

from price_predictor.domain.value_objects import WUBRG
from sealed.domain import card_embedding_layout as layout
from sealed.domain.deck import Deck
from sealed.domain.scorer_model import SetTransformerScorer
from sealed.infrastructure.converted_card_locator import (
    BASIC_LAND_NAMES,
    ConvertedCardLocator,
)
from sealed.infrastructure.delimited import parse_pipe_list
from sealed.infrastructure.match_data_loader import MatchOutcome, parse_match_outcome
from sealed.infrastructure.pool_file_reader import GeneratedDeck, SealedPool
from sealed.infrastructure.scorer_store import ScorerStore

SCORER_CKPT = MODELS / "sealed/scorer/512-best_l6_h4_s4_ff2176_mlp512_lr1e-05_mwlog.pt"
CARDS_PATH = OUTPUT / "cardsfolder-512"
YDATA = SEALED_TRAINING_DATA
WIN_RATES = YDATA / "matches-bo1" / "cards-win-rates.txt"
DATA_DIR = Path(__file__).resolve().parent
SCRATCH = OUTPUT / "scorer-probes"  # generated artifacts (gitignored)
SCRATCH.mkdir(parents=True, exist_ok=True)

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
MATCH_PATHS = [YDATA / rel for rel in MATCH_FILES]


def is_basic(name: str) -> bool:
    return name.lower() in BASIC_LAND_NAMES


def nonbasic(deck: Iterable[str]) -> list[str]:
    return [c for c in deck if not is_basic(c)]


def land_mask(mat):
    """Per row of a stacked (..., d_model) embedding array or tensor, whether
    its deterministic block flags a land."""
    return mat[..., -FC + layout.IS_LAND] > 0.5


@dataclass(frozen=True)
class Preset:
    """A run-size knob's full-run value and the tiny value ``--smoke`` uses."""

    full: Any
    smoke: Any


def apply_presets(args, presets: dict[str, Preset]) -> None:
    """Fill every ``args.<name>`` left at None (or never declared) from its
    preset, picked by ``args.smoke``: an explicit flag always wins."""
    for name, preset in presets.items():
        if getattr(args, name, None) is None:
            setattr(args, name, preset.smoke if args.smoke else preset.full)


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

    def deck_matrix_or_none(self, deck: Iterable[str]) -> np.ndarray | None:
        """``deck_matrix``, or None when a nonbasic card has no embedding."""
        try:
            return self.deck_matrix(list(deck))
        except KeyError:
            return None

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


def _head(path: Path, parse: Callable[[str], Any], limit: int | None) -> list:
    """``parse`` of each of the first ``limit`` (all when falsy) non-blank lines,
    read lazily: the corpora run to hundreds of MB and some callers want a head."""
    with open(path, encoding="utf-8") as handle:
        return [parse(line) for line in islice(filter(None, map(str.strip, handle)),
                                               limit or None)]


def _pool(line: str) -> SealedPool:
    set_code, _, names = line.partition(";")
    return SealedPool(set_code, parse_pipe_list(names))


def _generated_deck(line: str) -> GeneratedDeck:
    label, set_code, names = line.split(";", 2)
    return GeneratedDeck(label=label, set_code=set_code, deck=Deck.of(parse_pipe_list(names)))


def read_pools(path: Path, limit: int | None = None) -> list[SealedPool]:
    """The first ``limit`` (all when falsy) pools of a pools file."""
    return _head(path, _pool, limit)


def read_generated_decks(path: Path, limit: int | None = None) -> list[GeneratedDeck]:
    """The first ``limit`` (all when falsy) decks of a generated-decks file."""
    return _head(path, _generated_deck, limit)


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


@dataclass(frozen=True)
class MatchDeck:
    """One side of a match line: the deck a builder method built for a set."""

    method: str
    set_code: str
    cards: tuple[str, ...]
    line: int  # 0-based line of its match file

    @property
    def key(self) -> tuple[str, ...]:
        """The sorted nonbasic multiset: what ``DeckBank`` scores the deck by."""
        return tuple(sorted(nonbasic(self.cards)))


def sample_match_decks(files: Iterable[Path], cap: int | Callable[[Hashable], int] | None = None,
                       *, group: Callable[[MatchDeck], Hashable | None] = lambda d: d.method,
                       make: Callable[[MatchDeck], Any] | None = None, sides: str = "AB",
                       lines: int | None = None) -> Iterator:
    """The distinct decks of match files, in file order, at most ``cap`` per group.

    Two decks are the same when their ``MatchDeck.key`` is: the scorer never
    sees basics, so builds differing only in their basics count once.
    ``group(deck)`` names the cap a deck counts towards, or None to skip it;
    ``cap`` is one number for every group or a ``cap(group)`` callable.
    ``make(deck)`` turns a deck into what is yielded, or None to reject it: a
    rejected deck counts towards no cap and is not offered again. ``sides``
    picks the match sides read, ``lines`` the head of each file read (all
    when falsy). A missing file is reported and skipped.
    """
    seen: set[tuple[str, ...]] = set()
    taken: Counter = Counter()
    for path in files:
        if not path.exists():
            print("missing:", path, flush=True)
            continue
        for line, m in enumerate(islice(corpora.iter_matches(path), lines or None)):
            for side in sides:
                method, deck = (m.method_a, m.deck_a) if side == "A" else (m.method_b, m.deck_b)
                d = MatchDeck(method, m.set_code, tuple(deck.cards), line)
                g = group(d)
                if g is None or (cap is not None
                                 and taken[g] >= (cap(g) if callable(cap) else cap)):
                    continue
                if d.key in seen:
                    continue
                seen.add(d.key)
                out = d if make is None else make(d)
                if out is None:
                    continue
                taken[g] += 1
                yield out


_CLASS_TRICK = re.compile(r"gets? \+\d+/\+\d+ until end of turn")
_CLASS_TOKENS = re.compile(r"create.{0,60}token")


def card_classes(text: str) -> dict:
    """The card classes the T2 card-value slices cut on, from converted card
    text ("" for a card without one)."""
    txt = text.lower()
    types = re.search(r"types:.*", txt)
    types = types.group(0) if types else ""
    cost = re.search(r"mana cost:.*", txt)
    cost = cost.group(0) if cost else ""
    return {
        "is_trick": bool("instant" in types and _CLASS_TRICK.search(txt)),
        "is_counter": "counter target" in txt and "spell" in txt,
        "is_pw": "planeswalker" in types,
        "is_vehicle": "vehicle" in types,
        "makes_tokens": bool(_CLASS_TOKENS.search(txt)),
        "x_cost": "{x}" in cost,
        "hyb": "/" in cost,
        "tok_len": len(txt.split()),
    }

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

    type_line = line_val("types") or ""
    types = type_line.split()
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
        "type_line": type_line,
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
            "type_line": feats["type_line"],
            "mana_cost": (mana_cost or "").lower(),
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

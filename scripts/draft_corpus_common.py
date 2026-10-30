"""Shared helpers for the draft-corpus colour/lane diagnostics.

Imported by the draft-corpus scripts beside it and by ``draft_probes/``.

What lives here: parsing ``--drafts LABEL=PATH`` arguments, streaming
``drafts.jsonl`` with each record's geometry, the prefix-deck-scores reader and
the models that produce those scores, the ``shrunk_score_play`` card grader,
resolving a card's WUBRG colour identity from its converted-card ``.txt``, and
the colour summaries the lane diagnostics share (top-two lane, mana-base width,
effective colour count).
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path

from _lib.corpora import load_win_rates

from draft.domain.draft_geometry import DraftGeometry, DraftRecord
from draft.infrastructure import draft_record_io
from sealed.infrastructure.converted_card_locator import ConvertedCardLocator

WUBRG = "WUBRG"
BASIC_LAND_TYPES = frozenset(
    {"Plains", "Island", "Swamp", "Mountain", "Forest", "Wastes"}
)


class ColourResolver:
    """Card name -> WUBRG colour identity, memoized, from the converted corpus.

    An empty frozenset means colourless (artifact / land / devoid), which every
    caller treats as "always on-colour" and therefore excludes from the
    off-lane statistics.
    """

    def __init__(self, cards_path: Path) -> None:
        self._locator = ConvertedCardLocator(cards_path)
        self._cache: dict[str, frozenset] = {}
        self.unresolved: Counter = Counter()

    def __call__(self, card_name: str) -> frozenset:
        cached = self._cache.get(card_name)
        if cached is not None:
            return cached
        text = self._locator.load_text(card_name)
        out: frozenset = frozenset()
        if text is None:
            self.unresolved[card_name] += 1
        else:
            cost = (text.mana_cost_line() or "").upper()
            if cost:
                out = frozenset(c for c in WUBRG if "{%s}" % c in cost)
                if not out:
                    # hybrid / phyrexian pips ({W/U}, {W/P}) miss the exact-pip
                    # test above, so fall back to scanning inside the braces.
                    inner = "".join(re.findall(r"\{([^}]*)\}", cost))
                    out = frozenset(c for c in WUBRG if c in inner)
        self._cache[card_name] = out
        return out


def parse_drafts_arg(value: str) -> tuple[str, Path]:
    """``LABEL=PATH`` -> (label, path); a bare ``PATH`` labels by file stem."""
    if "=" in value:
        label, _, raw = value.partition("=")
        return label.strip(), Path(raw.strip())
    path = Path(value)
    return path.stem, path


def read_records(path: Path) -> Iterator[tuple[DraftRecord, DraftGeometry]]:
    """Stream one ``(record, geometry)`` per complete JSONL line."""
    for record in draft_record_io.read_records(path):
        yield record, DraftGeometry.from_record(record)


def read_prefix_scores(paths: Iterable[Path], key_by_file: bool = False):
    """``(rows, agent_of)`` from prefix-deck-scores JSONL files.

    ``rows`` maps a seat to ``{n_picks: deck_score}`` and ``agent_of`` the seat to
    its agent. A seat is ``(draft_id, seat)``, prefixed by the file name when
    ``key_by_file`` (corpora whose draft ids may collide). A torn final line from
    an interrupted run is skipped.
    """
    rows: dict = defaultdict(dict)
    agent_of: dict = {}
    for path in map(Path, paths):
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = (r["draft_id"], r["seat"])
                if key_by_file:
                    key = (path.name, *key)
                rows[key][r["n_picks"]] = r["deck_score"]
                agent_of[key] = r["agent"]
    return rows, agent_of


def load_labeler_and_scorer(cards_path: Path, scorer_path: Path, picker_path: Path,
                            build_method: str):
    """The deck labeler for full pools and the bare scorer for scoring a pool as is,
    over one shared card locator: ``(labeler, scorer, locator)``."""
    import torch

    from draft.application.generate_draft_data import GenerateDraftDataConfig, build_labeler
    from sealed.domain.scorer_model import SetTransformerScorer
    from sealed.infrastructure.scorer_store import ScorerStore

    locator = ConvertedCardLocator(cards_path)
    labeler = build_labeler(GenerateDraftDataConfig(
        n_drafts=0, agent_mix=[], scorer_checkpoint=scorer_path,
        picker_checkpoint=picker_path, cards_path=cards_path,
        build_method=build_method,     # never leave this defaulted: it defaults to picker
    ), locator=locator)
    ckpt = ScorerStore().load_checkpoint(scorer_path)
    scorer = SetTransformerScorer(ckpt.config)
    scorer.load_state_dict(ckpt.model_state_dict)
    scorer.eval()
    scorer.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return labeler, scorer, locator


def score_play(path: Path, min_in_deck: int = 0) -> dict[str, float]:
    """card -> ``shrunk_score_play`` from ``cards-win-rates.txt``, for cards seen in
    at least ``min_in_deck`` decks; blank cells (no signal) are skipped."""
    return {name: row["shrunk_score_play"]
            for name, row in load_win_rates(path, min_in_deck).items()
            if row["shrunk_score_play"] is not None}


def top_two_colours(pool: list[str], colours: ColourResolver) -> set[str]:
    """The two colours the seat drafted most cards in - its eventual "lane".

    Anchoring the off-lane statistics on the top *two* rather than on the built
    deck's colours is deliberate: a deck that ends up five colours would score
    as perfectly on-colour against itself, which is exactly the behaviour these
    diagnostics exist to detect.
    """
    counts: Counter = Counter()
    for card in pool:
        for colour in colours(card):
            counts[colour] += 1
    # Ties are broken by colour letter, not by Counter insertion order: the
    # latter derives from frozenset iteration, whose order follows per-process
    # randomized string hashing, and would make the whole report irreproducible
    # for any seat whose 2nd and 3rd colours are level.
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return {colour for colour, _ in ranked[:2]}


def mana_base_width(deck: list[str]) -> int:
    """Distinct basic land types in a built deck - its effective colour count.

    Read off the deck the builder actually assembled, so it reflects the pool's
    coherence rather than the pick-level colour spread.
    """
    return len({card for card in deck if card in BASIC_LAND_TYPES})


def effective_colour_count(counts: Counter, coverage: float = 0.8) -> int:
    """Smallest k whose top-k colours hold ``coverage`` of the coloured cards.

    Size-robust, so a 15-card pool after pack 1 and a 45-card pool after pack 3
    are directly comparable.
    """
    total = sum(counts.values())
    if total == 0:
        return 0
    running = 0
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    for rank, (_, n) in enumerate(ranked, start=1):
        running += n
        if running >= coverage * total:
            return rank
    return len(counts)

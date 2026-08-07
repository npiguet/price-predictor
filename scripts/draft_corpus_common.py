"""Shared helpers for the draft-corpus colour/lane diagnostics.

Imported by ``analyze_draft_lanes.py`` and ``analyze_pick_quality.py``; both are
run as ``python scripts/<name>.py``, which puts this directory on ``sys.path``.

What lives here: parsing ``--drafts LABEL=PATH`` arguments, streaming
``drafts.jsonl`` into the ``draft.domain.draft_geometry`` dataclasses, resolving
a card's WUBRG colour identity from its converted-card ``.txt``, and the two
colour summaries both scripts need (mana-base width, effective colour count).
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Iterator

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from draft.domain.draft_geometry import (  # noqa: E402
    Booster,
    DraftGeometry,
    DraftRecord,
    Seat,
)
from sealed.infrastructure.converted_card_locator import ConvertedCardLocator  # noqa: E402

WUBRG = "WUBRG"
BASIC_LAND_TYPES = frozenset(
    {"Plains", "Island", "Swamp", "Mountain", "Forest", "Wastes"}
)
_MANA_RE = re.compile(r"^mana cost:\s*(.*)$", re.IGNORECASE | re.MULTILINE)


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
        path = self._locator.text_path(card_name)
        out: frozenset = frozenset()
        if path is None:
            self.unresolved[card_name] += 1
        else:
            match = _MANA_RE.search(path.read_text(encoding="utf-8", errors="replace"))
            if match:
                cost = match.group(1).upper()
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
    """Stream one ``(record, geometry)`` per JSONL line.

    A trailing partial line (JVM crash mid-write) is skipped rather than fatal,
    matching every other reader of this corpus.
    """
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue
            record = DraftRecord(
                draft_id=raw["draft_id"],
                run_id=raw["run_id"],
                timestamp=raw["timestamp"],
                seats=[
                    Seat(s["agent"], s["deck"], s["deck_score"]) for s in raw["seats"]
                ],
                boosters=[Booster(b["set_code"], b["picks"]) for b in raw["boosters"]],
            )
            yield record, DraftGeometry.from_record(record)


def seat_pool(record: DraftRecord, geometry: DraftGeometry, seat: int) -> list[str]:
    """The seat's 45 picks in (pack, pick) order."""
    return [
        geometry.taken_card(record, seat, pack, pick)
        for pack in range(1, geometry.packs + 1)
        for pick in range(1, geometry.pack_size + 1)
    ]


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

"""Which records a curated training corpus keeps, and how many times.

Every decision is shaped so a worker can apply it to one shard without talking
to any other worker. The survey keeps each text's smallest record hashes per
selection cell; the main process turns each text's quota into a
:class:`CopyPlan` — a base copy count and a hash threshold — and a write worker
then decides a record's copies by comparing one number. The class mixture sets
each class's budget, the game-disjoint stratum is a per-game hash test, and
both are pure functions of the corpus, the flags and the seed (FR-052).

A text over its cap contributes a *random* subset rather than its first N: a
text's earliest records are its earliest games, and keeping those would
describe one opening board over and over — the bias the mana reservoir sampler
exists to avoid, on the other side of the pipeline.
"""

from __future__ import annotations

import hashlib
import heapq
import zlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

#: uint64, which is the width the threshold comparison is over.
_HASH_BYTES = 8


def record_hash(record_id: str, *, seed: int) -> int:
    """A stable uniform draw in [0, 2^64) for one record.

    ``blake2b`` keyed by the seed rather than the built-in ``hash()``, which is
    salted per process: a threshold computed in the survey pass would admit a
    different set of records in the write pass, and nothing would say so
    (FR-088a).
    """
    digest = hashlib.blake2b(
        record_id.encode("utf-8"),
        digest_size=_HASH_BYTES,
        key=str(seed).encode("utf-8"),
    ).digest()
    return int.from_bytes(digest, "big")


class CapHeap:
    """The ``cap`` smallest record hashes seen for one ability text.

    A max-heap of negated values, so the largest retained hash sits at the root
    and is the first one a smaller arrival displaces. Held bounded rather than
    collecting every hash: the corpus has tens of millions of records, and no
    quota a text is given exceeds the cap.
    """

    __slots__ = ("cap", "_heap")

    def __init__(self, cap: int) -> None:
        self.cap = cap
        self._heap: list[int] = []

    def offer(self, value: int) -> None:
        if self.cap <= 0:
            return
        if len(self._heap) < self.cap:
            heapq.heappush(self._heap, -value)
        elif -value > self._heap[0]:
            heapq.heappushpop(self._heap, -value)

    def merge(self, other: CapHeap) -> None:
        """Absorb another heap's values, keeping the cap smallest of the union.

        Valid only when both heaps share the same cap, because the k smallest of
        a union are drawn from each side's own k smallest. This precondition is
        enforced to prevent silent data loss: if the other heap discarded values
        due to a smaller cap, they are irrecoverably lost here.
        """
        if self.cap != other.cap:
            raise ValueError(
                f"cannot merge heaps with mismatched caps: "
                f"self.cap={self.cap}, other.cap={other.cap}"
            )
        for negated in other._heap:
            self.offer(-negated)

    def values(self) -> tuple[int, ...]:
        return tuple(-negated for negated in self._heap)


def class_targets(
    available: Mapping[str, int],
    mix: Mapping[str, float],
    *,
    ceiling: int = 0,
) -> tuple[dict[str, int], dict[str, int]]:
    """Per-class write targets, and the shortfall against a requested ceiling.

    The scarcest class sets the size of the whole dataset: the largest total
    the mixture can be written at is ``min(available[c] / share[c])``, and
    every class then takes its share of that. Filling the abundant classes to
    their own capacity instead would write the mixture the disk happens to
    hold, which is the shape the dataset exists to correct.

    ``ceiling`` (``--training-records``) lowers that total. A ceiling above
    what the mixture supports cannot be met, and the difference is returned per
    class rather than silently under-filled (FR-139).
    """
    shares = {
        name: float(share)
        for name, share in mix.items()
        if share > 0 and available.get(name, 0) > 0
    }
    if not shares:
        raise ValueError(f"no positive share with records available: mix={dict(mix)}")
    total_share = sum(shares.values())
    supported = min(
        available[name] * total_share / share for name, share in shares.items()
    )
    total = min(supported, float(ceiling)) if ceiling > 0 else supported
    targets = {
        name: min(available[name], int(total * share / total_share))
        for name, share in shares.items()
    }
    shortfall: dict[str, int] = {}
    if ceiling > 0 and ceiling > supported:
        for name, share in shares.items():
            missing = int(ceiling * share / total_share) - targets[name]
            if missing > 0:
                shortfall[name] = missing
    return targets, shortfall


# ── gen-2 selection (FR-049) ────────────────────────────────────────────

#: The sampling classes whose families split further by outcome signature
#: (FR-049 step 2). The others have no per-entity outcome to sort by: a cost
#: half's outcome is its payment, a trigger's is whether it fired.
SIGNATURE_CLASSES: frozenset[str] = frozenset({
    "resolution-effect", "rewrite", "combat", "continuous",
})


def outcome_signature(record) -> frozenset[tuple[str, bool]]:
    """The distinct ``(zone outcome, changed)`` pairs over a record's entities.

    From the targets the head trains on (``derive_targets``): the entity's
    ``zone_outcome`` or ``"stayed"`` when it has none, paired with whether any
    event touched it. A set, not a multiset, so a wipe that kills three
    creatures and one that kills seven share a signature rather than splitting
    one rule into thin cells by board size (FR-049 step 2).

    The import is local because ``effect_targets`` reads the head's field
    layout, which pulls in torch; this module is otherwise torch-free.
    """
    from effects.domain.effect_model import ZONE_OUTCOMES
    from effects.domain.effect_targets import derive_targets

    pairs = set()
    for entry in derive_targets(record).values():
        zone = entry.fields.get("zone_outcome")
        pairs.add((ZONE_OUTCOMES[zone] if zone is not None else "stayed", entry.affected))
    return frozenset(pairs)


def signature_label(signature: frozenset[tuple[str, bool]]) -> str:
    """A signature as one stable string, for a cell key and the manifest."""
    if not signature:
        return "(empty)"
    return ",".join(sorted(
        f"{zone}:{'changed' if changed else 'unchanged'}" for zone, changed in signature
    ))


def text_capacity(distinct: int, *, text_cap: int, reuse_cap: int) -> int:
    """How many written records one text can supply to one cell (FR-049).

    ``min(reuse_cap × n, text_cap)``, repeats included, so no text is replayed
    past the cap however short its family is. ``text_cap <= 0`` is no cap, as
    it always was; a record with no text has no text cap either, and its cell
    passes ``text_cap=0``.
    """
    repeated = max(0, reuse_cap) * distinct
    return repeated if text_cap <= 0 else min(repeated, text_cap)


@dataclass(frozen=True, slots=True)
class CopyPlan:
    """How many copies each record of one text in one cell is written.

    Every record gets ``base`` copies and one more when its hash is at most
    ``threshold``. Two numbers rather than a table, so a write worker applies
    it to a record it has never seen listed, by comparing one hash.
    """

    base: int = 0
    threshold: int | None = None

    def copies(self, value: int) -> int:
        extra = 1 if self.threshold is not None and value <= self.threshold else 0
        return self.base + extra


def copy_plan(smallest: Sequence[int], distinct: int, quota: int) -> CopyPlan:
    """The plan that writes ``quota`` records of a text with ``distinct`` records.

    ``smallest`` holds the text's smallest record hashes in the cell, at least
    ``min(distinct, quota)`` of them — the survey keeps the ``--text-cap``
    smallest, and a quota never exceeds the cap. A quota at or below
    ``distinct`` takes that many records once each, by smallest hash: a random
    subset under ``--seed`` (FR-049 step 4). A larger quota spreads evenly, each
    record ``quota // distinct`` times and the remainder once more on the
    smallest hashes, so repeats differ by at most one.
    """
    ordered = sorted(smallest)
    if quota <= 0 or distinct <= 0:
        return CopyPlan()
    if quota <= distinct:
        return CopyPlan(base=0, threshold=ordered[quota - 1])
    base, extra = divmod(quota, distinct)
    return CopyPlan(base=base, threshold=ordered[extra - 1] if extra else None)


def rate_plan(distinct: int, quota: int) -> CopyPlan:
    """A copy plan for records that carry no ability text, by hash rate.

    Combat and legality records name no acting line, so there is no text cap
    and no per-text heap: the cell is sampled at the rate its quota asks,
    ``quota / distinct`` copies a record on average, as a hash threshold. Exact
    in expectation rather than in count, which is what the class admission
    rate always was.
    """
    if quota <= 0 or distinct <= 0:
        return CopyPlan()
    base, extra = divmod(quota, distinct)
    if not extra:
        return CopyPlan(base=base)
    return CopyPlan(base=base, threshold=int(extra / distinct * 2**64) - 1)


def copies_for_text(
    record_hashes: Iterable[int], quota: int, text_cap: int, reuse_cap: int,
) -> dict[int, int]:
    """``record hash -> copies`` for one text's records in one cell (FR-049).

    The quota is first bounded by the text's capacity. Over the cap, the cap
    smallest hashes are written once each; otherwise the quota is spread
    evenly, the extra copies going to the smallest hashes.
    """
    hashes = sorted(set(record_hashes))
    capacity = text_capacity(len(hashes), text_cap=text_cap, reuse_cap=reuse_cap)
    plan = copy_plan(hashes, len(hashes), min(quota, capacity))
    return {value: plan.copies(value) for value in hashes}


def game_disjoint_draw(game_id: str) -> float:
    """A game's position in [0, 1), from its id alone (FR-044)."""
    return zlib.crc32(game_id.encode("utf-8")) % 10**6 / 10**6


def in_game_disjoint_stratum(
    game_id: str,
    names_held_out: bool,
    has_keyword_combat: bool,
    share: float,
    keyword_share: float,
) -> bool:
    """Whether a game enters the game-disjoint validation stratum (FR-044–046).

    A pure function of the game, so a rebuild over a grown corpus places every
    game it placed before (SC-007): the stratum grows with the corpus rather
    than being redrawn from it. A game naming a held-out card is card-disjoint
    and never placed here. A game holding a qualifying combat for a listed
    keyword is placed at the higher threshold, because those combats are what
    gate 2 scores and a 1% sample of them is too few to score.
    """
    if names_held_out:
        return False
    threshold = keyword_share if has_keyword_combat else share
    return game_disjoint_draw(game_id) < threshold

"""Typed-token draft-state reconstruction (FR-017 … FR-021).

For a target pick ``(seat s, pack p, pick i)`` this rebuilds the four
mutually-exclusive token groups the agent sees — ``POOL`` / ``PACK`` /
``PASSED`` / ``TAKEN`` — plus per-card recency ``(packs_ago, pick_ago)``, by
walking the seat's booster sightings through :mod:`draft.domain.draft_geometry`.
Pure domain logic, no torch.

Token-type lifecycle (data-model §3):

- ``POOL``   — cards the seat itself drafted; accumulates across all packs.
- ``PACK``   — the legal actions at this pick (cards currently in the held
  pack); reset each pick.
- ``PASSED`` — cards the seat saw earlier *this pack* and passed, whose fate is
  not yet known; emptied at every pack boundary.
- ``TAKEN``  — cards the seat knows were taken by *others*; accumulates across
  the draft. Cards enter ``TAKEN`` via (a) a wheel diff — when a booster the
  seat already saw returns, the cards it passed that are now gone were taken by
  others (FR-019a) — and (b) a pack-end flush — anything still in ``PASSED``
  when a pack is exhausted (FR-019b). Wheel *survivors* (still present on
  return) re-enter ``PACK`` instead.

Recency (FR-021):

- ``packs_ago ∈ {0,1,2}`` = packs since the card was last in the seat's pack
  (``0`` = this pack, wheel-capable), capped at 2.
- ``pick_ago ∈ {0,…,P-1}`` = picks since the card was last in the seat's pack
  *prior to this pick* (``0`` if never before), and **frozen** at its
  end-of-pack value (``P - last_pick``) once ``packs_ago ≥ 1``.
"""

from __future__ import annotations

from dataclasses import dataclass

from draft.domain.draft_geometry import DraftGeometry, DraftRecord

# Token-type indices (one-hot order: POOL, PACK, PASSED, TAKEN — data-model §3).
TYPE_POOL = 0
TYPE_PACK = 1
TYPE_PASSED = 2
TYPE_TAKEN = 3
NUM_TYPES = 4

_PACKS_AGO_CAP = 2  # packs_ago ∈ {0,1,2}


@dataclass(frozen=True, slots=True)
class CardInstance:
    """One observed card instance, in exactly one type at a time (FR-018)."""

    name: str
    token_type: int          # TYPE_POOL / TYPE_PACK / TYPE_PASSED / TYPE_TAKEN
    packs_ago: int           # {0,1,2}
    pick_ago: int            # {0,…,P-1}


@dataclass(frozen=True, slots=True)
class DraftState:
    """Reconstructed typed-token state for one ``(seat, pack, pick)`` example."""

    pack_number: int             # 1-based
    pick_number: int             # 1-based
    pack_actions: list[str]      # distinct PACK card names, in pick order
    target_index: int            # index into pack_actions of the card taken
    cards: list[CardInstance]    # POOL ∪ PACK ∪ PASSED ∪ TAKEN instances

    @property
    def pack_instances(self) -> list[CardInstance]:
        return [c for c in self.cards if c.token_type == TYPE_PACK]


def _recency(
    prior_last_seen: dict[str, tuple[int, int]],
    name: str,
    target_pack: int,
    target_pick: int,
    pack_size: int,
) -> tuple[int, int]:
    """Compute ``(packs_ago, pick_ago)`` from the pre-target last-sighting map."""
    if name not in prior_last_seen:
        return 0, 0  # never in the seat's pack before this pick (fresh PACK card)
    last_pack, last_pick = prior_last_seen[name]
    packs_ago = min(_PACKS_AGO_CAP, target_pack - last_pack)
    if packs_ago == 0:
        pick_ago = target_pick - last_pick
    else:
        # Frozen at the end-of-source-pack staleness (picks remaining after the
        # last sighting in that pack), independent of the current pick number.
        pick_ago = pack_size - last_pick
    pick_ago = max(0, min(pack_size - 1, pick_ago))
    return packs_ago, pick_ago


def build_state(
    record: DraftRecord,
    geometry: DraftGeometry,
    seat: int,
    pack: int,
    pick: int,
) -> DraftState:
    """Reconstruct the typed-token state for one target ``(seat, pack, pick)``.

    ``pack`` and ``pick`` are 1-based. The returned ``pack_actions`` are the
    distinct legal actions in pick order; ``target_index`` points at the card
    actually taken (the model is permutation-equivariant over card tokens, so
    this order is not a positional leak).
    """
    pod = geometry.pod_size
    P = geometry.pack_size

    pool: list[str] = []                       # self-drafted (multiset, ordered)
    taken: list[str] = []                      # known taken by others (one per instance)
    passed: dict[str, tuple[int, int]] = {}    # current-pack, unresolved: name -> last (pack,pick)
    last_seen: dict[str, tuple[int, int]] = {} # name -> last (pack,pick) in seat's pack

    def visible(p: int, i: int) -> tuple[int, int, list[str]]:
        k, offset = geometry.booster_for_pick(seat, p, i)
        return k, offset, record.boosters[k].picks[offset:]

    # Walk completed picks chronologically, up to (but excluding) the target take.
    for p in range(1, pack + 1):
        last_i = P if p < pack else pick - 1
        for i in range(1, last_i + 1):
            _k, offset, vis = visible(p, i)
            # Wheel diff: this booster was held pod picks ago (same booster).
            if offset >= pod:
                for c in record.boosters[_k].picks[offset - pod + 1: offset]:
                    passed.pop(c, None)
                    taken.append(c)  # taken by others; recency frozen at last sighting
            taken_card = vis[0]
            pool.append(taken_card)
            passed.pop(taken_card, None)
            for c in vis:
                last_seen[c] = (p, i)
            for c in vis[1:]:
                passed[c] = (p, i)
        if p < pack:
            # Pack boundary: flush remaining PASSED to TAKEN (FR-019b).
            for c in list(passed.keys()):
                taken.append(c)
            passed.clear()

    # Snapshot last_seen *before* the target observation so recency is "prior to
    # this pick" (FR-021); the target sighting itself does not count.
    prior_last_seen = dict(last_seen)

    # Target observation. If the target booster is wheeling back, resolve its
    # diff first (FR-019a): cards the seat passed from it that are gone now were
    # taken by others; survivors are the PACK below.
    _k, off_t, vis_t = visible(pack, pick)
    if off_t >= pod:
        for c in record.boosters[_k].picks[off_t - pod + 1: off_t]:
            passed.pop(c, None)
            taken.append(c)

    pack_actions: list[str] = []
    seen: set[str] = set()
    for c in vis_t:
        if c not in seen:
            seen.add(c)
            pack_actions.append(c)
    for c in pack_actions:
        passed.pop(c, None)  # FR-019a survivor: was passed, now back in pack

    instances: list[CardInstance] = []

    def add(name: str, token_type: int) -> None:
        packs_ago, pick_ago = _recency(prior_last_seen, name, pack, pick, P)
        instances.append(CardInstance(name, token_type, packs_ago, pick_ago))

    for name in pool:
        add(name, TYPE_POOL)
    for name in pack_actions:
        add(name, TYPE_PACK)
    for name in passed:
        add(name, TYPE_PASSED)
    for name in taken:
        add(name, TYPE_TAKEN)

    taken_card = vis_t[0]
    target_index = pack_actions.index(taken_card)
    return DraftState(
        pack_number=pack,
        pick_number=pick,
        pack_actions=pack_actions,
        target_index=target_index,
        cards=instances,
    )

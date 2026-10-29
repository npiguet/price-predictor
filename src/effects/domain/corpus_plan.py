"""What a curated corpus is planned to hold, and where each game goes (FR-049, FR-142).

Pure and torch-free: the survey fills a :class:`TrainingSupply` per shard, the
main process folds its keys to ability texts, and everything below turns the
folded supply into copy plans and the survey's game sets into strata. Nothing
here reads a file, so every rule is a function a test can call with numbers.

The selection hierarchy is a tree -- class, legality half, rule family, cell,
ability text -- and every level splits its budget the same way
(:func:`effects.domain.budget_allocation.allocate`): equal shares, capped by
what each member can supply with repeats, the remainder passed on. One
:class:`AllocationNode` walk does all of it and keeps each level's share and
planned count for the manifest.
"""

from __future__ import annotations

import dataclasses
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from effects.domain.corpus_curation import (
    CapHeap,
    CopyPlan,
    class_targets,
    copy_plan,
    game_disjoint_draw,
    in_game_disjoint_stratum,
    rate_plan,
    text_capacity,
)
from effects.domain.provenance import ProvenanceKey
from effects.domain.sampling import SamplingClass
from effects.domain.strata import Stratum

#: A record's acting keys: the whole tuple, hashable and picklable as it is.
Keys = tuple[ProvenanceKey, ...]


class BuildCorpusError(ValueError):
    """A build this corpus, these flags or this working directory cannot do.

    Separated from a bare ``ValueError`` so the CLI can report an operator
    condition as one logged line and a non-zero exit while letting an
    internal-invariant violation -- ``decide``'s cap mismatch,
    ``CapHeap.merge``'s -- keep its traceback. The two arrive at the same
    ``except`` clause otherwise, and a bug in the builder then reads exactly
    like a misconfigured run.

    A ``ValueError`` subclass rather than a fresh exception type, so a caller
    that already handles the broader class keeps working.
    """


# ── cells ───────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True, order=True)
class Cell:
    """One selection cell of FR-049: class, legality half, family, signature.

    ``half`` is set on ``playability-legality`` only (``real`` or
    ``what-if``), ``signature`` on the four classes FR-049 step 2 splits by
    outcome. A frozen dataclass rather than a joined string, so it pickles
    between processes and sorts the same way in all of them.

    ``modes`` is set on a survey's legality cell only, before the corpus-wide
    mode counts exist: a legality record's family is its rarest mode, which
    only the whole corpus can say. :meth:`resolved` turns it into the family,
    in this one place, and every cell the planner sorts is resolved.
    """

    klass: SamplingClass
    half: str = ""
    family: str = ""
    signature: str = ""
    modes: frozenset[str] = frozenset()

    def resolved(self, mode_counts: Mapping[str, int]) -> Cell:
        """This cell with its legality modes folded into their rarest mode's family."""
        if not self.modes:
            return self
        from effects.domain.rule_families import legality_family

        return dataclasses.replace(
            self, family=legality_family(self.modes, mode_counts), modes=frozenset(),
        )

    @property
    def label(self) -> str:
        """The family's name in the manifest: prefixed by its half on a legality cell."""
        return f"{self.half}:{self.family}" if self.half else self.family


# ── what the training-bound records can supply ──────────────────────────


@dataclass(slots=True)
class TrainingSupply:
    """Training-bound records per cell and acting keys, and their smallest hashes.

    Only records the write pass will route to training are counted, so every
    budget and threshold planned from it is one the write pass can fill: a
    record of a held-out, card-disjoint or game-disjoint game never reaches a
    copy plan's denominator.

    ``heaps`` keeps each slot's ``heap_cap`` smallest record hashes: enough to
    place any quota a text can be given, since no quota exceeds the cap, and
    bounded where every hash would not be. Records with no acting key --
    combat and legality, which have no text to cap -- are counted per cell in
    ``keyless`` and sampled by hash rate instead.
    """

    heap_cap: int
    keyed: Counter[tuple[Cell, Keys]] = field(default_factory=Counter)
    heaps: dict[tuple[Cell, Keys], CapHeap] = field(default_factory=dict)
    keyless: Counter[Cell] = field(default_factory=Counter)

    def add(self, cell: Cell, keys: Keys | None, value: int) -> None:
        """One training-bound record, by its cell, acting keys and record hash."""
        if not keys:
            self.keyless[cell] += 1
            return
        slot = (cell, keys)
        self.keyed[slot] += 1
        heap = self.heaps.get(slot)
        if heap is None:
            heap = self.heaps[slot] = CapHeap(self.heap_cap)
        heap.offer(value)

    def merge(self, other: TrainingSupply) -> None:
        """Absorb another shard's supply; both must keep the same heap size."""
        if other.heap_cap != self.heap_cap:
            raise ValueError(
                f"cannot merge supplies keeping {other.heap_cap} and "
                f"{self.heap_cap} hashes per slot"
            )
        self.keyed.update(other.keyed)
        self.keyless.update(other.keyless)
        for slot, heap in other.heaps.items():
            mine = self.heaps.get(slot)
            if mine is None:
                self.heaps[slot] = heap
            else:
                mine.merge(heap)

    def resolved(self, mode_counts: Mapping[str, int]) -> TrainingSupply:
        """Every legality cell folded into its family; cells that fold together merge."""
        out = TrainingSupply(self.heap_cap)
        for cell, count in self.keyless.items():
            out.keyless[cell.resolved(mode_counts)] += count
        for (cell, keys), count in self.keyed.items():
            slot = (cell.resolved(mode_counts), keys)
            out.keyed[slot] += count
            heap = self.heaps[cell, keys]
            mine = out.heaps.get(slot)
            if mine is None:
                out.heaps[slot] = CapHeap(self.heap_cap)
                mine = out.heaps[slot]
            mine.merge(heap)
        return out


#: The member name of a cell's records whose keys fold to no text, or that
#: carry none. No ability text is empty, so it cannot collide with one.
KEYLESS = ""


@dataclass(slots=True)
class Member:
    """What one ability text supplies to one cell (FR-049).

    ``heap`` holds the text's smallest record hashes, enough to place any
    quota; the keyless member has no heap and is sampled by hash rate.
    """

    distinct: int = 0
    heap: CapHeap | None = None

    def once(self, name: str, text_cap: int) -> int:
        """Records this member can supply once each: the text cap, or all of them."""
        if name == KEYLESS or text_cap <= 0:
            return self.distinct
        return min(self.distinct, text_cap)

    def capacity(self, name: str, *, text_cap: int, reuse_cap: int) -> int:
        """Records this member can supply with repeats."""
        return text_capacity(
            self.distinct, text_cap=0 if name == KEYLESS else text_cap,
            reuse_cap=reuse_cap,
        )


def members_of(
    supply: TrainingSupply, key_text: Mapping[Keys, str],
) -> tuple[dict[Cell, dict[str, Member]], dict[tuple[Cell, str], list[Keys]]]:
    """``cell -> member name -> Member``, and each member's acting keys.

    A member is an ability text -- several keys folding to one text share it,
    and its cap -- or :data:`KEYLESS` for the records whose keys fold to no
    text or that carry none.
    """
    members: dict[Cell, dict[str, Member]] = defaultdict(dict)
    keys_of: dict[tuple[Cell, str], list[Keys]] = defaultdict(list)
    for (cell, keys), count in supply.keyed.items():
        text = key_text.get(keys)
        if text is None:
            members[cell].setdefault(KEYLESS, Member()).distinct += count
            continue
        member = members[cell].setdefault(text, Member(heap=CapHeap(supply.heap_cap)))
        member.distinct += count
        member.heap.merge(supply.heaps[cell, keys])
        keys_of[cell, text].append(keys)
    for cell, count in supply.keyless.items():
        members[cell].setdefault(KEYLESS, Member()).distinct += count
    return dict(members), dict(keys_of)


# ── the allocation tree ─────────────────────────────────────────────────


@dataclass(slots=True)
class AllocationNode:
    """One level of the selection hierarchy, and what it was given.

    A leaf is one member of one cell: ``supply`` is what it can write with
    repeats and ``once`` what it can write without repeating a record. An
    inner node's capacity is its children's. :meth:`allocate` splits a
    budget down the tree and records at every node its ``share`` -- the equal
    split it would get if every node at every level could fill its own, so
    ``share - written`` is a shortfall -- and its ``planned`` count.
    """

    children: dict[Any, AllocationNode] = field(default_factory=dict)
    supply: int = 0
    once: int | None = None
    share: int = 0
    planned: int = 0

    @property
    def capacity(self) -> int:
        if self.children:
            return sum(child.capacity for child in self.children.values())
        return self.supply

    def child(self, key: Any) -> AllocationNode:
        node = self.children.get(key)
        if node is None:
            node = self.children[key] = AllocationNode()
        return node

    def allocate(self, budget: int, *, share: int | None = None) -> int:
        """Split ``budget`` down the tree; return what this node plans to write.

        Every level is the same equal split with redistribution. Where the
        children are a cell's members it runs in two rounds -- first over
        what each supplies once, so no record is repeated while another goes
        unused, then over the repeats the remainder still needs -- which is
        the only case FR-049 repeats at all: a family short of its share.
        """
        self.share = budget if share is None else share
        if not self.children:
            self.planned = min(max(0, budget), self.capacity)
            return self.planned
        split = _split(budget, self.children)
        fair = self.share // len(self.children)
        self.planned = sum(
            child.allocate(split[key], share=fair) for key, child in self.children.items()
        )
        return self.planned


def _split(budget: int, children: Mapping[Any, AllocationNode]) -> dict[Any, int]:
    from effects.domain.budget_allocation import allocate

    if any(child.once is None for child in children.values()):
        return allocate(budget, {key: child.capacity for key, child in children.items()})
    first = allocate(budget, {key: child.once for key, child in children.items()})
    left = budget - sum(first.values())
    if left <= 0:
        return first
    second = allocate(left, {
        key: child.capacity - first[key] for key, child in children.items()
    })
    return {key: first[key] + second.get(key, 0) for key in children}


@dataclass(frozen=True, slots=True)
class FamilyShare:
    """One family's place in its class budget, as decided (FR-053).

    ``share`` is the family's equal split of its class (or legality half)
    budget, before redistribution, so ``max(0, share - written)`` is its
    shortfall: the number the operator sends back to the coverage and variant
    collectors. ``planned`` is what the plan asked the write pass for.
    """

    available: int
    share: int
    planned: int


@dataclass(frozen=True, slots=True)
class SelectionPlan:
    """Every training record's copies, and what each level was given."""

    #: ``(cell, acting keys) -> copies``. Keys folding to one text share a plan.
    plans: dict[tuple[Cell, Keys], CopyPlan]
    #: ``cell -> copies`` for records whose keys fold to no text or that carry none.
    keyless: dict[Cell, CopyPlan]
    #: ``class -> family label -> FamilyShare``.
    families: dict[str, dict[str, FamilyShare]]
    #: Records each class is asked to write, copies included.
    targets: dict[str, int]
    #: What ``--training-records`` asked of a class beyond what it can supply.
    shortfall: dict[str, int]
    #: Distinct training-bound records each class can supply under the per-cell
    #: text cap, with no repeats: what the class mixture is divided over.
    available: dict[str, int]


def plan_selection(
    members: Mapping[Cell, Mapping[str, Member]],
    keys_of: Mapping[tuple[Cell, str], list[Keys]],
    *,
    mix: Mapping[str, float],
    ceiling: int,
    text_cap: int,
    reuse_cap: int,
) -> SelectionPlan:
    """FR-049 steps 1-4: class budget -> half -> family -> signature -> text.

    Raises:
        BuildCorpusError: ``mix`` names no class this corpus has a record of.
    """
    available: Counter[str] = Counter()
    for cell, cell_members in members.items():
        for name, member in cell_members.items():
            available[cell.klass] += member.once(name, text_cap)
    try:
        targets, shortfall = class_targets(dict(available), mix, ceiling=ceiling)
    except ValueError as exc:
        # `--class-mix` names classes this corpus has no records of: the
        # operator's flag against the operator's corpus, not a broken invariant.
        raise BuildCorpusError(str(exc)) from exc

    classes: dict[str, AllocationNode] = {}
    for cell, cell_members in members.items():
        node = classes.setdefault(cell.klass, AllocationNode())
        cell_node = node.child(cell.half).child(cell.family).child(cell)
        for name, member in cell_members.items():
            cell_node.children[name] = AllocationNode(
                supply=member.capacity(name, text_cap=text_cap, reuse_cap=reuse_cap),
                once=member.once(name, text_cap),
            )

    plans: dict[tuple[Cell, Keys], CopyPlan] = {}
    keyless: dict[Cell, CopyPlan] = {}
    families: dict[str, dict[str, FamilyShare]] = {}
    for klass, root in sorted(classes.items()):
        root.allocate(targets.get(klass, 0))
        for half, half_node in sorted(root.children.items()):
            for family, family_node in sorted(half_node.children.items()):
                families.setdefault(klass, {})[Cell(klass, half, family).label] = FamilyShare(
                    available=sum(
                        member.distinct
                        for cell in family_node.children for member in members[cell].values()
                    ),
                    share=family_node.share,
                    planned=family_node.planned,
                )
                for cell, cell_node in sorted(family_node.children.items()):
                    for name, leaf in cell_node.children.items():
                        member = members[cell][name]
                        if name == KEYLESS:
                            keyless[cell] = rate_plan(member.distinct, leaf.planned)
                            continue
                        plan = copy_plan(member.heap.values(), member.distinct, leaf.planned)
                        for keys in keys_of[cell, name]:
                            plans[cell, keys] = plan
    return SelectionPlan(
        plans=plans, keyless=keyless, families=families, targets=targets,
        shortfall=shortfall, available=dict(available),
    )


# ── where each game goes ────────────────────────────────────────────────


class GameRoute(StrEnum):
    """Where the survey places a game, before the card-disjoint cap is applied."""

    TRAINING = "training"
    GAME_DISJOINT = "game-disjoint"
    #: Card-disjoint or dropped: which one the cap decides once every held-out
    #: game is known, but never training either way.
    HELD_OUT = "held-out"


@dataclass(frozen=True, slots=True)
class GameRouting:
    """The per-game placement rule, decided once per game by the survey (FR-044-046).

    A pure function of the game, so a rebuild over a grown corpus places every
    game it placed before, and so the survey can apply it at the end of each
    shard: a raw shard is one JVM lifetime, and a game never outlives the JVM
    that played it. The write pass reads the game sets the survey built with
    it rather than deciding again.
    """

    game_disjoint_share: float
    keyword_share: float

    def route(self, game_id: str, *, held_out: bool, keyword_combat: bool) -> GameRoute:
        if held_out:
            return GameRoute.HELD_OUT
        if in_game_disjoint_stratum(
            game_id, keyword_combat, self.game_disjoint_share, self.keyword_share,
        ):
            return GameRoute.GAME_DISJOINT
        return GameRoute.TRAINING

    def under_keyword_threshold(self, game_id: str) -> bool:
        """A game-disjoint game placed only by the keyword threshold."""
        return game_disjoint_draw(game_id) >= self.game_disjoint_share


@dataclass(frozen=True, slots=True)
class GameStrata:
    """Every game's stratum, as the write pass applies it."""

    card_disjoint: frozenset[str]
    game_disjoint: frozenset[str]
    #: Every game naming a held-out card; those the cap did not admit are dropped.
    held_out: frozenset[str]

    def stratum_of(self, game_id: str) -> Stratum:
        if game_id in self.card_disjoint:
            return Stratum.CARD_DISJOINT
        if game_id in self.held_out:
            return Stratum.DROPPED_HELD_OUT
        if game_id in self.game_disjoint:
            return Stratum.GAME_DISJOINT
        return Stratum.TRAINING


def admit_card_disjoint(
    held_out_games: Iterable[str],
    text_games: Mapping[str, Iterable[str]],
    *,
    cap: int,
    seed: int,
) -> frozenset[str]:
    """Held-out games admitted while every held-out text they resolve is under cap.

    Games rather than records (FR-142, FR-136): a stratum holds whole games,
    the unit the split is decided in, so the cap admits or declines a game
    rather than trimming records inside one.

    ``text_games`` maps each held-out text to the games holding a
    **resolution** record acting through it: gate 1 and the gate-one slice
    score resolution records only, so a game whose only contact with a text is
    a decision candidate, an unfired trigger, a rewrite or a cost half holds
    nothing gate 1 measures and must not take one of the text's slots.

    A game is admitted only while **every** held-out text it carries is still
    under the cap, not merely one of them. Admitting on one never declined a
    game: its own private texts sit at zero forever. The cost is that a game
    pairing a saturated text with an unsaturated one is declined, which can
    leave the second short; the cap is a ceiling on how far any one held-out
    text may dominate the stratum, and a rule that can be pushed past its
    ceiling is not one.

    A held-out game carrying no held-out text is not admitted at all: it holds
    nothing gate 1 measures, and it is withheld from training either way.

    Walked in a seeded order so two builds of one corpus admit the same games.
    ``cap <= 0`` means no cap, matching ``--text-cap``'s own zero.
    """
    held = set(held_out_games)
    texts_of: dict[str, set[str]] = defaultdict(set)
    for text, games in text_games.items():
        for game in games:
            if game in held:
                texts_of[game].add(text)

    order = sorted(held)
    random.Random(seed).shuffle(order)
    taken: Counter[str] = Counter()
    admitted: set[str] = set()
    for game in order:
        texts = texts_of.get(game)
        if not texts:
            continue
        if cap > 0 and any(taken[text] >= cap for text in texts):
            continue
        admitted.add(game)
        for text in texts:
            taken[text] += 1
    return frozenset(admitted)

"""What a curated corpus is planned to hold, and where each game goes (FR-049, FR-142).

The survey fills a :class:`TrainingSupply` per shard; this turns the folded
supply into copy plans, splitting the budget down class, legality half, rule
family, cell and ability text with one
:func:`effects.domain.budget_allocation.allocate` rule at every level.
"""

from __future__ import annotations

import dataclasses
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from functools import partial
from typing import TYPE_CHECKING, Any, Literal

from effects.domain.budget_allocation import allocate
from effects.domain.corpus_curation import (
    SIGNATURE_CLASSES,
    CapHeap,
    CopyPlan,
    class_targets,
    copy_plan,
    game_disjoint_draw,
    outcome_signature,
    rate_plan,
    signature_label,
    text_capacity,
)
from effects.domain.provenance import ProvenanceKey, SidecarLookup
from effects.domain.records import EffectRecord, LegalityDecision
from effects.domain.sampling import SamplingClass, sampling_class
from effects.domain.strata import Stratum

if TYPE_CHECKING:
    from effects.domain.damage_step_keywords import KeywordResolver

#: A record's acting keys: the whole tuple, hashable and picklable as it is.
Keys = tuple[ProvenanceKey, ...]


class BuildCorpusError(ValueError):
    """A build this corpus, these flags or this working directory cannot do.

    The CLI reports it as one line and a non-zero exit, while an internal
    invariant's bare ``ValueError`` keeps its traceback.
    """


# ── cells ───────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True, order=True)
class Cell:
    """One selection cell of FR-049: class, legality half, family, signature.

    ``half`` is set on ``playability-legality`` only, ``signature`` on the
    four classes FR-049 step 2 splits by outcome. ``modes`` is set on a
    survey's legality cell until :meth:`resolved` folds it into the family
    once the corpus-wide mode counts exist.
    """

    klass: SamplingClass
    half: LegalityDecision | Literal[""] = ""
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

    @classmethod
    def of(
        cls, record: EffectRecord, sidecars: SidecarLookup, resolver: KeywordResolver,
        legality_mode_counts: Mapping[str, int] | None = None,
    ) -> Cell:
        """The cell a record is selected in. Both passes call this one function.

        ``random_seat`` is not a component (FR-050). A legality record's family is
        its rarest mode, which only the whole corpus can say, so without
        ``legality_mode_counts`` (the survey) its cell carries the modes for
        :meth:`resolved` to fold later.
        """
        from effects.domain.rule_families import FamilySentinel, legality_modes, rule_family

        klass = sampling_class(record)
        half = _selection_half(record, klass)
        if klass == SamplingClass.PLAYABILITY_LEGALITY:
            modes = legality_modes(record, sidecars)
            family = "" if modes else FamilySentinel.NONE
            cell = cls(klass=klass, half=half, family=family, modes=modes)
            return cell if legality_mode_counts is None else cell.resolved(legality_mode_counts)
        return cls(
            klass=klass,
            half=half,
            family=rule_family(record, sidecars, resolver=resolver),
            signature=(
                signature_label(outcome_signature(record))
                if klass in SIGNATURE_CLASSES else ""
            ),
        )

    @property
    def label(self) -> str:
        """The family's name in the manifest: prefixed by its half on a legality cell."""
        return f"{self.half}:{self.family}" if self.half else self.family


def _selection_half(
    record: EffectRecord, klass: SamplingClass,
) -> LegalityDecision | Literal[""]:
    """``real`` / ``what-if`` on a legality record (FR-049 step 3), empty on
    every other; a gen-1 record, whose decision is unknown, is a what-if."""
    if klass != SamplingClass.PLAYABILITY_LEGALITY:
        return ""
    if LegalityDecision.of(record) is LegalityDecision.REAL:
        return LegalityDecision.REAL
    return LegalityDecision.WHAT_IF


# ── what the training-bound records can supply ──────────────────────────


@dataclass(slots=True)
class TrainingSupply:
    """Training-bound records per cell and acting keys, and their smallest hashes.

    Only records the write pass routes to training are counted, so every
    budget planned from it can be filled. ``heaps`` keeps each slot's
    ``heap_cap`` smallest hashes, enough for any quota; records with no acting
    key are counted in ``keyless`` and sampled by hash rate.
    """

    heap_cap: int
    keyed: Counter[tuple[Cell, Keys]] = field(default_factory=Counter)
    heaps: defaultdict[tuple[Cell, Keys], CapHeap] = field(init=False)
    keyless: Counter[Cell] = field(default_factory=Counter)

    def __post_init__(self) -> None:
        # A partial rather than a lambda, so the supply pickles.
        self.heaps = defaultdict(partial(CapHeap, self.heap_cap))

    def add(self, cell: Cell, keys: Keys, value: int) -> None:
        """One training-bound record, by its cell, acting keys and record hash."""
        if not keys:
            self.keyless[cell] += 1
            return
        self.keyed[cell, keys] += 1
        self.heaps[cell, keys].offer(value)

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
            out.heaps[slot].merge(self.heaps[cell, keys])
        return out


#: The member name of a cell's records whose keys fold to no text, or that
#: carry none. No ability text is empty, so it cannot collide with one.
KEYLESS = ""


@dataclass(slots=True)
class Member:
    """What one ability text supplies to one cell (FR-049); the keyless member
    has no heap."""

    distinct: int = 0
    heap: CapHeap | None = None

    def once(self, text_cap: int) -> int:
        """Records this member can supply once each: the text cap, or all of
        them, which the keyless member always can."""
        if self.heap is None or text_cap <= 0:
            return self.distinct
        return min(self.distinct, text_cap)

    def capacity(self, *, text_cap: int, reuse_cap: int) -> int:
        """Records this member can supply with repeats."""
        return text_capacity(
            self.distinct, text_cap=0 if self.heap is None else text_cap,
            reuse_cap=reuse_cap,
        )


def members_of(
    supply: TrainingSupply, key_text: Mapping[Keys, str],
) -> tuple[dict[Cell, dict[str, Member]], dict[tuple[Cell, str], list[Keys]]]:
    """``cell -> member name -> Member``, and each member's acting keys.

    Keys folding to one text share its member and its cap.
    """
    members: dict[Cell, dict[str, Member]] = defaultdict(dict)
    keys_of: dict[tuple[Cell, str], list[Keys]] = defaultdict(list)
    for (cell, keys), count in supply.keyed.items():
        text = key_text.get(keys)
        if text is None:
            members[cell].setdefault(KEYLESS, Member()).distinct += count
            continue
        member = members[cell].get(text)
        if member is None:
            member = members[cell][text] = Member(heap=CapHeap(supply.heap_cap))
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
    repeats, ``once`` without. ``share`` is the equal split before
    redistribution, so ``share - written`` is a shortfall.
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

    def walk(self, depth: int) -> Iterator[tuple[tuple[Any, ...], AllocationNode]]:
        """``(path, node)`` for every node ``depth`` levels down, siblings in key order.

        Sorted so the plan comes out in one order in every process.
        """
        if depth == 0:
            yield (), self
            return
        for key in sorted(self.children):
            for path, node in self.children[key].walk(depth - 1):
                yield (key, *path), node

    def allocate(self, budget: int, *, share: int | None = None) -> int:
        """Split ``budget`` down the tree; return what this node plans to write.

        Over a cell's members it runs in two rounds — first what each supplies
        once, then repeats — so no record repeats while another goes unused.
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

    ``share`` is the equal split before redistribution, so
    ``max(0, share - written)`` is the family's shortfall.
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
    #: Distinct records each class can supply under the text cap, no repeats.
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
    available = _available(members, text_cap)
    try:
        targets, shortfall = class_targets(available, mix, ceiling=ceiling)
    except ValueError as exc:
        raise BuildCorpusError(str(exc)) from exc

    tree = _allocation_tree(members, text_cap=text_cap, reuse_cap=reuse_cap)
    # The class level is split by the mixture, not equally.
    for klass, root in tree.children.items():
        root.allocate(targets.get(klass, 0))
    plans, keyless = _copy_plans(tree, members, keys_of)
    return SelectionPlan(
        plans=plans, keyless=keyless, families=_family_shares(tree, members),
        targets=targets, shortfall=shortfall, available=available,
    )


#: How far below the tree's root each level sits: class, half, family, cell.
_FAMILY_DEPTH = 3
_CELL_DEPTH = 4


def _available(members: Mapping[Cell, Mapping[str, Member]], text_cap: int) -> dict[str, int]:
    """Distinct records each class can supply under the text cap, with no repeats."""
    available: Counter[str] = Counter()
    for cell, cell_members in members.items():
        for member in cell_members.values():
            available[cell.klass] += member.once(text_cap)
    return dict(available)


def _allocation_tree(
    members: Mapping[Cell, Mapping[str, Member]], *, text_cap: int, reuse_cap: int,
) -> AllocationNode:
    """class -> half -> family -> cell -> member, a member's leaf holding its supply."""
    tree = AllocationNode()
    for cell, cell_members in members.items():
        cell_node = tree.child(cell.klass).child(cell.half).child(cell.family).child(cell)
        for name, member in cell_members.items():
            cell_node.children[name] = AllocationNode(
                supply=member.capacity(text_cap=text_cap, reuse_cap=reuse_cap),
                once=member.once(text_cap),
            )
    return tree


def _family_shares(
    tree: AllocationNode, members: Mapping[Cell, Mapping[str, Member]],
) -> dict[str, dict[str, FamilyShare]]:
    """``class -> family label -> FamilyShare`` of an allocated tree."""
    families: dict[str, dict[str, FamilyShare]] = {}
    for (klass, half, family), node in tree.walk(_FAMILY_DEPTH):
        families.setdefault(klass, {})[Cell(klass, half, family).label] = FamilyShare(
            available=sum(
                member.distinct for cell in node.children for member in members[cell].values()
            ),
            share=node.share,
            planned=node.planned,
        )
    return families


def _copy_plans(
    tree: AllocationNode,
    members: Mapping[Cell, Mapping[str, Member]],
    keys_of: Mapping[tuple[Cell, str], list[Keys]],
) -> tuple[dict[tuple[Cell, Keys], CopyPlan], dict[Cell, CopyPlan]]:
    """Each member's planned count as copies: per acting key, or by rate when keyless.

    Members keep survey order; the tree sorts every level above them.
    """
    plans: dict[tuple[Cell, Keys], CopyPlan] = {}
    keyless: dict[Cell, CopyPlan] = {}
    for (*_, cell), cell_node in tree.walk(_CELL_DEPTH):
        for name, leaf in cell_node.children.items():
            member = members[cell][name]
            if name == KEYLESS:
                keyless[cell] = rate_plan(member.distinct, leaf.planned)
                continue
            plan = copy_plan(member.heap.values(), member.distinct, leaf.planned)
            plans.update(((cell, keys), plan) for keys in keys_of[cell, name])
    return plans, keyless


# ── where each game goes ────────────────────────────────────────────────


class GameRoute(StrEnum):
    """Where the survey places a game, before the card-disjoint cap is applied."""

    TRAINING = "training"
    GAME_DISJOINT = "game-disjoint"
    #: Card-disjoint or dropped, decided by the cap later.
    HELD_OUT = "held-out"


@dataclass(frozen=True, slots=True)
class GameRouting:
    """The per-game placement rule, decided once per game by the survey (FR-044-046).

    A pure function of the game, so a rebuild over a grown corpus places every
    game it placed before.
    """

    game_disjoint_share: float
    keyword_share: float

    def route(self, game_id: str, *, held_out: bool, keyword_combat: bool) -> GameRoute:
        if held_out:
            return GameRoute.HELD_OUT
        if self.game_disjoint(game_id, keyword_combat=keyword_combat):
            return GameRoute.GAME_DISJOINT
        return GameRoute.TRAINING

    def game_disjoint(self, game_id: str, *, keyword_combat: bool) -> bool:
        """Whether a game not held out enters the game-disjoint stratum (SC-007).

        A game holding a qualifying keyword combat takes the higher threshold,
        since those are what gate 2 scores.
        """
        threshold = self.keyword_share if keyword_combat else self.game_disjoint_share
        return game_disjoint_draw(game_id) < threshold

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

    Whole games (FR-142, FR-136). ``text_games`` maps each held-out text to the
    games holding a resolution record acting through it, the only kind gate 1
    scores; a game with none is not admitted. Requiring every text to be under
    cap, not just one, keeps the cap a true ceiling at the cost of leaving
    some texts short. Seeded order, so two builds admit the same games;
    ``cap <= 0`` means no cap.
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

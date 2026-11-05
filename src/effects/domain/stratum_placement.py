"""Which games go to which validation stratum and half (FR-059, FR-068a, FR-070;
research R3, E5).

Placement runs once, over the whole corpus, after the survey: gen-2 decided
each game alone, at the end of its shard, which cannot put both polarities of
every rule in both halves of both strata. It reads the survey's per-game rule
inventories, ``game -> {(rule, polarity): records}``, counted over budgeted
rules and rule-test-eligible records (:func:`rule_test_eligible`) of games
two Forge AIs played.

- Only Forge-AI games enter a validation stratum. A random-seat game naming a
  held-out card is dropped like a held-out game the cap declines; any other
  random-seat game trains.
- Card-disjoint: the text holdout fixes the candidate games. Games carrying a
  (rule, polarity) still short of its test records are admitted first, the
  scarcest cell first, then every other candidate in seeded order, all under
  the per-text cap and only while the game resolves a held-out text.
- Game-disjoint: gen-2's crc32 draw places the base set. For each
  (rule, polarity) short of its test records, unplaced games of the candidate
  pool carrying it are promoted, smallest game hash first, and then games of
  the pool until the stratum holds enough games for both halves.
- Halves: within each stratum, games in seeded order each join the half that
  keeps every (rule, polarity) count most even; a tie goes to the half with
  fewer games. A half left under ``min_games_per_half`` takes games from the
  other while the other has more.

A rule with test records of both polarities in the corpus whose half holds
none of one polarity is named as left out of that half.

:func:`rule_test_misses` is what ``build-corpus`` can already tell of the
rule-test component's precision target (FR-068b) from the same inventories,
before any rule test is frozen.

Placement is deterministic for a given corpus but not stable as it grows: a
promotion or a half depends on every other game (R3).

Torch-free: ``build-corpus`` places in its main process.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from effects.domain.corpus_curation import game_disjoint_draw, game_hash
from effects.domain.corpus_plan import GameStrata
from effects.domain.records import EffectRecord
from effects.domain.rule_catalogue import Polarity
from effects.domain.rule_detectors import RuleTag
from effects.domain.score_scales import auc_standard_error, component_standard_error
from effects.domain.seeded_draw import stable_order
from effects.domain.strata import Half, Stratum

#: ``(rule, polarity)``: one cell of a game's inventory.
RuleCell = tuple[str, Polarity]

#: The polarities a rule test reads (FR-068a); ``other`` draws none.
TEST_POLARITIES: tuple[Polarity, ...] = (Polarity.FIRES, Polarity.DOES_NOT_FIRE)

#: Each half of a stratum holds at least gen-2's whole probe set (E5).
MIN_GAMES_PER_HALF = 300
#: The rescaled standard error a rule test may have in a half (FR-068b).
MAX_ITEM_ERROR = 0.1
#: The rescaled standard error a component may have in a half (FR-068b).
MAX_COMPONENT_ERROR = 0.05
#: The candidate pool's game-disjoint draw, as a multiple of the share (R3).
POOL_FACTOR = 10.0


def rule_test_records_needed(max_item_error: float = MAX_ITEM_ERROR) -> int:
    """Records of each polarity a rule test needs in a half: the smallest n
    whose worst-case rescaled standard error with n records each side,
    ``2·√((2n + 1) / (12n²))``, is at most ``max_item_error`` (FR-068b): 68
    at 0.1."""
    n = 1
    while auc_standard_error(n, n) > max_item_error:
        n += 1
    return n


def rule_test_eligible(record: EffectRecord, tag: RuleTag) -> bool:
    """Whether one tag of one record may be a rule-test record (FR-072): not
    from a fork of any kind, an interventional resolution or collection aimed
    at a rule, not of polarity ``other``, and not on the review list."""
    return not (
        record.fork or record.interventional or record.synthetic
        or record.aimed_at is not None or record.directed is not None
        or tag.polarity not in TEST_POLARITIES or tag.needs_review
    )


@dataclass(frozen=True, slots=True)
class PlacementRule:
    """The flags placement follows, and its precision thresholds (E5)."""

    game_disjoint_share: float
    keyword_share: float
    #: ``--card-disjoint-text-cap``; ``<= 0`` is no cap.
    card_disjoint_text_cap: int
    seed: int
    min_games_per_half: int = MIN_GAMES_PER_HALF
    max_item_error: float = MAX_ITEM_ERROR
    pool_factor: float = POOL_FACTOR
    max_component_error: float = MAX_COMPONENT_ERROR

    def base_game_disjoint(self, game_id: str, *, keyword_combat: bool) -> bool:
        """Gen-2's draw: a game holding a qualifying keyword combat takes the
        higher threshold, since those are what gate 2 scores."""
        threshold = self.keyword_share if keyword_combat else self.game_disjoint_share
        return game_disjoint_draw(game_id) < threshold

    def under_keyword_threshold(self, game_id: str) -> bool:
        """A base game-disjoint game placed only by the keyword threshold."""
        return game_disjoint_draw(game_id) >= self.game_disjoint_share

    def in_pool(self, game_id: str, *, keyword_combat: bool) -> bool:
        """Whether placement may put the game in the game-disjoint stratum, so
        the survey keeps its training supply per game until placement (R3)."""
        return (
            self.base_game_disjoint(game_id, keyword_combat=keyword_combat)
            or game_disjoint_draw(game_id) < self.pool_factor * self.game_disjoint_share
        )


@dataclass(frozen=True, slots=True)
class GameCensus:
    """What the survey knows about every game, as placement reads it."""

    #: Every game holding a record the quality rules keep.
    games: frozenset[str]
    #: Games naming a held-out card.
    held_out: frozenset[str]
    #: Games any of whose records carries ``random_seat``.
    random_seat: frozenset[str]
    #: Games holding a qualifying combat for a ``--game-disjoint-keywords`` keyword.
    keyword_combat: frozenset[str]
    #: Held-out text -> games holding a resolution record acting through it.
    held_out_text_games: Mapping[str, Iterable[str]]
    #: ``game -> (rule, polarity) -> records``, Forge-AI games only, over
    #: budgeted rules and :func:`rule_test_eligible` records.
    inventories: Mapping[str, Mapping[RuleCell, int]] = field(default_factory=dict)


def place_games(census: GameCensus, rule: PlacementRule) -> GameStrata:
    """Every game's stratum and every validation game's half."""
    forge_ai = census.games - census.random_seat
    inventories = {
        game: {cell: n for cell, n in inventory.items()
               if cell[1] in TEST_POLARITIES and n > 0}
        for game, inventory in census.inventories.items() if game in forge_ai
    }
    need = 2 * rule_test_records_needed(rule.max_item_error)
    card = _admit_card_disjoint(
        census.held_out & forge_ai, census.held_out_text_games, inventories,
        cap=rule.card_disjoint_text_cap, need=need, seed=rule.seed,
    )
    game = _game_disjoint(
        forge_ai - census.held_out, census.keyword_combat, inventories, rule, need=need,
    )
    halves = {
        Stratum.CARD_DISJOINT: _assign_halves(card, inventories, rule),
        Stratum.GAME_DISJOINT: _assign_halves(game, inventories, rule),
    }
    return GameStrata(
        card_disjoint=card,
        game_disjoint=game,
        held_out=frozenset(census.held_out),
        halves=halves,
        left_out_of_half=_left_out(halves, inventories),
    )


def _carriers(inventories: Mapping[str, Mapping[RuleCell, int]],
              games: Iterable[str]) -> dict[RuleCell, list[str]]:
    """``cell -> games carrying it`` over ``games``."""
    out: dict[RuleCell, list[str]] = {}
    for game in games:
        for cell in inventories.get(game, ()):
            out.setdefault(cell, []).append(game)
    return out


def _scarcest_first(carriers: Mapping[RuleCell, list[str]],
                    inventories: Mapping[str, Mapping[RuleCell, int]]) -> list[RuleCell]:
    """Cells in the order placement fills them: fewest candidate records first,
    so a scarce cell is not crowded out by games admitted for a common one."""
    return sorted(carriers, key=lambda cell: (
        sum(inventories[game][cell] for game in carriers[cell]), cell))


def _admit_card_disjoint(
    candidates: frozenset[str],
    text_games: Mapping[str, Iterable[str]],
    inventories: Mapping[str, Mapping[RuleCell, int]],
    *, cap: int, need: int, seed: int,
) -> frozenset[str]:
    """Held-out games admitted while every held-out text they resolve is under
    cap, games carrying a short cell first (R3).

    A game resolving no held-out text is never admitted: gate 1 scores
    resolutions through a held-out text, and nothing else in the game is
    held out. Requiring every text under cap keeps the cap a true ceiling.
    """
    texts_of: dict[str, set[str]] = {}
    for text, games in text_games.items():
        for game in games:
            if game in candidates:
                texts_of.setdefault(game, set()).add(text)
    order = stable_order(candidates, seed)
    rank = {game: i for i, game in enumerate(order)}
    taken: Counter[str] = Counter()
    filled: Counter[RuleCell] = Counter()
    admitted: set[str] = set()

    def admit(game: str) -> None:
        texts = texts_of.get(game)
        if game in admitted or not texts:
            return
        if cap > 0 and any(taken[text] >= cap for text in texts):
            return
        admitted.add(game)
        taken.update(texts)
        filled.update(inventories.get(game, {}))

    carriers = _carriers(inventories, candidates)
    for cell in _scarcest_first(carriers, inventories):
        for game in sorted(carriers[cell], key=rank.__getitem__):
            if filled[cell] >= need:
                break
            admit(game)
    for game in order:
        admit(game)
    return frozenset(admitted)


def _by_game_hash(game: str) -> tuple[int, str]:
    return (game_hash(game), game)


def _game_disjoint(
    candidates: frozenset[str],
    keyword_combat: frozenset[str],
    inventories: Mapping[str, Mapping[RuleCell, int]],
    rule: PlacementRule,
    *, need: int,
) -> frozenset[str]:
    """Gen-2's draw, then promotions from the candidate pool: per short cell,
    then until the stratum holds two halves' worth of games."""
    placed = {
        game for game in candidates
        if rule.base_game_disjoint(game, keyword_combat=game in keyword_combat)
    }
    pool = sorted(
        (game for game in candidates - placed
         if rule.in_pool(game, keyword_combat=game in keyword_combat)),
        key=_by_game_hash,
    )
    filled: Counter[RuleCell] = Counter()
    for game in placed:
        filled.update(inventories.get(game, {}))
    carriers = _carriers(inventories, pool)
    for cell in _scarcest_first(carriers, inventories):
        for game in carriers[cell]:
            if filled[cell] >= need:
                break
            if game not in placed:
                placed.add(game)
                filled.update(inventories.get(game, {}))
    for game in pool:
        if len(placed) >= 2 * rule.min_games_per_half:
            break
        placed.add(game)
    return frozenset(placed)


def _assign_halves(
    games: frozenset[str],
    inventories: Mapping[str, Mapping[RuleCell, int]],
    rule: PlacementRule,
) -> dict[Half, frozenset[str]]:
    """Each game to the half that keeps every cell's two counts closest."""
    counts = {half: Counter() for half in Half}
    members: dict[Half, list[str]] = {half: [] for half in Half}
    order = stable_order(games, rule.seed)
    for game in order:
        inventory = inventories.get(game, {})
        half = min(Half, key=lambda h: (
            _imbalance_added(inventory, counts[h], counts[_other(h)]), len(members[h])))
        members[half].append(game)
        counts[half].update(inventory)
    _top_up(members, counts, inventories, rule.min_games_per_half)
    return {half: frozenset(members[half]) for half in Half}


def _other(half: Half) -> Half:
    return Half.REPORTING if half is Half.SELECTION else Half.SELECTION


def _imbalance_added(inventory: Mapping[RuleCell, int], mine: Counter, theirs: Counter) -> int:
    """How much ``Σ |mine − theirs|`` over the game's cells grows if the game joins ``mine``."""
    return sum(
        abs(mine[cell] + n - theirs[cell]) - abs(mine[cell] - theirs[cell])
        for cell, n in inventory.items()
    )


def _top_up(
    members: dict[Half, list[str]],
    counts: dict[Half, Counter],
    inventories: Mapping[str, Mapping[RuleCell, int]],
    minimum: int,
) -> None:
    """Move games into a half under ``minimum`` while the other holds more,
    each time the game whose move unbalances the cells least, the latest in
    seeded order on a tie."""
    for small in Half:
        large = _other(small)
        while len(members[small]) < minimum and len(members[large]) > len(members[small]) + 1:
            game = min(
                reversed(members[large]),
                key=lambda g: _move_cost(inventories.get(g, {}), counts[small], counts[large]),
            )
            members[large].remove(game)
            counts[large].subtract(inventories.get(game, {}))
            members[small].append(game)
            counts[small].update(inventories.get(game, {}))


def _move_cost(inventory: Mapping[RuleCell, int], to: Counter, source: Counter) -> int:
    """How much ``Σ |to − source|`` over the game's cells grows if the game
    moves from ``source`` to ``to``."""
    return sum(
        abs(to[cell] + n - (source[cell] - n)) - abs(to[cell] - source[cell])
        for cell, n in inventory.items()
    )


def _left_out(
    halves: Mapping[Stratum, Mapping[Half, frozenset[str]]],
    inventories: Mapping[str, Mapping[RuleCell, int]],
) -> dict[Stratum, dict[Half, tuple[str, ...]]]:
    """Rules with test records of both polarities in the corpus that hold none
    of one polarity in a half (FR-068a)."""
    seen: dict[str, set[Polarity]] = {}
    for inventory in inventories.values():
        for rule_id, polarity in inventory:
            seen.setdefault(rule_id, set()).add(polarity)
    testable = sorted(rule_id for rule_id, sides in seen.items()
                      if len(sides) == len(TEST_POLARITIES))
    out: dict[Stratum, dict[Half, tuple[str, ...]]] = {}
    for stratum, by_half in halves.items():
        for half, games in by_half.items():
            present = {cell for game in games for cell in inventories.get(game, ())}
            missing = tuple(
                rule_id for rule_id in testable
                if any((rule_id, polarity) not in present for polarity in TEST_POLARITIES)
            )
            if missing:
                out.setdefault(stratum, {})[half] = missing
    return out


@dataclass(frozen=True, slots=True)
class PrecisionMiss:
    """One half whose rule-test component already misses its target (FR-068b)."""

    stratum: Stratum
    half: Half
    #: Rules whose test meets the item target in the half: the component's items.
    items: int
    #: The component's standard error, ``√(Σ SEᵢ²) / m``; infinite with no item.
    standard_error: float

    def describe(self, target: float) -> str:
        if not self.items:
            return (f"{self.stratum} {self.half}: rule tests — no rule has enough "
                    "test records of both polarities to enter the component")
        return (f"{self.stratum} {self.half}: rule tests — standard error "
                f"{self.standard_error:.4f} over {self.items} rule(s), above {target}")


def rule_test_misses(
    strata: GameStrata,
    inventories: Mapping[str, Mapping[RuleCell, int]],
    rule: PlacementRule,
) -> list[PrecisionMiss]:
    """Every half whose rule-test component misses its precision target.

    A rule enters a half's component when its test's standard error
    (``score_scales.auc_standard_error``) is at most ``max_item_error``; the
    component's (``score_scales.component_standard_error``),
    ``√(Σ SEᵢ²) / m`` over its m rules, must be at most
    ``max_component_error``, and a half where no rule enters misses it. The
    counts are every rule-test-eligible record, before the freeze subsamples
    a polarity: the freeze can only raise a standard error, so a half missing
    here misses there too.
    """
    misses: list[PrecisionMiss] = []
    for stratum, by_half in sorted(strata.halves.items()):
        for half in Half:
            counts: Counter[RuleCell] = Counter()
            for game in by_half.get(half, ()):
                counts.update(inventories.get(game, {}))
            rules = sorted({rule_id for rule_id, _ in counts})
            errors = [
                error for error in (
                    auc_standard_error(counts[rule_id, Polarity.FIRES],
                                       counts[rule_id, Polarity.DOES_NOT_FIRE])
                    for rule_id in rules)
                if error <= rule.max_item_error
            ]
            component = component_standard_error(errors)
            if component > rule.max_component_error:
                misses.append(PrecisionMiss(stratum, half, len(errors), component))
    return misses

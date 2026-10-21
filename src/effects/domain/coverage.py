"""The coverage rules: what a coverage unit is, how it is weighted, when it is done.

``collect-coverage`` puts every card into a game, not just the popular ones.
These are the rules it plays by, kept apart from the orchestration that reads
shards and drives Forge so they can be read, and tested, without either.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from effects.domain.records import EffectRecord, events_of
from effects.domain.text_holdout import normalize_script_text

#: Card types that never go in a deck: Planechase planes and phenomena,
#: Archenemy schemes, Vanguard avatars, Conspiracy's draft cards, and the
#: contraptions and attractions Un-sets play from decks of their own. The
#: converted tree holds some 450 of them, and Forge's card database does not
#: load them as deck cards, so a coverage deck holding one fails to materialize
#: and every other card in it sits the match out. Having no records, they weigh
#: the most and land in most decks; leaving them in the unit made most of a
#: round's decks unplayable and charged a stalled round to every card those
#: decks held, until the cards retired.
NON_DECK_TYPES: frozenset[str] = frozenset({
    "scheme", "vanguard", "plane", "phenomenon", "conspiracy", "contraption",
    "attraction", "dungeon",
})


def is_deckable(converted_text: str) -> bool:
    """Whether a converted card's first ``types:`` line names no non-deck type.

    A card with no ``types:`` line is kept: the filter exists to drop known
    non-deck types, not to second-guess a converter that wrote no line.
    """
    for line in converted_text.splitlines():
        if line.startswith("types:"):
            return not NON_DECK_TYPES & set(line[len("types:"):].lower().split())
    return True


class ConsultVerdict(StrEnum):
    """What the castability consult said about a card."""

    CASTABLE = "castable"
    UNCASTABLE = "uncastable"
    UNKNOWN = "unknown"

    @classmethod
    def parse(cls, raw: str) -> ConsultVerdict:
        """The worker's verdict string, ``unknown`` when it is neither.

        ``unknown`` rather than an error, because an unrecognised verdict was
        always read as "not uncastable" and so ranked as castable -- the safe
        direction, since the consult only ranks.
        """
        try:
            return cls(raw)
        except ValueError:
            return cls.UNKNOWN


#: An uncastable card's share of a castable card's deck weight. Above zero,
#: because being in a game is the precondition an intervention forks from;
#: below one, because a slot holding a card the AI cannot cast mostly produces
#: no record toward any target, so castable cards should fill most slots.
UNCASTABLE_RANK_MULTIPLIER = 0.25


class CoverageUnit(Protocol):
    """What the round loop and ``retire_stalled`` read of a coverage unit.

    A card in the ordinary run, a held-out ability text in the held-out round:
    both progress toward a target, stall, and retire by the same rule.
    """

    retired: bool
    rounds_without_progress: int

    @property
    def records(self) -> int: ...

    def satisfied(self, target: int) -> bool: ...

    def done(self, target: int) -> bool: ...


@dataclass
class CardCoverage:
    """One card's progress toward its record target."""

    name: str
    records: int = 0
    rounds_without_progress: int = 0
    retired: bool = False
    verdict: ConsultVerdict = ConsultVerdict.UNKNOWN

    def satisfied(self, target: int) -> bool:
        return self.records >= target

    def done(self, target: int) -> bool:
        return self.retired or self.satisfied(target)


@dataclass
class TextCoverage:
    """One held-out ability text's progress toward its game floor (FR-036).

    Games rather than records: one long game can write hundreds of records of
    one ability, and a text seen in one game has been seen on one board. The
    fields ``retire_stalled`` reads are named as ``CardCoverage``'s are, so the
    same retirement rule serves both units.
    """

    text: str
    games: set[str] = field(default_factory=set)
    retired: bool = False
    rounds_without_progress: int = 0
    #: Whether any card carrying the text is castable by the consult (FR-037).
    castable: bool = True

    @property
    def records(self) -> int:
        """Distinct games, under the name ``retire_stalled`` compares."""
        return len(self.games)

    def satisfied(self, floor: int) -> bool:
        return len(self.games) >= floor

    def done(self, floor: int) -> bool:
        return self.retired or self.satisfied(floor)


def qualifying_cards(record: EffectRecord) -> set[str]:
    """Cards this record counts toward (FR-049).

    The acting line's host, every event subject, and every referenced ref —
    but **not** an entity that merely sat in the snapshot. A card on the
    battlefield while something else resolves teaches the model nothing about
    that card, and counting it would let a popular card satisfy every card
    beside it.
    """
    by_id: dict[str, str] = {
        entity.id: entity.name for entity in record.state.entities
    }
    names: set[str] = set()

    source = record.state.refs.source
    if source in by_id:
        names.add(by_id[source])
    for target in record.state.refs.targets:
        if target in by_id:
            names.add(by_id[target])
    for event in events_of(record):
        for subject in event.subjects:
            if subject in by_id:
                names.add(by_id[subject])
    return names


def deck_weights(
    coverage: Mapping[str, CardCoverage], target: int,
) -> dict[str, float]:
    """Weight each card by how far it is from its target (FR-046).

    A card with no records weighs most; one already satisfied weighs nothing.
    Linear in the shortfall rather than inverse in the count, so a card at 49
    of 50 still gets drawn occasionally and the last few records do not take
    unboundedly many rounds.
    """
    weights: dict[str, float] = {}
    for name, card in coverage.items():
        if card.done(target):
            continue
        weights[name] = float(target - card.records)
    return weights


def text_deck_weights(
    coverage: Mapping[str, TextCoverage],
    carriers: Mapping[str, frozenset[str]],
    floor: int,
) -> dict[str, float]:
    """A card's weight: the games its unfinished held-out texts still need."""
    weights: dict[str, float] = defaultdict(float)
    for text, unit in coverage.items():
        if unit.done(floor):
            continue
        for card in carriers[text]:
            weights[card] += float(floor - len(unit.games))
    return dict(weights)


def rank_by_consult(
    weights: Mapping[str, float], verdicts: Mapping[str, ConsultVerdict],
) -> dict[str, float]:
    """Apply the consult as a **ranking** input, never as a filter (FR-047).

    A card the consult judges uncastable keeps a weight, because being in a
    game is the precondition an intervention forks from — dropping it here
    would mean stage three had nothing to fork.
    """
    ranked: dict[str, float] = {}
    for name, weight in weights.items():
        verdict = verdicts.get(name, ConsultVerdict.UNKNOWN)
        multiplier = (
            UNCASTABLE_RANK_MULTIPLIER if verdict == ConsultVerdict.UNCASTABLE
            else 1.0
        )
        ranked[name] = weight * multiplier
    return ranked


def retire_stalled(
    coverage: Mapping[str, CoverageUnit],
    previous: Mapping[str, int],
    *,
    no_progress_rounds: int,
    decked: Collection[str],
) -> list[str]:
    """Retire units that gained no qualifying record this round (FR-050).

    Without this the run never ends: some cards Forge's AI cannot cast at all,
    and a round that makes no progress on them will make none on the next.

    ``decked`` is the units this round's decks actually held, and only those
    can stall: a card no deck contained was never attempted, so counting the
    round against it retires a card the run has not yet tried once. That is
    not a rare edge — at the defaults a round draws 500 x 23 slots against
    33,680 candidates, so a given card sits out a whole round with probability
    about 0.71 and roughly a third of the corpus would retire after three
    rounds having never been in a game. The run still terminates, because a
    live card keeps its weight and is drawn eventually; it just stops
    terminating with almost nothing collected and a residue list the size of
    the corpus.

    Required rather than defaulted to "everything": a default would restore
    the exact silent behaviour this parameter exists to remove.
    """
    decked_names = frozenset(decked)
    retired: list[str] = []
    for name, unit in coverage.items():
        if unit.retired:
            continue
        if unit.records > previous.get(name, 0):
            unit.rounds_without_progress = 0
            continue
        if name not in decked_names:
            continue
        unit.rounds_without_progress += 1
        if unit.rounds_without_progress >= no_progress_rounds:
            unit.retired = True
            retired.append(name)
    return retired


def is_complete(coverage: Mapping[str, CoverageUnit], target: int) -> bool:
    """The run ends when every unit is satisfied or retired (FR-050)."""
    return all(unit.done(target) for unit in coverage.values())


def is_text_round_complete(coverage: Mapping[str, TextCoverage], floor: int) -> bool:
    """Every held-out text with a castable carrier satisfied or retired (FR-037)."""
    return is_complete(awaited_texts(coverage), floor)


def awaited_texts(coverage: Mapping[str, TextCoverage]) -> dict[str, TextCoverage]:
    """The held-out texts whose completion ends the round (FR-037).

    A text no castable card carries is reported rather than awaited: no round
    can make the AI cast it, so waiting on it is waiting forever.
    """
    return {text: unit for text, unit in coverage.items() if unit.castable}


def held_out_texts_of(
    listed: frozenset[str], texts_by_card: Mapping[str, list[str]],
) -> dict[str, frozenset[str]]:
    """``text -> its listed carriers``, for the texts only listed cards carry.

    The depletion list names every card carrying a held-out text, so every
    held-out text is carried by listed cards alone; a text an unlisted card
    also carries is not held out and is not this round's business. The rule
    reads the list rather than recomputing the holdout, so the round needs none
    of the holdout's flags — at the cost of also covering a text that only
    listed cards carry without being held out itself, which is harmless.
    """
    carriers: dict[str, set[str]] = defaultdict(set)
    elsewhere: set[str] = set()
    for card, texts in texts_by_card.items():
        for text in texts:
            normalized = normalize_script_text(text)
            if card in listed:
                carriers[normalized].add(card)
            else:
                elsewhere.add(normalized)
    return {
        text: frozenset(cards) for text, cards in carriers.items() if text not in elsewhere
    }

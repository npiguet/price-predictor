"""The coverage rules: what a coverage unit is, how it is weighted, when it is done.

The rules ``collect-coverage`` plays by, apart from the orchestration that
reads shards and drives Forge.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from itertools import chain

from effects.domain.records import EffectRecord, events_of
from effects.domain.text_holdout import normalize_script_text

#: Card types Forge's card database does not load as deck cards, so a deck
#: holding one fails to materialize. Having no records, they would weigh the
#: most and land in most decks.
NON_DECK_TYPES: frozenset[str] = frozenset({
    "scheme", "vanguard", "plane", "phenomenon", "conspiracy", "contraption",
    "attraction", "dungeon",
})


def is_deckable(converted_text: str) -> bool:
    """Whether a converted card's first ``types:`` line names no non-deck type.

    A card with no ``types:`` line is kept.
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
        """The worker's verdict string; ``unknown``, ranked as castable, when
        it is neither — the safe direction, since the consult only ranks."""
        try:
            return cls(raw)
        except ValueError:
            return cls.UNKNOWN


#: Distinct games a held-out text must act in: the floor the held-out coverage
#: round collects to (FR-036), and below which the manifest lists the text
#: (FR-053). Both count the kept resolution records' games.
HELD_OUT_GAME_FLOOR = 5

#: An uncastable card's share of a castable card's deck weight: above zero so
#: interventions have it in play, below one since the AI mostly cannot cast it.
UNCASTABLE_RANK_MULTIPLIER = 0.25


@dataclass(frozen=True, slots=True)
class DeckCandidate:
    """A card a coverage deck can hold, as the manabase heuristic reads it.

    Carries the one value ``compute_basic_lands`` reads, ``mana_cost_line()``,
    because FR-002's import surface excludes the ``ConvertedCardText`` its
    signature is typed against — and a variant has no converted text (FR-056).
    """

    name: str
    #: Brace-delimited like the converted corpus (``{2}{R}{R}``), the shape
    #: ``convert_mana_cost`` parses; None when the card names no cost.
    mana_cost: str | None

    @classmethod
    def from_converted_text(cls, name: str, converted_text: str) -> DeckCandidate:
        for line in converted_text.splitlines():
            if line.strip().lower().startswith("mana cost:"):
                return cls(name, line.split(":", 1)[1].strip() or None)
        return cls(name, None)

    def mana_cost_line(self) -> str | None:
        return self.mana_cost


class CoverageUnit:
    """What the round loop and ``retire_stalled`` read of a coverage unit.

    A card in the ordinary run, a held-out ability text in the held-out round:
    both progress toward a target, stall, and retire by the same rule, so a
    subclass says only what its :attr:`progress` counts.
    """

    retired: bool
    rounds_without_progress: int

    @property
    def progress(self) -> int:
        """What a round must raise for the unit to count as progressing."""
        raise NotImplementedError

    def satisfied(self, target: int) -> bool:
        return self.progress >= target

    def done(self, target: int) -> bool:
        return self.retired or self.satisfied(target)

    def shortfall(self, target: int) -> int:
        """How far :attr:`progress` is below ``target``."""
        return target - self.progress


@dataclass
class CardCoverage(CoverageUnit):
    """One card's progress toward its record target."""

    name: str
    records: int = 0
    rounds_without_progress: int = 0
    retired: bool = False
    verdict: ConsultVerdict = ConsultVerdict.UNKNOWN

    @property
    def progress(self) -> int:
        return self.records


@dataclass
class TextCoverage(CoverageUnit):
    """One held-out ability text's progress toward its game floor (FR-036).

    Games rather than records: one long game can write hundreds of records of
    one ability, all on one board.
    """

    text: str
    games: set[str] = field(default_factory=set)
    retired: bool = False
    rounds_without_progress: int = 0
    #: Whether any card carrying the text is castable by the consult (FR-037).
    castable: bool = True

    @property
    def progress(self) -> int:
        """Distinct games."""
        return len(self.games)


def qualifying_cards(record: EffectRecord) -> set[str]:
    """Cards this record counts toward (FR-049).

    The acting line's host, every event subject and every referenced ref, but
    not an entity that merely sat in the snapshot.
    """
    by_id: dict[str, str] = {
        entity.id: entity.name for entity in record.state.entities
    }
    refs = record.state.refs
    return {
        by_id[entity] for entity in chain(
            (refs.source,), refs.targets,
            (subject for event in events_of(record) for subject in event.subjects),
        )
        if entity in by_id
    }


def deck_weights(
    coverage: Mapping[str, CardCoverage], target: int,
) -> dict[str, float]:
    """Weight each card by how far it is from its target (FR-046).

    Linear in the shortfall rather than inverse in the count, so the last few
    records do not take unboundedly many rounds.
    """
    weights: dict[str, float] = {}
    for name, card in coverage.items():
        if card.done(target):
            continue
        weights[name] = float(card.shortfall(target))
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
            weights[card] += float(unit.shortfall(floor))
    return dict(weights)


def rank_by_consult(
    weights: Mapping[str, float], verdicts: Mapping[str, ConsultVerdict],
) -> dict[str, float]:
    """Apply the consult as a **ranking** input, never as a filter (FR-047).

    An uncastable card keeps a weight: being in a game is the precondition an
    intervention forks from.
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

    Only units in ``decked``, the ones this round's decks held, can stall: at
    the defaults most cards sit out any given round, and counting it against
    them would retire cards never tried. Required rather than defaulted to
    everything for that reason.
    """
    decked_names = frozenset(decked)
    retired: list[str] = []
    for name, unit in coverage.items():
        if unit.retired:
            continue
        if unit.progress > previous.get(name, 0):
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


def awaited_texts(coverage: Mapping[str, TextCoverage]) -> dict[str, TextCoverage]:
    """The held-out texts whose completion ends the round (FR-037).

    A text no castable card carries is reported rather than awaited, since no
    round can make the AI cast it.
    """
    return {text: unit for text, unit in coverage.items() if unit.castable}


def held_out_texts_of(
    listed: frozenset[str], texts_by_card: Mapping[str, list[str]],
) -> dict[str, frozenset[str]]:
    """``text -> its listed carriers``, for the texts only listed cards carry.

    Reads the depletion list rather than recomputing the holdout, so the round
    needs none of the holdout's flags; a text an unlisted card also carries is
    not held out. It may also cover a text only listed cards carry without it
    being held out, which is harmless.
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

"""``collect-coverage --only-cards``: the held-out coverage round (FR-035 to
FR-038).

Decks are drawn from the listed cards alone (the ``holdout-cards`` list), and
each unit is a held-out ability text rather than a card: a text is satisfied
once it acted in ``--min-text-games`` distinct games, and awaited only while
some castable listed card carries it.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, ClassVar

from effects.application.collector_common import read_name_list
from effects.application.coverage_counting import TextCoverageCounter
from effects.application.coverage_rounds import (
    CoverageRounds,
    consult_castability,
    count_existing,
    no_converted_cards,
    non_deck_names,
    play_to_completion,
    unresolvable,
)
from effects.domain.coverage import (
    ConsultVerdict,
    DeckCandidate,
    TextCoverage,
    awaited_texts,
    held_out_texts_of,
    rank_by_consult,
    text_deck_weights,
)

if TYPE_CHECKING:
    from effects.application.collect_coverage import CollectCoverageConfig
    from effects.domain.split import HeldOutCards
    from effects.infrastructure.converted_trees import ConvertedTrees

logger = logging.getLogger(__name__)


@dataclass
class TextResidues:
    """What the held-out round could not reach (FR-037, SC-004)."""

    under_floor: dict[str, int] = field(default_factory=dict)
    uncastable: tuple[str, ...] = ()

    def render(self, floor: int) -> str:
        lines = [
            f"{len(self.under_floor)} held-out text(s) with a castable carrier "
            f"under {floor} games; {len(self.uncastable)} with no castable "
            "carrier, never counted as satisfied."
        ]
        lines += [f"  under floor ({games} games): {text}"
                  for text, games in sorted(self.under_floor.items())]
        lines += [f"  uncastable: {text}" for text in self.uncastable]
        return "\n".join(lines)


def text_residues(coverage: dict[str, TextCoverage], floor: int) -> TextResidues:
    under = {
        text: len(unit.games) for text, unit in coverage.items()
        if unit.castable and not unit.satisfied(floor)
    }
    uncastable = tuple(sorted(
        text for text, unit in coverage.items() if not unit.castable
    ))
    return TextResidues(under, uncastable)


@dataclass(kw_only=True)
class TextRounds(CoverageRounds):
    """The held-out round's rounds: each held-out text a unit, awaited only
    when some castable card carries it."""

    coverage: dict[str, TextCoverage]
    #: ``held-out text -> the deckable listed cards carrying it``.
    carriers: Mapping[str, frozenset[str]]
    counter: TextCoverageCounter
    no_decks: ClassVar[str] = "No unfinished held-out text has a card with converted text."

    @cached_property
    def awaited(self) -> dict[str, TextCoverage]:
        # Fixed for the run: castability comes from the consult, once.
        return awaited_texts(self.coverage)

    def weights(self) -> dict[str, float]:
        return rank_by_consult(
            text_deck_weights(self.coverage, self.carriers, self.target),
            self.verdicts,
        )

    def units_decked(self, cards: set[str]) -> set[str]:
        return {
            text for text, carrying in self.carriers.items() if carrying & cards
        }

    def report(self) -> str:
        return text_residues(self.coverage, self.target).render(self.target)


def _held_out_carriers(
    listed: Iterable[str], texts_by_card: Mapping[str, list[str]],
    non_deck: set[str], held_out: HeldOutCards,
) -> dict[str, frozenset[str]]:
    """``held-out text -> its deckable listed carriers``, for each text some
    deckable card carries.

    Held-out texts are decided over every listed card, non-deck ones
    included, or those would read as unlisted carriers and take their texts
    out of the round; only then do non-deck cards leave as carriers. A text
    only listed cards carry is a unit only when ``held_out`` holds it: a held-out
    card's other lines are carried by it alone too, and under ``template`` their
    template is often printed elsewhere, so no game could ever be credited to
    them (:meth:`HeldOutCards.text_acted_through` credits held-out texts only)
    and the round would deck them until each retired.
    """
    carriers = held_out_texts_of(frozenset(listed), texts_by_card)
    return {
        text: cards - non_deck
        for text, cards in carriers.items()
        if cards - non_deck and held_out.holds_text(text)
    }


def _text_coverage(
    carriers: Mapping[str, frozenset[str]], verdicts: Mapping[str, ConsultVerdict],
) -> dict[str, TextCoverage]:
    """One unit per held-out text, castable while any carrier is not judged
    uncastable."""
    return {
        text: TextCoverage(text=text, castable=any(
            verdicts.get(card, ConsultVerdict.UNKNOWN) != ConsultVerdict.UNCASTABLE
            for card in cards
        ))
        for text, cards in carriers.items()
    }


@dataclass(frozen=True)
class _Listing:
    """The listed cards the round decks: each one's converted text, those of
    a deck type (``present``), the holdout, and ``held-out text -> its
    deckable listed carriers``."""

    texts: dict[str, str]
    present: frozenset[str]
    held_out: HeldOutCards
    carriers: dict[str, frozenset[str]]

    def without(self, missing: frozenset[str]) -> _Listing:
        """The listing with ``missing`` (cards Forge's database does not
        hold) taken out of the deck and of every text's carriers."""
        carriers = {
            text: cards - missing for text, cards in self.carriers.items() if cards - missing
        }
        return _Listing(self.texts, self.present - missing, self.held_out, carriers)

    def candidates(self) -> dict[str, DeckCandidate]:
        return {
            name: DeckCandidate.from_converted_text(name, text)
            for name, text in self.texts.items() if name in self.present
        }


def _held_out_listing(config: CollectCoverageConfig, trees: ConvertedTrees) -> _Listing | None:
    """What the ``--only-cards`` list decks, or None (logged) when there is
    no converted card or no listed card carries a held-out text."""
    converted = trees.converted_cards
    if not converted:
        logger.error("%s", no_converted_cards(trees))
        return None
    only_cards = config.deck_holdout.only_cards
    listed = read_name_list(only_cards)
    listed_files = {
        name: script for name, script in trees.card_files.items()
        if name in listed
    }
    texts = {name: converted[name].text for name in listed_files}
    non_deck = non_deck_names(texts)
    present = frozenset(name for name in listed_files if name not in non_deck)
    held_out = trees.holdout(config.holdout)
    carriers = _held_out_carriers(listed_files, trees.card_texts(), non_deck, held_out)
    if non_deck:
        logger.info(
            "Held-out round: %d listed card(s) of a non-deck type left out",
            len(non_deck),
        )
    if not carriers:
        logger.error(
            "None of the %d card(s) in %s carries a text only listed cards "
            "carry; nothing to cover.", len(listed), only_cards,
        )
        return None
    return _Listing(texts, present, held_out, carriers)


def _text_rounds(config: CollectCoverageConfig, trees: ConvertedTrees) -> TextRounds | None:
    """The round's units, consulted and counted as the corpus stands; None
    when there is nothing to cover."""
    listing = _held_out_listing(config, trees)
    if listing is None:
        return None
    verdicts = consult_castability(sorted(listing.present))
    unresolved = unresolvable(listing.present, verdicts)
    listing = listing.without(frozenset(unresolved.cards))
    coverage = _text_coverage(listing.carriers, verdicts)
    floor = config.min_text_games
    counter = TextCoverageCounter(config.source_trees.sidecar_roots(), listing.held_out)
    count_existing(counter, coverage, config)
    logger.info(
        "Held-out round: %d text(s) on %d card(s); %d already at %d games; %d "
        "with no castable carrier",
        len(coverage), len(listing.present),
        sum(1 for unit in coverage.values() if unit.satisfied(floor)), floor,
        sum(1 for unit in coverage.values() if not unit.castable),
    )
    return TextRounds(
        coverage=coverage, carriers=listing.carriers, verdicts=verdicts,
        unresolved=unresolved, counter=counter, records_dir=config.effect_records,
        target=floor, workers=config.workers, candidates=listing.candidates(),
    )


def run_text_coverage(config: CollectCoverageConfig) -> int:
    """The held-out coverage round: listed cards only, keyed by held-out text."""
    rounds = _text_rounds(config, config.source_trees.converted())
    if rounds is None:
        return 1
    # The run's own caps, the three decision-sampling rates included: every
    # stratum has one record composition (FR-059a).
    return play_to_completion(rounds, config)

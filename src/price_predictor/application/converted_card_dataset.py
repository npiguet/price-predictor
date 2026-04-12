"""Walk a converted-cards directory and join entries to prices + printing metadata.

Two views over the same directory tree:

* ``load_parsed_cards`` — fully parses each ``.txt`` into a :class:`Card`
  and enriches it with :class:`PrintingData`. Used by the sklearn pipeline.
* ``load_training_samples`` — keeps the raw card text and bundles it into
  :class:`TrainingSample`. Used by the transformer pipeline, which feeds the
  raw text through its own tokenizer.

Both functions skip files that fail to parse, lack a name, or whose name
can't be resolved against the price map. They share the same
:class:`CardNameResolver`-based join, eliminating four near-identical inline
loops in train/evaluate (sklearn) and train_transformer/evaluate_transformer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from pathlib import Path

from price_predictor.application.training_sample import TrainingSample
from price_predictor.domain.card_name_resolver import CardNameResolver
from price_predictor.domain.entities import Card
from price_predictor.domain.tokenizer import extract_card_name
from price_predictor.domain.value_objects import PrintingData
from price_predictor.infrastructure.converted_card_parser import parse_converted_cards

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ParsedCardDataset:
    """A list of parsed, price-joined cards plus per-reason skip counters."""

    cards: list[Card]
    prices: list[float]
    skipped_reasons: dict[str, int] = field(default_factory=dict)

    @property
    def total_skipped(self) -> int:
        return sum(self.skipped_reasons.values())


def load_parsed_cards(
    output_dir: Path,
    price_map: dict[str, float],
    metadata_map: dict[str, PrintingData] | None = None,
) -> ParsedCardDataset:
    """Parse cards from ``output_dir`` and enrich them with price + printing data.

    Skip reasons are reported under ``parse_error`` (file failed to parse) and
    ``no_printings_match`` (parsed but no price/metadata found).
    """
    cards, parse_errors = parse_converted_cards(output_dir)
    logger.info("Parsed %d cards (%d parse errors)", len(cards), len(parse_errors))

    resolver = CardNameResolver(price_map, metadata_map)
    enriched: list[Card] = []
    prices: list[float] = []
    skipped_reasons: dict[str, int] = {
        "parse_error": len(parse_errors),
        "no_printings_match": 0,
    }

    for card in cards:
        resolved = resolver.resolve(card.name)
        if resolved is None:
            skipped_reasons["no_printings_match"] += 1
            continue
        enriched.append(replace(card, printing_data=resolved.printing_data))
        prices.append(resolved.price_eur)

    logger.info(
        "Joined %d cards to prices (skipped %d total)",
        len(enriched), sum(skipped_reasons.values()),
    )
    return ParsedCardDataset(
        cards=enriched, prices=prices, skipped_reasons=skipped_reasons,
    )


def load_training_samples(
    output_dir: Path,
    price_map: dict[str, float],
    metadata_map: dict[str, PrintingData] | None = None,
) -> list[TrainingSample]:
    """Walk ``output_dir`` and emit :class:`TrainingSample` for every resolved file.

    Files that fail to read, lack a ``name:`` line, or whose name can't be
    resolved are silently skipped (read failures and missing names are
    logged in aggregate).
    """
    resolver = CardNameResolver(price_map, metadata_map)
    samples: list[TrainingSample] = []
    unreadable_or_unnamed = 0

    for txt_file in sorted(output_dir.rglob("*.txt")):
        try:
            text = txt_file.read_text(encoding="utf-8")
        except OSError:
            unreadable_or_unnamed += 1
            continue

        card_name = extract_card_name(text)
        if not card_name:
            unreadable_or_unnamed += 1
            continue

        resolved = resolver.resolve(card_name)
        if resolved is None:
            continue

        samples.append(TrainingSample(
            name=card_name,
            text=text,
            price_eur=resolved.price_eur,
            printing_data=resolved.printing_data,
        ))

    if unreadable_or_unnamed > 0:
        logger.info(
            "Skipped %d text files (unreadable or missing name)",
            unreadable_or_unnamed,
        )
    return samples

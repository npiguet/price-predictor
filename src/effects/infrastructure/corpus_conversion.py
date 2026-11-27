"""``build-corpus`` joins its shards only against the conversion they were
collected against: the records directory's recorded
conversions (``price_predictor.infrastructure.records_conversion``) must be
the card trees' own."""

from __future__ import annotations

from pathlib import Path

from effects.infrastructure.shard_io import iter_shards
from price_predictor.infrastructure.converted_provenance import trees_conversion
from price_predictor.infrastructure.records_conversion import require_joinable


def require_collected_against(records_dir: Path, cards_folders: tuple[Path, ...]) -> None:
    """Refuse (``ConversionRefused``) a records directory whose shards name no
    conversion or one differing from ``cards_folders``'; a directory holding
    no shard is left to ``build-corpus``'s own refusal."""
    shards = iter_shards(records_dir)
    if shards:
        require_joinable(records_dir, shards, trees_conversion(cards_folders))

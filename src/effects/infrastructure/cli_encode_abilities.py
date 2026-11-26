"""``encode-abilities``: the per-ability embedding cache."""

from __future__ import annotations

import argparse
from pathlib import Path

from effects.infrastructure.cli_common import (
    _EFFECT_MODEL_STORE,
    _LAYOUT_REFUSED,
    _add_cards_folder,
    _add_recorded_tokenizer_paths,
    _add_variant,
    _add_variant_scripts,
    config_from_args,
    exits_with,
)


def _encode_abilities_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "encode-abilities",
        help="Compute the per-ability embedding cache from a trained checkpoint",
    )
    parser.set_defaults(func=run_encode_abilities)
    _add_variant(
        parser, help_text="Names the output suffix; the checkpoint is the baseline's own",
    )
    parser.add_argument(
        "--checkpoint", type=Path, required=True,
        help="The checkpoint to encode with (a baseline needs its own)",
    )
    _add_cards_folder(parser)
    _add_variant_scripts(parser)
    _add_recorded_tokenizer_paths(parser)
    parser.add_argument(
        "--output-root", type=Path, required=True,
        help="Cache run directory the ability cache is written under",
    )
    parser.add_argument(
        "--clean", action="store_true",
        help="Remove only the files this variant wrote, then re-encode",
    )


@exits_with({
    f"{_EFFECT_MODEL_STORE}.HashMismatchError": 1, FileNotFoundError: 1, _LAYOUT_REFUSED: 1,
    # --variant identity or state-only, whose models read no cache (finding 11.7).
    "effects.application.encode_abilities.UnencodedVariantError": 1,
})
def run_encode_abilities(args: argparse.Namespace) -> int:
    from effects.application.encode_abilities import EncodeAbilitiesConfig
    from effects.application.encode_abilities import run as encode

    encode(config_from_args(EncodeAbilitiesConfig, args))
    return 0

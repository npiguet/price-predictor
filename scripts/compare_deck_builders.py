#!/usr/bin/env python
"""Build decks three ways on one shared set of pools, for side-by-side comparison.

Generates (or reuses) N sealed pools, then builds one 40-card deck per pool with
each of three methods:

  * ``forge-best``   - Forge's own optimal deckbuilder (``DeckBuilderMain``)
  * ``deck-builder`` - the ``sealed build-decks`` scorer-guided greedy/SA search
  * ``deck-picker``  - the ``sealed pick-decks`` one-shot policy picker

It writes one file per method, each in the ``sealed build-decks --print-decks``
human-readable format (``=== Deck N  score=... ===`` header + one card per line
with mana cost). Every deck's score header is rated by the *same* teacher
scorer, so the three files are directly comparable pool-by-pool: ``Deck N`` is
the same pool in all three files.

This deliberately reuses the real use-case build methods
(``BuildDecksUseCase`` / ``PickDecksUseCase``) so the decks match exactly what
the CLI commands would produce.

Example:
    python scripts/compare_deck_builders.py \
        --size 50 \
        --scorer-checkpoint models/sealed/scorer/gen-4/512-...mwlog.pt \
        --picker-checkpoint models/sealed/picker/best_20260521_071908.pt \
        --cards-path output/cardsfolder-512/ \
        --output-dir output/sealed/compare/
"""

from __future__ import annotations

import argparse
import secrets
import sys
from pathlib import Path

import torch

from sealed.application.build_decks import BuildDecksConfig, BuildDecksUseCase
from sealed.application.evaluate_scorer import format_decks_for_display, score_decks
from sealed.application.pick_decks import PickDecksUseCase
from sealed.domain.greedy_deck_builder import NONLAND_DECK_SIZE
from sealed.infrastructure.converted_card_locator import ConvertedCardLocator
from sealed.infrastructure.evaluation_connector import EvaluationConnector
from sealed.infrastructure.pool_connector import PoolConnector
from sealed.infrastructure.pool_file_reader import parse_pools


def _log(msg: str) -> None:
    print(msg, flush=True)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--size", type=int, default=50,
                    help="Number of sealed pools to generate (default: 50).")
    ap.add_argument("--set", dest="set_code", default=None,
                    help="Set code (e.g. BLB). Omit for a random sealed-legal "
                         "set per pool.")
    ap.add_argument("--scorer-checkpoint", required=True, type=Path,
                    help="Teacher scorer checkpoint. Builds the deck-builder "
                         "decks AND scores all three methods' decks.")
    ap.add_argument("--picker-checkpoint", required=True, type=Path,
                    help="Trained picker checkpoint for the deck-picker method.")
    ap.add_argument("--cards-path", type=Path, default=Path("output/cardsfolder/"),
                    help="Converted-card / .npz cache dir (must match the "
                         "encoder the scorer and picker were trained on).")
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="Directory for the three output files. Default: a fresh "
                         "output/decks_<randomId>/ folder created per run.")
    ap.add_argument("--pools-path", type=Path, default=None,
                    help="Reuse an existing pools.txt instead of generating "
                         "new pools (skips the Forge pool-generation step).")
    # build-decks search knobs; defaults mirror `sealed build-decks`.
    ap.add_argument("--sa-temperature", type=float, default=0.8,
                    help="deck-builder SA initial temperature (0 = pure greedy).")
    ap.add_argument("--sa-cooling", type=float, default=0.85)
    ap.add_argument("--sa-max-iterations", type=int, default=200)
    return ap.parse_args(argv)


def _load_pools(args: argparse.Namespace) -> tuple[Path, list[tuple[str, list[str]]]]:
    if args.pools_path is not None:
        _log(f"Using existing pools: {args.pools_path}")
        return args.pools_path, parse_pools(args.pools_path)
    pools_dir = args.output_dir / "pools"
    pools_dir.mkdir(parents=True, exist_ok=True)
    _log(f"Generating {args.size} pools (set={args.set_code or 'random'})...")
    PoolConnector().generate(args.set_code, args.size, pools_dir)
    pools_file = pools_dir / "pools.txt"
    return pools_file, parse_pools(pools_file)


def _keep_buildable(
    raw_pools: list[tuple[str, list[str]]], locator: ConvertedCardLocator,
) -> list[tuple[str, list[str]]]:
    """Keep only pools with >=23 embeddable cards.

    The scorer and picker builders skip pools below that threshold, so
    pre-filtering keeps all three output files line-aligned (Deck N is the
    same pool in each).
    """
    kept: list[tuple[str, list[str]]] = []
    for set_code, names in raw_pools:
        n_embeddable = sum(1 for nm in names if locator.load_embedding(nm) is not None)
        if n_embeddable >= NONLAND_DECK_SIZE:
            kept.append((set_code, names))
    return kept


def _build_builder_decks(
    kept: list[tuple[str, list[str]]],
    builder: BuildDecksUseCase,
    scorer,
    locator: ConvertedCardLocator,
    config: BuildDecksConfig,
) -> list[list[str]]:
    decks: list[list[str]] = []
    step = max(1, len(kept) // 20)
    for i, (_set, names) in enumerate(kept, 1):
        deck = builder._build_one_deck(scorer, names, locator, config)
        decks.append(deck if deck is not None else [])
        if i % step == 0 or i == len(kept):
            _log(f"  deck-builder {i}/{len(kept)}")
    return decks


def _build_picker_decks(
    kept: list[tuple[str, list[str]]],
    picker_uc: PickDecksUseCase,
    picker,
    picker_dim: int,
    locator: ConvertedCardLocator,
    device: torch.device,
    picker_checkpoint: Path,
    cards_path: Path,
) -> list[list[str]]:
    decks: list[list[str]] = []
    step = max(1, len(kept) // 20)
    width_checked = False
    for i, (_set, names) in enumerate(kept, 1):
        embs, valid = picker_uc._load_pool(names, locator)
        if not width_checked and embs:
            picker_uc._check_width(picker_dim, embs[0].shape[0], picker_checkpoint, cards_path)
            width_checked = True
        decks.append(picker_uc._build_one_deck(picker, embs, valid, locator, device))
        if i % step == 0 or i == len(kept):
            _log(f"  deck-picker {i}/{len(kept)}")
    return decks


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    if args.output_dir is None:
        args.output_dir = Path("output") / f"decks_{secrets.token_hex(4)}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _log(f"Output dir: {args.output_dir}")
    locator = ConvertedCardLocator(args.cards_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pools_file, raw_pools = _load_pools(args)
    kept = _keep_buildable(raw_pools, locator)
    skipped = len(raw_pools) - len(kept)
    _log(f"{len(kept)} usable pools "
         f"({skipped} skipped for <{NONLAND_DECK_SIZE} embeddable cards)")
    if not kept:
        _log("No usable pools; aborting.")
        return 1
    pool_name_lists = [names for _, names in kept]

    # One teacher scorer instance both builds the deck-builder decks and scores
    # every method's decks, so the score headers are directly comparable.
    builder = BuildDecksUseCase()
    scorer = builder._load_model(args.scorer_checkpoint)
    picker_uc = PickDecksUseCase()
    picker, picker_dim = picker_uc._load_picker(args.picker_checkpoint, device)
    bd_config = BuildDecksConfig(
        pools_path=pools_file, label="deck-builder",
        checkpoint=args.scorer_checkpoint, cards_path=args.cards_path,
        sa_temperature=args.sa_temperature, sa_cooling=args.sa_cooling,
        sa_max_iterations=args.sa_max_iterations,
    )

    _log("Building forge-best decks (Forge JVM)...")
    forge_decks = EvaluationConnector().build_forge_decks(pool_name_lists)
    if len(forge_decks) != len(kept):
        _log(f"  WARNING: forge returned {len(forge_decks)} decks for "
             f"{len(kept)} pools; forge-best.txt may not be pool-aligned.")

    _log("Building deck-builder (scorer greedy/SA) decks...")
    builder_decks = _build_builder_decks(kept, builder, scorer, locator, bd_config)

    _log("Building deck-picker decks...")
    picker_decks = _build_picker_decks(
        kept, picker_uc, picker, picker_dim, locator, device,
        args.picker_checkpoint, args.cards_path,
    )

    _log("\nScoring all decks with the teacher scorer and writing files:")
    for name, decks in (
        ("forge-best", forge_decks),
        ("deck-builder", builder_decks),
        ("deck-picker", picker_decks),
    ):
        scores = score_decks(scorer, decks, locator)
        out = args.output_dir / f"{name}.txt"
        out.write_text(
            format_decks_for_display(decks, locator, scores), encoding="utf-8",
        )
        mean = sum(scores) / len(scores) if scores else 0.0
        _log(f"  {name:<13} {len(decks):>4} decks  mean_score={mean:+.4f}  -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

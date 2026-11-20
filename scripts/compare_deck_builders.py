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
the CLI commands would produce. Those per-deck methods are private: the use
cases expose only whole-file ``execute``, which cannot share one scorer across
the three methods or keep the outputs pool-aligned.

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
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

from price_predictor.infrastructure.run_manifest import ScriptRun, run_script
from sealed.application.build_decks import BuildDecksConfig, BuildDecksUseCase
from sealed.application.deck_assembly import load_pool_embeddings
from sealed.application.evaluate_scorer import format_decks_for_display, score_decks
from sealed.application.pick_decks import PickDecksUseCase
from sealed.domain.greedy_deck_builder import NONLAND_DECK_SIZE
from sealed.infrastructure.converted_card_locator import ConvertedCardLocator
from sealed.infrastructure.evaluation_connector import EvaluationConnector
from sealed.infrastructure.pool_connector import PoolConnector
from sealed.infrastructure.pool_file_reader import SealedPool, parse_pools


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
    ap.add_argument("--cards-path", type=Path, required=True,
                    help="Converted-card / .npz cache dir (must match the "
                         "encoder the scorer and picker were trained on).")
    ap.add_argument("--output-dir", type=Path, required=True,
                    help="Directory for the three output files (the run directory).")
    ap.add_argument("--pools-path", type=Path, default=None,
                    help="Reuse an existing pools.txt instead of generating "
                         "new pools (skips the Forge pool-generation step).")
    # build-decks search knobs; defaults mirror `sealed build-decks`.
    ap.add_argument("--sa-temperature", type=float, default=0.8,
                    help="deck-builder SA initial temperature (0 = pure greedy).")
    ap.add_argument("--sa-cooling", type=float, default=0.85)
    ap.add_argument("--sa-max-iterations", type=int, default=200)
    return ap.parse_args(argv)


def _load_pools(args: argparse.Namespace) -> tuple[Path, list[SealedPool]]:
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
    raw_pools: list[SealedPool], locator: ConvertedCardLocator,
) -> list[list[str]]:
    """Keep only pools with >=23 embeddable cards.

    The scorer and picker builders skip pools below that threshold, so
    pre-filtering keeps all three output files line-aligned (Deck N is the
    same pool in each).
    """
    return [pool.cards for pool in raw_pools
            if len(load_pool_embeddings(pool.cards, locator)[1]) >= NONLAND_DECK_SIZE]


@dataclass(frozen=True)
class BuilderSetup:
    """What the deck-builder method needs to build one deck per pool."""

    use_case: BuildDecksUseCase
    scorer: object
    locator: ConvertedCardLocator
    config: BuildDecksConfig


@dataclass(frozen=True)
class PickerSetup:
    """What the deck-picker method needs to build one deck per pool."""

    use_case: PickDecksUseCase
    picker: object
    picker_dim: int
    locator: ConvertedCardLocator
    device: torch.device
    picker_checkpoint: Path
    cards_path: Path


def _log_progress(label: str, i: int, total: int, step: int) -> None:
    if i % step == 0 or i == total:
        _log(f"  {label} {i}/{total}")


def _build_builder_decks(kept: list[list[str]], setup: BuilderSetup) -> list[list[str]]:
    decks: list[list[str]] = []
    step = max(1, len(kept) // 20)
    for i, names in enumerate(kept, 1):
        deck = setup.use_case._build_one_deck(
            setup.scorer, names, setup.locator, setup.config)
        decks.append(list(deck.cards) if deck is not None else [])
        _log_progress("deck-builder", i, len(kept), step)
    return decks


def _build_picker_decks(kept: list[list[str]], setup: PickerSetup) -> list[list[str]]:
    decks: list[list[str]] = []
    step = max(1, len(kept) // 20)
    width_checked = False
    for i, names in enumerate(kept, 1):
        embs, valid = load_pool_embeddings(names, setup.locator)
        if not width_checked:
            setup.use_case._check_width(setup.picker_dim, embs[valid[0]].shape[0],
                                        setup.picker_checkpoint, setup.cards_path)
            width_checked = True
        decks.append(list(setup.use_case._build_one_deck(
            setup.picker, embs, valid, setup.locator, setup.device).cards))
        _log_progress("deck-picker", i, len(kept), step)
    return decks


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    return run_script(
        args.output_dir, lambda: _run(args),
        ScriptRun(
            script=__file__, argv=argv,
            inputs=[args.scorer_checkpoint, args.picker_checkpoint, args.cards_path,
                    args.pools_path],
        ),
    )


def _run(args: argparse.Namespace) -> int:
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

    # One teacher scorer instance both builds the deck-builder decks and scores
    # every method's decks, so the score headers are directly comparable.
    builder = BuildDecksUseCase()
    scorer = builder._load_model(args.scorer_checkpoint)
    picker_uc = PickDecksUseCase()
    picker, picker_dim = picker_uc._load_picker(args.picker_checkpoint, device)
    bd_config = BuildDecksConfig(
        pools_path=pools_file, label="deck-builder",
        checkpoint=args.scorer_checkpoint, cards_path=args.cards_path,
        output=args.output_dir / "deck-builder.txt",  # never written: decks stay in memory
        sa_temperature=args.sa_temperature, sa_cooling=args.sa_cooling,
        sa_max_iterations=args.sa_max_iterations,
    )

    _log("Building forge-best decks (Forge JVM)...")
    forge_decks = EvaluationConnector().build_forge_decks(kept)
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

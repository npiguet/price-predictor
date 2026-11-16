"""Cut a small, kind-balanced sample of real records into a test fixture.

Tests of the gen-2 readers, families, value targets and probes need records
shaped exactly like the collector's output, so the fixture is cut from real
shards rather than built by hand. Every category below is filled with a few
records; a chosen cost half brings its effect half, a chosen probe brings the
combat record it mirrors, and a chosen fired trigger brings the resolution
records that follow it in the same game, so the joins the probes make (an
interaction, a link) have both ends present.

The sidecars and converted texts every chosen record names, anywhere in its
envelope, state or payload, are copied beside it under the same tree layout,
so a ``SidecarCache`` rooted there resolves every key.

    python scripts/make_fixture_records.py --generation 3 --per-category 2
        --records-dir output/effects/gen3/scratch/2026-11-15-pilot
        --cards-folder output/effects/gen3/converted/cardsfolder
        --cards-folder output/effects/gen3/converted/tokenscripts
        --output tests/fixtures/effects/gen3-records.jsonl.gz
        --sidecars-out tests/fixtures/effects/gen3-sidecars

(one command line).
Every path is named on the command line: the records, the converted trees
and both outputs (the ``artifact-layout`` skill). Gen-1 and gen-2 are cut the
same way from their records' NAS copies.

``--generation 2`` adds the categories the rule detectors are tested on: a
resolution whose effect half kills a creature at toughness 0 or less in the
state-based check (704.5f fires), one that leaves a creature at toughness 1
(704.5f does not fire), a creature killed by damage (704.5g), the legend rule
(704.5j), a planeswalker at loyalty 0 (704.5i) and an Aura whose host left
(704.5m), a damage-step probe stripping deathtouch, triggers a keyword
generates (fired and not), a predefined token
kind made (111.10) and an ability whose source is one, and records from a
random-seat game and from a game the Forge AI played on both seats, and the
shapes the detector families' parameters read: a resolution and a decision
candidate acting through a keyword's own line, a candidate whose card prints a
keyword, a candidate whose cost has a part (``Sac<...>``), a candidate listing
targets, a damage rewrite a prevention shield replaced, a creature leaving the
battlefield in combat, a token's static and a counter-conditioned static.

Gen-2 also pins records by id (``GEN2_PINNED`` in ``fixture_records/pinned.py``):
the shapes the rule detectors were corrected against in T036, each one record
read and judged by hand, found by reading the shards their ids name.
``--append`` adds the pinned records to the fixture already on disk without
rescanning, so every record an existing test names stays.

``--generation 3`` reads each ``--records-dir`` in its own interleaved order,
a later one filling only what the earlier left, and takes records only from
games that ended: each game the fixture holds a record of brings its
game-end marker, and every record sharing a ``link_id`` with a chosen one or
mirrored by one, wherever it sits in the game. Records are written as the
collector wrote their lines, in shard order. It adds, before
gen-2's categories, one per story-2 acceptance scenario (``a1-`` … ``a11-``: a
permanent with a static, a trigger and a replacement line; a creature stripped
of every ability; flying granted by another permanent's static; ward and
annihilator fragments; protection; a first-strike combat; a burn kill, which
also brings the dies-trigger records of its game; a sorcery off turn and a
candidate of the seat without priority; an unmet condition; a declared anchor
tapped and one with vigilance), one per ``turn_action`` subkind and per failed
check, the anchored legality shapes, and the new snapshot members (floating
effects of both kinds, delayed triggers, dungeons, day and night, each
extra-entity role, a face-down card, dice), and land-play candidates: playable,
past the turn's limit, off their player's turn, and on it at a time no sorcery
could be cast, the combats ``combat_keyword_effect`` reads (a vigilant
attacker, a first striker in the regular step), a directed fork, a fork
whose copy dropped something, and the rarer members ``field-coverage`` checks
vary (the initiative, a land-play limit other than one, a buffered ability's
events, a blocker required elsewhere or forbidden by its own line), and an
effect half creating one continuous effect or delayed trigger. Face-down keys
name ``variant-scripts/``: pass that tree as a ``--cards-folder`` too when the
records name it. The sidecars directory is emptied first, so it holds exactly
the sidecars the records name.

The categories, the scan and the joins live in ``fixture_records/``."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from fixture_records.categories import categories_for
from fixture_records.pinned import GEN2_PINNED, append_pinned, pinned_records
from fixture_records.scan import Selection, scan
from fixture_records.shards import interleaved, write_fixture
from fixture_records.sidecars import Sidecars, copy_sidecars

from effects.infrastructure.record_codec import record_from_dict
from effects.infrastructure.shard_io import iter_shards


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--generation", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--records-dir", type=Path, action="append", required=True,
                        help="a records run directory; repeatable, read in order, a later "
                             "one filling only the categories the earlier left unfilled")
    parser.add_argument("--cards-folder", action="append", type=Path, required=True,
                        help="a converted tree, named by its last path segment "
                             "(cardsfolder, tokenscripts, variant-scripts); repeatable")
    parser.add_argument("--output", type=Path, required=True,
                        help="the fixture, e.g. tests/fixtures/effects/gen3-records.jsonl.gz")
    parser.add_argument("--sidecars-out", type=Path, required=True,
                        help="the fixture's sidecar tree, e.g. "
                             "tests/fixtures/effects/gen3-sidecars")
    parser.add_argument("--per-category", type=int, default=3)
    parser.add_argument("--max-shards", type=int, default=400)
    parser.add_argument("--append", action="store_true",
                        help="gen-2: add the pinned records to the fixture on disk, no scan")
    return parser


def _checked_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    args = parser.parse_args()
    empty = next((directory for directory in args.records_dir if not iter_shards(directory)),
                 None)
    if empty is not None:
        parser.error(f"--records-dir {empty} holds no shard")
    not_a_tree = next((folder for folder in args.cards_folder if not folder.is_dir()), None)
    if not_a_tree is not None:
        parser.error(f"--cards-folder {not_a_tree} is no directory")
    if args.append and args.generation != 2:
        parser.error("--append adds gen-2's pinned records")
    return args


def _without_tags(record: dict) -> dict:
    """The record as the collector wrote it, without the scan's ``_`` members."""
    return {key: value for key, value in record.items() if not key.startswith("_")}


def _print_names(heading: str, names: list[str]) -> None:
    if names:
        print(f"{heading}: {', '.join(names)}", file=sys.stderr)


def _write(args: argparse.Namespace, selection: Selection, sidecars: Sidecars) -> int:
    """Write the fixture and the sidecars its records name, and report them."""
    records = selection.records()
    if args.generation == 2:
        records.update(pinned_records(args.records_dir[0], GEN2_PINNED))
    missing = selection.missing()
    # Every record must parse with the current reader before it is a fixture.
    for record in records.values():
        record_from_dict(_without_tags(record))
    write_fixture(args.output, records)
    if args.generation == 3 and args.sidecars_out.exists():
        # Exactly the sidecars these records name: none left from an earlier cut.
        shutil.rmtree(args.sidecars_out)
    copied, unresolved = copy_sidecars(records, sidecars, args.sidecars_out,
                                       keep_existing=False)
    selection.print_counts()
    games = len({record["game_id"] for record in records.values()})
    print(f"{len(records)} records ({len(selection.extras)} joined) in {games} games -> "
          f"{args.output}; {copied} sidecars -> {args.sidecars_out}", file=sys.stderr)
    _print_names("no record found for", missing)
    _print_names("no sidecar in the given trees for", unresolved)
    return 0


def main() -> int:
    args = _checked_args(_parser())
    sidecars = Sidecars({folder.name: folder for folder in args.cards_folder})
    if args.append:
        return append_pinned(args.records_dir[0], args.output, args.sidecars_out, sidecars)
    selection = Selection(categories_for(args.generation, sidecars, args.cards_folder),
                          args.per_category, forge_ai_wanted=args.generation >= 2)
    # Interleave the collection runs rather than reading one run's shards first,
    # so a category only another run reaches (probes, variants) is found early;
    # a later directory only fills what the earlier ones left unfilled.
    shards = [shard for directory in args.records_dir for shard in interleaved(directory)]
    scan(shards[: args.max_shards], selection, gen3=args.generation == 3)
    return _write(args, selection, sidecars)


if __name__ == "__main__":
    raise SystemExit(main())

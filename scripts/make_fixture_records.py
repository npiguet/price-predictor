"""Cut a small, kind-balanced sample of real records into a test fixture.

Tests of the readers, families, value targets and probes need records
shaped exactly like the collector's output, so the fixture is cut from real
shards rather than built by hand. Every category below is filled with a few
records; a chosen cost half brings its effect half, a chosen probe brings the
combat record it mirrors, and a chosen fired trigger brings the resolution
records that follow it in the same game, so the joins the probes make (an
interaction, a link) have both ends present.

The sidecars and converted texts every chosen record names, anywhere in its
envelope, state or payload, are copied beside it under the same tree layout,
so a ``SidecarCache`` rooted there resolves every key.

    python scripts/make_fixture_records.py --per-category 2
        --records-dir output/effects/gen3/scratch/2026-11-21-review-pilot-4
        --cards-folder output/effects/gen3/converted/cardsfolder
        --cards-folder output/effects/gen3/converted/tokenscripts
        --output tests/fixtures/effects/gen3-records.jsonl.gz
        --sidecars-out tests/fixtures/effects/gen3-sidecars

(one command line).
Every path is named on the command line: the records, the converted trees
and both outputs (the ``artifact-layout`` skill).

It reads each ``--records-dir`` in its own interleaved order,
a later one filling only what the earlier left, and takes records only from
games that ended: each game the fixture holds a record of brings its
game-end marker, and every record sharing a ``link_id`` with a chosen one or
mirrored by one, wherever it sits in the game. Records are written as the
collector wrote their lines, in shard order. The categories are, matched first
because a pilot holds few of each, what became of a stack item and which line
acted: an effect half fizzled, one partially fizzled and one declined, an
intervention's forced half, an ability countered (ward), a saga's chapter II
resolving, a Class at level 2 whose level-2 line acts, mana a pool lost (a
drain's effect half or a ``LoseMana`` rewrite), and colourless mana paid by a
cost half or floating in a pool; then the shapes the
rule detectors and detector families read (a creature killed in the state-based
check, the legend rule, a damage-step probe, a keyword's own line, a random-seat
game and a game the Forge AI played on both seats), one per story-2 acceptance
scenario (``a1-`` … ``a11-``: a permanent with a static, a trigger and a
replacement line; a creature stripped of every ability; flying granted by
another permanent's static; ward and annihilator fragments; protection; a
first-strike combat; a burn kill, which also brings the dies-trigger records of
its game; a sorcery off turn and a candidate of the seat without priority; an
unmet condition; a declared anchor tapped and one with vigilance), one per
``turn_action`` subkind and per failed check, the anchored legality shapes, and
the new snapshot members (floating effects of both kinds, delayed triggers,
dungeons, day and night, each extra-entity role, a face-down card, dice), and
land-play candidates: playable, past the turn's limit, off their player's turn,
and on it at a time no sorcery could be cast, the combats
``combat_keyword_effect`` reads (a vigilant attacker, a first striker in the
regular step), a directed fork, a fork whose copy dropped something, and the
rarer members ``field-coverage`` checks vary (the initiative, a land-play limit
other than one, a buffered ability's events, a blocker required elsewhere or
forbidden by its own line), and an effect half creating one continuous effect or
delayed trigger. Matched last, taking only what no other category took: the
members a pilot varies on few records (an event of a combat or a turn-based
action a clause made or that lasts, a buffered event other than a counter
change, a rewrite into another event than damage, a blocker needed twice or
forbidden by a line keying to nothing, a spell targeting a spell, a choice
made, a battle, an emblem, energy, unlimited land plays, a static's
contribution setting P/T, colours, control or a hidden keyword), and the
shapes ``effect_family`` names a family by (a charm's cost half, a rewrite
through a damage replacement line, a blocker a ``CantBlock`` static forbids),
a static's contribution adding a colour rather than setting the colours, the
game-end record, an Aura whose host an effect touches and leaves in place, a
zone change a replacement replaced, a rewrite through a keyword's line, an
activated line's effect half dealing damage to any target, and an attackers
record listing no legal attacker beside a noncreature permanent.
Face-down keys name ``variant-scripts/``: pass that tree as a ``--cards-folder``
too when the records name it. The sidecars directory is
emptied first, so it holds exactly the sidecars the records name.

The categories, the scan and the joins live in ``fixture_records/``."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from fixture_records.categories import categories_for
from fixture_records.scan import Selection, scan
from fixture_records.shards import interleaved, write_fixture
from fixture_records.sidecars import Sidecars, copy_sidecars

from effects.infrastructure.record_codec import record_from_dict
from effects.infrastructure.shard_io import iter_shards


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
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
    missing = selection.missing()
    # Every record must parse with the current reader before it is a fixture.
    for record in records.values():
        record_from_dict(_without_tags(record))
    write_fixture(args.output, records)
    if args.sidecars_out.exists():
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
    selection = Selection(categories_for(sidecars, args.cards_folder),
                          args.per_category, forge_ai_wanted=True)
    # Interleave the collection runs rather than reading one run's shards first,
    # so a category only another run reaches (probes, variants) is found early;
    # a later directory only fills what the earlier ones left unfilled.
    shards = [shard for directory in args.records_dir for shard in interleaved(directory)]
    scan(shards[: args.max_shards], selection)
    return _write(args, selection, sidecars)


if __name__ == "__main__":
    raise SystemExit(main())

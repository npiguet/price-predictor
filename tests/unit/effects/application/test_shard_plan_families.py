"""A shard's effect families are resolved for the records a batch draws.

The epoch line counts each trained record by rarity bucket and effect family.
The rarity weights read every record's text, but a family is read only for a
record a batch draws, once however often it is drawn, so a shard's join
tallies (``ShardClosed.unresolved``) count the drawn records' lookups alone.
"""

from __future__ import annotations

from effects.domain.provenance import ProvenanceKey
from effects.domain.records import RecordKind
from tests.unit.effects.builders import (
    GEN3_SIDECAR_FOLDERS,
    curated_fixture_records,
    make_preparer,
    make_record,
)

#: A script no fixture sidecar describes: the join reads it as unconverted.
UNCONVERTED = ProvenanceKey("cardsfolder/z/never_converted.txt", 0, "spell", 0)


def _plan(tmp_path, records):
    preparer = make_preparer(tmp_path, cards_folders=GEN3_SIDECAR_FOLDERS)
    texts = preparer._texts(records)
    return preparer, preparer._shard_plan(records, texts, preparer._games(records, texts))


def _shard():
    unconverted = make_record(
        RecordKind.RESOLUTION, game_id="g-unconverted", ability=(UNCONVERTED,),
    )
    return unconverted, [*curated_fixture_records(), unconverted]


def _lookups(preparer) -> int:
    """The unconverted script's lookups the cache has tallied."""
    return preparer.sidecars.join_gaps().unresolved.get(UNCONVERTED.script_file, 0)


def test_planning_a_shard_reads_each_text_and_no_family(tmp_path):
    """The rarity weights read every record's text: one lookup for each."""
    unconverted, records = _shard()
    preparer, _ = _plan(tmp_path, records)
    assert _lookups(preparer) == 1


def test_an_undrawn_record_leaves_no_family_lookup(tmp_path):
    unconverted, records = _shard()
    preparer, plan = _plan(tmp_path, records)
    plan.shares([record for record in records if record is not unconverted])
    assert _lookups(preparer) == 1


def test_a_drawn_record_resolves_its_family_once_however_often_it_is_drawn(tmp_path):
    unconverted, records = _shard()
    preparer, plan = _plan(tmp_path, records)
    _, families = plan.shares([unconverted])
    assert _lookups(preparer) == 2
    _, again = plan.shares([unconverted, unconverted])
    assert families == {plan.family(unconverted): 1}
    assert sum(again.values()) == 2 and _lookups(preparer) == 2

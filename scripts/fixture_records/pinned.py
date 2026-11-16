"""Gen-2 records pinned by id: the shapes the rule detectors were corrected
against in T036, each one record read and judged by hand."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

from effects.infrastructure.record_codec import record_from_dict
from effects.infrastructure.shard_io import iter_shard_lines, iter_shards
from fixture_records.shards import read_fixture, write_fixture
from fixture_records.sidecars import Sidecars, copy_sidecars

#: Gen-2 records pinned by id, with what each shows (T036, the rule detectors'
#: review groups). A shard is named by its record id's prefix.
GEN2_PINNED: dict[str, str] = {
    "71253f20-5472-4ddd-8c58-27c78651878b.3-mv4d43zn1o4f.6392":
        "burn takes a player to 0 life; the game end is in the next record",
    "71253f20-5472-4ddd-8c58-27c78651878b.3-mv4d43zn1o4f.6393":
        "the game-end rescue of orphaned events holding player_won",
    "71253f20-5472-4ddd-8c58-27c78651878b.8-mv3ts9u2nlt0.12970":
        "a game-end rescue spanning a turn, its board read after its events",
    "71253f20-5472-4ddd-8c58-27c78651878b.0-mv46oleh5eqw.8786":
        "a combat dealing lethal damage and poison at once",
    "71253f20-5472-4ddd-8c58-27c78651878b.7-mv4cjj66gz7g.13429":
        "Disintegrate: lethal damage, the creature exiled instead of dying",
    "71253f20-5472-4ddd-8c58-27c78651878b.9-mv42599vpk5m.16775":
        "Stuffy Doll: lethal damage to an indestructible creature",
    "71253f20-5472-4ddd-8c58-27c78651878b.9-mv42599vpk5m.1744":
        "Pestilence ends the game in the check: no creature dies",
    "71253f20-5472-4ddd-8c58-27c78651878b.8-mv3ts9u2nlt0.11931":
        "Nyx Infusion: a death with no recorded P/T change",
    "e4713c96-1647-4d93-ac89-039a435f0363.11-mv4w45ojam2q.11464":
        "Bite Down: one damage from a deathtouch source kills",
    "e4713c96-1647-4d93-ac89-039a435f0363.0-mv4yz1t23qrs.5464":
        "Undercity Uprising: deathtouch granted by the resolution, then a fight",
    "e4713c96-1647-4d93-ac89-039a435f0363.11-mv4w45ojam2q.15348":
        "Kami of the Painted Road gains protection from white: its white Aura falls",
    "e4713c96-1647-4d93-ac89-039a435f0363.2-mv541ovjbkr3.2538":
        "a bestowed Aura's host leaves",
    "71253f20-5472-4ddd-8c58-27c78651878b.2-mv465aywnc5h.14696":
        "Ordeal of Thassa sacrifices itself, attributed unresolved",
    "71253f20-5472-4ddd-8c58-27c78651878b.9-mv4cne1d8rhq.2995":
        "an Aura destroyed as the target, attributed unresolved",
    "71253f20-5472-4ddd-8c58-27c78651878b.9-mv4cne1d8rhq.2092":
        "the host destroyed, its Aura exiled in the check",
    "50e3aa10-aa8d-4153-a614-e786a1dc6a6d.11-mv5g0s6bka1j.14623":
        "a lone legend entering at 0/0 dies",
    "71253f20-5472-4ddd-8c58-27c78651878b.10-mv4aecvvg3eo.3650":
        "a dies trigger whose board shows its dead host at toughness 0",
    "e4713c96-1647-4d93-ac89-039a435f0363.7-mv4yq1tr68fr.2370":
        "an Aura's toughness boost lost: marked damage now lethal",
    "50e3aa10-aa8d-4153-a614-e786a1dc6a6d.11-mv5epq5qc9mg.3703":
        "the host stops being a creature: its Aura falls",
    "71253f20-5472-4ddd-8c58-27c78651878b.6-mv461fz6wqim.13501":
        "an endure resolution writing its counters as P1P1",
    "2ed74263-ee28-4948-9621-55dba1160c44.1-mv5im7i940u2.36687":
        "a Saga's last chapter draws, then the Saga is sacrificed in the check",
}


def _by_shard(record_ids) -> dict[str, set[str]]:
    by_shard: dict[str, set[str]] = defaultdict(set)
    for record_id in record_ids:
        by_shard[record_id.rsplit(".", 1)[0]].add(record_id)
    return by_shard


def _records_in(path: Path, ids: set[str]) -> dict[str, dict]:
    found = (json.loads(line) for line in iter_shard_lines(path))
    return {record["record_id"]: record for record in found if record["record_id"] in ids}


def pinned_records(records_dir: Path, wanted: dict[str, str]) -> dict[str, dict]:
    """The pinned records, read from the shards their ids name."""
    paths = {path.name.split(".jsonl")[0]: path for path in iter_shards(records_dir)}
    found: dict[str, dict] = {}
    for name, ids in sorted(_by_shard(wanted).items()):
        if name in paths:
            found.update(_records_in(paths[name], ids))
    return found


def append_pinned(records_dir: Path, output: Path, sidecars_out: Path,
                  sidecars: Sidecars) -> int:
    """Add the pinned records to the fixture on disk, leaving every record in it."""
    records = read_fixture(output)
    pinned = pinned_records(records_dir, GEN2_PINNED)
    for record in pinned.values():
        record_from_dict(record)
    added = {rid: r for rid, r in pinned.items() if rid not in records}
    records.update(added)
    write_fixture(output, records)
    copied, _ = copy_sidecars(added, sidecars, sidecars_out, keep_existing=True)
    missing = sorted(set(GEN2_PINNED) - set(pinned))
    print(f"{len(added)} pinned records added ({len(records)} in all) -> {output}; "
          f"{copied} sidecars -> {sidecars_out}", file=sys.stderr)
    if missing:
        print(f"pinned records not found: {', '.join(missing)}", file=sys.stderr)
    return 1 if missing else 0

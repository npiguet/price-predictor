"""The effects tests' fixture files: the gen-3 fixture records and their
sidecars, the Java characterization goldens, and the settings of the short
real runs over them.

The builders of records, files, batchers, training loops and selection sets
are in the ``*_builders`` modules beside this one.
"""

from __future__ import annotations

from pathlib import Path

EFFECTS_FIXTURES = Path(__file__).parents[2] / "fixtures" / "effects"
#: Forge's keyword table, as the fixture records' runs read it.
FIXTURE_KEYWORD_DEFINITIONS = EFFECTS_FIXTURES / "keyword-definitions.json"
#: The seed and shard count of the short real runs over the gen-3 records.
FIXTURE_TRAINING_SEED = 11
FIXTURE_TRAINING_SHARDS = 3

#: Real gen-3 JSON: the goldens the connector's characterization tests emit
#: (``-Dcharacterization.update=true``), read until the gen-3 fixture exists.
JAVA_GOLDENS = (
    Path(__file__).parents[3] / "forge-connector" / "src" / "test" / "resources"
    / "characterization"
)
#: The one golden that is no collected record: an AI what-if blockers record
#: pinned for its envelope alone, with an empty payload no collector writes.
_ENVELOPE_ONLY_GOLDENS = frozenset({"record-what-if.jsonl"})


def java_golden_records() -> list[dict]:
    """Every whole record the Java goldens hold, as the raw JSON objects."""
    import json

    return [
        json.loads(line)
        for path in sorted(JAVA_GOLDENS.glob("*.jsonl"))
        if path.name not in _ENVELOPE_ONLY_GOLDENS
        for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def java_golden_entities() -> list[dict]:
    """Every entity of the Java goldens, the standalone snapshots' included."""
    import json

    snapshots = [record["state"] for record in java_golden_records()]
    snapshots += [json.loads(path.read_text(encoding="utf-8"))
                  for path in sorted(JAVA_GOLDENS.glob("snapshot-*.json"))]
    return [entity for snapshot in snapshots for entity in snapshot["entities"]]


#: Real gen-3 records, each line as the collector wrote it, with the sidecars
#: every key they name resolves to: cut by ``scripts/make_fixture_records.py
#: --per-category 2`` from the fourth review pilot
#: (``output/effects/gen3/scratch/2026-11-21-review-pilot-4``, collected at
#: 4e8a8e57 / Forge e1acd9c8edd), from games that ended: each of its games ends
#: in its game-end marker. No byte of any line was changed since.
GEN3_RECORDS = EFFECTS_FIXTURES / "gen3-records.jsonl.gz"
GEN3_SIDECAR_FOLDERS = (
    EFFECTS_FIXTURES / "gen3-sidecars" / "cardsfolder",
    EFFECTS_FIXTURES / "gen3-sidecars" / "tokenscripts",
)
#: What the fixture holds, its game-end markers included (one per game), and
#: the SHA-256 of its decompressed lines, pinned by
#: ``test_record_codec_fixture.py::TestTheGen3Fixture``: a regeneration moves the
#: three together, and with them every count a test pins on the fixture.
GEN3_FIXTURE_RECORDS = 396
GEN3_FIXTURE_GAMES = 73
GEN3_FIXTURE_SHA256 = "052f4311a58f7a62b7d7f026bbb82a8ed865ec23bda9a9a112befbd110239f2e"


def gen3_fixture_records() -> list[dict]:
    """The gen-3 fixture's records, as the raw JSON objects the collector wrote."""
    import gzip
    import json

    with gzip.open(GEN3_RECORDS, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_cut_gen3_fixture(path: Path) -> Path:
    """The gen-3 fixture's records at ``path`` without their game-end
    markers: the shard a worker killed mid-game leaves, every game in it cut."""
    import gzip

    with gzip.open(GEN3_RECORDS, "rt", encoding="utf-8") as handle:
        kept = [line for line in handle if '"subkind":"game_end"' not in line]
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as out:
        out.writelines(kept)
    return path


def curated_fixture_records() -> list:
    """The gen-3 fixture's records as a curated corpus holds them, parsed:
    every record but the ``turn_action`` ones, which ``build-corpus`` writes
    into no stratum, so no trainer reads one."""
    from effects.domain.records import RecordKind
    from effects.infrastructure.shard_io import read_shard

    return [r for r in read_shard(GEN3_RECORDS) if r.kind is not RecordKind.TURN_ACTION]

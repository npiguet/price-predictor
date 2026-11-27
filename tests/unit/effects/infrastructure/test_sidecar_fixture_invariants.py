"""Whole-data invariants over the converter's own gen-3 sidecars and the
fixture records that name their keys.

* No key is declared by two lines of one face, and every line declares a key
  unless it is one of the listed exceptions.
* Every key a fixture record names joins exactly one line, is counted as
  dropped or runtime-only by design, or is a saga's bare keyword key, which a
  board line reads as one slot per chapter.
"""

from __future__ import annotations

from collections import Counter

import pytest

from effects.domain.provenance import KeyResolution
from effects.domain.records import key_sites
from effects.infrastructure.converted_trees import SourceTrees
from effects.infrastructure.shard_io import read_shard
from effects.infrastructure.sidecar_io import read_sidecar
from tests.unit.effects.builders import GEN3_RECORDS, GEN3_SIDECAR_FOLDERS

#: Line kinds that declare no key. A modal line's die-roll outcomes are unkeyed
#: ``option`` lines below their spell; a ``draft`` line is drafting text, no
#: ability; an additional-cost or alternate-cost line that states only a cost
#: declares no spell key (text-encoding.md, the spell line).
UNKEYED_KINDS = {"option", "draft", "additional cost", "alternate cost"}


@pytest.fixture(scope="module")
def sidecars():
    return [read_sidecar(path) for folder in GEN3_SIDECAR_FOLDERS
            for path in sorted(folder.rglob("*.provenance.json"))]


def test_no_key_is_declared_by_two_lines_of_one_sidecar(sidecars):
    assert len(sidecars) > 600
    repeated = []
    for sidecar in sidecars:
        declared = Counter(key for line in sidecar.lines for key in set(line.provenance))
        repeated += [(sidecar.script_file, key.label) for key, n in declared.items() if n > 1]
    assert not repeated, repeated[:5]


def test_every_line_declares_a_key_unless_it_is_a_listed_kind(sidecars):
    unkeyed = Counter(line.line_kind for sidecar in sidecars
                      for line in sidecar.lines if not line.provenance)
    assert set(unkeyed) <= UNKEYED_KINDS, unkeyed


def test_an_unkeyed_cost_line_states_a_cost_alone(sidecars):
    for sidecar in sidecars:
        for line in sidecar.lines:
            if not line.provenance and line.line_kind.endswith(" cost"):
                assert {span.role for span in line.role_spans} == {"cost"}, sidecar.script_file


def test_every_key_a_fixture_record_names_joins_a_line_or_is_counted_by_design():
    cache = SourceTrees.of(list(GEN3_SIDECAR_FOLDERS)).sidecar_cache()
    by_resolution: Counter = Counter()
    for record in read_shard(GEN3_RECORDS):
        for key in set(key_sites(record)):
            joined = cache.join(key)
            by_resolution[joined.resolution] += 1
            if joined.resolution is KeyResolution.AMBIGUOUS:
                chapters = cache.get(key.script_file).board_keys(key)
                assert len(chapters) >= 2, key
                assert all(cache.join(each).resolution is KeyResolution.LINE
                           for each in chapters), key
    assert set(by_resolution) <= {KeyResolution.LINE, KeyResolution.DROPPED,
                                  KeyResolution.RUNTIME_ONLY, KeyResolution.AMBIGUOUS}
    assert by_resolution[KeyResolution.LINE] > 10_000

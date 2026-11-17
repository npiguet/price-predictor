"""A sidecar's ``built`` keys on disk.

A keyword that renders several lines (a saga's chapters) declares on each
line the ``built`` key of the trait that line renders, and its bare keyword
key on no line. The fixture is Binding the Old Gods' real sidecar, its chapter
lines rewritten to that shape.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from effects.domain.provenance import BuiltTrait, ProvenanceKey, ProvenanceSidecar
from effects.infrastructure.sidecar_io import read_sidecar, sidecar_to_dict

_SIDECAR = (
    Path(__file__).parents[3] / "fixtures" / "effects" / "gen3-sidecars"
    / "cardsfolder" / "b" / "binding_the_old_gods.provenance.json"
)
_SAGA = ProvenanceKey("cardsfolder/b/binding_the_old_gods.txt", 0, "keyword", 0)


def _built_sidecar() -> ProvenanceSidecar:
    """The real sidecar, each chapter line keyed by its own chapter trigger."""
    sidecar = read_sidecar(_SIDECAR)
    chapters = [line for line in sidecar.lines if line.line_kind == "chapter"]
    assert len(chapters) == 3
    rewritten = {
        line.line_index: replace(line, provenance=(
            replace(_SAGA, built=BuiltTrait("trigger", position)),))
        for position, line in enumerate(chapters)
    }
    return replace(sidecar, lines=tuple(
        rewritten.get(line.line_index, line) for line in sidecar.lines))


def test_a_chapter_line_writes_its_built_key_in_the_sidecar_form():
    data = sidecar_to_dict(_built_sidecar())
    chapters = [line for line in data["lines"] if line["line_kind"] == "chapter"]
    assert [line["provenance"] for line in chapters] == [
        [{"face": 0, "trait_kind": "keyword", "index_within_kind": 0,
          "built": {"kind": "trigger", "index": index}}]
        for index in range(3)
    ]


def test_built_keys_survive_a_write_and_a_read(tmp_path):
    sidecar = _built_sidecar()
    path = tmp_path / _SIDECAR.name
    path.write_text(json.dumps(sidecar_to_dict(sidecar)), encoding="utf-8")
    read = read_sidecar(path)
    assert read == sidecar
    assert [key.built for line in read.lines if line.line_kind == "chapter"
            for key in line.provenance] == [BuiltTrait("trigger", i) for i in range(3)]

"""The value head's targets are keyed by line, not by text (task G13b, finding 1.10).

Two lines can share an encoding text (the prose of a trigger whose script
differs by a granted ability) and so one row of ``e``; each has its own targets,
so the head reads that row once per line.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from effects.application.head_target_builders import value_batch
from effects.application.surface_batching import BatchLines
from effects.domain.provenance import ProvenanceKey, SidecarLine
from effects.domain.value_targets import VALUE_TARGET_NAMES, ValueLine

_SIDECARS = Path(__file__).parents[3] / "fixtures" / "effects" / "gen3-sidecars" / "cardsfolder"
_KEY = ProvenanceKey("cardsfolder/s/shock.txt", 0, "spell", 0)
_BACKUP = "backup 1"
_DRAWN = VALUE_TARGET_NAMES.index("cards_drawn")


def _sidecar_line(card: str, line_index: int) -> SidecarLine:
    """The fixture card's sidecar line at ``line_index``."""
    data = json.loads((_SIDECARS / f"{card}.provenance.json").read_text(encoding="utf-8"))
    (found,) = (line for line in data["lines"] if line["line_index"] == line_index)
    return SidecarLine(
        line_index=line_index, line_kind=found["line_kind"], provenance=(_KEY,),
        script_api_type=found["script_api_type"], script_text=found["script_text"],
        prose_offset=found["prose_offset"],
    )


#: Golden Scale Aeronaut's and Scorn Blade Berserker's backup 1: one prose text,
#: two scripts, the second granting an ability that draws (which the trigger's
#: own targets leave out: G13d).
_AERONAUT = _sidecar_line("g/golden_scale_aeronaut", 4)
_BERSERKER = _sidecar_line("s/scorn_blade_berserker", 4)


def test_two_lines_sharing_a_text_are_two_rows_reading_one_row_of_e():
    lines = [ValueLine.of(_BACKUP, _AERONAUT), ValueLine.of(_BACKUP, _BERSERKER)]
    targets = value_batch(lines, {_BACKUP: 3})
    values, mask, rows = targets.values, targets.mask, targets.matrix_rows
    assert rows.tolist() == [3, 3]
    assert rows.dtype == torch.long
    assert values.shape[0] == mask.shape[0] == 2
    assert values[0].tolist() == values[1].tolist()
    assert float(values[1, _DRAWN]) == 0.0, "the granted draw is the granted ability's, G13d"


def test_the_batch_keeps_each_script_a_text_has_not_the_first_met():
    found = BatchLines()
    found.note(_BACKUP, _BERSERKER)
    found.note(_BACKUP, _AERONAUT)
    found.note(_BACKUP, _BERSERKER)
    assert list(found.texts) == [_BACKUP]
    assert found.texts[_BACKUP] is _BERSERKER
    assert len(found.value_lines()) == 2
    assert {line.script for line in found.value_lines()} == {
        _AERONAUT.script_text, _BERSERKER.script_text,
    }


def test_the_batch_orders_its_lines_the_same_whichever_it_met_first():
    forward, backward = BatchLines(), BatchLines()
    for line in (_AERONAUT, _BERSERKER):
        forward.note(_BACKUP, line)
    for line in (_BERSERKER, _AERONAUT):
        backward.note(_BACKUP, line)
    assert forward.value_lines() == backward.value_lines()

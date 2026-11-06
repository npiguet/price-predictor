"""Which of a rule's map outputs its tagged records can carry (FR-060a; research R1).

The rules-to-fields map names, per rule, the outputs its outcome moves. A
record of the kind a rule's detector tags may carry none of them: a
``trigger`` record holds no per-entity target (what the trigger does arrives
as its resolution's record), a cost half holds no events, a decision record
supervises its verdict and target legality and nothing a resolution changes.
A rule's **effective outputs** are the map's outputs at least one of its
tagged records carries; ``build-corpus`` writes them to the manifest beside
the map's, and the rule mask and the rule tests read them alone.

A record carries:

- a per-entity output (a field of the per-entity head) where it supervises
  the field at one of the tag's outcome entities, by the presence training
  reads (``effect_targets.derive_targets``, ``FieldSpec.supervised_value``:
  the field written or silent under its scope and the record's sampling
  class), a gated field only where the entity is affected;
- the affected gate where the record trains the gate (it is no playability
  record) and the tag names an outcome entity the snapshot holds;
- a verdict or global field where the record's row masks it in;
  ``created_objects`` where the record supervises the created-objects head;
- an output no shipped head carries never.

Torch-free: the ``build-corpus`` survey workers call it.
"""

from __future__ import annotations

from collections.abc import Iterable

from effects.domain.aux_head_targets import (
    global_targets,
    supervises_created_objects,
    verdict_targets,
)
from effects.domain.effect_head_layout import FIELDS_BY_NAME, GLOBAL_LAYOUT, VERDICT_LAYOUT
from effects.domain.effect_targets import RecordTargets, derive_targets
from effects.domain.records import EffectRecord
from effects.domain.rule_mask import CREATED_OBJECTS_OUTPUT, GATE_OUTPUT


def carried_outputs(
    record: EffectRecord, outcome: Iterable[str], outputs: Iterable[str], *,
    targets: RecordTargets | None = None,
) -> frozenset[str]:
    """The outputs of ``outputs`` ``record`` carries for a tag naming
    ``outcome`` as its outcome entities; ``targets`` is the record's
    ``derive_targets`` when the caller already holds it."""
    outputs = tuple(outputs)
    if not outputs:
        return frozenset()
    outcome = tuple(outcome)
    on_board = {e.id for e in record.state.entities} | {p.id for p in record.state.players}
    outcome = tuple(e for e in outcome if e in on_board)
    verdict = global_ = None
    found: set[str] = set()
    for output in outputs:
        if output == GATE_OUTPUT or output in FIELDS_BY_NAME:
            if not outcome:
                continue
            if targets is None:
                targets = derive_targets(record)
            if _entity_carries(targets, output, outcome):
                found.add(output)
        elif output == CREATED_OBJECTS_OUTPUT:
            if supervises_created_objects(record):
                found.add(output)
        elif output in VERDICT_LAYOUT.by_name:
            if verdict is None:
                verdict = verdict_targets(record) or False
            if verdict and verdict[1][VERDICT_LAYOUT.index[output]]:
                found.add(output)
        elif output in GLOBAL_LAYOUT.by_name:
            if global_ is None:
                global_ = global_targets(record) or False
            if global_ and global_[1][GLOBAL_LAYOUT.index[output]]:
                found.add(output)
    return frozenset(found)


def _entity_carries(targets: RecordTargets, output: str, outcome: tuple[str, ...]) -> bool:
    if output == GATE_OUTPUT:
        return targets.gated
    spec = FIELDS_BY_NAME[output]
    for entity in outcome:
        entry = targets.get(entity)
        if entry is None or (spec.gated and not entry.affected):
            continue
        if entry.supervised(spec) is not None:
            return True
    return False


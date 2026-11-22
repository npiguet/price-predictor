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

A per-entity output (a per-entity field or the gate, :func:`per_entity_output`)
is read at entities, every other at the record; a rule test of a rule whose
effective outputs are all per-entity (:func:`read_at_entities_only`) reads
nothing on a tag naming no outcome entity.

Torch-free: the ``build-corpus`` survey workers call it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

from effects.domain.aux_head_targets import (
    global_targets,
    supervises_created_objects,
    verdict_targets,
)
from effects.domain.effect_head_layout import (
    FIELDS_BY_NAME,
    GLOBAL_LAYOUT,
    VERDICT_LAYOUT,
    FieldSpec,
    RecordHeadLayout,
)
from effects.domain.effect_targets import RecordTargets, derive_targets
from effects.domain.records import EffectRecord
from effects.domain.rule_mask import CREATED_OBJECTS_OUTPUT, GATE_OUTPUT


def per_entity_output(output: str) -> bool:
    """Whether ``output`` is read at an entity: a per-entity field or the
    affected gate, never a verdict, global or created-objects output."""
    return output == GATE_OUTPUT or output in FIELDS_BY_NAME


def read_at_entities_only(outputs: Iterable[str]) -> bool:
    """Whether every one of a rule's ``outputs`` is read at an entity, so a
    rule test reads it nowhere on a tag naming no outcome entity; False for
    no output at all."""
    outputs = tuple(outputs)
    return bool(outputs) and all(per_entity_output(output) for output in outputs)


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
    carrier = _Carrier(record, outcome, targets)
    return frozenset(output for output in outputs if carrier.carries(output))


class _Carrier:
    """One record's answers to "does it carry this output for the tag": the
    outcome entities on its board, and the targets, verdict and global rows
    each derived once, the first time an output needs them."""

    def __init__(self, record: EffectRecord, outcome: Iterable[str],
                 targets: RecordTargets | None) -> None:
        self.record = record
        on_board = {e.id for e in record.state.entities} | {p.id for p in record.state.players}
        self.outcome = tuple(e for e in tuple(outcome) if e in on_board)
        self.targets = targets
        self.rows: dict[Callable, tuple | bool] = {}

    def carries(self, output: str) -> bool:
        if per_entity_output(output):
            return self._carries_at_entities(output)
        if output == CREATED_OBJECTS_OUTPUT:
            return supervises_created_objects(self.record)
        if output in VERDICT_LAYOUT.by_name:
            return self._flagged(verdict_targets, VERDICT_LAYOUT, output)
        if output in GLOBAL_LAYOUT.by_name:
            return self._flagged(global_targets, GLOBAL_LAYOUT, output)
        return False

    def _carries_at_entities(self, output: str) -> bool:
        if not self.outcome:
            return False
        if self.targets is None:
            self.targets = derive_targets(self.record)
        return _entity_carries(self.targets, output, self.outcome)

    def _flagged(self, derive: Callable, layout: RecordHeadLayout, output: str) -> bool:
        """Whether the record's row of a verdict or global head masks ``output`` in."""
        if derive not in self.rows:
            self.rows[derive] = derive(self.record) or False
        row = self.rows[derive]
        return bool(row and row[1][layout.index[output]])


def _entity_carries(targets: RecordTargets, output: str, outcome: tuple[str, ...]) -> bool:
    if output == GATE_OUTPUT:
        return targets.gated
    spec = FIELDS_BY_NAME[output]
    return any(_supervised_at(targets, entity, spec) for entity in outcome)


def _supervised_at(targets: RecordTargets, entity: str, spec: FieldSpec) -> bool:
    entry = targets.get(entity)
    if entry is None or _unaffected_gated(entry, spec):
        return False
    return entry.supervised(spec) is not None


def _unaffected_gated(entry, spec: FieldSpec) -> bool:
    return spec.gated and not entry.affected


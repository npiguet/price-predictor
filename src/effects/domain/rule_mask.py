"""The rule mask (FR-061, research T1): how much one copy of a curated record
trains each output.

Rule budgets add copies of a record for one rule (FR-054). Unmasked, an added
copy repeats every other output the record carries and distorts them; the
mask decides, per copy, which outputs train and at what weight, from the
copy's ``curation`` and the outputs the manifest's rule map names per rule
(FR-060a).

``hard`` (the default): an added copy trains the outputs of every rule its
record was drawn for, and the affected gate on its drawn rule's own entities
only; every other output is masked. ``soft``: every copy trains every output,
outputs other than the drawn rules' weighing ``natural_share / copies``.
Under both a natural copy of a record no copy was drawn for trains everything
at full weight. A natural copy of an aimed record (a directed fork's, or one
collected aimed at a rule) trains everything at full weight; its added copies
are masked like any other. A record with no ``curation`` (a lethality copy, a
record no build curated) trains at full weight.

An output is named as the map names it: a per-entity field, a verdict or
global field, ``created_objects`` for the whole created-objects head, or
``gate`` for the affected gate. The training-only value, script-API and MLM
heads read encoded texts, not records, and no mask reaches them.

Torch-free, so the surface collation and the heads' targets read one answer.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

#: The affected gate, as the rule map names it.
GATE_OUTPUT = "gate"
#: The created-objects head, which the map names whole.
CREATED_OBJECTS_OUTPUT = "created_objects"


class RuleMask(StrEnum):
    """``--rule-mask``: how an added copy trains the outputs its rule does not
    name (FR-061)."""

    HARD = "hard"
    SOFT = "soft"


@dataclass(frozen=True, slots=True)
class CopyWeights:
    """One copy's weight on each output.

    ``full`` names the outputs that train at weight one, ``elsewhere`` is
    every other output's weight. ``gate_entities``, when set, are the only
    entities the affected gate trains on, at weight one; None trains the gate
    on every entity at its output weight.
    """

    elsewhere: float = 1.0
    full: frozenset[str] = frozenset()
    gate_entities: frozenset[str] | None = None

    def weight(self, output: str) -> float:
        """The weight of ``output`` on this copy."""
        return 1.0 if output in self.full else self.elsewhere

    def gate_weight(self, subject: str | None) -> float:
        """The affected gate's weight at one entity of this copy."""
        if self.gate_entities is None:
            return self.weight(GATE_OUTPUT)
        return 1.0 if subject in self.gate_entities else 0.0

    @property
    def unmasked(self) -> bool:
        """Whether every output trains at full weight."""
        return self.elsewhere == 1.0 and self.gate_entities is None


#: A copy every output trains at full weight.
FULL_WEIGHT = CopyWeights()


def copy_weights(
    record, mode: RuleMask | None, rule_outputs: Mapping[str, tuple[str, ...]],
) -> CopyWeights:
    """How ``record``'s copy trains each output under ``mode`` (FR-061);
    None trains every record at full weight, as evaluation reads them.

    ``rule_outputs`` is the manifest's ``rule -> outputs`` (FR-060a). An
    added copy whose drawn rule names no output the record carries trains
    nothing but the gate on its rule's entities, since a masked-in output
    with no target trains nothing; one whose rule names no entity (an
    outcome on a created token, a ``does_not_fire`` tag) trains no gate.
    """
    curation = getattr(record, "curation", None)
    if mode is None or curation is None:
        return FULL_WEIGHT
    if not curation.is_added and getattr(record, "aimed_at", None):
        return FULL_WEIGHT
    full = frozenset(
        output for rule in curation.drawn_rules for output in rule_outputs.get(rule, ())
    )
    if RuleMask(mode) is RuleMask.HARD:
        if not curation.is_added:
            return FULL_WEIGHT
        return CopyWeights(0.0, full, frozenset(curation.rule_entities))
    share = min(curation.natural_share / max(curation.copies, 1), 1.0)
    if share == 1.0:
        return FULL_WEIGHT
    return CopyWeights(share, full)

"""The rule mask (FR-061) through the batcher."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import torch
import torch.nn.functional as functional

from effects.application import step_preparation
from effects.domain.effect_head_layout import (
    CREATED_OBJECTS_WIDTH,
    DISPERSED_FIELDS,
    FIELD_SLICES,
    FIELDS_BY_NAME,
    GATE_INDEX,
    PER_ENTITY_WIDTH,
    VERDICT_LAYOUT,
)
from effects.domain.effect_losses import EntityLossScoring, _gate_term, per_entity_loss
from effects.domain.input_layout import EffectHeadInput, Slot, SlotKind
from effects.domain.records import (
    Curation,
)
from effects.domain.rule_mask import FULL_WEIGHT, RuleMask, copy_weights
from effects.domain.surface_collation import entity_target_tensors

# ── the rule mask (FR-061) ──


class _Entry:
    """One entity's targets: affected, and each field's value where present."""

    def __init__(self, values: dict, *, affected: bool = True) -> None:
        self.affected = affected
        self.values = values

    def supervised(self, spec):
        return self.values.get(spec.name)


_MASK_ENTITIES = ("E1", "E2", "P0")
_MASK_FIELDS = tuple(
    FIELDS_BY_NAME[name]
    for name in ("zone_outcome", "damage_taken", "life_delta", "poison_delta")
)
#: The manifest's outputs per rule (FR-060a).
_RULE_OUTPUTS = {"702.2": ("zone_outcome",), "702.15": ("poison_delta",)}


def _mask_surface() -> EffectHeadInput:
    slots = (
        Slot(SlotKind.CARD, 0, entity_id="E1"),
        Slot(SlotKind.CARD, 1, entity_id="E2"),
        Slot(SlotKind.PLAYER, 2, player_id="P0"),
    )
    return EffectHeadInput(slots=slots, entity_slots=(0, 1, 2))


def _mask_targets() -> dict:
    """Two creatures that died to damage, and a player who lost life."""
    return {
        "E1": _Entry({"zone_outcome": 1, "damage_taken": 2.0}),
        "E2": _Entry({"zone_outcome": 1, "damage_taken": 3.0}),
        "P0": _Entry({"life_delta": -2.0}),
    }


@dataclass(frozen=True)
class _Copy:
    """One curated copy of a record, as the rule mask reads it."""

    natural_share: int
    copies: int
    drawn_rule: str | None = None
    drawn_rules: tuple = ()
    rule_entities: tuple = ()
    aimed_at: tuple | None = None

    @property
    def curation(self) -> Curation:
        return Curation(self.natural_share, self.copies, self.drawn_rule,
                        tuple(self.drawn_rules), tuple(self.rule_entities))


def _mask_batch(records, mode, targets=None):
    weights = [copy_weights(record, mode, _RULE_OUTPUTS) for record in records]
    return entity_target_tensors(
        [_mask_surface()] * len(records),
        [targets or _mask_targets() for _ in records], _MASK_FIELDS, weights,
    )


def _gradients(records, mode) -> torch.Tensor:
    """The per-entity loss's gradient at each output column of a batch of
    copies of one board, under ``mode``."""
    outputs = torch.zeros(
        len(records), len(_MASK_ENTITIES), PER_ENTITY_WIDTH, requires_grad=True,
    )
    loss, _parts = per_entity_loss(
        outputs, _mask_batch(records, mode),
        EntityLossScoring(fields=_MASK_FIELDS, log_dispersion=torch.zeros(len(DISPERSED_FIELDS))),
    )
    loss.backward()
    return outputs.grad


def _output_columns(output: str) -> slice:
    if output == "gate":
        return slice(GATE_INDEX, GATE_INDEX + 1)
    return slice(*FIELD_SLICES[output])


def _trains(gradient: torch.Tensor, row: int, output: str) -> set[str]:
    """The entities at which ``output``'s columns take a gradient on ``row``."""
    return {
        entity for column, entity in enumerate(_MASK_ENTITIES)
        if gradient[row, column, _output_columns(output)].abs().sum() > 0
    }


def test_a_hard_masked_copy_drawn_for_702_2_trains_only_its_rules_outputs():
    """Story 7 acceptance 1: only the outputs the map names for 702.2, plus
    the affected gate on the rule's own entities; a natural copy of the same
    record trains everything."""
    added = _Copy(natural_share=1, copies=2, drawn_rule="702.2",
                  drawn_rules=("702.2",), rule_entities=("E1",))
    natural = _Copy(natural_share=1, copies=2, drawn_rules=("702.2",))
    gradient = _gradients([added, natural], RuleMask.HARD)

    assert _trains(gradient, 0, "zone_outcome") == {"E1", "E2"}
    assert _trains(gradient, 0, "gate") == {"E1"}
    assert _trains(gradient, 0, "damage_taken") == set()
    assert _trains(gradient, 0, "life_delta") == set()
    assert _trains(gradient, 1, "zone_outcome") == {"E1", "E2"}
    assert _trains(gradient, 1, "damage_taken") == {"E1", "E2"}
    assert _trains(gradient, 1, "life_delta") == {"P0"}
    assert _trains(gradient, 1, "gate") == set(_MASK_ENTITIES)


def test_a_hard_masked_copy_trains_the_gate_on_its_rules_entities_alone():
    """The gate's mask lives in ``effect_losses._gate_term``, apart from the
    host-built field masks (plan: easy to lose): the copy's gate loss is the
    BCE at its rule's entities, the others adding neither loss nor gradient."""
    added = _Copy(natural_share=0, copies=1, drawn_rule="702.2",
                  drawn_rules=("702.2",), rule_entities=("E2", "P0"))
    assert _trains(_gradients([added], RuleMask.HARD), 0, "gate") == {"E2", "P0"}

    targets = _mask_targets()
    targets["P0"].affected = False
    logits = torch.tensor([0.5, -1.0, 2.0])
    flat = torch.zeros(len(_MASK_ENTITIES), PER_ENTITY_WIDTH)
    flat[:, GATE_INDEX] = logits
    expected = functional.binary_cross_entropy_with_logits(
        logits[1:], torch.tensor([1.0, 0.0]), reduction="sum",
    )
    gate = _gate_term(flat, _mask_batch([added], RuleMask.HARD, targets))
    assert torch.allclose(gate, expected)


def test_a_hard_masked_copy_whose_rule_names_no_entity_trains_no_gate():
    """A detector naming no entity (an outcome on a created token, a
    ``does_not_fire`` tag) leaves the gate nothing to train on; the rule's
    outputs still train."""
    added = _Copy(natural_share=0, copies=1, drawn_rule="702.2",
                  drawn_rules=("702.2",), rule_entities=())
    gradient = _gradients([added], RuleMask.HARD)
    assert _trains(gradient, 0, "gate") == set()
    assert _trains(gradient, 0, "zone_outcome") == {"E1", "E2"}


def test_a_hard_masked_copy_whose_rule_names_a_missing_output_trains_only_the_gate():
    """Edge case: 702.15 names ``poison_delta``, which this record does not
    carry, so its copy trains the gate on the rule's entities and nothing
    else."""
    added = _Copy(natural_share=1, copies=2, drawn_rule="702.15",
                  drawn_rules=("702.15",), rule_entities=("E1",))
    gradient = _gradients([added], RuleMask.HARD)
    assert _trains(gradient, 0, "gate") == {"E1"}
    for output in ("zone_outcome", "damage_taken", "life_delta", "poison_delta"):
        assert _trains(gradient, 0, output) == set(), output


def test_a_soft_masked_record_weighs_other_outputs_n_over_d():
    """Story 7 acceptance 2: natural share 1, written 4 times; outputs other
    than those of its drawn rules weigh 1/4 on each copy, natural and added,
    its drawn rules' outputs 1 on every copy."""
    natural = _Copy(natural_share=1, copies=4, drawn_rules=("702.2",))
    added = _Copy(natural_share=1, copies=4, drawn_rule="702.2",
                  drawn_rules=("702.2",), rule_entities=("E1",))
    soft = _gradients([natural, added], RuleMask.SOFT)
    full = _gradients([natural, added], None)

    for row in (0, 1):
        for output in ("gate", "damage_taken", "life_delta"):
            columns = _output_columns(output)
            assert torch.allclose(soft[row, :, columns], full[row, :, columns] / 4), output
        columns = _output_columns("zone_outcome")
        assert soft[row, :, columns].abs().sum() > 0
        assert torch.allclose(soft[row, :, columns], full[row, :, columns])


def test_an_uncurated_record_trains_at_full_weight():
    """A lethality copy or an uncurated record carries no curation."""
    uncurated = SimpleNamespace(curation=None, aimed_at=None)
    for mode in RuleMask:
        assert copy_weights(uncurated, mode, _RULE_OUTPUTS) is FULL_WEIGHT


def test_an_aimed_natural_copy_follows_n_over_d_under_soft_and_trains_fully_under_hard():
    """D15: a directed fork's natural copy is a natural record like any other
    (FR-061); ``aimed_at: []`` (a coverage or variants run naming no rule) is
    aimed all the same (FR-009), and neither is exempt from the soft share."""
    for aimed_at in (("704.5f",), ()):
        aimed = _Copy(natural_share=1, copies=4, drawn_rules=("702.2",), aimed_at=aimed_at)
        assert copy_weights(aimed, RuleMask.HARD, _RULE_OUTPUTS) is FULL_WEIGHT
        assert copy_weights(aimed, RuleMask.SOFT, _RULE_OUTPUTS).elsewhere == 0.25
    natural = _Copy(natural_share=1, copies=4, drawn_rules=("702.2",), aimed_at=None)
    assert copy_weights(natural, RuleMask.SOFT, _RULE_OUTPUTS).elsewhere == 0.25


def test_an_aimed_records_added_copy_is_masked_like_any_other():
    added = _Copy(natural_share=1, copies=2, drawn_rule="702.2", drawn_rules=("702.2",),
                  rule_entities=("E1",), aimed_at=("704.5f",))
    assert copy_weights(added, RuleMask.HARD, _RULE_OUTPUTS).elsewhere == 0.0


def test_the_verdict_head_takes_the_rule_mask():
    """A verdict or global field is an output the map names too (FR-060a)."""
    names = [str(spec.name) for spec in VERDICT_LAYOUT.fields]
    outputs = {"603.2": ("trigger_fired",)}
    hard = _Copy(natural_share=1, copies=2, drawn_rule="603.2", drawn_rules=("603.2",))
    soft = _Copy(natural_share=1, copies=4, drawn_rules=("603.2",))
    weights = [copy_weights(hard, RuleMask.HARD, outputs),
               copy_weights(soft, RuleMask.SOFT, outputs)]
    every = [True] * len(names)

    _target, mask = step_preparation._record_head_rows(
        [hard, soft], weights, VERDICT_LAYOUT,
        lambda record: (VERDICT_LAYOUT.blank()[0], every),
    )
    fired = names.index("trigger_fired")
    assert mask[0].tolist() == [1.0 if i == fired else 0.0 for i in range(len(names))]
    assert mask[1].tolist() == [1.0 if i == fired else 0.25 for i in range(len(names))]


def test_the_created_objects_head_is_masked_whole(monkeypatch):
    monkeypatch.setattr(step_preparation, "supervises_created_objects", lambda record: True)
    monkeypatch.setattr(
        step_preparation, "created_objects_targets",
        lambda record: ([1.0] * CREATED_OBJECTS_WIDTH, [True] * CREATED_OBJECTS_WIDTH),
    )
    outputs = {"111.5": ("created_objects",), "702.2": ("zone_outcome",)}
    named = _Copy(natural_share=0, copies=1, drawn_rule="111.5", drawn_rules=("111.5",))
    other = _Copy(natural_share=0, copies=1, drawn_rule="702.2", drawn_rules=("702.2",))
    soft = _Copy(natural_share=1, copies=4, drawn_rules=("702.2",))
    weights = [copy_weights(named, RuleMask.HARD, outputs),
               copy_weights(other, RuleMask.HARD, outputs),
               copy_weights(soft, RuleMask.SOFT, outputs)]
    _target, mask, known = step_preparation._created_objects_rows(
        [named, other, soft], weights,
    )
    assert mask.tolist() == [True, False, True]
    assert known[0].unique().tolist() == [1.0]
    assert known[2].unique().tolist() == [0.25]

"""Gate 2 reads the model through the one chunked forward, on pairs that differ
in the keyword alone (13.4, 13.7).

The pairs are the fixture's real combats, each built twice in one batch: as
collected, and with the carrier stripped of the row's keyword. Every collated
tensor of the pair is diffed slot by slot: the twin may differ from the record
only at the carrier's keyword flags and its lines (a dropped slot, the
positions after it, the links that ended at it), so nothing else, the damage
assignment above all, states the keyword's consequence in the twin.
"""

from __future__ import annotations

import dataclasses
from contextlib import contextmanager

import pytest
import torch

from effects.application import gate_two
from effects.application.model_loading import Runnable
from effects.application.record_forward import RecordForward
from effects.domain.card_features import card_segments
from effects.domain.damage_step_keywords import (
    DAMAGE_STEP_KEYWORDS,
    KeywordResolver,
    qualifying_observations,
)
from effects.domain.input_layout import SlotKind
from effects.domain.surface_builder import NO_STRIP
from effects.domain.surface_collation import TrunkInputs, entity_index
from effects.infrastructure.record_codec import record_from_dict
from tests.unit.effects.builders import gen3_fixture_records
from tests.unit.effects.training_builders import fixture_run


@pytest.fixture(scope="module")
def run(corpus, records, tmp_path_factory):
    return fixture_run(corpus, records, tmp_path_factory.mktemp("gate-two"))


def _batcher(run):
    batcher = run.preparer.scoring_batcher()
    batcher.materializer = run.scoring
    return batcher


def _observations(batcher) -> list[gate_two.Observation]:
    resolver = KeywordResolver(batcher.sidecars)
    return [
        gate_two.Observation(record, row, participant)
        for record in map(record_from_dict, gen3_fixture_records())
        for row, participant in qualifying_observations(record, DAMAGE_STEP_KEYWORDS, resolver)
    ]


@pytest.fixture(scope="module")
def observations(run):
    found = _observations(_batcher(run))
    assert {o.row.keyword for o in found} >= {"first_strike", "trample"}
    return found


def _strip(observation) -> dict[str, frozenset[str]]:
    return {observation.participant.carrier.id: frozenset({observation.row.keyword})}


# ── the twin differs in the keyword alone ────────────────────────────────


def _kept(whole, twin) -> list[int]:
    """The record's slot each twin slot is, in order; every other slot of the
    record is one the strip dropped."""
    def same(a, b):
        return (a.kind, a.entity_id, a.player_id, a.e_row, a.e_vector, a.option) == (
            b.kind, b.entity_id, b.player_id, b.e_row, b.e_vector, b.option)

    kept, at = [], 0
    for slot in twin.slots:
        while not same(whole.slots[at], slot):
            at += 1
        kept.append(at)
        at += 1
    return kept


def _carrier_card(surface, carrier: str) -> int:
    return next(index for index, slot in enumerate(surface.slots)
                if slot.kind is SlotKind.CARD and slot.entity_id == carrier)


def _differing(tensor: torch.Tensor, kept: list[int]) -> set[tuple[int, ...]]:
    """``(twin slot, column…)`` where the twin's row differs from the record's."""
    record, twin = tensor[0][kept], tensor[1][: len(kept)]
    return {tuple(index) for index in (record != twin).nonzero().tolist()}


def _slot_tensors(inputs: TrunkInputs) -> dict[str, torch.Tensor]:
    """Every per-slot tensor of the collation, by name."""
    found = {f"slot_features[{kind.name}]": tensor
             for kind, tensor in inputs.slot_features.items()}
    for spec in dataclasses.fields(TrunkInputs):
        value = getattr(inputs, spec.name)
        if isinstance(value, torch.Tensor) and spec.name != "relations":
            found[spec.name] = value
    return found


def _allowed(name: str, twin, carrier: str, flags: slice):
    """The differences the strip may make in one tensor: the carrier's keyword
    flags, and the positions of its lines after a dropped one."""
    card = _carrier_card(twin, carrier)
    if name == f"slot_features[{SlotKind.CARD.name}]":
        return lambda at: at[0] == card and flags.start <= at[1] < flags.stop
    if name == "positions":
        return lambda at: twin.slots[at[0]].entity_id == carrier and at[0] > card
    return lambda at: False


def _relations(inputs: TrunkInputs, row: int, kept=None) -> set[tuple[int, int, int]]:
    rows = inputs.relations[inputs.relations[:, 0] == row][:, 1:].tolist()
    if kept is None:
        return {tuple(link) for link in rows}
    where = {slot: index for index, slot in enumerate(kept)}
    return {(where[a], where[b], kind) for a, b, kind in rows if a in where and b in where}


def _pair(run, observation):
    return _batcher(run).prepare(
        [observation.record] * 2, [NO_STRIP, _strip(observation)])


def test_no_card_part_states_the_damage_assignment(observations):
    """G03 took the assignment out of the input, so the twin asks nothing of it."""
    entity = observations[0].participant.carrier
    assert not any("assign" in name for name in card_segments(entity, observations[0].record))


def test_the_twin_differs_from_its_record_in_the_carriers_keyword_alone(run, observations):
    for observation in observations:
        prepared = _pair(run, observation)
        whole, twin = prepared.surfaces
        carrier = observation.participant.carrier.id
        kept = _kept(whole, twin)
        dropped = [whole.slots[i] for i in range(len(whole.slots)) if i not in set(kept)]
        assert dropped, f"{observation.row.keyword}: the strip dropped no line"
        assert all(s.kind is SlotKind.ABILITY and s.entity_id == carrier for s in dropped)
        flags = card_segments(observation.participant.carrier, observation.record)[
            "keyword_flags"]
        moved = set()
        for name, tensor in _slot_tensors(prepared.inputs).items():
            allowed, differing = _allowed(name, twin, carrier, flags), _differing(tensor, kept)
            assert all(allowed(at) for at in differing), (name, sorted(differing)[:5])
            moved |= {name} if differing else set()
        assert f"slot_features[{SlotKind.CARD.name}]" in moved
        assert int(prepared.inputs.attention_mask[1].sum()) == len(kept)
        _assert_links_kept(prepared.inputs, kept, _carrier_card(twin, carrier))


def _assert_links_kept(inputs: TrunkInputs, kept: list[int], card: int) -> None:
    """Every link between kept slots stays; a new one only re-ends a dropped
    line's link at its card."""
    record, twin = _relations(inputs, 0, kept), _relations(inputs, 1)
    assert record <= twin
    assert all(card in (a, b) for a, b, _ in twin - record)


# ── through the one forward, inside running() ────────────────────────────


class _Watched(RecordForward):
    """The real forward, noting what gate 2 asks of it and the mode the
    modules are in when the heads are read."""

    def __init__(self, base: RecordForward) -> None:
        super().__init__(base.encoder, base.model, base.materializer)
        self.strips: list = []
        self.depth = 0

    @contextmanager
    def running(self):
        self.depth += 1
        try:
            with super().running():
                yield
        finally:
            self.depth -= 1

    def records(self, records, batcher, *, size=32, strips=None):
        self.strips += list(strips)
        yield from super().records(records, batcher, size=size, strips=strips)


@dataclasses.dataclass(frozen=True, slots=True)
class _WatchedRunnable(Runnable):
    watched: list = dataclasses.field(default_factory=list)

    def forward(self) -> RecordForward:
        self.watched.append(_Watched(Runnable.forward(self)))
        return self.watched[-1]


def _runnable(run, *, watched: bool = False) -> Runnable:
    kind = _WatchedRunnable if watched else Runnable
    return kind(run.encoder, run.model, _batcher(run), tuple(run.fields[1]))


def _depth(runnable: _WatchedRunnable) -> int:
    """How deep in ``running()`` the last forward gate 2 asked for is; 0 for none."""
    return runnable.watched[-1].depth if runnable.watched else 0


def test_gate_two_scores_its_pairs_through_the_one_forward(run, observations, monkeypatch):
    runnable = _runnable(run, watched=True)
    reads = []
    score = gate_two.observation_moves
    monkeypatch.setattr(gate_two, "observation_moves", lambda *a, **k: (
        reads.append((_depth(runnable), run.model.training, torch.is_grad_enabled())),
        score(*a, **k))[1])

    gate_two.score_keywords([o.record for o in observations], DAMAGE_STEP_KEYWORDS, runnable)

    assert runnable.watched, "gate 2 never ran the one forward"
    stripped = [strip for strip in runnable.watched[-1].strips if strip]
    assert stripped == [_strip(o) for o in observations]
    assert reads and all(depth > 0 and not training and not grad
                         for depth, training, grad in reads)
    assert run.model.training


def _reference(run, observation) -> tuple[torch.Tensor, torch.Tensor]:
    """The pair's per-entity head in float32, as gate 2 read it before it ran
    the one forward (``Runnable.per_entity_outputs``)."""
    run.encoder.eval()
    run.model.eval()
    with torch.inference_mode():
        built = _batcher(run).build(
            [observation.record] * 2, run.encoder, [NO_STRIP, _strip(observation)])
        hidden = run.model(built.inputs)
        outputs = run.model.per_entity_at(hidden, entity_index(built.surfaces)).float()
    run.encoder.train()
    run.model.train()
    return outputs[0], outputs[1]


def test_on_the_cpu_the_pairs_read_as_the_float32_path_did(run, observations):
    """No autocast on the CPU, so the one forward reads the very numbers the
    float32 path did."""
    forward = Runnable.forward(_runnable(run))
    for observation in observations:
        with forward.running():
            (chunk,) = forward.records(
                [observation.record] * 2, _batcher(run), size=2,
                strips=[NO_STRIP, _strip(observation)])
            got = chunk.at_entities()
        whole, twin = _reference(run, observation)
        assert torch.equal(got[0], whole) and torch.equal(got[1], twin)

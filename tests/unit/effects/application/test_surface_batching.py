"""``SurfaceBatcher`` hands the builder the sidecar's answer to "is this a line".

The rule itself lives in the domain builder; this pins the wiring, because a
batcher that forgot to pass the predicate would silently keep every phantom
token and every number it reports would still move.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from effects.application.surface_batching import NoiseState, SurfaceBatcher
from effects.application.train_effect_model import VariantMasks
from effects.domain.effect_head_input import SlotKind
from effects.domain.provenance import ProvenanceKey, SidecarLine
from effects.domain.records import (
    EffectRecord,
    Moment,
    PlayabilityDecisionPayload,
    RecordKind,
    ResolutionPayload,
)
from effects.domain.state_snapshot import EntityState, GlobalState, StateSnapshot
from effects.infrastructure.record_io import read_shard
from effects.infrastructure.sidecar_io import SidecarCache

_ACT = ProvenanceKey("cardsfolder/l/lightning_bolt.txt", 0, "spell", 0)
_ANTHEM = ProvenanceKey("cardsfolder/a/anthem.txt", 0, "static", 0)
_PHANTOM = ProvenanceKey("cardsfolder/a/anthem.txt", 0, "spell", 0)

_LINES = {
    _ACT: SidecarLine(line_index=0, line_kind="spell", provenance=(_ACT,),
                      script_text="deals 3 damage to any target"),
    _ANTHEM: SidecarLine(line_index=0, line_kind="static", provenance=(_ANTHEM,),
                         script_text="creatures you control get +1/+1"),
}


class _Sidecars:
    """The two answers a real cache gives: a line, or None for a dropped key."""

    def line_for(self, key):
        return _LINES.get(key)

    def prose_for(self, key):
        return None


def _record() -> EffectRecord:
    return EffectRecord(
        record_id="run.0.1", run_id="run", timestamp="t", game_id="run.0.1",
        kind=RecordKind.RESOLUTION, moment=Moment.RESOLUTION, actor_player="P0",
        ability=(_ACT,), payload=ResolutionPayload(),
        state=StateSnapshot(
            global_=GlobalState(turn=1, phase="main1", active="P0", priority="P0", stack_size=1),
            players=(),
            entities=(EntityState(id="E1", name="Anthem", zone="battlefield",
                                  controller="P0", printed=(_PHANTOM, _ANTHEM)),),
        ),
    )


def _batcher(**masks) -> SurfaceBatcher:
    return SurfaceBatcher(
        tokenizer=None, sidecars=_Sidecars(), masks=VariantMasks(**masks),
        surface="script", e_dim=4, widths={}, device=torch.device("cpu"),
    )


def test_a_dropped_key_gets_no_ability_token():
    rows = {"creatures you control get +1/+1": 0, "deals 3 damage to any target": 1}
    surface = _batcher().surface_for(_record(), rows)
    assert [slot.e for slot in surface.of_kind(SlotKind.ABILITY)] == [0]


def test_the_state_only_variant_keeps_the_zero_token_for_a_real_line():
    rows = {"creatures you control get +1/+1": 0, "deals 3 damage to any target": 1}
    surface = _batcher(zero_e=True).surface_for(_record(), rows)
    abilities = surface.of_kind(SlotKind.ABILITY)
    assert len(abilities) == 1
    assert abilities[0].e == (0.0,) * 4


class _Counting(_Sidecars):
    """Records every join, which is what the memo exists to stop repeating."""

    def __init__(self) -> None:
        self.calls: list = []

    def line_for(self, key):
        self.calls.append(key)
        return _LINES.get(key)


class _Mismatching(_Sidecars):
    """The sidecar disagrees with the record: the fail-loudly case."""

    def line_for(self, key):
        raise KeyError(
            f"provenance key {key} appears in neither the lines nor the "
            "dropped_keys"
        )


def test_a_key_is_joined_once_however_often_the_batch_asks():
    # Three keys, asked for by `batch_texts` and then three more times per
    # record inside `surface_for`, over two passes.
    sidecars = _Counting()
    batcher = _batcher()
    batcher.sidecars = sidecars
    records = [_record()]
    for _ in range(2):
        texts = batcher.batch_texts(records)
        rows = {text: row for row, text in enumerate(sorted(texts))}
        batcher.surface_for(records[0], rows)

    assert sidecars.calls == [_ACT, _PHANTOM, _ANTHEM]


def test_a_sidecar_mismatch_is_never_memoised():
    # A mismatch is the one answer the memo must not keep: swallowing it once
    # would turn "the sidecar does not describe this card" into no text at all.
    batcher = _batcher()
    batcher.sidecars = _Mismatching()
    for _ in range(3):
        with pytest.raises(KeyError, match="neither the lines"):
            batcher.text_of(_ACT)


def test_a_sidecar_mismatch_raises_out_of_the_batcher():
    batcher = _batcher()
    batcher.sidecars = _Mismatching()
    with pytest.raises(KeyError, match="neither the lines"):
        batcher.batch_texts([_record()])


# ── noise on e (FR-056, T089) ────────────────────────────────────────────

def _spread_matrix(rows: int = 4000, dim: int = 3) -> torch.Tensor:
    generator = torch.Generator().manual_seed(3)
    scale = torch.tensor([3.0, 1.0, 0.25])[:dim]
    return torch.randn(rows, dim, generator=generator) * scale


def test_the_added_noise_has_covariance_r_squared_sigma():
    torch.manual_seed(0)
    matrix = _spread_matrix()
    noise = NoiseState(ratio=0.5, ramp_steps=10)
    noise.step = 10
    added = noise.apply(matrix) - matrix
    # Sigma is held in float64 so the factorization stays exact.
    expected = 0.25 * noise.sigma.float()
    observed = torch.cov(added.T)
    assert torch.allclose(observed, expected, atol=0.05 * expected.abs().max())


def test_sigma_starts_from_the_first_batch_and_decays_at_0_99():
    noise = NoiseState(ratio=0.1, ramp_steps=1)
    first = _spread_matrix()
    noise.update(first)
    assert torch.allclose(noise.sigma.float(), torch.cov(first.T), atol=1e-4)
    second = first * 2.0
    noise.update(second)
    expected = 0.99 * torch.cov(first.T) + 0.01 * torch.cov(second.T)
    assert torch.allclose(noise.sigma.float(), expected, atol=1e-4)


def test_no_gradient_flows_through_sigma():
    matrix = _spread_matrix(rows=50).requires_grad_(True)
    noise = NoiseState(ratio=0.2, ramp_steps=1)
    noise.step = 1
    noise.apply(matrix).sum().backward()
    assert not noise.sigma.requires_grad
    # The noise is additive and detached: the gradient is the identity's.
    assert torch.equal(matrix.grad, torch.ones_like(matrix))


def test_the_ratio_ramps_linearly_over_the_first_epoch():
    noise = NoiseState(ratio=0.1, ramp_steps=5000)
    noise.step = 0
    assert noise.scale() == 0.0
    noise.step = 2500
    assert noise.scale() == pytest.approx(0.05)
    noise.step = 9000
    assert noise.scale() == pytest.approx(0.1)


def test_at_step_zero_nothing_is_added():
    matrix = _spread_matrix(rows=20)
    noise = NoiseState(ratio=0.1, ramp_steps=100)
    assert torch.equal(noise.apply(matrix), matrix)


def test_a_scoring_batcher_carries_no_noise():
    """Validation, evaluation and encoding never perturb e (scenario 4)."""
    assert _batcher().noise is None


# ── decision records and option rows (FR-060a, FR-063a) ─────────────────

_FIXTURES = Path(__file__).parents[3] / "fixtures" / "effects"


def _fixture_batcher() -> tuple[SurfaceBatcher, list[EffectRecord]]:
    records = list(read_shard(_FIXTURES / "gen1-records.jsonl.gz"))
    sidecars = SidecarCache({
        "cardsfolder": _FIXTURES / "gen1-sidecars" / "cardsfolder",
        "tokenscripts": _FIXTURES / "gen1-sidecars" / "tokenscripts",
    })
    batcher = SurfaceBatcher(
        tokenizer=None, sidecars=sidecars, masks=VariantMasks(),
        surface="script", e_dim=4, widths={}, device=torch.device("cpu"),
    )
    return batcher, records


def test_a_decision_records_act_slot_reads_its_candidates_e():
    batcher, records = _fixture_batcher()
    decisions = [
        r for r in records
        if isinstance(r.payload, PlayabilityDecisionPayload)
        and batcher.text_of(r.payload.candidates[0].ability[0]) is not None
    ]
    assert decisions, "the fixture holds a decision with a convertible candidate"
    record = decisions[0]
    texts = batcher.batch_texts([record])
    rows = {text: row for row, text in enumerate(sorted(texts))}
    surface = batcher.surface_for(record, rows)
    act = surface.slots[surface.act_index]
    candidate_text = batcher.text_of(record.payload.candidates[0].ability[0])
    assert act.e == rows[candidate_text]


def test_gen1_sidecars_add_no_option_rows():
    batcher, records = _fixture_batcher()
    texts = batcher.batch_texts(records)
    rows = {text: row for row, text in enumerate(sorted(texts))}
    for record in records:
        surface = batcher.surface_for(record, rows)
        assert not any(slot.option for slot in surface.slots)


def test_a_real_gen2_charm_gets_its_four_mode_rows_after_its_root():
    """FR-063a on Cryptic Command as the gen-2 converter writes it."""
    gen2 = _FIXTURES / "gen2-sidecars"
    sidecars = SidecarCache({"cardsfolder": gen2 / "cardsfolder"})
    root = ProvenanceKey("cardsfolder/c/cryptic_command.txt", 0, "spell", 0)
    batcher = SurfaceBatcher(
        tokenizer=None, sidecars=sidecars, masks=VariantMasks(),
        surface="script", e_dim=4, widths={}, device=torch.device("cpu"),
    )
    modes = batcher.options_for(root)
    assert [mode.option for mode in modes] == [0, 1, 2, 3]
    record = EffectRecord(
        record_id="run.0.1", run_id="run", timestamp="t", game_id="run.0.1",
        kind=RecordKind.RESOLUTION, moment=Moment.RESOLUTION, actor_player="P0",
        ability=(root,), payload=ResolutionPayload(),
        state=StateSnapshot(
            global_=GlobalState(turn=1, phase="main1", active="P0", priority="P0",
                                stack_size=1),
            players=(),
            entities=(EntityState(id="E1", name="Cryptic Command", zone="stack",
                                  controller="P0", printed=(root,)),),
        ),
    )
    texts = batcher.batch_texts([record])
    assert len(texts) == 5  # the root and each mode's own chain
    rows = {text: row for row, text in enumerate(sorted(texts))}
    abilities = batcher.surface_for(record, rows).of_kind(SlotKind.ABILITY)
    assert [slot.option for slot in abilities] == [False, True, True, True, True]
    assert [slot.e for slot in abilities[1:]] == [
        rows[batcher.text_of(mode)] for mode in modes
    ]


def test_a_rank_deficient_batch_under_bf16_autocast_still_draws_noise():
    """The training condition that broke the Cholesky factor.

    Training runs under bf16 autocast, and a batch often carries fewer unique
    texts than ``e`` has dimensions, so the batch covariance is singular. The
    covariance and the draw must leave autocast and stay exact, or the factor
    sees a matrix that is not positive semi-definite.
    """
    generator = torch.Generator().manual_seed(5)
    dim = 64
    # Twelve texts in a 64-wide space, at the scale a trained e reaches.
    matrix = torch.randn(12, dim, generator=generator) * 40.0
    noise = NoiseState(ratio=0.1, ramp_steps=1)
    noise.step = 1
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        for _ in range(20):
            out = noise.apply(matrix)
    assert torch.isfinite(out).all()
    assert out.dtype == matrix.dtype
    assert torch.allclose(noise.sigma, noise.sigma.T)

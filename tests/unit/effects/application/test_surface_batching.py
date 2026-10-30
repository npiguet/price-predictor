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
from effects.application.text_staging import MlmMasker
from effects.domain.baselines import BaselineMasks
from effects.domain.effect_head_input import ACT_SLOT, SlotKind
from effects.domain.provenance import ProvenanceKey, SidecarLine
from effects.domain.records import (
    EffectRecord,
    Moment,
    PlayabilityDecisionPayload,
    RecordKind,
    ResolutionPayload,
)
from effects.domain.state_snapshot import EntityState, GlobalState, StateSnapshot
from effects.infrastructure.shard_io import read_shard
from effects.infrastructure.sidecar_io import SidecarCache
from tests.unit.effects.builders import (
    make_batcher,
    make_entity,
    make_record,
    make_snapshot,
    slots_of_kind,
)

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

    def get(self, script_file):
        raise KeyError(f"{script_file}: no sidecar, so no charm options")


def _record() -> EffectRecord:
    return make_record(
        record_id="run.0.1", run_id="run", game_id="run.0.1", ability=(_ACT,),
        state=make_snapshot(
            global_=GlobalState(turn=1, phase="main1", active="P0", priority="P0", stack_size=1),
            entities=(make_entity("E1", "Anthem", printed=(_PHANTOM, _ANTHEM)),),
        ),
    )


def _batcher(sidecars=None, **masks) -> SurfaceBatcher:
    return make_batcher(
        tokenizer=None, sidecars=sidecars or _Sidecars(), masks=BaselineMasks(**masks),
        surface="script", e_dim=4, widths={}, device=torch.device("cpu"),
    )


def test_a_dropped_key_gets_no_ability_token():
    rows = {"creatures you control get +1/+1": 0, "deals 3 damage to any target": 1}
    surface = _batcher().surface_builder(rows).build(_record())
    assert [slot.e for slot in slots_of_kind(surface, SlotKind.ABILITY)] == [0]


def _prepared(variant: str):
    """One record prepared under ``variant``'s masks, through the real pipeline."""
    from effects.domain.ability_tokenizer import AbilityTokenizer
    from effects.domain.baselines import baseline_masks
    from effects.domain.effect_head_input import feature_widths

    record = _record()
    batcher = make_batcher(
        tokenizer=AbilityTokenizer({"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[MASK]": 3}),
        sidecars=_Sidecars(), masks=baseline_masks(variant), surface="script",
        e_dim=4, widths=feature_widths([record]), device=torch.device("cpu"),
    )
    prepared = batcher.prepare([record])
    return prepared.surfaces[0], prepared.inputs


def _columns(surface, kind):
    return [index for index, slot in enumerate(surface.slots) if slot.kind is kind]


def test_the_state_only_variant_keeps_the_zero_token_for_a_real_line():
    surface, inputs = _prepared("state-only")
    (ability,) = _columns(surface, SlotKind.ABILITY)
    assert inputs.e_rows[0, ability] == -1
    assert torch.all(inputs.e_vectors[0, ability] == 0.0)


def test_the_state_only_variant_zeroes_the_acting_lines_e():
    surface, inputs = _prepared("state-only")
    (act,) = _columns(surface, SlotKind.ACT)
    assert inputs.e_rows[0, act] == -1


def test_the_no_state_variant_keeps_the_acting_lines_e():
    """FR-094: every input zeroed except [ACT] and the record-kind flag."""
    surface, inputs = _prepared("no-state")
    (act,) = _columns(surface, SlotKind.ACT)
    (ability,) = _columns(surface, SlotKind.ABILITY)
    assert inputs.e_rows[0, act] >= 0
    assert inputs.e_rows[0, ability] == -1


def test_the_no_state_variant_keeps_only_the_record_kind_flags_of_global():
    from effects.domain.effect_head_input import KIND_FLAG_WIDTH

    full = _prepared("full")[1].slot_features[SlotKind.GLOBAL][0, 0]
    _surface, inputs = _prepared("no-state")
    blank = inputs.slot_features[SlotKind.GLOBAL][0, 0]
    assert torch.all(blank[:-KIND_FLAG_WIDTH] == 0.0)
    assert torch.equal(blank[-KIND_FLAG_WIDTH:], full[-KIND_FLAG_WIDTH:])
    assert full[:-KIND_FLAG_WIDTH].abs().sum() > 0
    assert torch.all(inputs.slot_features[SlotKind.CARD] == 0.0)


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
    # record inside `SurfaceBuilder.build`, over two passes.
    sidecars = _Counting()
    batcher = _batcher(sidecars)
    records = [_record()]
    for _ in range(2):
        texts = batcher.batch_texts(records)
        rows = {text: row for row, text in enumerate(sorted(texts))}
        batcher.surface_builder(rows).build(records[0])

    assert sidecars.calls == [_ACT, _PHANTOM, _ANTHEM]


def test_a_sidecar_mismatch_is_never_memoised():
    # A mismatch is the one answer the memo must not keep: swallowing it once
    # would turn "the sidecar does not describe this card" into no text at all.
    batcher = _batcher(_Mismatching())
    for _ in range(3):
        with pytest.raises(KeyError, match="neither the lines"):
            batcher.text_of(_ACT)


def test_a_sidecar_mismatch_raises_out_of_the_batcher():
    batcher = _batcher(_Mismatching())
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
    added = noise.apply(matrix, step=10) - matrix
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
    noise.apply(matrix, step=1).sum().backward()
    assert not noise.sigma.requires_grad
    # The noise is additive and detached: the gradient is the identity's.
    assert torch.equal(matrix.grad, torch.ones_like(matrix))


def test_the_ratio_ramps_linearly_over_the_first_epoch():
    noise = NoiseState(ratio=0.1, ramp_steps=5000)
    assert noise.scale(0) == 0.0
    assert noise.scale(2500) == pytest.approx(0.05)
    assert noise.scale(9000) == pytest.approx(0.1)


def test_at_step_zero_nothing_is_added():
    matrix = _spread_matrix(rows=20)
    noise = NoiseState(ratio=0.1, ramp_steps=100)
    assert torch.equal(noise.apply(matrix, step=0), matrix)


def test_a_resume_file_holding_the_old_step_key_still_loads():
    """Resume files written before the state dropped its unread ``step`` carry
    it; an interrupted run must continue across the change."""
    noise = NoiseState(ratio=0.1, ramp_steps=1)
    noise.apply(_spread_matrix(rows=20), step=1)
    saved = noise.state_dict()
    restored = NoiseState(ratio=0.1, ramp_steps=1)
    restored.load_state_dict({**saved, "step": 7}, torch.device("cpu"))
    assert torch.equal(restored.sigma, noise.sigma)
    assert torch.equal(restored._factor, noise._factor)
    assert restored._draws_since_factor == noise._draws_since_factor


def test_a_scoring_batcher_carries_no_noise(tmp_path):
    """Validation, evaluation and encoding never perturb e (scenario 4): the
    materializer a scoring batch is built with has none unless handed one."""
    from effects.application.surface_batching import Materializer
    from effects.domain.baselines import baseline_masks

    assert Materializer(torch.device("cpu"), baseline_masks("full")).noise is None


# ── decision records and option rows (FR-060a, FR-063a) ─────────────────

_FIXTURES = Path(__file__).parents[3] / "fixtures" / "effects"


def _fixture_batcher() -> tuple[SurfaceBatcher, list[EffectRecord]]:
    records = list(read_shard(_FIXTURES / "gen1-records.jsonl.gz"))
    sidecars = SidecarCache({
        "cardsfolder": _FIXTURES / "gen1-sidecars" / "cardsfolder",
        "tokenscripts": _FIXTURES / "gen1-sidecars" / "tokenscripts",
    })
    batcher = make_batcher(
        tokenizer=None, sidecars=sidecars, masks=BaselineMasks(),
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
    surface = batcher.surface_builder(rows).build(record)
    act = surface.slots[ACT_SLOT]
    candidate_text = batcher.text_of(record.payload.candidates[0].ability[0])
    assert act.e == rows[candidate_text]


def test_gen1_sidecars_add_no_option_rows():
    batcher, records = _fixture_batcher()
    texts = batcher.batch_texts(records)
    rows = {text: row for row, text in enumerate(sorted(texts))}
    for record in records:
        surface = batcher.surface_builder(rows).build(record)
        assert not any(slot.option for slot in surface.slots)


def test_a_real_gen2_charm_gets_its_four_mode_rows_after_its_root():
    """FR-063a on Cryptic Command as the gen-2 converter writes it."""
    gen2 = _FIXTURES / "gen2-sidecars"
    sidecars = SidecarCache({"cardsfolder": gen2 / "cardsfolder"})
    root = ProvenanceKey("cardsfolder/c/cryptic_command.txt", 0, "spell", 0)
    batcher = make_batcher(
        tokenizer=None, sidecars=sidecars, masks=BaselineMasks(),
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
    abilities = slots_of_kind(batcher.surface_builder(rows).build(record), SlotKind.ABILITY)
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
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        for _ in range(20):
            out = noise.apply(matrix, step=1)
    assert torch.isfinite(out).all()
    assert out.dtype == matrix.dtype
    assert torch.allclose(noise.sigma, noise.sigma.T)


# ── the MLM pass (FR-060b) ───────────────────────────────────────────────

class _MaskTokens:
    pad_id, mask_id = 0, 1


class _EchoEncoder:
    """Returns each position's (masked) token id as its hidden state."""

    def __call__(self, token_ids, **_rest):
        return None, token_ids.unsqueeze(-1).float().repeat(1, 1, 2)


def _mlm_lines(lengths, *, number: float | None = None) -> list:
    """Prepared lines of ``lengths`` tokens, each token after [CLS] a number
    when ``number`` is given."""
    from effects.domain.ability_encoder import EncodedLine

    return [
        EncodedLine(
            token_ids=[2] + list(range(10, 10 + n - 1)), role_ids=[0] * n,
            numbers=[0.0] + [number or 0.0] * (n - 1),
            number_mask=[0] + [int(number is not None)] * (n - 1),
        )
        for n in lengths
    ]


def _mlm_batcher() -> SurfaceBatcher:
    return make_batcher(tokenizer=_MaskTokens(), sidecars=_Sidecars())


def _mlm_pass(batcher, lines, mask_prob):
    masked = batcher.draw_mlm_mask(lines, mask_prob)
    if masked is None:
        return None
    return MlmMasker.forward(_EchoEncoder(), masked, torch.device("cpu"))


def test_the_mlm_pass_returns_only_the_masked_positions():
    batcher = _mlm_batcher()
    hidden, targets = _mlm_pass(batcher, _mlm_lines([6, 3, 9]), 0.5)
    # Every gathered row is a position the pass replaced with [MASK]...
    assert torch.all(hidden[:, 0] == _MaskTokens.mask_id)
    # ...and its target is the token that stood there: never [CLS] (id 2),
    # never padding (id 0).
    assert targets.numel() == hidden.shape[0] > 0
    assert torch.all(targets >= 10)


def test_the_mlm_pass_masks_nothing_at_probability_zero():
    assert _mlm_pass(_mlm_batcher(), _mlm_lines([5, 5]), 0.0) is None


def test_a_masked_number_carries_no_value():
    """The number embedding would otherwise hand the masked token's value back."""
    masked = _mlm_batcher().draw_mlm_mask(_mlm_lines([6, 4], number=3.0), 0.5)
    picks = masked.inputs["token_ids"] == _MaskTokens.mask_id
    assert picks.any()
    assert torch.all(masked.inputs["numbers"][picks] == 0.0)
    assert torch.all(masked.inputs["number_mask"][picks] == 0)
    # An unmasked number keeps its value.
    kept = (masked.inputs["attention_mask"] == 1) & ~picks
    kept[:, 0] = False
    assert torch.all(masked.inputs["numbers"][kept] == 3.0)


def test_a_stand_in_key_reads_its_text_with_the_real_line():
    """A counterfactual copy's acting key names no sidecar (FR-063e)."""
    batcher = _batcher()
    stand = ProvenanceKey("counterfactual/5/cardsfolder/l/lightning_bolt.txt", 0, "spell", 0)
    batcher.stand_in(stand, "deals 5 damage to any target", _ACT)
    assert batcher.text_of(stand) == "deals 5 damage to any target"
    assert batcher.line_of(stand) is _LINES[_ACT]
    assert batcher.options_for(stand) == ()


def test_a_withheld_keyword_is_read_as_its_definition_never_its_token():
    """``--withhold-keyword``: the line still says what the keyword does, so the
    zero-shot check has a definition the model trained on reading."""
    from effects.domain.ability_tokenizer import AbilityTokenizer
    from effects.domain.keyword_definitions import KeywordDefinition

    words = ["[PAD]", "[UNK]", "cardname", "[MASK]", "[CLS]", "flying", "reach",
             "cant", "be", "blocked", "except", "by", "creatures", "with", "or"]
    tokenizer = AbilityTokenizer(
        {word: index for index, word in enumerate(words)},
        {"Flying": KeywordDefinition(
            keyword="Flying",
            reminder_template="cant be blocked except by creatures with flying or reach",
        )},
    )
    batcher = make_batcher(
        tokenizer=tokenizer, sidecars=_Sidecars(), masks=BaselineMasks(),
        surface="prose", e_dim=4, widths={}, device=torch.device("cpu"),
        withhold_keyword="Flying",
    )
    (line,) = batcher.stage({"flying": None}).prepared
    ids = {word: index for index, word in enumerate(words)}
    # [CLS], then the reminder text in place of the bare token.
    assert line.token_ids[1:4] == [ids["cant"], ids["be"], ids["blocked"]]


def test_a_withheld_keywords_token_never_appears_even_inside_an_expansion():
    """Flying's own reminder text says "flying", and so does Reach's."""
    from effects.domain.ability_tokenizer import AbilityTokenizer
    from effects.domain.keyword_definitions import KeywordDefinition

    words = ["[PAD]", "[UNK]", "cardname", "[MASK]", "[CLS]", "flying", "reach",
             "cant", "be", "blocked", "except", "by", "creatures", "with", "or",
             "can", "block"]
    tokenizer = AbilityTokenizer(
        {word: index for index, word in enumerate(words)},
        {
            "Flying": KeywordDefinition(
                keyword="Flying",
                reminder_template="cant be blocked except by creatures with flying or reach",
            ),
            "Reach": KeywordDefinition(
                keyword="Reach", reminder_template="can block creatures with flying",
            ),
        },
    )
    batcher = make_batcher(
        tokenizer=tokenizer, sidecars=_Sidecars(), masks=BaselineMasks(),
        surface="prose", e_dim=4, widths={}, device=torch.device("cpu"),
        withhold_keyword="Flying", keyword_expand_p=1.0,
    )
    flying = words.index("flying")
    texts = {"flying": None, "reach": None, "creatures with flying": None}
    for line in batcher.stage(texts).prepared:
        assert flying not in line.token_ids


# ── the identity baseline's table (FR-094) ──────────────────────────────


def test_every_known_text_has_a_row_of_its_own():
    """An exact index: two texts never share a free vector, however many there are."""
    from effects.application.surface_batching import IdentityTable

    texts = [f"deals {n} damage to any target" for n in range(50_000)]
    table = IdentityTable(texts, e_dim=2)
    rows = table.rows(texts).tolist()
    assert len(set(rows)) == len(texts)
    assert IdentityTable.UNKNOWN_ROW not in rows


def test_a_text_the_index_never_saw_reads_the_shared_unknown_row():
    from effects.application.surface_batching import IdentityTable

    table = IdentityTable(["flying"], e_dim=2)
    assert table.rows(["vigilance", "trample"]).tolist() == [IdentityTable.UNKNOWN_ROW] * 2


def test_the_identity_batch_reads_each_texts_own_row():
    from effects.application.surface_batching import IdentityTable

    texts = ["creatures you control get +1/+1", "deals 3 damage to any target"]
    table = IdentityTable(texts, e_dim=4)
    batcher = make_batcher(
        tokenizer=None, sidecars=_Sidecars(), masks=BaselineMasks(identity_embedding=True),
        surface="script", e_dim=4, widths={}, device=torch.device("cpu"),
        identity_table=table,
    )
    rows, matrix = batcher.encode_texts({text: None for text in texts}, encoder=None)
    for text, row in rows.items():
        assert torch.equal(matrix[row], table(table.rows([text]))[0])


# ── the prose surface (FR-069) ───────────────────────────────────────────


def test_a_prose_text_is_encoded_with_its_lines_role_spans():
    from effects.domain.ability_encoder import role_index
    from effects.domain.ability_tokenizer import AbilityTokenizer
    from effects.domain.provenance import RoleSpan

    prose = "{T}: draw a card."
    line = SidecarLine(
        line_index=0, line_kind="activated", provenance=(_ACT,),
        script_text="AB$ Draw | Cost$ T",
        role_spans=(RoleSpan(0, 4, "cost"), RoleSpan(5, len(prose), "effect")),
    )
    batcher = make_batcher(
        tokenizer=AbilityTokenizer({"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[MASK]": 3}),
        sidecars=_Sidecars(), masks=BaselineMasks(), surface="prose", e_dim=4,
        widths={}, device=torch.device("cpu"),
    )
    (encoded,) = batcher.stage({prose: line}).prepared
    assert encoded.role_ids[1] == role_index("cost")
    assert encoded.role_ids[-1] == role_index("effect")
    # The script surface's spans would index the wrong text; a script fallback
    # on the prose surface reads none.
    (fallback,) = batcher.stage({line.script_text: line}).prepared
    assert set(fallback.role_ids) == {0}

"""``SurfaceBatcher`` hands the builder the sidecar's answer to "is this a line".

The rule itself lives in the domain builder; this pins the wiring, because a
batcher that forgot to pass the predicate would silently keep every phantom
token and every number it reports would still move.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as functional

from effects.application import step_preparation
from effects.application.mlm_masking import MlmMasker
from effects.application.surface_batching import NoiseState, SurfaceBatcher
from effects.domain.baselines import BaselineMasks
from effects.domain.effect_head_input import ACT_SLOT, EffectHeadInput, Slot, SlotKind
from effects.domain.effect_head_layout import (
    CREATED_OBJECTS_WIDTH,
    DISPERSED_FIELDS,
    FIELD_SLICES,
    FIELDS_BY_NAME,
    GATE_INDEX,
    PER_ENTITY_WIDTH,
    VERDICT_LAYOUT,
)
from effects.domain.effect_losses import _gate_term, per_entity_loss
from effects.domain.provenance import ProvenanceKey, SidecarLine
from effects.domain.records import (
    Curation,
    EffectRecord,
    Moment,
    PlayabilityDecisionPayload,
    RecordKind,
    ResolutionPayload,
)
from effects.domain.rule_mask import FULL_WEIGHT, RuleMask, copy_weights
from effects.domain.state_snapshot import EntityState, GlobalState, StateSnapshot
from effects.domain.surface_collation import entity_target_tensors
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
    """A batcher over the gen-3 fixture's sidecars. Its tokenizer knows only
    the specials but holds the keyword definitions a keyless keyword line is
    read through."""
    from effects.domain.ability_tokenizer import SPECIAL_TOKENS, AbilityTokenizer
    from effects.infrastructure.tokenizer_loading import load_keyword_definitions
    from tests.unit.effects.builders import FIXTURE_KEYWORD_DEFINITIONS, GEN3_SIDECAR_FOLDERS

    records = list(read_shard(_FIXTURES / "gen3-records.jsonl.gz"))
    cards, tokens = GEN3_SIDECAR_FOLDERS
    sidecars = SidecarCache({"cardsfolder": cards, "tokenscripts": tokens})
    tokenizer = AbilityTokenizer(
        {token: index for index, token in enumerate(SPECIAL_TOKENS)},
        load_keyword_definitions(FIXTURE_KEYWORD_DEFINITIONS), surface="script",
    )
    batcher = make_batcher(
        tokenizer=tokenizer, sidecars=sidecars, masks=BaselineMasks(),
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


def test_every_option_row_on_a_real_surface_follows_its_root():
    """FR-063a on the gen-3 fixture: a mode row comes right after its root's
    row or another mode row of the same card, never first."""
    batcher, records = _fixture_batcher()
    texts = batcher.batch_texts(records)
    rows = {text: row for row, text in enumerate(sorted(texts))}
    for record in records:
        slots = batcher.surface_builder(rows).build(record).slots
        for before, slot in zip(slots, slots[1:]):
            if slot.option:
                assert before.kind is SlotKind.ABILITY
                assert before.entity_id == slot.entity_id


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


def _mlm_inputs(lines) -> dict:
    from effects.domain.ability_encoder import collate_lines

    return collate_lines(lines, _MaskTokens.pad_id)


def _mlm_pass(batcher, lines, mask_prob):
    masked = batcher.draw_mlm_mask(_mlm_inputs(lines), mask_prob)
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
    masked = _mlm_batcher().draw_mlm_mask(_mlm_inputs(_mlm_lines([6, 4], number=3.0)), 0.5)
    picks = masked.inputs["token_ids"] == _MaskTokens.mask_id
    assert picks.any()
    assert torch.all(masked.inputs["numbers"][picks] == 0.0)
    assert torch.all(masked.inputs["number_mask"][picks] == 0)
    # An unmasked number keeps its value.
    kept = (masked.inputs["attention_mask"] == 1) & ~picks
    kept[:, 0] = False
    assert torch.all(masked.inputs["numbers"][kept] == 3.0)


def test_an_overridden_key_reads_its_text_with_the_real_line():
    """A counterfactual copy's acting key names no sidecar (FR-063e)."""
    batcher = _batcher()
    copy = ProvenanceKey("counterfactual/5/cardsfolder/l/lightning_bolt.txt", 0, "spell", 0)
    batcher.override_text(copy, "deals 5 damage to any target", _ACT)
    assert batcher.text_of(copy) == "deals 5 damage to any target"
    assert batcher.line_of(copy) is _LINES[_ACT]
    assert batcher.options_for(copy) == ()


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


class TestRelationsUnderTheMasks:
    """T102: the board's links are state, so the no-state baseline reads none."""

    def _inputs(self):
        from effects.domain.effect_head_input import EffectHeadInput, Slot, SlotKind
        from effects.domain.surface_collation import collate_surfaces

        slots = tuple(Slot(kind=SlotKind.CARD, position=i, features=(1.0,)) for i in range(3))
        return collate_surfaces([EffectHeadInput(slots=slots, relations=((0, 1, 2),))],
                                e_dim=4, widths={SlotKind.CARD: 1, SlotKind.GLOBAL: 40,
                                                 SlotKind.PLAYER: 1})

    def test_no_state_drops_every_relation(self):
        from effects.domain.baselines import Baseline, baseline_masks
        from tests.unit.effects.builders import make_batcher

        batcher = make_batcher(masks=baseline_masks(Baseline.NO_STATE))
        assert batcher._masked(self._inputs()).relations.shape == (0, 4)

    def test_the_full_model_keeps_them(self):
        from effects.domain.baselines import Baseline, baseline_masks
        from tests.unit.effects.builders import make_batcher

        batcher = make_batcher(masks=baseline_masks(Baseline.FULL))
        assert batcher._masked(self._inputs()).relations.tolist() == [[0, 0, 1, 2]]


def _flying_batcher(**settings):
    from effects.domain.ability_tokenizer import AbilityTokenizer
    from effects.domain.keyword_definitions import KeywordDefinition

    words = ["[PAD]", "[UNK]", "cardname", "[MASK]", "[CLS]", "flying", "cant", "be"]
    tokenizer = AbilityTokenizer(
        {word: index for index, word in enumerate(words)},
        {"Flying": KeywordDefinition(
            keyword="Flying",
            reminder_template="cant be blocked except by creatures with flying or reach",
        )},
    )
    return make_batcher(
        tokenizer=tokenizer, sidecars=_Sidecars(), masks=settings.pop("masks", BaselineMasks()),
        surface="prose", e_dim=4, device=torch.device("cpu"), **settings,
    )


def _resolution(entities) -> EffectRecord:
    return make_record(
        record_id="run.0.1", run_id="run", game_id="run.0.1", ability=(_ACT,),
        state=make_snapshot(entities=entities),
    )


def test_a_keyword_line_with_no_script_reads_its_reminder_text():
    """Root spec § 9.2: a keyword an effect granted is a line read through its
    reminder text; a keyword the definitions do not know reads as written."""
    from effects.domain.effect_head_input import keyword_line_key

    batcher = _flying_batcher(widths={})
    flying = keyword_line_key("Flying")
    assert batcher.text_of(flying) == "cant be blocked except by creatures with flying or reach"
    assert batcher.line_of(flying) is None
    assert batcher.options_for(flying) == ()
    assert batcher.keyword_of_key(flying) == "flying"
    assert batcher.text_of(keyword_line_key("Mystery:3")) == "Mystery:3"


def test_a_batch_encodes_the_reminder_text_of_a_granted_keyword_line():
    from effects.domain.state_snapshot import CardLine

    entity = EntityState(
        id="E1", name="hornclaw", zone="battlefield", controller="P0",
        lines=(CardLine(keys=(), kind="keyword", granted=True, fragment="Flying"),),
    )
    record = _resolution(entities=(entity,))
    texts = _flying_batcher(widths={}).batch_texts([record])
    assert "cant be blocked except by creatures with flying or reach" in texts


def test_no_state_zeroes_the_line_flags_and_the_card_ids():
    from effects.domain.effect_head_input import feature_widths

    entity = EntityState(
        id="E1", name="soldier", zone="battlefield", controller="P0",
        subtypes=("human",), counters={"OIL": 2}, printed=(_ACT,),
    )
    record = _resolution(entities=(entity,))
    widths = feature_widths([record])
    full = _flying_batcher(widths=widths).prepare([record]).inputs
    blind = _flying_batcher(
        widths=widths, masks=BaselineMasks(zero_state=True)).prepare([record]).inputs
    assert full.slot_features[SlotKind.ABILITY].abs().sum() > 0
    assert full.subtype_ids.abs().sum() > 0 and full.counter_counts.abs().sum() > 0
    assert blind.slot_features[SlotKind.ABILITY].abs().sum() == 0
    for name in ("subtype_ids", "chosen_type_ids", "counter_ids", "counter_counts"):
        assert getattr(blind, name).abs().sum() == 0, name


def test_a_floating_effects_card_reads_its_source_lines_text():
    """Root spec § 9.3 (T114): a timestamped floating effect's ``[CARD]``
    is followed by its source line, encoded with the batch like any other."""
    from effects.domain.effect_head_input import command_zone_entities
    from effects.infrastructure.record_codec import record_from_dict
    from tests.unit.effects.builders import (
        GEN3_SIDECAR_FOLDERS,
        gen3_fixture_records,
    )

    record = next(
        record_from_dict(raw) for raw in gen3_fixture_records()
        if raw["record_id"].endswith(".1-mvyw1pozylvp.13630"))
    floating, = command_zone_entities(record)
    sidecars = SidecarCache({folder.name: folder for folder in GEN3_SIDECAR_FOLDERS})
    batcher = make_batcher(
        tokenizer=None, sidecars=sidecars, masks=BaselineMasks(),
        surface="script", e_dim=4, widths={}, device=torch.device("cpu"),
    )
    source = batcher.text_of(floating.lines[0].keys[0])
    texts = batcher.batch_texts([record])
    assert source is not None and source in texts
    rows = {text: row for row, text in enumerate(sorted(texts))}
    surface = batcher.surface_builder(rows).build(record)
    at = next(index for index, slot in enumerate(surface.slots)
              if slot.kind is SlotKind.CARD and slot.entity_id == floating.id)
    assert surface.slots[at + 1].e == rows[source]


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


def _copy(*, natural_share: int, copies: int, drawn_rule=None, drawn_rules=(),
          rule_entities=(), aimed_at=None) -> SimpleNamespace:
    """One curated copy of a record, as the rule mask reads it."""
    return SimpleNamespace(
        curation=Curation(natural_share, copies, drawn_rule, tuple(drawn_rules),
                          tuple(rule_entities)),
        aimed_at=aimed_at,
    )


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
        outputs, _mask_batch(records, mode), fields=_MASK_FIELDS,
        log_dispersion=torch.zeros(len(DISPERSED_FIELDS)),
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
    added = _copy(natural_share=1, copies=2, drawn_rule="702.2",
                  drawn_rules=("702.2",), rule_entities=("E1",))
    natural = _copy(natural_share=1, copies=2, drawn_rules=("702.2",))
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
    added = _copy(natural_share=0, copies=1, drawn_rule="702.2",
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
    added = _copy(natural_share=0, copies=1, drawn_rule="702.2",
                  drawn_rules=("702.2",), rule_entities=())
    gradient = _gradients([added], RuleMask.HARD)
    assert _trains(gradient, 0, "gate") == set()
    assert _trains(gradient, 0, "zone_outcome") == {"E1", "E2"}


def test_a_hard_masked_copy_whose_rule_names_a_missing_output_trains_only_the_gate():
    """Edge case: 702.15 names ``poison_delta``, which this record does not
    carry, so its copy trains the gate on the rule's entities and nothing
    else."""
    added = _copy(natural_share=1, copies=2, drawn_rule="702.15",
                  drawn_rules=("702.15",), rule_entities=("E1",))
    gradient = _gradients([added], RuleMask.HARD)
    assert _trains(gradient, 0, "gate") == {"E1"}
    for output in ("zone_outcome", "damage_taken", "life_delta", "poison_delta"):
        assert _trains(gradient, 0, output) == set(), output


def test_a_soft_masked_record_weighs_other_outputs_n_over_d():
    """Story 7 acceptance 2: natural share 1, written 4 times; outputs other
    than those of its drawn rules weigh 1/4 on each copy, natural and added,
    its drawn rules' outputs 1 on every copy."""
    natural = _copy(natural_share=1, copies=4, drawn_rules=("702.2",))
    added = _copy(natural_share=1, copies=4, drawn_rule="702.2",
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


def test_an_aimed_natural_copy_and_an_uncurated_record_train_at_full_weight():
    """A directed fork's natural copy is a natural record like any other
    (FR-061), and a lethality copy or uncurated record carries no curation."""
    aimed = _copy(natural_share=1, copies=4, drawn_rules=("702.2",), aimed_at=("704.5f",))
    uncurated = SimpleNamespace(curation=None, aimed_at=None)
    for mode in RuleMask:
        assert copy_weights(aimed, mode, _RULE_OUTPUTS) is FULL_WEIGHT
        assert copy_weights(uncurated, mode, _RULE_OUTPUTS) is FULL_WEIGHT


def test_the_verdict_head_takes_the_rule_mask():
    """A verdict or global field is an output the map names too (FR-060a)."""
    names = [str(spec.name) for spec in VERDICT_LAYOUT.fields]
    outputs = {"603.2": ("trigger_fired",)}
    hard = _copy(natural_share=1, copies=2, drawn_rule="603.2", drawn_rules=("603.2",))
    soft = _copy(natural_share=1, copies=4, drawn_rules=("603.2",))
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
    named = _copy(natural_share=0, copies=1, drawn_rule="111.5", drawn_rules=("111.5",))
    other = _copy(natural_share=0, copies=1, drawn_rule="702.2", drawn_rules=("702.2",))
    soft = _copy(natural_share=1, copies=4, drawn_rules=("702.2",))
    weights = [copy_weights(named, RuleMask.HARD, outputs),
               copy_weights(other, RuleMask.HARD, outputs),
               copy_weights(soft, RuleMask.SOFT, outputs)]
    _target, mask, known = step_preparation._created_objects_rows(
        [named, other, soft], weights,
    )
    assert mask.tolist() == [True, False, True]
    assert known[0].unique().tolist() == [1.0]
    assert known[2].unique().tolist() == [0.25]

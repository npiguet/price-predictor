"""The MLM pass (FR-060b), keyword overrides and the identity baseline's table (FR-094)."""

from __future__ import annotations

import torch

from effects.application.mlm_masking import MlmMasker
from effects.application.surface_batching import SurfaceBatcher
from effects.domain.baselines import BaselineMasks
from effects.domain.provenance import ProvenanceKey
from tests.unit.effects.application.surface_batching_support import (
    _ACT,
    _LINES,
    _batcher,
    _Sidecars,
)
from tests.unit.effects.builders import (
    make_batcher,
)

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

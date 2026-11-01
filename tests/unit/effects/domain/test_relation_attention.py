"""The relation-aware layer's properties (T098, FR-046, FR-049, root spec § 10.5).

Written before the layer: each test states one property the root spec asks of
it. All run on CPU at a tiny size.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from effects.domain.relation_attention import RelationEncoderLayer

D, HEADS, FF, TYPES = 16, 2, 32, 5


def _stock(dropout: float = 0.0) -> nn.TransformerEncoderLayer:
    torch.manual_seed(0)
    return nn.TransformerEncoderLayer(
        d_model=D, nhead=HEADS, dim_feedforward=FF, dropout=dropout,
        batch_first=True, norm_first=True)


def _layer(dropout: float = 0.0, relation_values: bool = True) -> RelationEncoderLayer:
    torch.manual_seed(0)
    return RelationEncoderLayer(D, HEADS, FF, dropout, relation_types=TYPES,
                                relation_values=relation_values)


def _batch(seed: int = 1, batch: int = 3, slots: int = 7):
    torch.manual_seed(seed)
    x = torch.randn(batch, slots, D)
    pad = torch.zeros(batch, slots, dtype=torch.bool)
    if batch > 1:
        pad[1, -2:] = True
    return x, pad


def _relations(rows):
    return torch.tensor(rows, dtype=torch.long).reshape(-1, 4)


class TestZeroTablesAreTheStockLayer:
    def test_outputs_and_gradients_match_in_fp32(self):
        stock, layer = _stock(), _layer()
        missing, unexpected = layer.load_state_dict(stock.state_dict(), strict=False)
        assert not unexpected and set(missing) == {"rel_k", "rel_v"}
        stock.train()
        layer.train()
        x, pad = _batch()
        x1, x2 = x.clone().requires_grad_(), x.clone().requires_grad_()
        expected = stock(x1, src_key_padding_mask=pad)
        found = layer(x2, src_key_padding_mask=pad,
                      relations=_relations([[0, 0, 1, 2], [2, 3, 0, 4]]))
        keep = ~pad
        assert torch.allclose(found[keep], expected[keep], atol=1e-6)
        (expected[keep].sum()).backward()
        (found[keep].sum()).backward()
        assert torch.allclose(x1.grad, x2.grad, atol=1e-6)
        for name, param in stock.named_parameters():
            assert torch.allclose(param.grad, dict(layer.named_parameters())[name].grad,
                                  atol=1e-6), name

    def test_the_stock_parameter_names_are_kept(self):
        names = set(dict(_layer().named_parameters()))
        assert set(dict(_stock().named_parameters())) <= names


class TestPermutation:
    def test_permuting_slots_and_relations_together_permutes_the_outputs(self):
        layer = _layer().eval()
        with torch.no_grad():
            layer.rel_k.normal_()
            layer.rel_v.normal_()
        x, _ = _batch(batch=1, slots=6)
        rel = _relations([[0, 0, 3, 1], [0, 4, 2, 0], [0, 1, 1, 3]])
        perm = torch.tensor([5, 2, 0, 4, 1, 3])
        inverse = torch.argsort(perm)
        moved = rel.clone()
        moved[:, 1], moved[:, 2] = inverse[rel[:, 1]], inverse[rel[:, 2]]
        with torch.no_grad():
            out = layer(x, relations=rel)
            out_perm = layer(x[:, perm], relations=moved)
        assert torch.allclose(out_perm, out[:, perm], atol=1e-5)


class TestDirection:
    def test_a_relation_from_i_to_j_changes_i_and_not_j(self):
        layer = _layer().eval()
        with torch.no_grad():
            layer.rel_k.normal_()
            layer.rel_v.normal_()
        x, _ = _batch(batch=1, slots=5)
        with torch.no_grad():
            plain = layer(x, relations=_relations([]))
            related = layer(x, relations=_relations([[0, 1, 3, 2]]))
        assert not torch.allclose(related[0, 1], plain[0, 1])
        for slot in (0, 2, 3, 4):
            assert torch.allclose(related[0, slot], plain[0, slot], atol=1e-6), slot


class _PartnerReader(nn.Module):
    """One layer over identical-looking slots; the answer is the partner's bit."""

    def __init__(self, use_relations: bool) -> None:
        super().__init__()
        self.embed = nn.Linear(2, D)
        self.layer = RelationEncoderLayer(D, HEADS, FF, 0.0, relation_types=1)
        self.head = nn.Linear(D, 1)
        if not use_relations:
            self.layer.rel_k.requires_grad_(False)
            self.layer.rel_v.requires_grad_(False)

    def forward(self, x, rel):
        return self.head(self.layer(self.embed(x), relations=rel)[:, 0]).squeeze(-1)


def _partner_task(generator, batch: int = 64, slots: int = 7):
    """Slot 0 is the reader; one other slot, at random, is its partner, linked
    only by a relation. The label is the partner's bit. The six other slots
    always carry exactly three 1s, so their bits together say nothing about the
    partner's: without the relation the best guess is a coin flip. Slot 0's
    input carries a reader flag and no bit."""
    others = torch.stack([torch.randperm(slots - 1, generator=generator) < (slots - 1) // 2
                          for _ in range(batch)]).float()
    bits = torch.cat([torch.zeros(batch, 1), others], dim=1)
    partner = torch.randint(1, slots, (batch,), generator=generator)
    x = torch.stack([bits, torch.zeros_like(bits)], dim=-1)
    x[:, 0, 0] = 0.0
    x[:, 0, 1] = 1.0
    rel = torch.stack([torch.arange(batch), torch.zeros(batch, dtype=torch.long), partner,
                       torch.zeros(batch, dtype=torch.long)], dim=1)
    return x, rel, bits[torch.arange(batch), partner]


@pytest.mark.parametrize("use_relations, learns", [(True, True), (False, False)])
def test_a_toy_task_only_a_relation_solves(use_relations, learns):
    torch.manual_seed(0)
    model = _PartnerReader(use_relations)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    generator = torch.Generator().manual_seed(0)
    for _ in range(300):
        x, rel, y = _partner_task(generator)
        loss = nn.functional.binary_cross_entropy_with_logits(model(x, rel), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    x, rel, y = _partner_task(torch.Generator().manual_seed(1), batch=2000)
    with torch.no_grad():
        accuracy = ((model(x, rel) > 0).float() == y).float().mean().item()
    assert (accuracy > 0.9) if learns else (accuracy < 0.6), accuracy


def test_with_relation_values_off_the_value_table_stays_zero_and_untrained():
    layer = _layer(relation_values=False)
    assert "rel_v" not in dict(layer.named_parameters())
    assert torch.count_nonzero(layer.rel_v) == 0
    x, _ = _batch()
    layer(x, relations=_relations([[0, 0, 1, 2]])).sum().backward()
    assert layer.rel_k.grad is not None
    assert torch.count_nonzero(layer.rel_v) == 0


def test_the_value_term_reads_the_dropped_out_attention_weights():
    """With the value projection zeroed, the output at i carries only the relation
    value term: equal to the relation-free output exactly when dropout drops a(i, j)."""
    layer = RelationEncoderLayer(D, HEADS, FF, 0.5, relation_types=1).train()
    with torch.no_grad():
        layer.self_attn.in_proj_weight[2 * D:].zero_()
        layer.self_attn.in_proj_bias[2 * D:].zero_()
        layer.rel_v.normal_()
    x, _ = _batch(batch=1, slots=4)
    rel = _relations([[0, 0, 2, 0]])
    attend = layer._attend
    dropped = kept = 0
    for seed in range(40):
        torch.manual_seed(seed)
        with_rel = attend(layer.norm1(x), None, rel)
        torch.manual_seed(seed)
        without = attend(layer.norm1(x), None, _relations([]))
        same = torch.allclose(with_rel[0, 0], without[0, 0], atol=1e-6)
        dropped += same
        kept += not same
    assert dropped and kept

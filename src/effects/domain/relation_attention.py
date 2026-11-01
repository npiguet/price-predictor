"""The relation-aware encoder layer of the gen-3 trunk (root spec § 10.1–10.2, research M2).

A relation is a directed, typed link from slot i to slot j. In each head of
each layer it adds a learned key vector ``K_r`` to j's key as i reads it, and a
learned value vector ``V_r`` to what i receives from j:

    score(i, j) = q_i · (k_j + Σ_{r ∈ R(i, j)} K_r) / √d
    a(i, j)     = softmax_j score(i, j), padding masked, then attention dropout
    out_i       = Σ_j a(i, j) · (v_j + Σ_{r ∈ R(i, j)} V_r)

The rest is ``nn.TransformerEncoderLayer`` with ``norm_first=True``, under the
same parameter names, so a stock layer's state dict loads into this one, and
with both tables at zero the two compute the same thing.

No vector per pair of slots is ever built. The key term is ``q_i · K_r`` per
slot and type, scattered onto the pairs the relations name; the value term
sums, per slot and type, the attention weights of the pairs carrying that
type, and multiplies the sums by the value table. Relations arrive as a
``(N, 4)`` long tensor of ``[batch, from, to, type]`` rows.

Prior art: ``nn.TransformerEncoderLayer`` as ``effect_model`` builds it, whose
parameter names and pre-norm layout this keeps. Not a subclass: the relation
terms enter the attention scores, which the stock layer offers no hook for.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class RelationEncoderLayer(nn.Module):
    """A pre-norm encoder layer whose attention reads typed slot-to-slot relations."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float, *,
                 relation_types: int, relation_values: bool = True,
                 layer_norm_eps: float = 1e-5) -> None:
        super().__init__()
        if d_model % nhead:
            raise ValueError(f"d_model {d_model} does not split into {nhead} heads")
        self.nhead = nhead
        self.head_width = d_model // nhead
        self.attention_dropout = dropout
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        shape = (nhead, relation_types, self.head_width)
        self.rel_k = nn.Parameter(torch.zeros(shape))
        if relation_values:
            self.rel_v = nn.Parameter(torch.zeros(shape))
        else:
            # Kept at zero and out of the optimizer (--relation-values off).
            self.register_buffer("rel_v", torch.zeros(shape))
        self.relation_values = relation_values

    def forward(self, src: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None,
                relations: torch.Tensor | None = None) -> torch.Tensor:
        x = src
        x = x + self.dropout1(self._attend(self.norm1(x), src_key_padding_mask, relations))
        return x + self.dropout2(self.linear2(self.dropout(F.relu(self.linear1(self.norm2(x))))))

    def _attend(self, x: torch.Tensor, padding: torch.Tensor | None,
                relations: torch.Tensor | None) -> torch.Tensor:
        batch, slots, width = x.shape
        heads, d = self.nhead, self.head_width
        q, k, v = F.linear(x, self.self_attn.in_proj_weight,
                           self.self_attn.in_proj_bias).chunk(3, dim=-1)

        def split(t: torch.Tensor) -> torch.Tensor:  # (B, S, D) -> (B, H, S, d)
            return t.view(batch, slots, heads, d).transpose(1, 2)

        q, k, v = split(q), split(k), split(v)
        scale = 1.0 / math.sqrt(d)
        scores = (q @ k.transpose(-1, -2)) * scale
        has_relations = relations is not None and relations.numel() > 0
        if has_relations:
            b, i, j, r = relations.unbind(1)
            # q_i · K_r for every slot and type, read off at each relation's (b, i, r).
            q_dot_k = torch.einsum("bhsd,htd->bhst", q, self.rel_k.to(q.dtype))
            added = q_dot_k[b, :, i, r] * scale  # (N, H)
            bias = torch.zeros(batch, slots, slots, heads, dtype=scores.dtype,
                               device=scores.device).index_put((b, i, j), added, accumulate=True)
            scores = scores + bias.permute(0, 3, 1, 2)
        if padding is not None:
            scores = scores.masked_fill(padding[:, None, None, :], float("-inf"))
        weights = F.dropout(torch.softmax(scores, dim=-1), self.attention_dropout,
                            training=self.training)
        out = weights @ v
        if has_relations:
            # Σ a(i, j) over the pairs carrying type r, per slot i: (B, H, S, T).
            picked = weights[b, :, i, j]  # (N, H)
            sums = torch.zeros(batch, slots, self.rel_v.shape[1], heads, dtype=weights.dtype,
                               device=weights.device).index_put((b, i, r), picked,
                                                                accumulate=True)
            out = out + torch.einsum("bsth,htd->bhsd", sums, self.rel_v.to(weights.dtype))
        out = out.transpose(1, 2).reshape(batch, slots, width)
        return self.self_attn.out_proj(out)

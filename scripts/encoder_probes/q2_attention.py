"""R6 (c) — what each of the 8 pool queries attends to.

``_MultiQueryAttentionPool`` discards the attention weights, so this script
re-runs the pool by hand from the loaded model's own modules: the same
``kv_proj`` slice per query, the same learned query vector, the same shared
``nn.MultiheadAttention``, but with ``need_weights=True``. The pooled output
is checked against ``model.encode`` so the replay is provably the real thing.

Token positions are bucketed by the converted-text line they came from
(``mana cost:`` value, ``types:`` value, ``power toughness:`` value,
``static:`` value = keywords, every other ability line = body, and the
field-label words themselves).
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_lib as pl  # noqa: E402
import q_common as qc  # noqa: E402

N_CARDS = 320
BUCKETS = ("cost", "types", "pt", "keyword", "body", "label")


def normalize_text(raw: str) -> str:
    """Name-stripped text with the tokenizer's implicit ``mana cost: none``."""
    lines = [l for l in raw.splitlines() if not l.startswith("name:") and l.strip()]
    if not any(l.startswith("mana cost:") for l in lines):
        lines.append("mana cost: none")
    return "\n".join(lines)


def line_bucket(line: str) -> str:
    if line.startswith("mana cost:"):
        return "cost"
    if line.startswith("types:"):
        return "types"
    if line.startswith("power toughness:"):
        return "pt"
    if line.startswith("static:"):
        return "keyword"
    return "body"


_NONE_COST = ["mana", "cost", ":", "none"]


def _frag(fragment: str, tokenizer) -> list[str]:
    """Tokens of a text fragment, minus the tokenizer's implicit cost line.

    ``MtgTokenizer.tokenize`` appends ``mana cost: none`` to any input that
    has no ``mana cost:`` line, so a per-line replay has to strip it back off.
    """
    toks = tokenizer.tokenize(fragment)
    if "mana cost:" not in fragment.lower():
        assert toks[-4:] == _NONE_COST, toks
        toks = toks[:-4]
    return toks


def token_buckets(text: str, tokenizer) -> list[str] | None:
    """Per-token bucket labels, or None when the line-wise split disagrees."""
    labels: list[str] = []
    toks: list[str] = []
    for line in text.splitlines():
        head, sep, tail = line.partition(":")
        bucket = line_bucket(line)
        head_toks = _frag(head + sep, tokenizer) if sep else []
        tail_toks = _frag(tail if sep else line, tokenizer)
        toks.extend(head_toks + tail_toks)
        labels.extend(["label"] * len(head_toks) + [bucket] * len(tail_toks))
    if tuple(toks) != tuple(tokenizer.tokenize(text)):
        return None
    return labels


@torch.no_grad()
def pool_attention(runner: pl.EncoderRunner, texts: list[str]):
    """``(attn, pooled)`` — attn is ``(B, 8, T)``, pooled ``(B, 512)``."""
    tok, dev = runner.tokenizer, runner.device
    encoded = [tok.encode(t, runner.max_seq_len) for t in texts]
    ids = torch.tensor([e[0] for e in encoded], dtype=torch.long, device=dev)
    mask = torch.tensor([e[1] for e in encoded], dtype=torch.long, device=dev)
    model = runner.model
    x = model.token_encoder(ids)
    enc = model.card_encoder.encoder(x, src_key_padding_mask=(mask == 0))
    pool = model.card_encoder.attn_pool
    kv = pool.kv_proj(enc).view(len(texts), enc.size(1), pool.n_pool_queries,
                                pool.head_dim)
    kpm = mask == 0
    attns, outs = [], []
    for k in range(pool.n_pool_queries):
        q = pool.queries[:, k:k + 1, :].expand(len(texts), -1, -1)
        kv_k = kv[:, :, k, :]
        out, w = pool.attn(q, kv_k, kv_k, key_padding_mask=kpm,
                           need_weights=True, average_attn_weights=True)
        attns.append(w.squeeze(1))
        outs.append(out.squeeze(1))
    return torch.stack(attns, dim=1).cpu().numpy(), torch.cat(outs, -1).cpu().numpy()


def main() -> None:
    join, _ = qc.load_frame()
    rng = np.random.default_rng(42)
    # A deliberately mixed sample: creatures, spells, lands, other permanents.
    strata = {
        "creature": join["is_creature"].fillna(0).to_numpy(float) > 0,
        "spell": (join["is_instant"].fillna(0).to_numpy(float) > 0)
                 | (join["is_sorcery"].fillna(0).to_numpy(float) > 0),
        "land": join["is_land"].fillna(0).to_numpy(float) > 0,
    }
    other = ~(strata["creature"] | strata["spell"] | strata["land"])
    strata["other"] = other
    picks: list[int] = []
    per = N_CARDS // len(strata)
    for name, m in strata.items():
        idx = np.flatnonzero(m)
        picks.extend(rng.choice(idx, size=min(per, len(idx)), replace=False))
    picks = sorted(set(picks))

    runner = pl.EncoderRunner()
    tok = runner.tokenizer
    texts, buckets, kinds, names = [], [], [], []
    kind_of = {i: k for k, m in strata.items() for i in np.flatnonzero(m)}
    dropped = 0
    for i in picks:
        raw = Path(join.loc[i, "txt_path"]).read_text(encoding="utf-8",
                                                      errors="replace")
        text = normalize_text(raw)
        b = token_buckets(text, tok)
        if b is None or len(b) > runner.max_seq_len:
            dropped += 1
            continue
        texts.append(text)
        buckets.append(b)
        kinds.append(kind_of[i])
        names.append(join.loc[i, "name"])
    print(f"cards={len(texts)} dropped={dropped}", flush=True)

    attn, pooled = pool_attention(runner, texts)
    cached = pl.load_embedding_matrix(names, join)
    print("replay vs cache max|d| =", float(np.abs(pooled - cached).max()), flush=True)

    n_q = attn.shape[1]
    mass = np.zeros((n_q, len(BUCKETS)))
    counts = np.zeros(len(BUCKETS))
    ent, ent_norm, top1 = [[] for _ in range(n_q)], [[] for _ in range(n_q)], \
        [[] for _ in range(n_q)]
    tok_mass: list[Counter] = [Counter() for _ in range(n_q)]
    tok_freq: Counter = Counter()
    per_card_rows = []
    for c, (text, b) in enumerate(zip(texts, buckets)):
        t = len(b)
        toks = tok.tokenize(text)
        a = attn[c, :, :t]
        a = a / a.sum(axis=1, keepdims=True)
        for bi, name in enumerate(BUCKETS):
            sel = np.array([x == name for x in b])
            if sel.any():
                mass[:, bi] += a[:, sel].sum(axis=1)
        counts += [sum(1 for x in b if x == name) / t for name in BUCKETS]
        for k in range(n_q):
            p = np.clip(a[k], 1e-12, None)
            h = float(-(p * np.log(p)).sum())
            ent[k].append(h)
            ent_norm[k].append(h / np.log(t) if t > 1 else 0.0)
            top1[k].append(float(a[k].max()))
            for tk_str, w in zip(toks, a[k]):
                tok_mass[k][tk_str] += float(w)
        tok_freq.update(toks)
        per_card_rows.append({"name": names[c], "kind": kinds[c], "n_tok": t,
                              **{f"ent{k}": ent[k][-1] for k in range(n_q)}})

    n = len(texts)
    share = pd.DataFrame(mass / n, columns=list(BUCKETS))
    share.insert(0, "query", range(n_q))
    share.loc[len(share)] = ["token-share", *(counts / n)]
    share.to_csv(qc.SCRATCH / "q2_bucket_share.csv", index=False)

    ent_tab = pd.DataFrame({
        "query": range(n_q),
        "mean_entropy": [float(np.mean(e)) for e in ent],
        "mean_entropy_norm": [float(np.mean(e)) for e in ent_norm],
        "mean_top1_mass": [float(np.mean(t)) for t in top1],
    })
    ent_tab.to_csv(qc.SCRATCH / "q2_entropy.csv", index=False)

    # Per-query token preference: mean attention weight a token receives,
    # relative to the uniform share it would get by chance.
    rows = []
    for k in range(n_q):
        lift = {}
        for tk_str, m_ in tok_mass[k].items():
            if tok_freq[tk_str] < 8:
                continue
            lift[tk_str] = m_ / tok_freq[tk_str]
        best = sorted(lift.items(), key=lambda kv: -kv[1])[:12]
        rows.append({"query": k, "top_tokens": ", ".join(
            f"{t}({v:.3f})" for t, v in best)})
    pd.DataFrame(rows).to_csv(qc.SCRATCH / "q2_top_tokens.csv", index=False)

    # Query-to-query similarity of the attention distribution, per card.
    cross = np.zeros((n_q, n_q))
    for c, b in enumerate(buckets):
        t = len(b)
        a = attn[c, :, :t]
        a = a / a.sum(axis=1, keepdims=True)
        cross += np.corrcoef(a)
    cross /= n
    pd.DataFrame(cross).to_csv(qc.SCRATCH / "q2_query_corr.csv", index=False)

    pd.DataFrame(per_card_rows).to_csv(qc.SCRATCH / "q2_per_card.csv", index=False)
    summary = {
        "n_cards": n,
        "mean_tokens": float(np.mean([len(b) for b in buckets])),
        "query_attn_corr_offdiag_mean": float(
            cross[np.triu_indices(n_q, 1)].mean()),
        "query_attn_corr_offdiag_min": float(cross[np.triu_indices(n_q, 1)].min()),
        "replay_max_abs_diff": float(np.abs(pooled - cached).max()),
    }
    with open(qc.SCRATCH / "q2_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(share.round(4).to_string(index=False))
    print(ent_tab.round(4).to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

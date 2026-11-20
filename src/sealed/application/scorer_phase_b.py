"""Phase B's encoder forward in ``train-scorer``: each batch's unique cards
tokenized, encoded in chunks under gradient checkpointing and spliced into
the embedding table, the step-0 reference batch, and the embedding drift
measured against it each epoch."""

from __future__ import annotations

import torch

from sealed.application.scorer_log import _log
from sealed.application.scorer_training_context import ReferenceBatch, _TrainingContext
from sealed.infrastructure.match_data_loader import TrainingBatch


def _phase_b_inject_encoder(ctx: _TrainingContext, batch: TrainingBatch) -> None:
    """Run the encoder over the batch's unique cards and splice the resulting
    text vectors into the embedding-table rows referenced by this batch
    (FR-007, Decision §4). The encoder forward runs once per unique card; the
    autograd graph released after ``optimizer.step()`` ensures the within-batch
    cache does not survive between training steps."""
    assert ctx.encoder is not None
    assert ctx.tokenizer is not None
    assert ctx.card_token_cache is not None
    assert ctx.locator is not None
    assert ctx.idx_to_name is not None
    assert ctx.max_seq_len is not None

    unique = torch.unique(
        torch.cat([batch.winner_indices.flatten(), batch.loser_indices.flatten()]),
    )
    input_ids_list: list[torch.Tensor] = []
    attention_mask_list: list[torch.Tensor] = []
    for idx_t in unique:
        idx = int(idx_t.item())
        cached = ctx.card_token_cache.get(idx)
        if cached is None:
            cached = _tokenize_card(ctx, idx)
            ctx.card_token_cache[idx] = cached
        input_ids_list.append(cached[0])
        attention_mask_list.append(cached[1])

    input_ids = torch.stack(input_ids_list).to(ctx.device)
    attention_mask = torch.stack(attention_mask_list).to(ctx.device)
    text_vectors = _encode_chunked(
        ctx.encoder, input_ids, attention_mask,
        chunk_size=ctx.encoder_chunk_size, training=True,
    )
    ctx.embedding_table.set_text_vectors(unique.to(ctx.device), text_vectors)

    if ctx.reference_batch is None:
        # The encoder is already in eval mode for the entire Phase B training
        # loop (no dropout), so the forward we just computed for the scorer's
        # loss is identical to the forward `_embedding_drift` will run each
        # epoch on the same inputs. Snapshot it directly — no extra forward
        # needed.
        ctx.reference_batch = ReferenceBatch(
            card_indices=unique.detach().clone().to(ctx.device),
            input_ids=input_ids.detach().clone(),
            attention_mask=attention_mask.detach().clone(),
            step0_text_vectors=text_vectors.detach().clone(),
        )
        _log(f"Reference batch captured: {unique.numel()} unique cards (Phase B step 0)")


def _encode_chunked(
    encoder: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    chunk_size: int,
    training: bool,
) -> torch.Tensor:
    """Run the encoder forward in fixed-size chunks, applying gradient
    checkpointing per chunk during training so peak activation memory is
    bounded by ``chunk_size`` cards instead of the full unique-card count.

    ``encoder`` must expose ``_encode_and_pool(input_ids, attention_mask) ->
    (B, text_dim)`` — both ``CardPriceTransformerModel`` and ``SealedEncoderModel``
    do.

    A typical Phase B step sees ~1000–3000 unique cards in a single batch
    (`batch_size * cards_per_deck * 2` minus duplicates). One forward pass at
    that size on a 6-layer transformer overflows commodity GPUs; chunking
    keeps each forward bounded by ``chunk_size``. Gradient checkpointing
    drops activations after each chunk and recomputes them during backward,
    trading ~1.5× compute for ~10× less memory.
    """
    n = input_ids.size(0)
    if n <= chunk_size or not training:
        return encoder._encode_and_pool(input_ids, attention_mask)
    outputs: list[torch.Tensor] = []
    for start in range(0, n, chunk_size):
        end = start + chunk_size
        chunk_ids = input_ids[start:end]
        chunk_mask = attention_mask[start:end]
        chunk_out = torch.utils.checkpoint.checkpoint(
            encoder._encode_and_pool, chunk_ids, chunk_mask,
            use_reentrant=False,
        )
        outputs.append(chunk_out)
    return torch.cat(outputs, dim=0)


def _tokenize_card(
    ctx: _TrainingContext, idx: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert ctx.tokenizer is not None
    assert ctx.locator is not None
    assert ctx.idx_to_name is not None
    assert ctx.max_seq_len is not None
    name = ctx.idx_to_name[idx]
    converted = ctx.locator.load_text(name)
    if converted is None:
        raise FileNotFoundError(
            f"Phase B: no converted card .txt for {name!r} under "
            f"{ctx.locator._cards_path}",
        )
    text = converted.without_name_line()
    input_ids, attention_mask = ctx.tokenizer.encode(text, ctx.max_seq_len)
    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(attention_mask, dtype=torch.long),
    )


def _embedding_drift(ctx: _TrainingContext) -> float:
    """Mean L2 distance between current encoder text vectors on the reference
    batch and their step-0 values (Decision §5, FR-012). Computed under
    ``model.eval()`` so dropout doesn't muddle the signal — the step-0
    reference is captured the same way."""
    assert ctx.encoder is not None
    assert ctx.reference_batch is not None
    was_training = ctx.encoder.training
    ctx.encoder.eval()
    try:
        with torch.no_grad():
            current = _encode_chunked(
                ctx.encoder,
                ctx.reference_batch.input_ids,
                ctx.reference_batch.attention_mask,
                chunk_size=ctx.encoder_chunk_size,
                training=False,
            )
        diff = current - ctx.reference_batch.step0_text_vectors
        return float(diff.norm(dim=-1).mean().item())
    finally:
        if was_training:
            ctx.encoder.train()

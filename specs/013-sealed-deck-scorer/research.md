# Research: Sealed Deck Scorer

**Feature**: 013-sealed-deck-scorer | **Date**: 2026-04-11

## R-01: Set Transformer Architecture for Deck Scoring

**Decision**: Use standard Self-Attention Blocks (SAB) + Pooling by Multihead Attention (PMA) from Lee et al. 2019, implemented directly in PyTorch using `nn.MultiheadAttention`.

**Rationale**: The deck scorer needs a permutation-invariant architecture that accepts variable-length unordered sets of card vectors and produces a single scalar score. The Set Transformer is designed exactly for this: SAB layers let cards attend to each other (learning interactions like "this removal spell is more valuable because the deck is slow"), and PMA uses learned seed vectors to pool card representations into a fixed-size deck vector.

With deck sizes of 20-29 cards, the O(n^2) cost of standard self-attention is negligible (~800 entries). Induced Set Attention Blocks (ISAB), which reduce complexity to O(nm) using inducing points, are unnecessary at this scale and add architectural complexity.

**Alternatives considered**:
- **DeepSets** (Zaheer et al. 2017): Simpler (element-wise transform + sum pooling), but cannot model pairwise card interactions. A 2-card synergy package would be invisible to the model. Rejected because card interactions are central to deck quality.
- **GNN / Message Passing**: Cards as nodes with fully-connected edges achieves similar expressiveness to self-attention but adds graph construction overhead with no benefit for fully-connected sets.
- **Standard Transformer with CLS token**: Requires positional encodings (which would break permutation invariance) or special handling. PMA is a cleaner solution for set-to-scalar mapping.

## R-02: Bradley-Terry Pairwise Loss Implementation

**Decision**: Use `torch.nn.functional.binary_cross_entropy_with_logits` with logit = `score_winner - score_loser` and target = 1.0. This is mathematically equivalent to the Bradley-Terry model: P(A beats B) = sigmoid(score_A - score_B).

**Rationale**: Standard BCE on score differences is the textbook implementation. PyTorch's `binary_cross_entropy_with_logits` is numerically stable (uses log-sum-exp internally), handles extreme score differences without overflow, and supports automatic differentiation through both scores.

**Alternatives considered**:
- **Custom loss with manual sigmoid**: Numerically unstable when score differences are large. No benefit over the built-in function.
- **Margin-based loss (hinge)**: Doesn't produce calibrated probabilities. Bradley-Terry is preferred because the sigmoid interpretation aligns with how Elo-style ratings work.

## R-03: Normalization Statistics as register_buffer

**Decision**: Store per-feature mean and std vectors as PyTorch `register_buffer` on the model. Compute once at training startup from the full training corpus.

**Rationale**: `register_buffer` creates non-trainable tensors that:
- Are included in `model.state_dict()` — saved/loaded with checkpoints automatically
- Move to GPU with `.to(device)` — no manual device management
- Are not included in `model.parameters()` — optimizer ignores them
- Cannot get out of sync with the model weights

This is the standard PyTorch pattern for batch normalization running statistics, and the exact same use case applies here.

**Alternatives considered**:
- **Separate normalization file alongside checkpoint**: Risk of file going missing or out of sync with model. Rejected.
- **Recompute from data at inference**: Requires access to the full training corpus at inference time. Rejected.
- **nn.BatchNorm1d**: Operates per-batch rather than using corpus-wide statistics. Would give different normalization during training vs. inference unless manually overridden.

## R-04: Differential Learning Rates for Embedding Fine-Tuning

**Decision**: Use PyTorch optimizer parameter groups to set separate learning rates for the card embedding lookup table vs. the rest of the model.

**Rationale**: Standard PyTorch pattern. The optimizer accepts a list of parameter group dicts, each with its own `lr`:

```python
optimizer = Adam([
    {'params': scorer_params, 'lr': 1e-3},
    {'params': embedding_params, 'lr': 1e-5},
])
```

When embeddings are frozen (Phase A), the embedding parameters are simply excluded from the optimizer entirely (or their `requires_grad` is set to False).

**Alternatives considered**:
- **Manual gradient scaling**: Multiply embedding gradients by a factor. Fragile, harder to configure, interacts poorly with adaptive optimizers (Adam already scales gradients).
- **Two separate optimizers**: More complex step/zero_grad management. No benefit over parameter groups.

## R-05: Variable-Length Deck Batching with Attention Masking

**Decision**: Within each training batch, pad shorter decks to the length of the longest deck in that batch. Use a boolean attention mask to prevent padding tokens from influencing attention computation.

**Rationale**: PyTorch's `nn.MultiheadAttention` natively supports `key_padding_mask` — a boolean tensor of shape `(batch, seq_len)` where `True` positions are ignored in attention. This is the standard approach for variable-length sequences/sets in PyTorch.

For the PMA pooling layer, the same mask is used as `key_padding_mask` when seed vectors attend over card representations, ensuring padding cards contribute nothing to the deck vector.

**Alternatives considered**:
- **Fixed-length padding to maximum possible deck size**: Wastes computation on padding. Dynamic batching is standard and straightforward.
- **Per-example forward passes (batch_size=1)**: Eliminates padding but prevents GPU parallelism. Not viable for training efficiency.

## R-06: Evaluation Pipeline — Python/Java Coordination

**Decision**: Reuse the existing flat-file coordination pattern from feature 012 (match-outcomes). The Python script writes a validation matches file, splits it across workers, each Java worker processes its subset and writes outcomes to a companion file. Python collects and aggregates.

**Rationale**: The existing match-outcomes pipeline already proves this pattern works: flat-text files for IPC, one file per worker, append-only outcomes. The evaluation pipeline uses the same forge-connector JAR and Forge classes, just with a different entry point (ValidationWorkerMain) that reads two pre-built decks from the match file and plays the game.

Key difference from match-outcomes: workers are finite (process a fixed number of matches and exit) rather than infinite loops. The Python supervisor does a simple run-and-wait with retry, not continuous monitoring.

**Alternatives considered**:
- **Socket-based IPC**: More complex, no benefit for a batch evaluation.
- **Single Java process**: Slower (no parallelism), and Forge JVMs crash — parallelism provides both speed and resilience.

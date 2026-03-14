# Research: Transformer Model Architecture

**Feature Branch**: `007-transformer-model-arch`
**Date**: 2026-03-14

## 1. Maximum Sequence Length (`max_seq_len`)

**Decision**: Computed automatically at the start of each training run from the actual dataset. Not hardcoded.

**Algorithm** (executed as the first step of `train-transformer`, before building `TransformerTrainingDataset`):

1. Load all matched card texts from `output/` (same card set that will be used for training).
2. Tokenize every card with `BertTokenizer("bert-base-uncased").encode()` (no padding, no truncation).
3. Compute the distribution of raw token counts (including the `[CLS]` token added by `.encode()`).
4. Set `max_seq_len` to the **95th percentile** of the distribution, rounded up to the nearest multiple of 8 (for GPU memory alignment). Clamp to a minimum of 64.
5. Log to CLI: chosen `max_seq_len`, 95th/99th/max percentile values, and the percentage of cards that will be truncated.
6. Store the computed `max_seq_len` in `TransformerConfig`, which is saved inside the `.pt` artifact. Inference and evaluation reload this value from the artifact — they never recompute it.

**Why per-run, not hardcoded**: The output corpus changes when cards are re-converted (feature 006 updates). A stale hardcoded value could silently truncate new cards or waste memory on padding. Computing from the data ensures the value is always appropriate.

**Expected range**: 256–384 tokens based on current card corpus. Starting estimate 256 (used only for VRAM budget calculations in the spec, not in code).

**Rationale**: The converted card scripts in `output/*.txt` vary from short vanilla creatures (~20 tokens) to complex cards with extensive triggered/activated abilities (~300+ tokens). The BERT WordPiece tokenizer (`bert-base-uncased`) will further expand some MTG-specific terms (e.g., "trample" stays as one token, but "hexproof" may split into "hex" + "##proof").

**Alternatives considered**:
- Fixed 512 (BERT default): Wasteful — most cards are much shorter, and longer sequences increase VRAM usage quadratically in attention.
- Fixed 128: Too short — would truncate many cards with multiple abilities.
- Fixed 256: Could silently under- or over-fit if the corpus changes. Data-driven is safer.

## 2. Training Data Pipeline: Matching Card Scripts to Prices

**Decision**: Reuse the existing `forge_parser.py` + `mtgjson_loader.py` pipeline to build the card→price mapping, then load the corresponding converted text file from `output/` for each matched card.

**Rationale**: The existing pipeline (feature 001) already handles:
- Parsing Forge card scripts to extract card names
- Building name→UUID mappings from `AllPrintings.json`
- Building name→price mappings from `AllPricesToday.json`
- Handling split card naming conventions

The transformer pipeline adds one step: for each card matched to a price, resolve the corresponding `output/<first_letter>/<slug>.txt` file to get the tokenizer input text. Cards without a matching output file are skipped (with logging).

**Alternatives considered**:
- Build a separate data loader from scratch: Rejected — duplicates effort and risks diverging from the sklearn model's dataset, making evaluation comparison invalid.
- Index output files directly and reverse-match to prices: Rejected — the existing pipeline is the authoritative source for which cards have prices.

## 3. Shifted-Log Price Transform

**Decision**: Train on `log(price + 2)`, convert output via `exp(x) - 2`.

**Rationale**: The spec defines this clearly. The shift of 2 EUR (the community "bulk" threshold) compresses price differences below ~2 EUR so the model doesn't waste capacity distinguishing between €0.02 and €0.10 cards.

Note: The existing sklearn model uses `np.log(price)` (no shift). This is a deliberate difference — the two models use different target spaces. The transformer evaluation must use `exp(x) - 2` to convert predictions back to EUR, while the sklearn evaluation uses `exp(x)`.

**Alternatives considered**:
- Same `log(price)` as sklearn: Rejected by spec — the shifted-log was chosen specifically for the transformer to focus on meaningful price ranges.
- Raw EUR prices: Rejected — MSE loss on raw prices would be dominated by expensive cards.

## 4. VRAM Budget Verification

**Decision**: The architecture (~2M parameters) with batch size 64 fits comfortably in 8GB VRAM under fp32 training.

**Rationale**: Memory breakdown estimate:

| Component | Size (fp32) |
|---|---|
| Model parameters | ~8 MB (2M × 4 bytes) |
| Gradients | ~8 MB |
| AdamW optimizer states (m, v) | ~16 MB |
| Activations (batch 64, seq 256, d_model 128, 4 layers) | ~100–200 MB |
| Input batch (token IDs, int64) | ~0.1 MB |
| **Total estimate** | **~250–350 MB** |

This leaves >7 GB headroom. Mixed precision (fp16) is not required but could be used as a safety margin if `max_seq_len` increases significantly.

**Alternatives considered**:
- Mixed precision by default: Not needed given the large headroom. Can be added later if `max_seq_len` exceeds 512 or batch size needs increasing.
- Gradient accumulation: Not needed — full batch of 64 fits in memory.

## 5. BERT Tokenizer Integration

**Decision**: Use `transformers.BertTokenizer.from_pretrained("bert-base-uncased")` at both training and inference time. Cache the tokenizer locally.

**Rationale**: The BERT WordPiece tokenizer provides:
- 30,522 vocabulary entries (fixed, pre-trained)
- Built-in `[CLS]` (token ID 101) and `[PAD]` (token ID 0) tokens
- `tokenizer.encode()` with `max_length`, `truncation=True`, `padding="max_length"` handles all sequence formatting

The tokenizer is loaded once at model initialization (both training and inference). On first use, the `transformers` library downloads and caches the vocabulary files (~230 KB). Subsequent loads are from cache.

**Alternatives considered**:
- Custom tokenizer trained on MTG text: Rejected by spec — BERT tokenizer provides good English coverage and avoids a custom training step.
- Byte-pair encoding (GPT-2 tokenizer): Rejected — BERT tokenizer is simpler and the spec explicitly names it.

## 6. Model Artifact Format

**Decision**: Save as a single `.pt` file containing a dict with keys `"state_dict"` and `"config"`.

**Rationale**: The config dict stores all hyperparameters needed to reconstruct the model architecture (`d_model`, `n_layers`, `n_heads`, `ff_dim`, `max_seq_len`, `vocab_size`, `dropout`). This makes the artifact self-describing — loading requires no external configuration.

```python
torch.save({
    "state_dict": model.state_dict(),
    "config": {
        "d_model": 128,
        "n_layers": 4,
        "n_heads": 4,
        "ff_dim": 512,
        "max_seq_len": 256,  # actual value
        "vocab_size": 30522,
        "dropout": 0.1,
    },
}, "models/transformer/model.pt")
```

**Alternatives considered**:
- `torch.save(model)` (pickle entire model): Fragile — breaks if class definition changes. State dict is the PyTorch-recommended approach.
- ONNX export: Rejected — adds complexity without benefit for this use case (no cross-framework serving needed).

## 7. API Response Format Change (FR-012)

**Decision**: The `POST /api/v1/evaluate` response changes from flat fields to a nested structure grouped by model type:

```json
{
  "sklearn": {
    "predicted_price_eur": 2.35,
    "model_version": "20260301-143601"
  },
  "transformer": {
    "predicted_price_eur": 2.18,
    "model_version": "transformer-v1"
  }
}
```

**Rationale**: FR-012 requires both models to always run with both predictions returned. A nested structure groups each model's outputs cleanly, avoids field name proliferation (no `predicted_price_eur_sklearn` / `predicted_price_eur_transformer`), and makes it trivial to add future models as new top-level keys. The `transformer` object is `null` if the model artifact is unavailable (graceful degradation).

This is a **breaking change** to the API response format. The `eval` CLI subcommand (which consumes this endpoint) must be updated to read from `response["sklearn"]["predicted_price_eur"]` instead of `response["predicted_price_eur"]`.

**Migration approach**:
- Update `create_app()` to accept both model artifacts
- Run sklearn prediction as before, nest under `"sklearn"` key
- Attempt transformer prediction; nest under `"transformer"` key or set to `null` on failure
- Update `run_eval()` in CLI to display both prices from nested structure
- Update all server tests to expect new nested response shape

**Alternatives considered**:
- Flat fields with model-prefixed names (`predicted_price_eur_sklearn`, etc.): Rejected — field proliferation scales poorly with additional models and mixes concerns at the same JSON level.
- Version the endpoint (v2): Overkill — this is an internal API with no external consumers yet.

## 8. CLI Subcommand Design

**Decision**: Add `train-transformer` and `evaluate-transformer` as new subcommands alongside existing `train` and `evaluate`.

**Rationale**: The transformer model has a fundamentally different training pipeline (PyTorch, BERT tokenizer, GPU, epochs with early stopping) vs. the sklearn model (feature engineering, GBR). Separate subcommands keep each pipeline self-contained and avoid overloading the existing `train` subcommand with conditional logic.

**CLI interface**:
```
price_predictor train-transformer
    --output-dir PATH       (default: output/)
    --prices-path PATH      (default: resources/AllPricesToday.json)
    --printings-path PATH   (default: resources/AllPrintings.json)
    --forge-cards-path PATH (default: ../forge/forge-gui/res/cardsfolder/)
    --model-output PATH     (default: models/transformer/)
    --batch-size INT        (default: 64)
    --epochs INT            (default: 20)
    --lr FLOAT              (default: 1e-4)
    --patience INT          (default: 5)
    --random-seed INT       (default: 42)

price_predictor evaluate-transformer
    --model-path PATH       (default: models/transformer/)
    --output-dir PATH       (default: output/)
    --prices-path PATH      (default: resources/AllPricesToday.json)
    --printings-path PATH   (default: resources/AllPrintings.json)
    --forge-cards-path PATH (default: ../forge/forge-gui/res/cardsfolder/)
    --random-seed INT       (default: 42)
    --output-csv PATH       (optional)
```

**Alternatives considered**:
- Extend existing `train` with `--model-type` flag: Rejected — mixes two different training pipelines in one handler, violates Simplicity First.
- Standalone `train_transformer.py` script: Rejected by spec — must be a subcommand on the existing CLI entry point.

## 9. Device Handling (GPU/CPU Fallback)

**Decision**: Training requires CUDA GPU and fails fast with a clear error if unavailable. Inference auto-detects device (GPU if available, CPU otherwise).

**Rationale**: Per spec (FR-013), GPU is required for training but inference supports CPU fallback. Implementation:

```python
# Training
device = torch.device("cuda" if torch.cuda.is_available() else None)
if device is None:
    raise RuntimeError("CUDA GPU required for training. No GPU detected.")

# Inference
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
```

The model artifact stores the config but not the device — the device is always determined at load time.

**Alternatives considered**:
- Allow CPU training: Rejected by spec — training on CPU would be impractically slow (~hours vs. minutes on GPU).
- MPS (Apple Silicon) support: Not required — spec targets GeForce RTX 3060 Ti specifically.

## 10. Positional Encoding

**Decision**: Use learned positional embeddings (`nn.Embedding(max_seq_len, d_model)`), not sinusoidal.

**Rationale**: The spec says "learned positional encoding." Learned embeddings are simpler to implement (just another `nn.Embedding` layer) and sufficient for the fixed `max_seq_len` used in this model. They are added to the token embeddings before the first transformer layer.

**Alternatives considered**:
- Sinusoidal positional encoding (Vaswani et al.): Rejected — spec explicitly says learned. Also, sinusoidal is primarily useful for extrapolating to unseen sequence lengths, which is not needed here (all inputs are ≤ `max_seq_len`).
- Rotary position embeddings (RoPE): Over-engineered for this model size and task.

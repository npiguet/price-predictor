# Research: Encoder Fine-Tuning (Phase B) for Sealed Scorer

**Feature**: 015-encoder-fine-tuning
**Date**: 2026-04-29

## Codebase Survey

### Overlapping Domain Vocabulary

| Existing Concept | Location | Decision | Notes |
|---|---|---|---|
| `TrainScorerConfig` | `src/sealed/application/train_scorer.py:33` | **Extend** | Add `embedding_lr`, `encoder_checkpoint`, `scorer_checkpoint`, `patience` fields; remove `unfreeze_embeddings`. The `embedding_lr` field already exists (default `1e-5`) — repurposed: `0` keeps encoder frozen (Phase A), non-zero opens Phase B. Default flips to `0`. |
| `TrainingMetrics` | `src/sealed/application/train_scorer.py:71` | **Reuse** | Already carries `embedding_drifts: list[float]`. Phase B writes one entry per epoch. Spec FR-012 explicitly names this field. |
| `EpochStats` | `src/sealed/application/train_scorer.py:79` | **Extend** | Replace per-component scorer-only `grad_norms: dict[str, float]` with two grouped values: scorer-group L2 norm and encoder-group L2 norm. (`_gradient_norms()` is currently scorer-internal-only.) |
| `ResumeState` | `src/sealed/application/train_scorer.py:96` | **Extend** | Carries scorer model + optimizer state. Add encoder weights (loaded from resumed checkpoint or from `--encoder-checkpoint`) and the per-resume phase indicator so phase-lock validation lives in one place. |
| `_TrainingContext` | `src/sealed/application/train_scorer.py:111` | **Extend** | Add the live `CardPriceTransformerModel` (encoder), the per-card text/token cache, and the step-0 reference embeddings (replacing the current `initial_embeddings` lookup-table snapshot). |
| `LoadedScorerCheckpoint` | `src/sealed/infrastructure/scorer_store.py:19` | **Extend** | Add an optional `encoder_state_dict: dict[str, Any] \| None` field. Phase A checkpoints leave it `None`; Phase B checkpoints populate it. The `config` dict is widened to carry every training flag (FR-009). |
| `ScorerStore.save_checkpoint` / `load_checkpoint` | `src/sealed/infrastructure/scorer_store.py:33` | **Extend** | Save accepts an optional encoder state dict and a richer config payload. Load returns the new field. |
| `EmbeddingTable` | `src/sealed/infrastructure/match_data_loader.py:78` | **Reuse (with role change)** | In Phase A this lookup table holds frozen `.npz` rows and is iterated for batch lookups. In Phase B the same table still holds the *current* per-card vectors (pre-deterministic-feature concat is unchanged), but each batch step **overwrites the rows referenced in that batch** with freshly-encoded vectors before scorer forward pass. The `unfreeze()`/`is_frozen()` API is removed — the encoder, not the table, is the parameter group. The table itself is no longer in any optimizer's parameter list. |
| `CardEncoder` (sealed) | `src/sealed/domain/card_encoder.py:14` | **Extend** | Currently wraps the encoder in a `@torch.no_grad()` `encode()` for one-card-at-a-time inference. Phase B needs (a) batched encoding, and (b) a **gradient-tracking** path that returns the live `(2*d_model,)` text vector (deterministic features still concatenated downstream). The two-call surface stays inside the same module; the existing single-card no-grad call site (`encode-cards` use case) is unchanged in semantics. |
| `CardPriceTransformerModel.encode` | `src/price_predictor/infrastructure/transformer_model.py:73` | **Reuse + add a sibling** | The existing `@torch.no_grad()` `encode()` is the right helper for `encode-cards`. Phase B needs the same pooling logic *with* gradients — call `_encode_and_pool` directly (already a `nn.Module` method). The `nn.Module` is the encoder parameter group: token embedding, position embedding, `encoder` (Transformer stack), output dropout. The output head and meta-projection are not in the parameter group (the scorer replaces them). |
| `ConvertedCardLocator` | `src/sealed/infrastructure/converted_card_locator.py:27` | **Reuse + extend** | Already resolves card names → `.txt`/`.npz`. Phase B needs the `.txt` path for every card seen in any training example (so we can tokenize). Use the existing `load_text` and `text_path` methods. Token IDs/masks are computed once per unique card across the run and cached. |
| `MtgTokenizer` | `src/price_predictor/domain/tokenizer.py:18` | **Reuse** | Loaded once via `load_tokenizer(vocab_path)` — same call site as `encode-cards`. |
| `MatchTrainingExample` / `TrainingBatch` / `collate_training_examples` | `src/sealed/infrastructure/match_data_loader.py` | **Reuse** | The (winner_indices, loser_indices, masks) shapes are unchanged — the indices still point into `EmbeddingTable`. Phase B re-fills the relevant rows of the table from the encoder before scorer forward pass; downstream code is identical. |
| `EncodeCardsConfig` | `src/sealed/application/encode_cards.py:13` | **Reuse** | Already carries `cards_path` and `clean`. The new flag plumbing happens in `cli.py`/`run_encode_cards`; the `EncodeCardsUseCase` itself takes an `encoder` object — no change to that use case. |
| `EncodeCardsUseCase` | `src/sealed/application/encode_cards.py:25` | **Reuse** | Already idempotent; spec confirms (FR/clarification) that `--clean` continues to be the user's tool to force a refresh. No semantic change required. |
| `CardEncoder.__init__` | `src/sealed/domain/card_encoder.py:22` | **Reuse** | The `(model, tokenizer, max_seq_len)` constructor is the only assembly site needed. The new `--scorer-checkpoint` flag changes *where the model state_dict comes from*, not the encoder shape. |
| Reference batch / `embedding_drift` | `src/sealed/application/train_scorer.py:507` `_embedding_drift()` | **Replace logic, keep field name** | The current implementation diffs the *full* `EmbeddingTable.embedding.weight` against a snapshot — it's the per-card lookup-row drift used by the `--unfreeze-embeddings` path. Spec § 7 redefines the metric: a fixed reference batch (the unique cards in step 0's first Phase B batch) re-encoded each epoch, distance against their step-0 vectors. The `embedding_drifts: list[float]` field on `TrainingMetrics` is reused (FR-012); a new helper computes it. |
| `_select_device` / `_move_optimizer_state` | `src/sealed/application/train_scorer.py:370,376` | **Reuse** | Both phases share these. The encoder module gets `.to(device)` alongside the scorer. |
| `CardPriceTransformerModel.config.max_seq_len` | reachable via `load_model` return | **Reuse** | Same plumbing as `encode-cards` — the encoder's `max_seq_len` controls tokenization length. |

No parallel domain concepts introduced. The notable convergence: the `EmbeddingTable.unfreeze()` API and the `--unfreeze-embeddings` flag both go away — Phase B replaces lookup-table fine-tuning with encoder fine-tuning, and the spec's FR-002 is explicit about removing the boolean flag.

### Adjacent Prior Art

#### Encoder-loading pipeline (already in use by `encode-cards`)

- `cli.py:314-370` (`run_encode_cards`) — loads the price-predictor transformer via `load_model(encoder_path)`, the tokenizer via `load_tokenizer(vocab_path)`, and assembles a `CardEncoder`.
- **Reuse**: `train-scorer` Phase B does *exactly* the same load (FR-003 default points at the same `latest.pt`) — the helper functions are reusable as-is.
- **New twist**: when `--scorer-checkpoint <phaseB>.pt` is supplied to `encode-cards`, the encoder weights come from a different file. The model class (`CardPriceTransformerModel`) and config (`TransformerConfig`) are still loaded from the price-predictor side; only `model.load_state_dict(...)` reads from the scorer checkpoint's `encoder_state_dict` slice.

#### Checkpoint persistence

- `src/price_predictor/infrastructure/torch_checkpoint.py` provides shared `save_checkpoint(path, payload, config)` / `load_checkpoint(path, config_cls)` helpers. Both `ScorerStore` (sealed) and `transformer_store` (price-predictor) build on these.
- **Reuse**: Phase B checkpoints continue using `save_checkpoint` — just with a richer `payload` dict (new `encoder_state_dict` key) and a richer `config` (the dataclass `ScorerConfig` now carries every training flag, OR a separate `config` dict is passed in. See Decision §1 below).

#### Optimizer with multiple parameter groups

- The current `_build_optimizer` (`src/sealed/application/train_scorer.py:337`) already conditionally constructs a two-group `Adam` optimizer when `EmbeddingTable.unfreeze()` was called: scorer params at `--lr` and lookup-table params at `--embedding-lr`. **The shape is correct; only the second group's contents change** — encoder parameters instead of `EmbeddingTable.parameters()`.
- **Reuse**: Same dispatch (`if encoder is in graph → two groups else one group`), same `_move_optimizer_state` helper for resume.
- **Switch**: `Adam` → `AdamW` for both branches (FR-005a). The pre-existing `Adam` is the only class swap; momentum/variance buffer shapes carry across cleanly within the same per-group state-loading flow.

#### CLI flag registration

- `add_dataclass_arg` (`src/price_predictor/infrastructure/cli_helpers.py:10`) keys CLI defaults off dataclass fields. All Phase B flags are dataclass fields on `TrainScorerConfig`/(none for `EncodeCardsConfig`), so registration follows the existing pattern.
- **Reuse**: All four new `train-scorer` flags (`--embedding-lr`, `--encoder-checkpoint`, `--scorer-checkpoint`, `--patience`) and the two new `encode-cards` flags use `add_dataclass_arg` or plain `add_argument` (matching the existing `--encoder-path` / `--vocab-path` pattern).
- **Naming nudge**: the existing `encode-cards` argument is `--encoder-path`; the spec uses `--encoder-checkpoint`. Decision: rename `--encoder-path` → `--encoder-checkpoint` to match (one CLI rename, mirrors `train-scorer`'s `--encoder-checkpoint`). Decision logged below.

#### Validation cadence + early stopping

- The existing `--val-interval` flag and the `if (epoch - ctx.start_epoch + 1) % config.val_interval == 0` block already gate validation. Spec § Clarifications fixes the cadence at "once per epoch", and FR-011 introduces `--patience` (epochs since last new peak `val_acc`).
- **Reuse**: The validation block stays. `--val-interval` becomes redundant. Decision: keep `--val-interval` as a deprecated no-op or remove it. See Decision §6.

#### Within-batch caching (no prior art)

- Nothing in the codebase batches per-unique-card encoder calls. This is a new helper that lives close to `train_scorer.py` because it knows the batch's index→card-name mapping (held by `EmbeddingTable.name_to_idx`). Implementation: deduplicate the `(winner_indices ∪ loser_indices)` set in a batch, run the encoder once per unique card on a padded `(unique_cards, max_seq_len)` token tensor, and write the resulting `(2*d_model,)` rows back into the per-card vectors before deterministic-feature concat.

### Convention Alignment

**Sibling module to mirror**: The `sealed` package — same hexagonal layout (`domain` / `application` / `infrastructure`).

| Convention | Pattern | Source |
|---|---|---|
| CLI registration | `_build_X_parser(subparsers)`, `set_defaults(func=run_X)` | `cli.py:154-227` |
| Application use case | `XUseCase.execute(config: XConfig) -> XResult` | `train_scorer.py:170` |
| Config dataclass | Fields with defaults, optional methods (`scorer_config()`, `best_checkpoint_name()`). | `train_scorer.py:33` |
| Persistence | Domain-agnostic `save_checkpoint`/`load_checkpoint` in `price_predictor.infrastructure.torch_checkpoint` | `torch_checkpoint.py` |
| Test style | `unittest.mock.MagicMock` for adapters; `tmp_path` for file I/O; class-per-behavior. | `test_train_scorer.py`, `test_encode_cards.py` |
| Logging | `_log(message)` timestamps every line. End-of-epoch report goes through `_print_epoch_report`. | `train_scorer.py:27,513` |

**Deviation**: None anticipated. Phase B implementation follows existing patterns.

### Third-Instance Check

| Sub-problem | Instance 1 | Instance 2 | Action |
|---|---|---|---|
| Encoder weight loading | `transformer_store.load_model()` (price-predictor side, full file) | `encode-cards`/`run_encode_cards` (passes through `load_model`) | Phase B's `train-scorer` and `encode-cards --scorer-checkpoint` both extract `encoder.state_dict` from a Phase B checkpoint. **Not a third instance**: both read from a *scorer* checkpoint, not a price-predictor one. The shared helper is `LoadedScorerCheckpoint.encoder_state_dict` access — one place. No new abstraction needed. |
| Optimizer two-group construction | `_build_optimizer` (current `--unfreeze-embeddings` path) | (none) | Phase B replaces the `EmbeddingTable.parameters()` second group with the encoder's. Same function, same dispatch — not a duplication. |
| Reference-batch capture for drift | `_embedding_drift()` (lookup-table snapshot) | (none) | Spec replaces the algorithm. One call site. Not a third instance. |
| Token+mask precomputation | `CardEncoder.encode()` (single card, no-grad) | (none) | Phase B needs batched + gradient-tracking. Not a duplication: the no-grad path stays (used by `encode-cards`); the new path is a batched sibling. |

No third instances found. No shared abstraction extraction needed.

## Design Decisions

### Decision 1 — `config` payload widening for self-describing checkpoints (FR-009)

**Decision**: Persist a flat `config` dict containing every `train-scorer` CLI flag value alongside the existing `ScorerConfig` (architecture-only) dataclass. Implementation: extend `ScorerStore.save_checkpoint` to accept the full `TrainScorerConfig` (or a `dict[str, Any]` of its fields) and serialize it under a new `train_config` key in the payload, while the architecture-only `ScorerConfig` continues to be serialized under `config` for backward compatibility with existing checkpoints. On `--resume`, the resumed `train_config` is loaded as defaults; CLI-supplied flags override per FR-010.

**Rationale**: The existing `ScorerConfig` only holds architecture (`d_model`, `n_layers`, `n_heads`, `n_seeds`, `d_ff`, `mlp_hidden`, `dropout`); making it carry optimizer/data/schedule flags would mix domain (model shape) and infra concerns (training schedule) into one dataclass and force every scorer constructor to take training-only fields. A separate `train_config` dict is a cleaner split: domain dataclass stays minimal; infrastructure persists the full training context.

**Alternatives considered**:
- Widen `ScorerConfig`: rejected — pollutes the domain entity that downstream code (build-decks, evaluate-scorer, scorer_model itself) depends on.
- Reconstruct flags from the optimizer state: rejected — only learning rates survive; architecture/schedule flags don't.

### Decision 2 — Encoder weight loading on a fresh Phase B run

**Decision**: Always load encoder weights when starting Phase B (`--embedding-lr` non-zero, `--scorer-checkpoint` present). The default `--encoder-checkpoint` value resolves to `models/price-predictor/transformer/latest.pt`. The encoder is constructed with the price-predictor's `TransformerConfig` from that file, then `model.load_state_dict(...)` populates the weights. On `--resume <phaseB>.pt`, encoder weights come from the resumed checkpoint's `encoder_state_dict`; the price-predictor file is not touched.

**Rationale**: Phase A trained against `.npz` files produced by a specific encoder checkpoint. The fresh Phase B bootstrap MUST start from the *same* encoder so the first scorer forward pass matches the cached `.npz` rows the scorer learned to score. The price-predictor's `latest.pt` is the stable pointer to that encoder; default-resolving it eliminates an easy-to-forget user step.

**Alternatives considered**:
- Bake the encoder weights into the Phase A checkpoint: rejected — Phase A doesn't run the encoder, so persisting weights it never updates is misleading and would mask the moment Phase A's `.npz` cache and the price-predictor's encoder diverge.
- Make `--encoder-checkpoint` mandatory on Phase B: rejected — every Phase B kickoff would repeat the same path; defaulting is more ergonomic and the spec explicitly authorizes a default (FR-003).

### Decision 3 — Mutual-exclusivity rule for `--encoder-checkpoint` in Phase B `--resume`

**Decision**: The validation MUST distinguish "user explicitly passed `--encoder-checkpoint`" from "argparse filled in the default". `argparse.Namespace` doesn't expose this, so register `--encoder-checkpoint` with `default=None` and resolve to `models/price-predictor/transformer/latest.pt` *after* the conflict check. If `args.encoder_checkpoint is not None and args.resume is not None and resumed-checkpoint-is-Phase-B`, reject. Otherwise apply the default.

**Rationale**: FR-004 carves out the default value from the conflict, so we need a way to tell "user passed the flag" from "default was used". Late-resolving the default is the standard argparse pattern; it lives in `run_train_scorer` (CLI layer) where the `args.encoder_checkpoint is None` check is unambiguous.

**Alternatives considered**:
- Use `argparse.Namespace`'s `vars(args)` membership check: rejected — `add_argument` always adds the dest; the explicit-pass signal is lost.
- Sentinel object as default: rejected — same pattern as `None`, with extra ceremony.

### Decision 4 — Encoder cache lifetime (FR-007)

**Decision**: The encoder cache is a `dict[int, torch.Tensor]` (card row → encoder text vector) constructed *inside* the per-batch step. After the optimizer step it goes out of scope and PyTorch's autograd graph is released. The cache itself does not survive between batches — there is no instance attribute holding it across steps.

**Rationale**: Spec § Edge Cases is explicit: "the encoder cache must clear between training steps so each step builds a fresh computation graph". A function-local dict is the simplest correct implementation; gradients accumulate naturally because PyTorch's autograd already supports duplicate references to the same tensor (each loss path adds to the encoder parameters' `.grad`).

**Alternatives considered**:
- Caching across the whole epoch and `retain_graph=True`: rejected — explodes memory and breaks the standard one-graph-per-step pattern.
- Caching token IDs but re-running the encoder per reference: rejected — spec § 3 calls out the 2–10× speedup from collapsing duplicate forward passes.

### Decision 5 — Reference-batch drift metric (FR-012)

**Decision**: At step 0 (the very first Phase B forward pass), capture the unique card row indices in the batch and the encoder output for each (a `(num_unique, 2*d_model)` tensor). Detach and store on the device. At end of every epoch, run a separate forward pass with `model.eval()` over the same token IDs (cached at capture time) and compute mean L2 distance to the stored step-0 vectors.

**Rationale**: "Captured during that step's forward pass before the optimizer step" (per spec clarification 3) means we read the encoder output we already computed for free, before `optimizer.step()` shifts the weights. No separate reference-batch sampling code; no extra forward passes at step 0.

**Alternatives considered**:
- Capture before training starts (separate forward pass over a fixed pool sample): rejected — needs a separate dataset path and adds a "step -1" baseline that doesn't reflect the actual training loop.
- Use only step-0 token IDs and re-tokenize each epoch: equivalent in result, more redundant work — rejected for simplicity.

### Decision 6 — `--val-interval` removal

**Decision**: Remove `--val-interval` from `train-scorer`. Validation runs once per epoch unconditionally. The `--patience` flag (FR-011) replaces interval-based skipping: if the user wants fewer "expensive" validation runs they can shrink the validation set, not the cadence.

**Rationale**: Spec § Clarifications: "Once per epoch (end of epoch). … `--patience` counts epochs without a new peak `val_acc`". A per-N-epoch interval would make `--patience` ambiguous (epochs since last validation? since last peak? since last skipped peak?) and adds no real benefit — validation is fast (a single pass over 20% of the corpus).

**Alternatives considered**:
- Keep `--val-interval` and clamp `--patience` to count validation events: rejected — bookkeeping for two cadences with overlapping intent.
- Default `--val-interval=1` and let users override: rejected — same ambiguity, less explicit.

### Decision 7 — Renaming `encode-cards --encoder-path` → `--encoder-checkpoint`

**Decision**: Rename the existing `encode-cards --encoder-path` flag to `--encoder-checkpoint` so both subcommands speak the same vocabulary. Drop the deprecated alias — the `sealed` CLI doesn't promise external API stability and there are no checked-in scripts in the repo using `--encoder-path`.

**Rationale**: Spec FR-013 names the flag `--encoder-checkpoint` for `encode-cards`; the codebase had it as `--encoder-path`. Aligning eliminates a vocabulary mismatch between two flags that mean the same thing.

**Alternatives considered**:
- Add `--encoder-checkpoint` as an alias and keep `--encoder-path`: rejected — feature ships with one stable name.

### Decision 8 — `EmbeddingTable` post-Phase-B role

**Decision**: `EmbeddingTable` keeps its current role as the *batch-time* card-vector container, but in Phase B its `embedding.weight` rows are *overwritten* in each step by the encoder output (text slice) concatenated with the deterministic-feature slice. Concretely: the table is built from `.npz` files at startup as today, but in Phase B each batch step (a) computes encoder text vectors for the unique cards in that batch, (b) writes them into `embedding.weight[unique_indices, :2*d_model]`, and (c) calls `embedding(...)` to read the per-deck index sequences. The deterministic-feature slice (last `FEATURE_COUNT` columns) is never touched.

**The `EmbeddingTable.unfreeze()` / `is_frozen()` API is removed.** The encoder is the parameter group; the lookup table is no longer trained directly.

**Rationale**: Reusing the table preserves the indexing/masking pipeline that `MatchTrainingExample` and `collate_training_examples` already build on top of — the per-deck card-index plumbing doesn't change between phases. Writing into `embedding.weight` (with `requires_grad=False` on the table itself) keeps autograd working: each batch's gradients flow back through the writes' source tensors (the encoder output) into the encoder parameters, not into the table.

**Alternatives considered**:
- Drop the lookup table in Phase B and feed scorer with `(batch, max_cards, d_model)` tensors built directly from encoder outputs: rejected — introduces a second batch-assembly path, duplicating padding/masking code that `collate_training_examples` already handles.
- Keep `unfreeze()` for compatibility: rejected — FR-002 is unambiguous about removing the boolean flag, and the table is no longer something the user can choose to fine-tune separately.

### Decision 9 — AdamW migration (FR-005a)

**Decision**: Switch the optimizer from `Adam` to `AdamW` in *both* phases. Phase A keeps a single parameter group at `--lr`; Phase B builds two groups (scorer at `--lr`, encoder at `--embedding-lr`). No weight-decay flag is exposed; `AdamW` runs at its default `weight_decay=0.01`. The Adam → AdamW switch is a hard cut — pre-feature Phase A checkpoints (which carry Adam optimizer state) are not expected to be resumable; the spec § Assumptions explicitly states Phase A will be retrained from scratch.

**Rationale**: Single optimizer family across phases is what FR-005a calls for; it removes an "optimizer-class swap" guard from the resume path. AdamW's decoupled weight decay is the modern default for transformer fine-tuning and gives the encoder a small, free regularizer that pulls weights back toward zero — useful counterweight to the catastrophic-forgetting risk in spec § Risks.

**Alternatives considered**:
- Keep Adam: rejected — the spec mandates AdamW.
- Make weight decay a CLI flag: rejected (YAGNI) — the default is fine for this feature; expose later if a sweep proves it useful.

### Decision 10 — Deterministic train/val split (FR-011a)

**Decision**: The split is already deterministic — `_load_dataset` calls `train_test_split(..., random_state=config.random_seed, shuffle=True)` with `random_seed=42`. Keep that mechanism. The split inputs (the parsed `MatchOutcome` list) are themselves deterministic given a fixed `match-outcomes.txt`. No code change required for this requirement; capture it as an existing-behavior assertion in tests.

**Rationale**: The existing `random_seed=42` field on `TrainScorerConfig` is the seed referenced by FR-011a. Phase A and Phase B end up with the same split because `random_seed` defaults to `42` everywhere and the split inputs are corpus-derived.

**Alternatives considered**:
- Hash the corpus contents into a separate seed: rejected — adds machinery without changing behavior; the fixed default seed is already functionally identical.

### Decision 11 — Per-parameter-group gradient clipping (FR-008)

**Decision**: After `loss.backward()` and before `optimizer.step()`, call `torch.nn.utils.clip_grad_norm_(group_params, max_norm=1.0)` once per parameter group. In Phase A there is one group (scorer); in Phase B there are two (scorer, encoder). The two groups are clipped independently.

**Rationale**: FR-008 + spec clarification mandates per-group clipping in both phases. Independent clipping prevents an encoder gradient spike (which can happen with cold encoder LR-up at step 0) from squeezing scorer gradients to nothing, and vice versa.

**Alternatives considered**:
- Single global `clip_grad_norm_` on `model.parameters() + encoder.parameters()`: rejected — couples the two groups.
- Per-parameter clipping (`clip_grad_value_`): rejected — spec is explicit about max-norm, and value-clipping is generally noisier.

## Open Risks

- **Token-level cache memory**: at ~26K cards × `max_seq_len=128` tokens × int64 = ~26MB token IDs + masks, plus per-card encoder outputs at ~26K × 512 floats = ~52MB. Both fit easily on CPU; only the per-batch active outputs go on GPU.
- **Per-step throughput**: spec § Caching savings projects 2–4× slowdown vs Phase A. The `--patience` early-stop window (default 5) keeps the absolute wall-clock cost predictable; a typical Phase B run is 5–15 epochs.
- **Optimizer state size**: AdamW two-group state = scorer params + encoder params (~3M each, 2× for moments) = ~50MB on disk. Within the existing checkpoint footprint.

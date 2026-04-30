# Goal

Fine-tune the price-predictor encoder during sealed scorer training so that card
embeddings shift from "what predicts a card's market price" to "what predicts deck
quality in sealed". This is the proper implementation of the "Phase B" stage
referenced in the embedding schedule of `sealed-deck-picker.md`.

# Background

The sealed scorer is built in two phases:

- **Phase A — Frozen embeddings.** The Set Transformer + scoring MLP train on top
  of fixed card embeddings produced ahead of time by `python -m sealed encode-cards`,
  which runs the price-predictor transformer over every card's text and writes a
  `.npz` file per card.
- **Phase B — Embedding fine-tuning.** Once Phase A plateaus on validation loss,
  the embeddings are unfrozen so they can shift to encode deckbuilding-relevant
  features (complementarity, role identification, format-relative value).

The current `--unfreeze-embeddings` flag flips `requires_grad = True` on the
precomputed embedding lookup table; the price-predictor encoder that produced
those vectors is not in the training graph. This spec replaces that decoupled
vector fine-tuning with proper encoder fine-tuning.

# Specification

This section is prescriptive. Everything below is what an implementer must
build; rationale and tradeoffs are deferred to the next section.

## 1. Forward and Backward Pass (Phase B)

For each training example (one match outcome):

1. For each of the 46 cards (23 nonland cards × 2 decks), look up its
   converted card text and tokenize it.
2. Run the encoder forward pass: token embeddings → 2 SAB layers →
   `cat([max_pool, mean_pool])`. Output shape: `(2 * encoder_d_model,)`.
3. Concatenate the encoder output with the deterministic feature vector
   parsed from the same card text (per `sealed-deck-picker.md` § Card
   Representation).
4. Pass the resulting per-card vectors through the scorer's SAB stack and
   PMA pooling, then through the scoring MLP.
5. Compute the Bradley-Terry pairwise loss on the `(score_winner, score_loser)`
   pair. Backpropagate through the entire graph: scorer → encoder → token
   embedding table.

Non-differentiable components: the tokenizer (gradients stop at the token
embedding lookup) and the deterministic feature parser (its outputs are
constant).

The encoder is run in **eval mode** during Phase B training so dropout
does not add stochasticity. Cached `.npz` files were produced by
`encode-cards` under `encoder.eval()`; running the encoder in train mode
during Phase B would feed the scorer noisier vectors than it was Phase-A
tuned to score, dragging early Phase B `val_acc` down regardless of any
encoder weight movement. Eval mode no-ops dropout and BatchNorm; gradient
flow is unaffected, so the encoder still trains.

The optimizer is a single `AdamW` instance with two parameter groups: the
**scorer group** (SAB stack + PMA + scoring MLP) at `--lr`, and the **encoder
group** (2 SAB layers + output projection + token embedding table) at
`--embedding-lr`.

## 2. CLI Flags

### `train-scorer` (Phase B-relevant flags)

| Flag                    | Default                                        | Meaning                                                                                                                                                                                                                                  |
|-------------------------|------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--embedding-lr`        | `0`                                            | Learning rate for the encoder parameter group. `0` keeps the encoder out of the training graph (Phase A). Any non-zero value puts it in (Phase B).                                                                                       |
| `--encoder-checkpoint`  | `models/price-predictor/transformer/latest.pt` | Source of encoder weights when starting a fresh Phase B run via `--scorer-checkpoint`. Forbidden if explicitly passed when resuming a Phase B checkpoint (which already carries encoder weights).                                        |
| `--scorer-checkpoint`   | _(none)_                                       | Bootstrap scorer weights from a Phase A checkpoint to start a fresh Phase B run. Loads scorer weights only — `optimizer.state_dict`, `epoch`, `best_val_accuracy`, and `encoder.state_dict` are ignored.                                 |
| `--encoder-chunk-size`  | `128`                                          | Phase B only: chunk the encoder forward pass over each step's unique cards into pieces of this size, with gradient checkpointing per chunk so peak activation memory is bounded by one chunk. Lower it on tight GPUs; raise it on roomy ones. |
| `--max-grad-norm`       | `100.0`                                        | Per-parameter-group L2 norm cap applied between backward and optimizer step. Default is loose so clipping acts as a NaN-spike guard, not an effective LR throttle. The per-epoch report prints both mean and max pre-clip norm per group; if max stays well under the cap, clipping is rare and the configured LRs take effect as expected. |

The boolean `--unfreeze-embeddings` flag is **removed**; the on/off semantics
are subsumed by `--embedding-lr` (`0` vs non-zero).

`--resume` is **phase-locked**: the resumed checkpoint's phase (presence
or absence of `encoder.state_dict`) must match the current run's phase
(`--embedding-lr` zero or non-zero). Cross-phase resume is rejected with
no override flag. `--resume` and `--scorer-checkpoint` are mutually
exclusive — `--resume` continues an existing run within its phase;
`--scorer-checkpoint` initializes a fresh run.

### `encode-cards`

| Flag                   | Default                                        | Meaning                                                                                          |
|------------------------|------------------------------------------------|--------------------------------------------------------------------------------------------------|
| `--encoder-checkpoint` | `models/price-predictor/transformer/latest.pt` | Load encoder weights from a price-predictor checkpoint.                                          |
| `--scorer-checkpoint`  | _(none)_                                       | Load encoder weights from a sealed scorer checkpoint (extracted from the scorer's `state_dict`). |

The two flags are **mutually exclusive**; passing both is an error.
Passing a Phase A checkpoint (no `encoder.state_dict`) to
`--scorer-checkpoint` is also an error — the user is directed to
`--encoder-checkpoint` for non-Phase-B sources. Output `.npz` files have
identical structure regardless of source.

## 3. Within-batch Encoder Caching

Within each training step, cache the encoder output for each unique card and
reuse the cached tensor for duplicate references in the same batch. PyTorch
autograd handles the shared computation graph automatically — gradients
through duplicate references all accumulate into the same encoder parameters.
The cache is per-batch; it must be cleared between batches so the next step
builds a fresh computation graph. Without this caching, a Phase B epoch is
roughly 10× slower than Phase A; with it, 2–4×.

The encoder forward pass over a step's unique cards is also **chunked** at
`--encoder-chunk-size` (default 128) with `torch.utils.checkpoint` per
chunk. A typical step at `--batch-size 64` sees ~1000–3000 unique cards;
running them through a 6-layer transformer in a single forward overflows
commodity GPUs. Chunking bounds peak activation memory at one chunk's
worth, and gradient checkpointing recomputes the chunk's forward during
backward instead of storing its activations. Cost: ~1.5× compute per
backward pass. Lower the chunk size if Phase B still hits CUDA OOM; raise
it for extra throughput when memory permits.

## 4. Hyperparameter Defaults

| Hyperparameter    | Default | Notes                                                                                                                                                                                                              |
|-------------------|---------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--lr`            | `1e-5`  | Unchanged from Phase A.                                                                                                                                                                                            |
| `--embedding-lr`  | `0`     | Non-zero activates Phase B. Recommended starting value: `1e-7` (see Rationale § Why the embedding LR has to be unusually low).                                                                                     |
| `--patience`      | `5`     | Stop training after this many epochs without a new peak `val_acc`. Applies to both phases. The best checkpoint to date is always preserved as `best_*.pt`.                                                          |
| `--max-grad-norm` | `100.0` | Per-parameter-group L2 norm cap. Loose enough to act as a NaN-spike guard rather than a throttle; the per-epoch report shows mean and max pre-clip norms so a too-low setting is visible.                          |

## 5. Checkpoint Format

A scorer checkpoint at `models/sealed/scorer/best_*.pt` is a single PyTorch
file containing:

| Key                                                                             | Phase A | Phase B |
|---------------------------------------------------------------------------------|---------|---------|
| `scorer.state_dict` (SAB + PMA + MLP)                                           | ✓       | ✓       |
| `encoder.state_dict` (token embedding table + 2 SAB layers + output projection) | —       | ✓       |
| `optimizer.state_dict`                                                          | ✓       | ✓       |
| `epoch`, `best_val_accuracy`, `config`                                          | ✓       | ✓       |

The presence of `encoder.state_dict` in the loaded checkpoint is the
authoritative signal that it was produced by Phase B. `config` records the
CLI flag values used to produce the checkpoint; on `--resume`, flags passed
on the resuming invocation override the stored config for that run.

### Checkpoint filenames

Phase A and Phase B write to **distinct filenames** in the same
`--checkpoint-dir`, so a Phase B run does not overwrite the Phase A
checkpoint it bootstrapped from:

| Phase   | Latest checkpoint   | Best checkpoint                                                          |
|---------|---------------------|--------------------------------------------------------------------------|
| Phase A | `latest.pt`         | `best_l<n>_h<n>_s<n>_ff<d>_mlp<d>_lr<f>.pt`                              |
| Phase B | `latest_phaseB.pt`  | `best_phaseB_l<n>_h<n>_s<n>_ff<d>_mlp<d>_lr<f>_emblr<f>.pt`              |

The Phase B `best_*.pt` filename encodes the active `--embedding-lr` because
two Phase B runs that differ only in `--embedding-lr` would otherwise
collide. Reverting Phase B in favor of Phase A is then a matter of pointing
downstream tools back at the Phase A files, which survived the Phase B run
intact.

## 6. Workflow

The two phases are **two separate `train-scorer` invocations** chained via
`--scorer-checkpoint`. Phase A and Phase B do not share a single
training-loop call.

### Step 1 — Phase A (frozen encoder)

```bash
python -m sealed train-scorer
```

Run with `--embedding-lr 0` (the default). Early stopping fires per the
`--patience` rule from § 4; typical run length 10–20 epochs.

### Step 2 — Phase B (encoder fine-tuning)

```bash
python -m sealed train-scorer \
    --scorer-checkpoint models/sealed/scorer/best_<phaseA>.pt \
    --encoder-checkpoint models/price-predictor/transformer/latest.pt \
    --embedding-lr 1e-7
```

Phase B writes its own `best_phaseB_*_emblr*.pt` and `latest_phaseB.pt`
into the same `--checkpoint-dir` (§ 5); the Phase A files used to
bootstrap the run are not modified. Same `val_acc`-based early stopping
as Phase A; typical run length 5–15 epochs. To continue an interrupted
Phase B run, re-invoke with
`--resume models/sealed/scorer/best_phaseB_<...>.pt --embedding-lr 1e-7`
(phase must match — see § 2).

### Step 3 — Re-cache embeddings

```bash
python -m sealed encode-cards \
    --scorer-checkpoint models/sealed/scorer/best_phaseB_<...>.pt \
    --clean
```

Refreshes every `.npz` file under `output/cardsfolder/` using the fine-tuned
encoder. Cards absent from the match-outcomes corpus are re-encoded too,
inheriting whatever generalization the encoder learned during Phase B.

### Step 4 — Evaluate

```bash
python -m sealed evaluate-scorer --set <SET>
```

Run twice — once on the Phase A checkpoint, once on the Phase B checkpoint —
and compare match-win rate against forge-best. This is the authoritative
metric for whether Phase B helped.

## 7. Monitoring

Validation runs **once per epoch at end of epoch**; `--patience` counts
epochs without a new peak `val_acc`. The following metrics are logged at
the same cadence:

| Metric                                                                                                                                                                                                       | Action threshold                                                                                                |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|
| `val_acc`                                                                                                                                                                                                    | Drives `--patience`-based early stopping.                                                                       |
| `embedding_drift` — mean L2 distance of post-encoder card vectors on a fixed reference batch from their step-0 values (reuses the `embedding_drifts` field on `TrainingMetrics`)                              | Drift > 1.0 within first 3 epochs → lower `--embedding-lr` and restart from Phase A checkpoint.                 |
| Encoder gradient norm                                                                                                                                                                                        | Logged for diagnostic use.                                                                                      |

The reference batch is the set of unique cards in the very first Phase B
training batch, with their post-encoder vectors captured during that
batch's forward pass before the first optimizer step. These vectors are
the step-0 baseline reused for every subsequent drift computation in the
run.

## 8. Out of Scope

- **Label-noise reduction.** Phase B operates on per-card features, not
  labels, so it cannot move a label-noise-bound val_acc ceiling (see
  Rationale § Expected impact). Interventions on labels themselves —
  repeated matchups, longer formats, fixed-opponent training — are out of
  scope.

# Rationale

This section explains the decisions in the Specification. Not implementation
guidance.

## Why fine-tune the encoder rather than the cached vectors

The decoupled approach gives each of ~26K cards an independent 544-dim row
in a lookup table (~14M parameters), against ~1.24M card references in the
corpus — ~47 updates per row on average, heavily skewed toward format
staples. The encoder is ~3M shared parameters: every reference contributes
gradient to that shared pool.

| Property                     | Decoupled vectors                                                                                | Encoder fine-tuning                                                                                          |
|------------------------------|--------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------|
| **Parameter sharing**        | None. Two similar cards (lifegain payoffs, flying creatures) update independently.               | A learned feature ("removal", "evasion") gets contributions from every card with that property.              |
| **Appearance skew**          | Format staples shift substantially; rare cards (1–3 references) barely move.                     | Rare cards inherit shared feature updates from common cards with similar text.                               |
| **Transfer to new cards**    | A card not in the corpus keeps its original price-predictor embedding forever.                   | Any card text can be re-encoded with updated weights; new sets get sealed-relevant embeddings automatically. |
| **Architectural constraint** | None. Gradient descent can shift vectors in ways that don't survive being re-derived from text.  | Every embedding stays derivable from text via the encoder, which acts as a strong regularizer.               |

## Why two separate `train-scorer` runs instead of a mid-training switch

`--patience` early stopping handles the Phase A → Phase B transition signal
automatically, but a single-run implementation would still need mid-run
optimizer reconfiguration to add the encoder parameter group at the moment
early stopping fires. The two-run design avoids that special case: Phase A
runs to its natural early stop, then a separate Phase B invocation builds
the optimizer with both parameter groups from scratch. The Phase A checkpoint
also stays cleanly intact as a fallback if Phase B regresses. Cost: one extra
shell invocation between Phase A and Phase B.

## Why the embedding LR has to be unusually low

For each training example, the encoder is called 46 times (23 cards × 2
decks), and PyTorch autograd accumulates gradients from all 46 paths into
each encoder parameter's `.grad`. With `--lr 1e-5` and the typical "10× lower
for fine-tuning" rule of thumb (`--embedding-lr 1e-6`):

- Scorer per-step movement: `1e-5 × |grad_scorer|`
- Encoder per-step movement: `1e-6 × 46 × |grad_per_card| ≈ 4.6e-5 × |grad_per_card|`

The encoder still moves ~5× *faster* than the scorer despite the nominally
lower LR. To get the encoder moving 10× *slower* (the actual goal of low-LR
fine-tuning):

```
--embedding-lr ≈ lr / (10 × 46) ≈ 2e-8
```

The recommended `1e-7` default sits one order of magnitude above this lower bound
to leave headroom for warmup and avoid stalling on small gradients; `1e-8` to `1e-7`
is the practical range.

## Why a single `--embedding-lr` flag instead of a separate boolean

A separate boolean alongside `--embedding-lr` would have allowed incoherent
combinations like `--unfreeze-embeddings --embedding-lr 0`. Collapsing the
two into one numeric flag (`0` = frozen, non-zero = unfrozen) eliminates
that surface area.

## Why fresh Phase B runs load the encoder from the price-predictor checkpoint

A fresh Phase B run bootstraps scorer weights from a Phase A checkpoint via
`--scorer-checkpoint`; that file contains scorer weights only. The encoder
weights have to come from somewhere compatible with the precomputed `.npz`
embeddings the Phase A scorer was trained against. The price-predictor
`latest.pt` is the only file that satisfies this consistency constraint by
construction — it produced those `.npz` embeddings via `encode-cards`. For
continuing Phase B runs (`--resume <phaseB>.pt`), the encoder weights are
already in the resumed checkpoint.

## Why re-cache `.npz` files instead of running the encoder at inference

The downstream tools (`build-decks`, `evaluate-scorer`, `match-outcomes`) all
consume `.npz` card embeddings without running the encoder. This is a
deliberate design choice: the encoder is much more expensive than a `.npz`
lookup, and inference-time encoding would require the encoder weights (and
matching tokenizer) wherever those tools run. Re-caching once at the end of
Phase B preserves the property — downstream tools see a single drop-in
replacement of the existing `.npz` files.

## Caching savings

At `batch_size=64` and 46 cards per example, a naive Phase B implementation
runs ~2,944 encoder calls per training step, but only ~500–1,500 unique
cards typically appear in a batch — format staples appear in many decks.
Caching by unique card collapses the redundant calls.

## Expected impact

`experiments/gen2-initial-training.md` records four interventions that all
converged to the same val_acc ceiling: depth sweep, dropout sweep, multi-view
pooling, and hand-computed deck statistics. All four target the scorer's
*aggregation* over per-card features without changing the per-card features
themselves. Phase B is the first intervention that operates on the per-card
features, so it can plausibly move the ceiling — unless the ceiling is set
by label noise, in which case it can't. The deployment metric (head-to-head
match-win rate vs forge-best) may move more than val_acc, since robust
per-card features matter more for ranking novel decks during search than
they do for predicting in-distribution validation labels.

## Risks and mitigations

| Risk                                                                                                                             | Mitigation                                                                                                                                                                                        |
|----------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Catastrophic forgetting** — encoder loses general card-text understanding while specializing for sealed quality                | Low `--embedding-lr` (Spec § 4); revert to Phase A checkpoint if Phase B regresses on the deployment metric.                                                                                      |
| **Overfitting through the encoder** — encoder shifts to fit specific training decks rather than general sealed-relevant features | Short Phase B duration via `val_acc`-based early stopping (Spec § 4).                                                                                                                             |
| **Inference-time staleness** — downstream tools using stale `.npz` files silently produce wrong scores                           | Surface in `encode-cards` CLI help text. Future enhancement: embed a checkpoint-hash reference into `.npz` files so downstream tools can detect a mismatch.                                       |
| **New cards after Phase B** — cards from sets released later still need embeddings                                               | Re-run `encode-cards --scorer-checkpoint` whenever new cards are added to `output/cardsfolder/`. The fine-tuned encoder generalizes to text it hasn't seen, but only when actually invoked on it. |

# Goal

Fine-tune the price-predictor encoder during sealed scorer training so that card
embeddings shift from "what predicts a card's market price" to "what predicts deck
quality in sealed". This is the proper implementation of the "Phase B" stage
referenced in the embedding schedule of `sealed-deck-picker.md`.

# Background

The sealed scorer is built in two phases per the embedding schedule defined in
`sealed-deck-picker.md`:

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
vector fine-tuning with proper encoder fine-tuning. `--unfreeze-embeddings` is
removed and `--embedding-lr` (`0` vs non-zero) becomes the on/off switch.

## Decoupled vectors vs. encoder fine-tuning

The decoupled approach treats each card's embedding as an independent
`d_model`-sized parameter row in a lookup table: ~26K unique cards × 544 dims =
**~14M parameters**, but only ~27K matches × 2 decks × 23 nonland cards =
**~1.24M card references** in the corpus — **~47 gradient updates per row** on
average.

The encoder is shared. The price-predictor transformer has approximately **3M
parameters** (2 encoder layers ~800K each, ~1.3M token embedding table, ~150K
output projection), and every one of the 1.24M references contributes gradient
to these shared parameters via the forward pass.

| Property                     | Decoupled vectors                                                                                | Encoder fine-tuning                                                                                          |
|------------------------------|--------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------|
| **Parameter sharing**        | None. Two similar cards (lifegain payoffs, flying creatures) update independently.               | A learned feature ("removal", "evasion") gets contributions from every card with that property.              |
| **Appearance skew**          | Format staples shift substantially; rare cards (1–3 references) barely move.                     | Rare cards inherit shared feature updates from common cards with similar text.                               |
| **Transfer to new cards**    | A card not in the corpus keeps its original price-predictor embedding forever.                   | Any card text can be re-encoded with updated weights; new sets get sealed-relevant embeddings automatically. |
| **Architectural constraint** | None. Gradient descent can shift vectors in ways that don't survive being re-derived from text.  | Every embedding stays derivable from text via the encoder, which acts as a strong regularizer.               |

# Specification

This section is prescriptive. Everything below is what an implementer must
build; rationale and tradeoffs are deferred to the next section.

## 1. Definitions

| Term        | Meaning                                                                                                                                                            |
|-------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Phase A** | A `train-scorer` invocation with `--embedding-lr 0`. The encoder is not in the training graph; the scorer trains on top of precomputed `.npz` card embeddings.     |
| **Phase B** | A `train-scorer` invocation with `--embedding-lr > 0`. The encoder is in the training graph; encoder weights are updated by backprop alongside the scorer weights. |
| **Encoder** | The price-predictor transformer (token embedding table → 2 SAB layers → `cat([max_pool, mean_pool])`), as defined in `specs/007-transformer-model-arch/`.          |
| **Scorer**  | The sealed deck scorer (per-card SAB stack + PMA pooling + scoring MLP), as defined in `sealed-deck-picker.md` § Architecture.                                     |

## 2. Forward and Backward Pass (Phase B)

For each training example (one match outcome):

1. For each of the 46 cards (23 nonland cards × 2 decks), look up its
   converted card text and tokenize it.
2. Run the encoder forward pass: token embeddings → 2 SAB layers →
   `cat([max_pool, mean_pool])`. Output shape: `(2 * encoder_d_model,)`.
3. Concatenate the encoder output with the 32-dim deterministic feature
   vector parsed from the same card text (per `sealed-deck-picker.md` § Card
   Representation). Per-card vector shape: `(2 * encoder_d_model + 32,)`.
4. Pass the resulting `(46, 2 * encoder_d_model + 32)` card vectors through
   the scorer's SAB stack and PMA pooling, then through the scoring MLP.
5. Compute the Bradley-Terry pairwise loss on the `(score_winner, score_loser)`
   pair. Backpropagate through the entire graph: scorer → encoder → token
   embedding table.

Non-differentiable components: the tokenizer (gradients stop at the token
embedding lookup) and the deterministic feature parser (its outputs are
constant).

The optimizer is a single `AdamW` instance with two parameter groups: the
**scorer group** (SAB stack + PMA + scoring MLP) at `--lr`, and the **encoder
group** (2 SAB layers + output projection + token embedding table) at
`--embedding-lr`.

## 3. Within-batch Encoder Caching

A naive Phase B implementation runs the encoder forward+backward for every
card reference in every batch. At `batch_size=64` and 46 cards per example,
that's ~2,944 encoder calls per training step, but typically only 500–1,500
unique cards (format staples appear in many decks).

The implementer must cache the encoder output for each unique card in the
batch and reuse the cached tensor for duplicate references. PyTorch autograd
handles the shared computation graph automatically — gradients through
duplicate references all accumulate into the same encoder parameters. Without
this caching, a Phase B epoch is ~10× slower than Phase A; with it, the
slowdown is 2–4× (a 25-epoch Phase B run takes ~30–45 minutes vs ~12 minutes
for Phase A on the current corpus).

## 4. CLI Flags

### `train-scorer` (Phase B-relevant flags)

| Flag                   | Default                                        | Required | Meaning                                                                                                                                                                         |
|------------------------|------------------------------------------------|----------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--embedding-lr`       | `0`                                            | no       | Learning rate for the encoder parameter group. `0` keeps the encoder out of the training graph (Phase A). Any non-zero value puts it in (Phase B).                              |
| `--encoder-checkpoint` | `models/price-predictor/transformer/latest.pt` | no       | Source of encoder weights when bootstrapping a fresh Phase B run from a Phase A checkpoint. Ignored when resuming a Phase B checkpoint (which already carries encoder weights). |

The boolean `--unfreeze-embeddings` flag is **removed**; the on/off semantics
are subsumed by `--embedding-lr` (`0` vs non-zero).

### `encode-cards`

| Flag                   | Default                                        | Meaning                                                                                          |
|------------------------|------------------------------------------------|--------------------------------------------------------------------------------------------------|
| `--encoder-checkpoint` | `models/price-predictor/transformer/latest.pt` | Load encoder weights from a price-predictor checkpoint.                                          |
| `--scorer-checkpoint`  | _(none)_                                       | Load encoder weights from a sealed scorer checkpoint (extracted from the scorer's `state_dict`). |

The two flags are **mutually exclusive**; passing both is an error. Output
`.npz` files have identical structure regardless of source.

## 5. Encoder Weight Loading Priority (Phase B `train-scorer`)

Applied in order at the start of every Phase B run:

1. If `--resume <path>` is supplied AND the loaded `state_dict` contains
   encoder keys → use the encoder weights from the resumed checkpoint;
   `--encoder-checkpoint` is ignored.
2. Otherwise → load encoder weights from the file at `--encoder-checkpoint`.
3. If neither source is available → error.

## 6. Checkpoint Format

A scorer checkpoint at `models/sealed/scorer/best_*.pt` is a single PyTorch
file containing:

| Key                                                                             | Phase A | Phase B |
|---------------------------------------------------------------------------------|---------|---------|
| `scorer.state_dict` (SAB + PMA + MLP)                                           | ✓       | ✓       |
| `encoder.state_dict` (token embedding table + 2 SAB layers + output projection) | —       | ✓       |
| `optimizer.state_dict`                                                          | ✓       | ✓       |
| `epoch`, `best_val_accuracy`, `config`                                          | ✓       | ✓       |

The presence of `encoder.state_dict` in the loaded checkpoint is the
authoritative signal that it was produced by Phase B.

## 7. Hyperparameter Defaults

| Hyperparameter                    | Default | Notes                                                                                                                          |
|-----------------------------------|---------|--------------------------------------------------------------------------------------------------------------------------------|
| `--lr`                            | `1e-5`  | Unchanged from Phase A.                                                                                                        |
| `--embedding-lr`                  | `0`     | Non-zero activates Phase B. Recommended starting value: `1e-7` (see Rationale § Why the embedding LR has to be unusually low). |
| `--patience`                      | `5`     | Stop training after this many epochs without a new peak `val_acc`. Applies to both phases. The peak epoch resets each time a new peak is observed; the best checkpoint to date is always preserved as `best_*.pt`. |
| Gradient clipping (encoder group) | 1.0     | Max-norm; caps peak per-step movement under unusual gradient spikes.                                                           |

## 8. Workflow

The two phases are **two separate `train-scorer` invocations** chained via
`--resume`. Phase A and Phase B do not share a single training-loop call.

### Step 1 — Phase A (frozen encoder)

```bash
python -m sealed train-scorer
```

Run with `--embedding-lr 0` (the default). Training stops automatically when
`val_acc` has not produced a new peak for `--patience` epochs (default 5);
the best checkpoint is preserved at `models/sealed/scorer/best_*.pt`.
Typical run length: 10–20 epochs.

### Step 2 — Phase B (encoder fine-tuning)

```bash
python -m sealed train-scorer \
    --resume models/sealed/scorer/best_<phaseA>.pt \
    --encoder-checkpoint models/price-predictor/transformer/latest.pt \
    --embedding-lr 1e-7
```

Encoder weight loading follows § 5. Same `val_acc`-based early stopping as
Phase A (`--patience`, default 5); typical run length 5–15 epochs.

### Step 3 — Re-cache embeddings

```bash
python -m sealed encode-cards \
    --scorer-checkpoint models/sealed/scorer/best_<phaseB>.pt \
    --clean
```

Refreshes every `.npz` file under `output/cardsfolder/` to match the
fine-tuned encoder.

### Step 4 — Evaluate

```bash
python -m sealed evaluate-scorer --set <SET>
```

Run twice — once on the Phase A checkpoint, once on the Phase B checkpoint —
and compare match-win rate against forge-best. This is the authoritative
metric for whether Phase B helped.

## 9. Monitoring

Required metrics, logged every validation interval:

| Metric                                                                                                                                                                                | Action threshold                                                                                                                         |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------|
| `val_acc`                                                                                                                                                                             | Drives `--patience`-based early stopping (Spec § 7). The peak epoch is updated whenever a new peak is observed.                          |
| `embedding_drift` (existing `embedding_drifts` field on `TrainingMetrics`): mean L2 distance of post-encoder card vectors from the values they had at Phase B step 0                  | Drift > 1.0 within first 3 epochs → lower `--embedding-lr` and restart from Phase A checkpoint. Drift < 0.05 after 10 epochs → raise it. |
| Encoder gradient norm                                                                                                                                                                 | Logged for diagnostic use.                                                                                                               |

## 10. Out of Scope

- **Label-noise reduction.** Phase B cannot move the label noise floor (see
  Rationale § Expected impact). Interventions that operate on labels themselves
  — repeated matchups, longer formats, fixed-opponent training — are out of
  scope.

# Rationale

This section explains the decisions in the Specification. Not implementation
guidance.

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

For each training example, the encoder is called 46 times (23 cards × 2 decks).
PyTorch autograd treats each call as a separate consumer of the encoder
parameters and **accumulates gradients from all 46 paths** into each encoder
parameter's `.grad`. This is the correct behavior for parameter sharing —
it's exactly *why* fine-tuning the encoder works at all — but it has a
consequence for LR.

The scorer's parameters see no comparable factor. With `--lr 1e-5` and the
typical "10× lower for fine-tuning" rule of thumb (`--embedding-lr 1e-6`):

- Scorer per-step movement: `1e-5 × |grad_scorer|`
- Encoder per-step movement: `1e-6 × 46 × |grad_per_card| ≈ 4.6e-5 × |grad_per_card|`

So the encoder still moves ~5× *faster* than the scorer despite the nominally
lower LR. To get the encoder moving 10× *slower* (the actual goal of low-LR
fine-tuning):

```
--embedding-lr ≈ lr / (10 × 46) ≈ 2e-8
```

The recommended `1e-7` default sits about one order of magnitude above this
lower bound to leave headroom for warmup and avoid stalling on small
gradients; `1e-8` to `1e-7` is the practical range. Gradient clipping on the
encoder group (Spec § 7) is a complementary mitigation for batches with
unusually large per-card gradients.

## Why a single `--embedding-lr` flag instead of a separate boolean

`--embedding-lr 0` is exactly equivalent to "frozen", and any non-zero value
to "unfrozen". Removing the boolean eliminates the surface area for
incoherent combinations like `--unfreeze-embeddings --embedding-lr 0`.

## Why the encoder is loaded from the price-predictor checkpoint at fresh Phase B start

A fresh Phase B run resumes from a Phase A scorer checkpoint, which contains
scorer weights only. The encoder weights have to come from somewhere
compatible with the precomputed `.npz` embeddings the Phase A scorer was
trained against. The price-predictor `latest.pt` is the only file that
satisfies this consistency constraint by construction — it's the file that
produced those `.npz` embeddings via `encode-cards`. For continuing Phase B
runs (resuming a Phase B checkpoint), the encoder weights are already in the
resumed checkpoint and are the correct source by construction.

## Why re-cache `.npz` files after Phase B instead of always using the encoder at inference

The downstream tools (`build-decks`, `evaluate-scorer`, `match-outcomes`) all
consume `.npz` card embeddings without running the encoder. This is a
deliberate design choice: the encoder is much more expensive than a `.npz`
lookup, and inference-time encoding would require the encoder weights (and
matching tokenizer) wherever those tools run. Re-caching once at the end of
Phase B preserves the property — downstream tools see a single drop-in
replacement of the existing `.npz` files. Cost: one full re-encode pass over
all cards in `output/cardsfolder/` (~30K cards, a few minutes) per Phase B
run.

## Expected impact

`experiments/gen2-initial-training.md` records four interventions that all
converged to the same ~0.70 val_acc ceiling: depth sweep (2–6 SAB layers),
dropout sweep (0.0–0.4), multi-view pooling (PMA + max + mean), and
hand-computed deck statistics. All four target the scorer's *aggregation*
over per-card features without changing the per-card features themselves.

Phase B is the first intervention that operates on the per-card features. If
the val_acc ceiling is set by under-expressive per-card features (the
price-predictor encoder optimizing for the wrong objective), Phase B can
plausibly move it. If the ceiling is set elsewhere — primarily by label noise
at ~0.72–0.78 (see `experiments/gen2-initial-training.md` § "The oracle
ceiling for this corpus") — Phase B can't.

Realistic outcome: val_acc moves from ~0.695 to somewhere in the 0.70–0.74
range. The deployment metric (head-to-head match-win rate vs forge-best) may
move more — robust per-card features matter more for ranking novel decks
during search than they do for predicting labels in the in-distribution
validation set.

## Risks and mitigations

| Risk                                                                                                                             | Mitigation                                                                                                                                                                                        |
|----------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Catastrophic forgetting** — encoder loses general card-text understanding while specializing for sealed quality                | Low `--embedding-lr` (Spec § 7), embedding-drift monitoring (Spec § 9), revert to Phase A checkpoint if regressed.                                                                                |
| **Overfitting through the encoder** — encoder shifts to fit specific training decks rather than general sealed-relevant features | Short Phase B duration, `val_acc`-based early stopping via `--patience` (Spec § 7, § 9).                                                                                                          |
| **Inference-time staleness** — downstream tools using stale `.npz` files silently produce wrong scores                           | Surface in `encode-cards` CLI help text. Future enhancement: embed a checkpoint-hash reference into `.npz` files so downstream tools can detect a mismatch.                                       |
| **New cards after Phase B** — cards from sets released later still need embeddings                                               | Re-run `encode-cards --scorer-checkpoint` whenever new cards are added to `output/cardsfolder/`. The fine-tuned encoder generalizes to text it hasn't seen, but only when actually invoked on it. |

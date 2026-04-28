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

The current implementation of "Phase B" is misleading. The `--unfreeze-embeddings`
flag on `train-scorer` simply flips `requires_grad = True` on the lookup table of
precomputed embedding vectors. The price-predictor encoder network that produced
those vectors is **not** in the training graph. This is decoupled vector
fine-tuning, not encoder fine-tuning, and it has fundamental problems that make
it ineffective in practice. This spec describes the proper implementation: bring
the encoder into the scorer's training graph and fine-tune it end-to-end at a low
learning rate. The existing boolean `--unfreeze-embeddings` flag is removed in
favor of letting `--embedding-lr` itself act as the on/off switch (`0` =
frozen, non-zero = fine-tuned).

## Why decoupled vector fine-tuning doesn't work

The decoupled approach treats each card's embedding as an independent
`d_model`-sized parameter row in a lookup table. Concretely:

- ~26K unique cards × 544 dims = **~14M independent parameters**
- ~27K matches × 2 decks × 23 nonland cards = **~1.24M card references** in the
  training corpus
- Average gradient updates per card row: **~47**

That is a tiny per-card learning signal, and four problems compound it:

1. **No parameter sharing across similar cards.** Two cards with similar text
   (two lifegain payoffs, two flying creatures) receive completely independent
   updates. A proper encoder would share learned features across them via the
   same SAB stack and tokenizer; the lookup table treats them as wholly distinct.
2. **Skewed appearance distribution.** Common format staples appear hundreds of
   times in the corpus; the long tail of rare cards appears 1–3 times. Common
   cards shift substantially while rare cards barely move, leaving the embedding
   space inconsistent — common neighborhoods reorganized for sealed quality,
   rare neighborhoods still reflecting price-prediction structure.
3. **No transfer to new cards.** A card not present in the training corpus keeps
   its original price-predictor embedding forever. New MTG sets released after
   training have to fall back to the price encoder for embeddings — but the
   scorer is now using a coordinate system that has drifted away from the price
   encoder's original output, so those new-set embeddings are inconsistent with
   the rest.
4. **No architectural constraint.** With nothing forcing embeddings to be
   derivable from card text, gradient descent can shift them in ways that fit
   the specific decks in the training set but wouldn't survive being re-derived
   from text. The encoder normally enforces this constraint as a strong
   regularizer.

## Why encoder fine-tuning works

Fine-tuning the encoder shares parameters across all cards. The price-predictor
transformer has approximately **3M parameters** (2 encoder layers × ~800K each
for attention and FFN, plus ~1.3M for the token embedding table, plus ~150K for
the output projection). Every one of the 1.24M card references in the training
corpus contributes gradient to these shared parameters via the encoder forward
pass.

This addresses all four problems above:

1. **Parameter sharing**: a learned feature like "removal spell" or "evasion
   creature" gets contributions from every card with that property — thousands
   of examples per feature instead of dozens per card.
2. **No skew problem**: rare cards inherit shared feature updates from common
   cards with similar text. A rare lifegain payoff benefits from the shared
   "lifegain matters" feature being learned from the common lifegain payoffs
   that show up frequently.
3. **Transfer to new cards**: any card text can be encoded fresh with the
   updated encoder weights. New sets get useful sealed-relevant embeddings
   automatically.
4. **Architectural constraint**: every embedding stays derivable from text via
   the encoder, which acts as a strong regularizer against overfitting to
   specific training-set decks.

# Specification

This section is prescriptive. Everything below is what an implementer must
build; rationale and tradeoffs are deferred to the next section.

## 1. Definitions

| Term                   | Meaning                                                                                                                                                            |
|------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Phase A**            | A `train-scorer` invocation with `--embedding-lr 0`. The encoder is not in the training graph; the scorer trains on top of precomputed `.npz` card embeddings.     |
| **Phase B**            | A `train-scorer` invocation with `--embedding-lr > 0`. The encoder is in the training graph; encoder weights are updated by backprop alongside the scorer weights. |
| **Encoder**            | The price-predictor transformer (token embedding table → 2 SAB layers → `cat([max_pool, mean_pool])`), as defined in `specs/007-transformer-model-arch/`.          |
| **Scorer**             | The sealed deck scorer (per-card SAB stack + PMA pooling + scoring MLP), as defined in `sealed-deck-picker.md` § Architecture.                                     |
| **Phase A checkpoint** | A scorer checkpoint produced by a Phase A run. Contains scorer weights only — no encoder weights.                                                                  |
| **Phase B checkpoint** | A scorer checkpoint produced by a Phase B run. Contains scorer weights, encoder weights, and the token embedding table.                                            |

## 2. Forward and Backward Pass (Phase B)

For each training example (one match outcome):

1. For each of the 46 cards (23 nonland cards × 2 decks), look up its
   converted card text and tokenize it.
2. Run the encoder forward pass: token embeddings → 2 SAB layers →
   `cat([max_pool, mean_pool])`. Output shape: `(2 * encoder_d_model,)`.
3. Concatenate the encoder output with the 32-dim deterministic feature
   vector parsed from the same card text (per `sealed-deck-picker.md` § Card
   Representation). Output shape: `(total_dim,)`.
4. Pass the resulting `(46, total_dim)` card vectors through the scorer's
   SAB stack and PMA pooling, then through the scoring MLP.
5. Compute the Bradley-Terry pairwise loss on the `(score_winner, score_loser)`
   pair. Backpropagate through the entire graph: scorer → encoder → token
   embedding table.

Non-differentiable components: the tokenizer (gradients stop at the token
embedding lookup) and the deterministic feature parser (its outputs are
constant).

## 3. Optimizer Parameter Groups

| Group   | Parameters                                               | Learning rate    |
|---------|----------------------------------------------------------|------------------|
| Scorer  | SAB stack + PMA + scoring MLP                            | `--lr`           |
| Encoder | 2 SAB layers + output projection + token embedding table | `--embedding-lr` |

A single `AdamW` optimizer with two parameter groups, one per row above.

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

| Key                                              | Phase A | Phase B |
|--------------------------------------------------|---------|---------|
| `scorer.state_dict` (SAB + PMA + MLP)            | ✓       | ✓       |
| `encoder.state_dict` (2 SAB + output projection) | —       | ✓       |
| `encoder.token_embedding`                        | —       | ✓       |
| `optimizer.state_dict`                           | ✓       | ✓       |
| `epoch`, `best_val_accuracy`, `config`           | ✓       | ✓       |

The presence of `encoder.*` keys in the loaded `state_dict` is the
authoritative signal that a checkpoint was produced by Phase B.

## 7. Hyperparameter Defaults

| Hyperparameter                    | Default                          | Notes                                                                                                                          |
|-----------------------------------|----------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| `--lr`                            | scorer's existing default (1e-5) | Unchanged from Phase A.                                                                                                        |
| `--embedding-lr`                  | `0`                              | Non-zero activates Phase B. Recommended starting value: `1e-7` (see Rationale § Why the embedding LR has to be unusually low). |
| `--epochs` (Phase B)              | 5–15                             | Much shorter than Phase A.                                                                                                     |
| Gradient clipping (encoder group) | max-norm 1.0                     | Caps peak per-step movement under unusual gradient spikes.                                                                     |

## 8. Workflow

The two phases are **two separate `train-scorer` invocations** chained via
`--resume`. Phase A and Phase B do not share a single training-loop call.

### Step 1 — Phase A (frozen encoder)

```bash
python -m sealed train-scorer
```

Run with `--embedding-lr 0` (the default). Train until validation loss
plateaus, judged manually from the training log; typical 10–20 epochs.

### Step 2 — Phase B (encoder fine-tuning)

```bash
python -m sealed train-scorer \
    --resume models/sealed/scorer/best_<phaseA>.pt \
    --encoder-checkpoint models/price-predictor/transformer/latest.pt \
    --embedding-lr 1e-7
```

`--encoder-checkpoint` may be omitted when its default points to the
intended encoder. To continue an existing Phase B run, replace `--resume`
with a Phase B checkpoint; `--encoder-checkpoint` is then ignored
automatically (see § 5).

Train for 5–15 epochs with val_loss-based early stopping (patience 3).

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

| Metric                                                                                                                                              | Action threshold                                                                                                                         |
|-----------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------|
| `val_loss`                                                                                                                                          | If rising for 3 consecutive validation intervals: stop early.                                                                            |
| `embedding_drift` (existing `embedding_drifts` field on `TrainingMetrics`): mean L2 distance of post-encoder card vectors from their initial values | Drift > 1.0 within first 3 epochs → lower `--embedding-lr` and restart from Phase A checkpoint. Drift < 0.05 after 10 epochs → raise it. |
| Encoder gradient norm                                                                                                                               | Logged for diagnostic use.                                                                                                               |

## 10. Out of Scope

- **Label-noise reduction.** Phase B cannot move the label noise floor at
  ~0.72–0.78 val_acc (see `experiments/gen2-initial-training.md` § "The
  oracle ceiling for this corpus"). Interventions that operate on labels
  themselves — repeated matchups, longer formats, fixed-opponent training —
  are out of scope.
- **Auto-detection of Phase A plateau.** Plateau detection remains a manual
  judgment call by the operator inspecting the training logs.
- **Backward compatibility for the removed `--unfreeze-embeddings` flag.**
  The flag is removed without an alias or deprecation warning. Existing
  shell scripts that relied on it must be updated.

# Rationale

This section explains the decisions in the Specification. Not implementation
guidance.

## Why two separate `train-scorer` runs instead of a mid-training switch

A single-run implementation that auto-unfreezes the encoder at some
detected plateau point would need: plateau detection logic in the training
loop, mid-run optimizer reconfiguration to add the encoder parameter group,
and graceful handling of the case where plateau detection fires too early
(or never).

The two-run design avoids all of that:

- Plateau detection is the operator's judgment call from the training log.
  Misjudgment costs a wasted Phase B run, not a corrupted training process.
- The Phase A checkpoint stays cleanly intact as a fallback. If Phase B
  regresses, revert by restarting from the Phase A checkpoint with a lower
  `--embedding-lr` (or back out entirely).
- The training-loop code stays unchanged from the existing implementation
  apart from the new optimizer parameter group. No special case for "the
  epoch where things change."

The cost is one extra shell invocation between Phase A and Phase B. Cheap.

## Why the embedding LR has to be unusually low

For each training example, the encoder is called 46 times (23 cards per deck
× 2 decks). PyTorch autograd treats each call as a separate consumer of the
encoder parameters and **accumulates gradients from all 46 paths** into each
encoder parameter's `.grad` tensor:

```
encoder_param.grad ≈ 46 × (mean per-card gradient contribution)
```

This is the correct behavior for parameter sharing — it's exactly *why*
fine-tuning the encoder works at all (every card's match outcome contributes
to learning shared features). But it has a consequence for LR.

The scorer's parameters (PMA seeds, scoring MLP, SAB layers) are mostly used
once per example and don't see the 46× factor. Setting `--embedding-lr` equal
to `--lr` would give the encoder ~46× larger per-step weight movement than
the scorer, which would shred the pretrained encoder structure within the
first few epochs.

Concretely, with the typical "10× lower than main LR" fine-tuning rule of
thumb (e.g., `--lr 1e-5`, `--embedding-lr 1e-6`):

- Scorer per-step weight movement: `1e-5 × |grad_scorer|`
- Encoder per-step weight movement: `1e-6 × 46 × |grad_per_card| ≈ 4.6e-5 × |grad_per_card|`

Assuming `|grad_per_card|` is roughly comparable to `|grad_scorer|` (both
downstream of the same loss), the encoder still moves ~5× *faster* than the
scorer despite the nominally lower LR. To get the encoder moving 10× *slower*
than the scorer (the actual goal of "low-LR fine-tuning"):

```
--embedding-lr ≈ lr / (10 × 46) ≈ 2e-8
```

Hence the `1e-7` default and `1e-7` to `1e-8` recommended range — much lower
than typical fine-tuning advice would suggest. Gradient clipping on the
encoder group (Spec § 7) is a complementary mitigation for the cases where
per-card gradients are unusually large for one batch.

## Why a single `--embedding-lr` flag instead of a separate boolean

The original CLI had `--unfreeze-embeddings` (bool) plus `--embedding-lr`
(float). The boolean is informationally redundant: `--embedding-lr 0` is
exactly equivalent to "frozen", and any non-zero value is exactly equivalent
to "unfrozen". Removing the boolean eliminates the surface area where a user
can pass an incoherent combination (`--unfreeze-embeddings --embedding-lr 0`,
or vice versa).

## Why caching encoded outputs within a batch

A naive Phase B implementation runs the encoder forward+backward for every
card reference in every batch — at `batch_size=64` and 23 cards per deck,
that's ~2,944 encoder calls per training step. Many of those references are
duplicate cards within the batch (format staples appear in many decks).

Caching the encoder output for each unique card in the batch (typically
500–1,500 unique from ~3,000 references) and reusing the cached tensor for
duplicate references reduces encoder forward passes by 2–5×. PyTorch
autograd handles the shared computation graph automatically — gradients
through duplicate references all accumulate into the same encoder
parameters.

This caching is what makes Phase B tractable on the existing hardware
(2–4× slower per epoch than Phase A; a 25-epoch Phase B run is ~30–45
minutes vs ~12 minutes for Phase A on the current corpus). Without it the
slowdown would be closer to 10×.

## Why the encoder is loaded from the price-predictor checkpoint at fresh Phase B start

A fresh Phase B run resumes from a Phase A scorer checkpoint, which
contains scorer weights only. The encoder weights have to come from
*somewhere* compatible with the precomputed `.npz` embeddings the Phase A
scorer was trained against. The price-predictor `latest.pt` is the only
file that satisfies this consistency constraint by construction — it's the
file that produced those `.npz` embeddings in the first place via
`encode-cards`.

For continuing Phase B runs (resuming a Phase B checkpoint), the encoder
weights are already in the resumed checkpoint and are the correct source
by construction.

## Why re-cache `.npz` files after Phase B instead of always using the encoder at inference

The downstream tools (`build-decks`, `evaluate-scorer`, `match-outcomes`)
all consume `.npz` card embeddings without running the encoder. This is
a deliberate design choice in the existing codebase: the encoder is much
more expensive than a `.npz` lookup, and inference-time encoding would
require the encoder weights (and matching tokenizer) to be available
wherever those tools run.

Re-caching once at the end of Phase B preserves this property — downstream
tools see a single drop-in replacement of the existing `.npz` files and
need no other changes. The cost is one full re-encode pass over all cards
in `output/cardsfolder/` (~30K cards, a few minutes on the existing
hardware) per Phase B run.

## Where Phase B fits in the roadmap

`experiments/gen2-initial-training.md` records four interventions that
all converged to the same ~0.70 val_acc ceiling: depth sweep (2–6 SAB
layers), dropout sweep (0.0–0.4), multi-view pooling (PMA + max + mean),
and hand-computed deck statistics (with magnitude scaling 1× to 200×).
All four target the scorer's *aggregation* over per-card features without
changing the per-card features themselves.

Phase B encoder fine-tuning is the first intervention that operates on the
per-card features. If the val_acc ceiling is set by under-expressive
per-card features (the price-predictor encoder optimizing for the wrong
objective), Phase B can plausibly move it. If the ceiling is set
elsewhere — primarily by label noise — Phase B can't.

## Realistic upside

Phase B addresses the per-card-features bottleneck in the model-side
error budget but cannot push past the label-noise ceiling at ~0.72–0.78
val_acc (see `experiments/gen2-initial-training.md` § "The oracle ceiling
for this corpus"). Realistic outcome: val_acc moves from ~0.695 to
somewhere in the 0.70–0.74 range. The deployment metric (head-to-head
match-win rate vs forge-best) may move more — robust per-card features
matter more for ranking novel decks during search than they do for
predicting labels in the in-distribution validation set.

## Risks and mitigations

| Risk                                                                                                                             | Mitigation                                                                                                                                                                                        |
|----------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Catastrophic forgetting** — encoder loses general card-text understanding while specializing for sealed quality                | Low `--embedding-lr` (Spec § 7), embedding-drift monitoring (Spec § 9), revert to Phase A checkpoint if regressed.                                                                                |
| **Overfitting through the encoder** — encoder shifts to fit specific training decks rather than general sealed-relevant features | Short Phase B duration (5–15 epochs), val_loss-based early stopping with patience 3 (Spec § 9).                                                                                                   |
| **Inference-time staleness** — downstream tools using stale `.npz` files silently produce wrong scores                           | Surface in `encode-cards` CLI help text. Future enhancement: embed a checkpoint-hash reference into `.npz` files so downstream tools can detect a mismatch.                                       |
| **New cards after Phase B** — cards from sets released later still need embeddings                                               | Re-run `encode-cards --scorer-checkpoint` whenever new cards are added to `output/cardsfolder/`. The fine-tuned encoder generalizes to text it hasn't seen, but only when actually invoked on it. |

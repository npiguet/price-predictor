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

# Architecture

Phase B brings the price-predictor encoder into the scorer's training graph. The
forward data flow becomes:

1. For each card name in the training batch, look up its converted card text
   and tokenize it (cached from disk or precomputed).
2. Run the price-predictor transformer (token embeddings → SAB stack →
   `cat([max_pool, mean_pool])`) to produce the **text portion** of the card
   embedding (`2 * encoder_d_model` features, e.g. 512 with the default
   encoder).
3. Concatenate with the **deterministic features** parsed deterministically
   from the same card text (32 features — mana cost, types, P/T, mana
   production, etc., per `sealed-deck-picker.md` § Card Representation).
4. Pass the resulting `total_dim` card vectors (default 544) to the scorer's
   SAB stack and scoring MLP, exactly as in Phase A.
5. Compute the Bradley-Terry pairwise loss on (winner, loser) deck pairs and
   backprop through everything: scorer → encoder → token embedding table.

The tokenizer is not differentiable; gradients stop at the token embedding
table. The deterministic features are also not differentiable — they are
computed deterministically from card text and remain fixed throughout
training.

## Tunable Parameter Groups

Phase B trains three sets of parameters, organized as two optimizer groups:

- **Scorer parameters** (SAB stack + PMA + scoring MLP): trained at the main
  learning rate `--lr` (typically 1e-5 with AdamW).
- **Encoder parameters** (price-predictor transformer SAB stack + output head)
  and the **token embedding table**: trained at `--embedding-lr`, **10–100×
  smaller than `--lr`**. The two share a single optimizer group; the token
  embedding table is large but sparse (each batch only touches the tokens
  appearing in cards in the batch), which makes the low LR particularly
  important to avoid catastrophic forgetting on rare tokens.

## Caching for Performance

A naive implementation runs the encoder forward+backward for every card
reference in every batch — at `batch_size=64` and 23 cards per deck, that's
~2,944 encoder calls per training step. Many of those cards are duplicates
within the batch (format staples appear in many decks).

Cache encoder outputs **within a single training step**:

1. Collect all unique card names in the batch (typically 500–1,500 unique from
   ~3,000 references).
2. Encode each unique card once.
3. Look up each card reference into the cached encoded representation.
4. Backprop normally — autograd handles the shared computation graph
   automatically once the cached tensor is reused for multiple references.

This reduces encoder forward passes by 2–5× per batch and is essential to
keeping Phase B training cost reasonable.

## Cost

The price-predictor encoder is a 2-layer transformer (per
`specs/007-transformer-model-arch/`), small enough that Phase B is tractable on
the same GPU used for Phase A.

- **Compute per epoch**: 2–4× slower than Phase A (with within-batch caching).
  A 25-epoch Phase B run goes from ~12 minutes (Phase A) to ~30–45 minutes.
- **Memory**: backpropping through 2 transformer layers for ~1,500 unique
  cards per batch fits comfortably with `batch_size=64` on a single GPU.
  Memory pressure is dominated by activations of the scorer's SAB stack, not
  the encoder.
- **Disk**: at training time, the precomputed `.npz` embeddings are no longer
  needed (they are regenerated on the fly by the encoder). The text files in
  `output/cardsfolder/` are still required.

# Hyperparameters

- **`--embedding-lr`**: defaults to `0` (encoder frozen, Phase A behavior). Any
  non-zero value triggers Phase B: the encoder is included in the training
  graph at this learning rate. Recommended range when activated: 1e-5 to 1e-7,
  i.e. 10–100× lower than `--lr`. Lower values are more conservative and
  preserve more of the original encoder structure. Start at the higher end of
  the range and lower it if embedding drift is too rapid.

  This single flag replaces the older two-flag scheme (`--unfreeze-embeddings`
  bool + `--embedding-lr` float) — the boolean was redundant since a learning
  rate of zero is exactly equivalent to "frozen", and a non-zero rate is
  exactly equivalent to "unfrozen". The decoupled-vector behavior of the old
  `--unfreeze-embeddings` flag is removed entirely; there is no useful middle
  ground worth supporting.
- **Phase B start**: after Phase A's val_loss plateaus (typically 10–20 epochs
  on the current corpus). Resume from the best Phase A checkpoint with
  `--resume <path> --embedding-lr 1e-6`.
- **Phase B duration**: 5–15 epochs. Phase B should be much shorter than Phase
  A; the encoder is making fine adjustments to an already-useful representation,
  not learning from scratch. Use val_loss-based early stopping with patience ~3.

Embedding drift monitoring (already in place via `embedding_drifts` in
`TrainingMetrics`): track average L2 distance of the post-encoder card vectors
from their initial values across training. If they drift too far too fast
(e.g., L2 > 1.0 within the first 3 epochs), lower `--embedding-lr`. If they
barely move (L2 < 0.05 after 10 epochs), raise it.

# Where the Fine-Tuned Encoder Lives

The encoder is part of the scorer's training graph during Phase B (it has to be,
for backprop to flow into it), so its weights are part of `model.state_dict()`
and get written to disk as part of the scorer checkpoint. A Phase B training run
produces a single artifact at `models/sealed/scorer/best_*.pt` that contains
everything: scorer SAB stack + scoring MLP + fine-tuned encoder + token
embedding table.

The original price-predictor checkpoint at
`models/price-predictor/transformer/latest.pt` is **not modified**. That file
is still the encoder for price prediction and for initial sealed-encoding (Phase
A's `encode-cards` invocation, which runs before any Phase B training has
happened).

# End-of-Training Re-Cache

After Phase B completes, downstream tools (`build-decks`, `evaluate-scorer`,
`match-outcomes` in self-play mode) consume `.npz` card embeddings without
running the encoder. Those `.npz` files were produced by Phase A's frozen
encoder and are now stale — they no longer match the encoder weights baked into
the scorer checkpoint.

`encode-cards` accepts a `--scorer-checkpoint <path>` flag. When supplied, it
extracts the fine-tuned encoder weights from the scorer checkpoint and uses
them to re-encode every card, overwriting the `.npz` files. Run this once
after Phase B completes:

```bash
python -m sealed encode-cards --scorer-checkpoint models/sealed/scorer/best.pt --clean
```

The `--clean` flag forces a full re-encode (otherwise `encode-cards` skips
files that already exist).

# Recommended Workflow

1. **Confirm Phase A is plateaued.** Phase B is not useful until the scorer
   has extracted what it can from the frozen embeddings. Inspect the Phase A
   training logs; val_loss should be flat (or rising due to overfitting) for
   the last 3–5 epochs.
2. **Resume from the best Phase A checkpoint** with `--embedding-lr 1e-6` (a
   safe starting point — non-zero is what activates Phase B).
3. **Track val_loss and embedding drift.** A meaningful Phase B run should
   show val_loss continuing to drop for 5–10 more epochs, with embedding drift
   increasing smoothly (not in spikes). If val_loss starts rising immediately,
   `--embedding-lr` is too high.
4. **Re-cache embeddings** at the end via `encode-cards --scorer-checkpoint`.
5. **Re-evaluate** with `evaluate-scorer` head-to-head against forge-best on a
   fixed pool set. This is the deployment metric (per
   `experiments/gen2-initial-training.md` recommendations) and the only one
   that reliably reflects whether Phase B actually helped.

# Risks

- **Catastrophic forgetting.** The encoder might lose its general card-text
  understanding while specializing for sealed quality, breaking generalization
  to held-out cards. Mitigations: low learning rate, embedding-drift
  monitoring, ability to revert to the Phase A checkpoint if Phase B regresses.
- **Overfitting through the encoder.** If Phase B trains too long, the encoder
  can shift to fit the specific training deck distribution rather than learning
  general sealed-relevant features. Mitigations: short Phase B duration (5–15
  epochs), val_loss-based early stopping with patience ~3.
- **Inference-time staleness.** Tools that use precomputed `.npz` embeddings
  will silently produce wrong scores if the cache hasn't been refreshed after
  Phase B. Mitigations: surface this in CLI help text for `encode-cards` and
  in the recommended workflow above. Consider embedding a checkpoint-hash
  reference into the `.npz` files so downstream tools can detect staleness.
- **Unsupported new cards.** If a new MTG set is released after Phase B, those
  cards' embeddings should be regenerated with the fine-tuned encoder weights
  (same `encode-cards --scorer-checkpoint` invocation). The fine-tuned encoder
  generalizes, but only if it is actually run on the new cards.

# Why This Order in the Roadmap

Several other interventions were tried first (architecture sweeps, dropout,
multi-pool, hand-computed deck stats — see `experiments/gen2-initial-training.md`
for the full record). All of them targeted the scorer's *aggregation* over
per-card features without changing the per-card features themselves. They all
hit the same ~0.70 val_acc ceiling.

Phase B encoder fine-tuning is the first intervention that targets the per-card
features. The diagnostic for whether it's the right next step is exactly the
val_acc ceiling: regularization/aggregation interventions can't move it because
they don't change the inputs to the aggregation. Encoder fine-tuning *can* in
principle move it because it changes what each card "looks like" to the rest of
the network.

# What This Doesn't Do

Phase B does not move the **label noise floor** estimated at ~0.72–0.78 val_acc
for the current corpus (see `experiments/gen2-initial-training.md` § "The
oracle ceiling for this corpus"). Bo7 outcomes are inherently noisy for close
matchups, and no model improvement — including encoder fine-tuning — can
predict labels that the labels themselves can't determine. The realistic upside
from Phase B is moving val_acc from ~0.695 to somewhere in the 0.70–0.74 range,
not breaking through the noise ceiling.

To push past the noise floor, label-side interventions are required (repeated
matchups, longer formats for close matchups, fixed-opponent training); these
are out of scope for this spec.

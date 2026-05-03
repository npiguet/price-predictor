# Goal

The scorer's per-card embedding is 544 dims: 512 from the price-predictor
encoder and 32 hand-extracted deterministic features (mana cost, types,
colors, P/T, mana production). Sealed scorers trained on this representation
discriminate deck shape — color count, mana curve, type balance, all of
which the deterministic block already exposes — but lose to hand-coded
heuristics at matched deck shape. The remaining gap sits in per-card quality
discrimination, which only the 512 transformer dims can carry. The
price-prediction signal those dims are pretrained on is loosely aligned with
sealed playability and confounded by collector and reserved-list effects.
See [`experiments/deterministic-feature-reliance.md`](../experiments/deterministic-feature-reliance.md)
for the within-bucket win-rate analysis that surfaced this gap.

A dedicated sealed encoder is trained from scratch on a per-card winnability
target derived from per-game play data. Its output replaces the price-predictor
encoder as the source of the 512 transformer dims fed into the sealed scorer.
Initializing from price-predictor weights would re-introduce the very bias
this spec is meant to remove, so the encoder is trained from random init and
shaped entirely by the winnability signal.

# Label

For each card:

```
label = wins_when_played / wins_when_in_deck
```

| Term                | Definition                                                                               |
|---------------------|------------------------------------------------------------------------------------------|
| `wins_when_played`  | Games where the card was played (entered the battlefield or stack) by the winning side.  |
| `wins_when_in_deck` | Games where the card was in the winning side's deck, whether or not it was played.       |

Losing-side games are excluded from both terms. The ratio answers: when the
card is in a winning deck, how often does it contribute by getting played?

## Denominator choice

| Alternative                    | Bias                                              |
|--------------------------------|---------------------------------------------------|
| `/ games_when_played`          | Rewards rarely-cast bombs that win when they hit. |
| `/ in_deck_games` (won + lost) | Penalizes cards stuck in losing decks.            |
| `/ wins_when_in_deck` (chosen) | Credit-assignment within winning decks only.      |

# Data collection

The Java match worker writes per-game records to
`output/sealed/cards-played.txt`, alongside the per-match
`output/sealed/match-outcomes.txt`.

- One line per game (a Bo7 match produces 4-7 lines).
- Append-only, line-buffered.
- Written automatically during every `python -m sealed match-outcomes` run.
  No opt-in flag.
- Row order matches `match-outcomes.txt`: a match's game lines appear
  contiguously, in game order.

## Format

```
set_code;method_A;method_B;cards_played_A;cards_played_B;cards_not_played_A;cards_not_played_B;winner;starter
```

| Column                 | Type                 | Description                                                                                              |
|------------------------|----------------------|----------------------------------------------------------------------------------------------------------|
| `set_code`             | string               | MTG set code; matches the per-match value.                                                               |
| `method_A`, `method_B` | string               | Generation-method tags.                                                                                  |
| `cards_played_X`       | pipe-separated names | Cards from side X's deck that entered the battlefield or stack during this game.                         |
| `cards_not_played_X`   | pipe-separated names | Remaining cards in side X's deck. `cards_played_X ∪ cards_not_played_X` reconstructs side X's full deck. |
| `winner`               | `A` or `B`           | Game winner.                                                                                             |
| `starter`              | `A` or `B`           | First-turn side.                                                                                         |

A card "is played" iff it enters the battlefield or the stack during the
game, controlled by the side that owns it.

# Aggregation

Single pass over `cards-played.txt`:

```
for each line:
    if winner == A:
        for c in cards_played_A:                       wins_when_played[c] += 1
        for c in cards_played_A ∪ cards_not_played_A:  wins_when_in_deck[c] += 1
    if winner == B:
        for c in cards_played_B:                       wins_when_played[c] += 1
        for c in cards_played_B ∪ cards_not_played_B:  wins_when_in_deck[c] += 1
```

`label[c] = wins_when_played[c] / wins_when_in_deck[c]`. Cards with
`wins_when_in_deck == 0` are excluded from training.

# Low-n regularization

Cards with few observations have noisy labels. Two compatible approaches:

| Approach           | Form                                                                             | Effect                                                                |
|--------------------|----------------------------------------------------------------------------------|-----------------------------------------------------------------------|
| Bayesian shrinkage | `label = (wins_when_played + k/2) / (wins_when_in_deck + k)`                     | Pulls low-n labels toward 0.5; converges to the raw ratio as n grows. |
| Sample weighting   | regression-loss weight per card = `wins_when_in_deck / (wins_when_in_deck + k)`  | Low-n cards contribute less gradient.                                 |

`k` is the prior weight — conceptually, the equivalent number of 50/50
observations baked into the estimate. Higher `k` shrinks more aggressively.
Worked example at `k = 20`:

| `wins_when_in_deck` | wins | raw label | shrunk label |
|---------------------|------|-----------|--------------|
| 2                   | 2    | 1.00      | 0.55         |
| 20                  | 15   | 0.75      | 0.625        |
| 1000                | 750  | 0.75      | 0.745        |

The two approaches stack.

# Architecture

The model has three components: a token encoder, a card encoder, and a
regression head. The token encoder and card encoder together form the
artifact saved at `models/sealed/encoder/latest.pt`; the regression head
exists only during training and is discarded afterward.

Data flow:

```
input: tokenized card text (sequence of token IDs)
    ↓
[token encoder]                 vocab lookup + positional encoding
    ↓
(T, d_token) token sequence
    ↓
[card encoder]                  transformer layers + pool
    ↓
card vector (d_card)
    ↓
[regression head]               training-only
    ↓
predicted winnability ∈ [0, 1]
```

## Token encoder

A learned lookup table mapping each token ID to a `d_token`-dim vector,
summed with a positional encoding for the token's position. No cross-token
mixing happens here — each token's output is a function of its own ID and
position only. The vocabulary is the existing MTG tokenizer (~5k tokens;
see spec 010).

Output shape: `(T, d_token)`, where `T` is the number of tokens in the
card's text.

## Card encoder

Takes the `(T, d_token)` token sequence and produces a `d_card`-dim card
vector. Two stages internally:

```
(T, d_token)
    ↓
[N transformer encoder layers]    self-attention + FFN, with residuals
    ↓
(T, d_token)                      contextualized tokens
    ↓
[pool layer]
    ↓
card vector (d_card)
```

**Transformer layers** (`N` typically 4-6). Standard transformer encoder
blocks (multi-head self-attention + FFN + residual). This stage is where
all cross-token mixing happens — abilities get linked to their costs,
triggered effects link to their trigger conditions, type-line tokens link
to body-text tokens. After the `N` layers, every token's output vector
reflects context from the entire card, not just the token itself.

**Pool layer** (single layer). Summarizes the `T` contextualized token
vectors into a single fixed-size card vector via two parallel operations
whose outputs are concatenated:

| Operation                  | Description                                                                                                                          |
|----------------------------|--------------------------------------------------------------------------------------------------------------------------------------|
| Multi-query attention pool | `K` learned query vectors cross-attend to the token sequence; the `K` outputs are concatenated. `K` typically 4.                     |
| Max pool                   | Element-wise max across the token sequence.                                                                                          |

`d_card = (multi-query attention output dims) + (max-pool output dims)`.
A natural default mirroring the price-predictor's split is `d_token = 256`
for both halves, giving `d_card = 512`.

**Why multi-query attention rather than stacked pool layers.** The
"linking abilities to costs and effects" intent is what motivates richer
pooling, but that linking already happens in the transformer-layer stack
above — every transformer layer is full self-attention over all tokens, so
by the end of the stack each token has absorbed context from every other.
Stacking multiple cross-attention pool layers on top would refine one
summary vector iteratively, but it can't surface information that isn't
already encoded in the contextualized tokens. The right axis for "richer
pooling" is parallel summary views, not iterative refinement: K learned
queries in a single pool layer each summarize the token sequence from a
different angle (one query may learn to focus on creature stats, another
on removal effects, another on cost structure, etc.) and their concatenated
outputs cover more ground than any single summary could.

If extra capacity is genuinely needed, it goes into the transformer-layer
stack (`N`), not into stacking pool layers.

## Regression head

```
card vector (d_card) → Linear(d_card, 1) → sigmoid → predicted winnability
```

Single linear layer to a scalar, followed by sigmoid since the label is
bounded to `[0, 1]`. This is deliberately the simplest possible head — a
strong architectural commitment that all representational work lives in
the encoder. If a linear projection of the card vector can't fit the
winnability target, the encoder is the problem, and a deeper head would
just paper over it.

# Training

The encoder and regression head are trained jointly from random
initialization on the per-card `(card_text, winnability_score)` map
produced by the aggregation step. Loss is MSE against the shrunk label.
The encoder is the only artifact preserved after training; the regression
head is discarded.

The train/val split is at the *card* level (held-out cards, never shared
with the train set) and stratified by winnability quartile so each split
covers the full label range. Without stratification a random per-card
split risks under-representing the tails (the strongest and weakest
cards), which are the most informative to evaluate on.

The trained encoder checkpoint at `models/sealed/encoder/latest.pt`
becomes the default `--encoder-checkpoint` for `train-scorer` Phase A and
Phase B. The price-predictor pipeline is no longer a dependency of the
sealed encoder; it remains a separate product for predicting card prices
for users.

# CLI

## Vocabulary

```
python -m sealed build-vocab
    [--cards-folder PATH]    default output/cardsfolder/
    [--vocab-path PATH]      default models/sealed/encoder/vocab.txt
    [--target-size N]        default ~5000
```

Thin sealed-side wrapper that delegates the corpus scan and tokenizer
fitting to the shared `price_predictor.application.build_vocabulary`
utility, with sealed-specific output defaults. The vocabulary algorithm
is task-agnostic (just "scan converted card text → emit token vocab"),
so sharing it does not re-introduce a price-task dependency on the
sealed encoder's training data.

## Training

```
python -m sealed train-encoder
    Inputs:
    [--cards-played-path PATH]   default output/sealed/cards-played.txt
    [--cards-folder PATH]        default output/cardsfolder/
    [--vocab-path PATH]          default models/sealed/encoder/vocab.txt
    [--model-output PATH]        default models/sealed/encoder/

    Training (mirrors price_predictor's `train transformer`):
    [--batch-size N]             default 64
    [--epochs N]                 default 100
    [--lr F]                     default 1e-4
    [--patience N]               default 20
    [--dropout F]                default 0.1

    Architecture / regularization (sealed-specific):
    [--n-layers N]               default 6
    [--n-heads N]                default 4
    [--n-pool-queries K]         default 4
    [--shrinkage-k F]            default 20
```

Remaining knobs are hardcoded:

| Constant             | Value                  | Note                                                                  |
|----------------------|------------------------|-----------------------------------------------------------------------|
| `d_model`            | 256                    | matches price-predictor (= `d_token` in the architecture section)     |
| `ff_dim`             | 1024                   | matches price-predictor                                               |
| `max_seq_len`        | computed from corpus   | matches price-predictor; max card length rounded up to multiple of 8  |
| `loss`               | MSE                    | regression on the [0, 1] target                                       |
| `val_split`          | 0.2                    | 20% held-out cards                                                    |
| `val_stratification` | winnability quartiles  | each split covers the full label range                                |
| `seed`               | 42                     | matches price-predictor                                               |

Aggregation (cards-played.txt → per-card winnability map) runs inline at
train start; no separate command, no on-disk cache. Best checkpoint by
val loss saves to `models/sealed/encoder/{timestamp}.pt`; `latest.pt`
updates after each successful run.

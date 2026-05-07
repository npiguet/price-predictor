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

A dedicated sealed encoder is trained from scratch on two per-card
targets derived from per-game play data: a signed net-influence score
(does this card pull toward winning or losing when in a deck?) and a
played rate (how often does this card actually get cast when in a
deck?). The two are jointly supervised through parallel regression
heads on top of a shared encoder. Its output replaces the
price-predictor encoder as the source of the 512 transformer dims fed
into the sealed scorer. Initializing from price-predictor weights would
re-introduce the very bias this spec is meant to remove, so the encoder
is trained from random init and shaped entirely by the score and
played-rate signals.

# Labels

Each card has two scalar targets, both derived from the same per-game
play counts:

| Head | Formula                                                                                          | Range      | Interpretation                                                            |
|------|--------------------------------------------------------------------------------------------------|------------|---------------------------------------------------------------------------|
| 1    | `(wins_when_played - losses_when_played) / (wins_when_in_deck + losses_when_in_deck)`            | [-1, +1]   | Net winning influence: when in a deck, does this card pull toward winning or losing? |
| 2    | `(wins_when_played + losses_when_played) / (wins_when_in_deck + losses_when_in_deck)`            | [0, 1]     | Played rate: when in a deck, how often does it actually get cast?         |

| Term                  | Definition                                                                                |
|-----------------------|-------------------------------------------------------------------------------------------|
| `wins_when_played`    | Games where the card was played (entered the battlefield or stack) by the winning side.   |
| `wins_when_in_deck`   | Games where the card was in the winning side's deck, whether or not it was played.        |
| `losses_when_played`  | Games where the card was played by the losing side.                                       |
| `losses_when_in_deck` | Games where the card was in the losing side's deck, whether or not it was played.         |

## Why two heads

Head 1 alone collapses two distinct phenotypes onto the same label:

|                       | Head 1 ≈ 0                                       | Head 1 ≫ 0                                  | Head 1 ≪ 0                                  |
|-----------------------|--------------------------------------------------|---------------------------------------------|---------------------------------------------|
| **Head 2 ≈ 0**        | Dead card: drafted but never cast (e.g. Shackles, 637 in-deck / 0 plays). | Niche bomb: rarely cast, mostly when winning. | Niche dud: rarely cast, mostly when losing.   |
| **Head 2 ≈ 0.5–1.0**  | Cast often but balanced — neutral when active.   | Cast often, correlates with winning (good card). | Cast often, correlates with losing (bad card). |

A dead card and a balanced-when-cast card both sit at head 1 = 0, so a
single-head model treats them identically. Head 2 supplies the
discrimination — `(head 1, head 2) ≈ (0, 0)` is dead weight, `(0, ≫0)`
is a card that gets cast but is neutral. The two heads share the
encoder, so head 2's gradient forces the encoder to place these cases in
different regions of embedding space even though head 1's gradient
cannot tell them apart.

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
timestamp;run_id;set_code;method_A;method_B;cards_played_A;cards_played_B;cards_not_played_A;cards_not_played_B;winner;starter
```

| Column                 | Type                 | Description                                                                                                                                                |
|------------------------|----------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `timestamp`            | ISO 8601 UTC         | Game completion time. Same format used by `match-outcomes.txt`.                                                                                            |
| `run_id`               | UUID                 | Identifies the supervisor invocation. Matches the parent match's `run_id` in `match-outcomes.txt`, so the two files are joinable on `run_id`.              |
| `set_code`             | string               | MTG set code; matches the per-match value.                                                                                                                 |
| `method_A`, `method_B` | string               | Generation-method tags.                                                                                                                                    |
| `cards_played_X`       | pipe-separated names | Non-basic cards from side X's deck that entered the battlefield or stack during this game.                                                                 |
| `cards_not_played_X`   | pipe-separated names | Remaining non-basic cards in side X's deck. `cards_played_X ∪ cards_not_played_X` reconstructs side X's full deck *minus basic lands*.                     |
| `winner`               | `A` or `B`           | Game winner.                                                                                                                                               |
| `starter`              | `A` or `B`           | First-turn side.                                                                                                                                           |

A card "is played" iff it enters the battlefield or the stack during the
game, controlled by the side that owns it.

Basic lands (any card whose type line contains the supertype `Basic` —
Plains, Island, Swamp, Mountain, Forest, Wastes, snow-covered variants,
and any future basic printings) are excluded by the Java worker at write
time. They never appear in `cards_played_X` or `cards_not_played_X` and
never enter the per-card label map. Their labels would be
near-deterministic (every winning deck taps lands every game) and would
dominate the label distribution without contributing any card-quality
signal.

Within a single `(run_id, set_code, method_A, method_B)` group, the i-th
contiguous block of game lines corresponds to the i-th matching line in
`match-outcomes.txt`. Concurrent supervisors writing with distinct
`run_id`s are joinable independently and never conflict.

# Aggregation

Single pass over `cards-played.txt`:

```
for each line:
    if winner == A:
        winner_played, winner_deck = cards_played_A, cards_played_A ∪ cards_not_played_A
        loser_played,  loser_deck  = cards_played_B, cards_played_B ∪ cards_not_played_B
    else:
        winner_played, winner_deck = cards_played_B, cards_played_B ∪ cards_not_played_B
        loser_played,  loser_deck  = cards_played_A, cards_played_A ∪ cards_not_played_A
    for c in winner_played:  wins_when_played[c]    += 1
    for c in winner_deck:    wins_when_in_deck[c]   += 1
    for c in loser_played:   losses_when_played[c]  += 1
    for c in loser_deck:     losses_when_in_deck[c] += 1
```

For each card,

```
in_deck_total     = wins_when_in_deck + losses_when_in_deck
played_total      = wins_when_played + losses_when_played
raw_score         = (wins_when_played - losses_when_played) / in_deck_total
raw_played_rate   = played_total / in_deck_total
```

Cards with `in_deck_total == 0` are excluded from training.

After aggregation, and before training begins, `train-encoder` writes
the entire per-card label map to `output/sealed/cards-win-rates.txt`,
sorted by raw score descending. One row per card included in training,
semicolon-separated, columns:

```
card_name;wins_when_played;wins_when_in_deck;losses_when_played;losses_when_in_deck;raw_score;shrunk_score;raw_played_rate;shrunk_played_rate
```

The file is overwritten on every run. Its purpose is to make the
shrinkage effect (and the label distribution generally) human-readable
without persisting the label map itself for cross-run reuse — diffing
two runs with different `--shrinkage-k` values is the supported way to
verify that low-observation cards shift while high-observation cards
stay close to their raw labels. The path is fixed and not configurable.

# Low-n regularization

Cards with few observations have noisy labels. Bayesian shrinkage is
applied to each head separately, with the prior chosen to match the
head's neutral point:

| Head | Shrunk form                                                                                                      | Prior                          |
|------|------------------------------------------------------------------------------------------------------------------|--------------------------------|
| 1    | `(wins_when_played - losses_when_played) / (wins_when_in_deck + losses_when_in_deck + k)`                        | 0   (neutral influence)        |
| 2    | `(wins_when_played + losses_when_played + k/2) / (wins_when_in_deck + losses_when_in_deck + k)`                  | 0.5 (no information either way)|

`k` is the prior weight — conceptually, the equivalent number of
neutral observations baked into the estimate. Higher `k` shrinks more
aggressively. A single `--shrinkage-k` flag drives both heads.

Worked example for head 1 at `k = 20`:

| `in_deck_total` | `wins_played` | `losses_played` | raw score | shrunk score |
|-----------------|---------------|-----------------|-----------|--------------|
| 2               | 2             | 0               | +1.00     | +0.091       |
| 20              | 15            | 5               | +0.50     | +0.250       |
| 1000            | 600           | 100             | +0.50     | +0.490       |

Sample weighting (`weight = in_deck_total / (in_deck_total + k)`)
stacks with shrinkage if low-n cards still introduce too much variance.

# Architecture

The model has three components: a token encoder, a card encoder, and
two regression heads. The token encoder and card encoder together form
the artifact saved at `models/sealed/encoder/latest.pt`; both regression
heads exist only during training and are discarded afterward.

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
    ↓                           training-only:
    ├──→ [score head]       →   predicted score        ∈ [-1, +1]
    └──→ [played-rate head] →   predicted played rate  ∈ [0, 1]
```

## Token encoder

A learned lookup table mapping each token ID to a `d_token`-dim vector,
summed with a positional encoding for the token's position. No cross-token
mixing happens here — each token's output is a function of its own ID and
position only. The vocabulary is the existing MTG tokenizer (~5k tokens;
see spec 010).

Before tokenizing, the card's `name:` line is stripped — exactly the same
transformation `python -m sealed encode-cards` applies at inference time.
This serves two purposes: it keeps the encoder's training and inference
inputs identically shaped (no train/inference mismatch), and it prevents
the encoder from shortcutting on `name → label` during training. Card
names are still required during aggregation to map labels to cards, but
they never reach the model.

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

| Operation                  | Description                                                                                                                                              |
|----------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
| Multi-query attention pool | `K` learned query vectors cross-attend to the token sequence; each query outputs `d_token / K` dims; the `K` outputs are concatenated to `d_token` dims. |
| Max pool                   | Element-wise max across the token sequence (output dims = `d_token`).                                                                                    |

The two halves are concatenated to give `d_card = 2 * d_token`. With
`d_token = 256` (mirroring the price-predictor), `d_card = 512`. `d_card`
is independent of `K`; `K` controls per-query capacity, not the
card-vector size. `d_token` must be divisible by `K` (default `K = 4`,
giving 64 dims per query).

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

## Regression heads

Two parallel heads, each a single linear projection of the card vector
to a scalar:

```
card vector (d_card) ─┬─→ Linear(d_card, 1) → tanh    → predicted score        ∈ [-1, +1]
                      └─→ Linear(d_card, 1) → sigmoid → predicted played rate  ∈ [0, 1]
```

The activation matches each head's range: `tanh` for the signed score,
`sigmoid` for the [0, 1]-bounded played rate. Each head is deliberately
the simplest possible projection — a strong architectural commitment
that all representational work lives in the encoder. If a linear
projection of the card vector can't fit either target, the encoder is
the problem, and a deeper head would just paper over it.

Sharing the encoder across the two heads is also the mechanism that
disambiguates dead cards from balanced-when-cast cards (see *Why two
heads* in the labels section): head 2's gradient pushes those two
phenotypes apart even though head 1's gradient cannot tell them apart.

# Training

The encoder and both regression heads are trained jointly from random
initialization on the per-card `(card_text, score, played_rate)` map
produced by the aggregation step. The training loss is the unweighted
sum of two MSE terms, one per head, against the shrunk targets:

```
loss = MSE(head_1_output, shrunk_score) + MSE(head_2_output, shrunk_played_rate)
```

The encoder is the only artifact preserved after training; both
regression heads are discarded.

`train-encoder` performs a corpus consistency check at start: every card
name referenced in `cards-played.txt` must have a corresponding `.txt`
file in the corpus folder. If any card is missing, training fails
immediately with an error naming the missing cards (capped at a
reasonable display count, with the total reported) and points the user
at `python -m price_predictor convert` to rebuild the corpus. Training
does not proceed by silently dropping the missing cards — the user must
either rebuild the corpus or delete the offending `cards-played.txt`
lines.

Vocabulary freshness is *not* validated. Tokens introduced after the
last `build-vocab` run fall back to UNK silently. Keeping
`models/sealed/encoder/vocab.txt` in sync with the corpus is the user's
responsibility — re-run `python -m sealed build-vocab` after any
material corpus change.

The train/val split is at the *card* level (held-out cards, never shared
with the train set) and stratified by score quartile (head 1 target) so
each split covers the full score range. Without stratification a random
per-card split risks under-representing the tails (the strongest and
weakest cards), which are the most informative to evaluate on.
Stratification on head 2 is not enforced separately — score is the
primary signal for downstream use, and the played-rate range is wide
enough across the score quartiles that one-axis stratification is
sufficient in practice.

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
| `loss`               | MSE(score) + MSE(played_rate) | unweighted sum of two per-head MSE terms                       |
| `val_split`          | 0.2                    | 20% held-out cards                                                    |
| `val_stratification` | score quartiles        | each split covers the full score range                                |
| `seed`               | 42                     | matches price-predictor                                               |

Aggregation (cards-played.txt → per-card score + played-rate map) runs
inline at train start; no separate command, no on-disk cache. Best
checkpoint by val loss saves to `models/sealed/encoder/{timestamp}.pt`;
`latest.pt` updates after each successful run.

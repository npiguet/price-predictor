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
See [`experiments/2026-05-02-deterministic-feature-reliance.md`](../experiments/2026-05-02-deterministic-feature-reliance.md)
for the within-bucket win-rate analysis that surfaced this gap.

A dedicated sealed encoder is trained from scratch on per-card targets
derived from per-game play data. Nine parallel regression heads on top
of a shared encoder supervise distinct facets of card behavior — net
winning influence on the play and on the draw, played rate, cast-lift,
and a per-color affinity for each of WUBRG — and a masked-token
reconstruction head provides dense self-supervision over the long-tail
cards that the regression heads under-train. The encoder output replaces
the price-predictor encoder as the source of the 512 transformer dims
fed into the sealed scorer. Initializing from price-predictor weights
would re-introduce the very bias this spec is meant to remove, so the
encoder is trained from random init and shaped entirely by the
per-card signals defined below.

# Labels

Every per-card target derives from the same per-game play counters in
`cards-played.txt`. Counters split four ways:

| Subscript          | Meaning                                                                                          |
|--------------------|--------------------------------------------------------------------------------------------------|
| `played` / `not_played` | the card was cast in the game / was in the deck but never cast                              |
| `won` / `lost`     | side that owned the card won / lost the game                                                     |
| `@play` / `@draw`  | side that owned the card was the starter / was not the starter                                   |
| `with_X`           | the card's owner ran at least one card with color X ∈ {W, U, B, R, G} in its mana cost           |

Counter shorthand: `W_played@play` is the count of games where the
owning side won, was on the play, and cast the card. `n_in_deck@play`
is the count of games where the card was in the owner's deck and the
owner was on the play. Equivalent definitions for the other axes; an
unsubscripted counter sums over all games.

Five head families:

| Head                       | Formula                                                                                                                                                              | Range      | Activation | Captures                                                  |
|----------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------|------------|-----------------------------------------------------------|
| `score_play`               | `(W_played@play - L_played@play) / n_in_deck@play`                                                                                                                   | [-1, +1]   | tanh       | net winning influence on the play (tempo-card value)      |
| `score_draw`               | `(W_played@draw - L_played@draw) / n_in_deck@draw`                                                                                                                   | [-1, +1]   | tanh       | net winning influence on the draw (reactive-card value)   |
| `played_rate`              | `(W_played + L_played) / n_in_deck`                                                                                                                                  | [0, 1]     | sigmoid    | cast frequency when in a deck                             |
| `cast_lift`                | `p_play - p_dead`, where `p_play = W_played / (W_played + L_played)` and `p_dead = W_not_played / (W_not_played + L_not_played)`                                     | [-1, +1]   | tanh       | causal lift from casting (independent of which deck)      |
| `color_lift_X` (X ∈ WUBRG) | `score_with_X − score_overall`, where `score_with_X = (W_played_with_X − L_played_with_X) / n_in_deck_with_X` and `score_overall = (W_played − L_played) / n_in_deck`| [-1, +1]   | tanh       | per-color affinity (cross-color synergy at deck level)    |

Cards with `n_in_deck == 0` are excluded entirely. For per-slice heads,
cells where the slice denominator is zero (`n_in_deck@play == 0`,
`n_in_deck_with_X == 0`, etc.) contribute zero loss — not a zero label.
The raw forms above are for label-inspection diagnostics; training uses
the shrunk forms in *Low-n regularization*.

# Why these heads

Each head targets a distinct axis of card behavior. Together they span
the information available in `cards-played.txt` without redundancy.

## `score_play` / `score_draw`: play/draw split of winning influence

Sealed has a structural play/draw asymmetry — tempo cards (one-drops,
hasty creatures, curve plays) gain value on the play; reactive cards
(sweepers, expensive removal, card draw) gain value on the draw. A
single net-influence label averages these into one scalar, so a card
that is +0.30 on the play and -0.10 on the draw and a card that is
+0.10 on both both land at ≈ +0.10 — the encoder sees identical labels
for two MTG-distinct phenotypes. With the split, the gradients become
opposite vectors in embedding space, and the downstream scorer can
balance tempo and reactive cards in a deck deliberately (sealed decks
mix both because game-1 play/draw is unknown).

A combined unsplit head is *not* trained alongside: with `n@play ≈
n@draw` per card (starter assignment is approximately uniform across
games), `unsplit ≈ 0.5·score_play + 0.5·score_draw` carries no
information the split heads don't already carry, and supervising on it
would just double the gradient pressure on the winning-influence axis
relative to played rate.

## `played_rate`: cast frequency

Disambiguates dead cards (in deck but never cast) from
balanced-when-cast cards. Both sit at score ≈ 0; `played_rate` ≈ 0 vs
≈ 0.5 distinguishes them.

| Phenotype                  | score ≈ 0                          | score ≫ 0                                  | score ≪ 0                            |
|----------------------------|------------------------------------|--------------------------------------------|--------------------------------------|
| **`played_rate` ≈ 0**      | dead card (drafted, never cast)    | niche bomb (rarely cast, mostly when winning) | niche dud (rarely cast, when losing) |
| **`played_rate` ≈ 0.5–1.0**| cast often, neutral effect         | cast often, correlates with winning        | cast often, correlates with losing   |

Without `played_rate` the encoder cannot distinguish those phenotypes
from any of the score-family heads alone.

`played_rate` is unsplit by play/draw — whether a card gets cast is
dominated by mana cost and draws, both ≈ insensitive to the play/draw
axis, so a played-rate split mostly buys √2 noise for negligible signal.

## `cast_lift`: casting effect isolated from deck quality

The score family sums over *all* in-deck observations regardless of
whether the card was actually cast. This conflates "casting this card
changes the outcome" with "this card tends to land in winning decks".
An auto-include card just along for the ride in strong decks and a
card that genuinely swings games when cast can land at the same
score value.

Worked example, all 100 in-deck observations:

| Card                    | n_cast | p_play | p_dead | score | played_rate | cast_lift |
|-------------------------|-------:|-------:|-------:|------:|------------:|----------:|
| Workhorse 2-drop        |     80 |   0.60 |   0.50 | +0.16 |        0.80 |     +0.10 |
| 6-drop bomb             |     30 |   0.70 |   0.50 | +0.12 |        0.30 |     +0.20 |
| Auto-include drag       |     60 |   0.55 |   0.55 | +0.10 |        0.60 |      0.00 |

The third row is the diagnostic: score looks like a useful card, but
the deck wins 55% whether or not the card hits the table. The
build-method × card-strength interaction (forge-best favors it over
`random`) gives it ride-along lift that `cast_lift` correctly
attributes to the deck rather than the card. The 6-drop bomb is the
symmetric case: its score is *attenuated* because two-thirds of in-deck
appearances are dead games even though every individual cast swings the
outcome by 20pp.

The four raw played/not-played × winner/loser counters have three
degrees of freedom after factoring out scale. Score, played rate, and
cast lift are jointly required to capture all of them: `(played_rate,
cast_lift)` recovers the casting gap but not the absolute winrate
level; `(score, played_rate)` recovers `p_play` but not `p_dead`. None
of the three is a linear combination of the other two. The loss-side
downforce metric `p_play_lose − p_dead_lose` equals `−cast_lift`
exactly, so a separate downforce head adds nothing.

## `color_lift_X`: cross-color affinity

Captures the per-card "this white removal pairs well with blue" axis
that the encoder cannot extract from token-level color-identity
signals alone. The downstream scorer reads these heads at
deck-building time to prefer cards whose embedding indicates synergy
with the deck's already-committed colors — directly relevant when
building a 2- or 3-color deck where the secondary color choice is
under-constrained.

The deviation framing (`score_with_X − score_overall`, not the raw
conditional `score_with_X`) is deliberate:

- The diagonal cell — a card's own color, mechanically present in any
  deck containing it — collapses to zero. The head doesn't carry a
  noisy copy of the score family along that direction, and gradient
  signal lives entirely in the off-diagonal cells where the affinity
  information actually sits.
- The card's overall winning-influence baseline cancels out, so a
  strong card and a weak card with identical color-affinity profiles
  receive identical labels. The encoder is forced to encode color
  affinity as a separate axis from raw card quality.

Five outputs (one per WUBRG) rather than a single multi-class head
because the deviations are independent quantities — a card can be
positively affine to U and negatively to R with no constraint between
them.

## MLM auxiliary loss

The five regression-axis families deliver dense per-token gradient on
high-observation cards but the long tail (cards seen in few decks)
contributes one noisy scalar per head per card after shrinkage. MLM
restores per-token gradient there: ~15% of input tokens are masked at
each forward pass, the contextualized token outputs are projected back
to vocab logits at the masked positions, and cross-entropy against the
original token feeds into the loss with a small weight.

| Mechanism                       | Effect                                                                                                          |
|---------------------------------|-----------------------------------------------------------------------------------------------------------------|
| Dense per-token signal          | Tail cards still contribute meaningful gradient — every token of every card is a training example.              |
| Self-supervised                 | No additional data collection or label noise.                                                                   |
| Joint with regression heads     | Avoids the catastrophic-forgetting risk of pre-train-then-fine-tune; one optimization, one set of hyperparameters. |
| Regularizer                     | Encoder cannot collapse onto a few regression-label-specific dimensions; those dimensions still need to support token reconstruction across the full corpus. |

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
| `starter`              | `A` or `B`           | First-turn side. Drives the play/draw split for `score_play` and `score_draw`.                                                                             |

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

Two passes over `cards-played.txt`. The first pass collects the primary
and play/draw counters. The second pass resolves per-card color
identity from the converted card corpus and fills the per-color slices.

## Pass 1: primary + play/draw counters

For each game:

```
winner_was_starter = (winner == starter)

if winner == A:
    winner_played, winner_deck = cards_played_A, cards_played_A ∪ cards_not_played_A
    loser_played,  loser_deck  = cards_played_B, cards_played_B ∪ cards_not_played_B
else:
    winner_played, winner_deck = cards_played_B, cards_played_B ∪ cards_not_played_B
    loser_played,  loser_deck  = cards_played_A, cards_played_A ∪ cards_not_played_A

for c in winner_deck:
    wins_when_in_deck[c] += 1
    if c in winner_played: wins_when_played[c] += 1
    if winner_was_starter:
        wins_when_in_deck@play[c]  += 1
        if c in winner_played: wins_when_played@play[c] += 1

for c in loser_deck:
    losses_when_in_deck[c] += 1
    if c in loser_played: losses_when_played[c] += 1
    if not winner_was_starter:
        losses_when_in_deck@play[c]  += 1
        if c in loser_played: losses_when_played@play[c] += 1
```

`@draw` counters are derived by subtraction:
`wins_when_played@draw = wins_when_played − wins_when_played@play`,
`losses_when_in_deck@draw = losses_when_in_deck − losses_when_in_deck@play`,
etc.

`cast_lift` reuses the four primary counters:
`wins_when_not_played = wins_when_in_deck − wins_when_played`,
`losses_when_not_played = losses_when_in_deck − losses_when_played`.
No new counters needed.

## Pass 2: per-color slices

After pass 1, each card's color identity is resolved from the
`mana cost:` line of its converted text file under
`output/cardsfolder/`. WUBRG letters in the cost (including hybrid
like `{W/U}` and Phyrexian like `{W/P}`) contribute; generic, colorless
and X costs contribute nothing. Lands and any cards whose converted
text has no `mana cost:` line (e.g. some split-card halves) contribute
no colors.

Each game's deck colors are the union of per-card colors over the
deck's contents. For each side, for each color present in that side's
deck, the per-color counters increment in parallel with the primary
counters: `wins_when_played_with_c`, `wins_when_in_deck_with_c`,
`losses_when_played_with_c`, `losses_when_in_deck_with_c`. Multi-color
decks contribute to multiple slices, so the per-color counters are not
disjoint and do not sum to the primary counters.

## Label-inspection file

`train-encoder` writes the per-card label map to
`output/sealed/cards-win-rates.txt` after aggregation and before
training, sorted by `shrunk_score_play` descending. The file is
overwritten on every run. Its purpose is to make the shrinkage effect
(and the label distribution generally) human-readable without
persisting the label map for cross-run reuse — diffing two runs with
different `--shrinkage-k` values is the supported way to verify that
low-observation cards shift while high-observation cards stay close to
their raw labels. The path is fixed and not configurable.

Schema (one row per card, semicolon-separated, no trailing `;`):

```
card_name;
wins_when_played;wins_when_in_deck;losses_when_played;losses_when_in_deck;
raw_score_play;shrunk_score_play;
raw_score_draw;shrunk_score_draw;
raw_played_rate;shrunk_played_rate;
raw_cast_lift;shrunk_cast_lift;
raw_color_lift_W;shrunk_color_lift_W;
raw_color_lift_U;shrunk_color_lift_U;
raw_color_lift_B;shrunk_color_lift_B;
raw_color_lift_R;shrunk_color_lift_R;
raw_color_lift_G;shrunk_color_lift_G
```

# Low-n regularization

Cards with few observations have noisy labels. Bayesian shrinkage is
applied to each head separately, with the prior chosen to match the
head's neutral point:

| Head             | Shrunk form                                                                                                                                            | Prior                          |
|------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------|
| `score_play`     | `(W_played@play − L_played@play) / (n_in_deck@play + k)`                                                                                               | 0   (neutral influence)        |
| `score_draw`     | `(W_played@draw − L_played@draw) / (n_in_deck@draw + k)`                                                                                               | 0   (neutral influence)        |
| `played_rate`    | `(played_total + k/2) / (n_in_deck + k)`                                                                                                               | 0.5 (no information either way)|
| `cast_lift`      | `(W_played + k/2) / (played_total + k) − (W_not_played + k/2) / ((n_in_deck − played_total) + k)`                                                      | 0   (casting changes nothing)  |
| `color_lift_X`   | `(W_played_with_X − L_played_with_X) / (n_in_deck_with_X + k) − (W_played − L_played) / (n_in_deck + k)`                                               | 0   (no color affinity)        |

A single `--shrinkage-k` flag drives every head. `k` is the prior
weight — conceptually, the equivalent number of neutral observations
baked into the estimate. Higher `k` shrinks more aggressively.

Worked example for `score_play` at `k = 20`:

| `n_in_deck@play` | `W_played@play` | `L_played@play` | raw   | shrunk |
|------------------|-----------------|-----------------|-------|--------|
| 2                | 2               | 0               | +1.00 | +0.091 |
| 20               | 15              | 5               | +0.50 | +0.250 |
| 1000             | 600             | 100             | +0.50 | +0.490 |

Per-head sample weights stack on top of shrinkage; cells with the
smaller-side denominator near zero have their loss contribution pulled
down regardless of the shrunk value's magnitude:

```
weight_score_play   = n_in_deck@play / (n_in_deck@play + k)
weight_score_draw   = n_in_deck@draw / (n_in_deck@draw + k)
weight_played_rate  = n_in_deck      / (n_in_deck      + k)
weight_cast_lift    = m / (m + k),  m = min(played_total, n_in_deck − played_total)
weight_color_lift_X = n_in_deck_with_X / (n_in_deck_with_X + k)
```

`cast_lift`'s weight uses the smaller of `played_total` and
`n_in_deck − played_total` because cast-lift labels degenerate when
either side is empty: a card cast nearly every draw has too few
`not_played` observations to estimate `p_dead`; symmetric problem at
`played_rate ≈ 0`. Whichever side has the smaller count sets the
cell's reliability — the middle of the played-rate range is where
`cast_lift` carries the most signal.

`color_lift_X`'s weight is necessary even after shrinkage: a R card
seen in W decks 5% of games has a `color_lift_W` cell whose noise
floor is set by the W slice's small denominator, not by the card's
overall observation count.

# Architecture

The model has three components: a token encoder, a card encoder, and
a collection of training-only heads. The token encoder and card encoder
together form the artifact saved at `models/sealed/encoder/latest.pt`;
all heads exist only during training and are discarded afterward.

Data flow:

```
input: tokenized card text (sequence of token IDs, ~15% replaced with [MASK])
    ↓
[token encoder]                      vocab lookup + positional encoding
    ↓
(T, d_token)
    ↓
[N transformer encoder layers]       self-attention + FFN, with residuals
    ↓
(T, d_token) contextualized tokens ──→ [MLM head]   →  vocab logits at masked positions
    ↓
[pool layer]                         multi-query attention pool ‖ max pool
    ↓
card vector (d_card)
    ↓                                training-only:
    ├──→ [score_play head]           →  ∈ [-1, +1]
    ├──→ [score_draw head]           →  ∈ [-1, +1]
    ├──→ [played_rate head]          →  ∈ [0, 1]
    ├──→ [cast_lift head]            →  ∈ [-1, +1]
    └──→ [color_lift head]           →  5 outputs ∈ [-1, +1] (one per WUBRG)
```

## Token encoder

A learned lookup table mapping each token ID to a `d_token`-dim vector,
summed with a positional encoding for the token's position. No cross-token
mixing happens here — each token's output is a function of its own ID and
position only. The vocabulary is the existing MTG tokenizer (~5k tokens;
see spec 010), with a reserved `[MASK]` token used by the MLM head and
never overlapping a real corpus token.

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
(T, d_token)                      contextualized tokens (also consumed by MLM head)
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

## Heads

All five regression heads are single linear projections of the card
vector, with activation matched to each head's range:

| Head           | Projection            | Activation       |
|----------------|-----------------------|------------------|
| `score_play`   | `Linear(d_card, 1)`   | tanh             |
| `score_draw`   | `Linear(d_card, 1)`   | tanh             |
| `played_rate`  | `Linear(d_card, 1)`   | sigmoid          |
| `cast_lift`    | `Linear(d_card, 1)`   | tanh             |
| `color_lift`   | `Linear(d_card, 5)`   | tanh (per output)|

Each head is deliberately the simplest possible projection — a strong
architectural commitment that all representational work lives in the
encoder. If a linear projection of the card vector can't fit a target,
the encoder is the problem, and a deeper head would just paper over it.

The MLM head reads the contextualized token sequence (before the pool
layer), projecting each token vector back to vocab-size logits:

```
contextualized token (d_token) → Linear(d_token, vocab_size) → token logits
```

Loss is computed only at masked positions; unmasked positions
contribute no MLM gradient. The MLM head is discarded after training
along with the regression heads.

# Training

The encoder and all heads are trained jointly from random initialization
on the per-card label map produced by aggregation. The training loss is:

```
L_reg = MSE(score_play) + MSE(score_draw) + MSE(played_rate) + MSE(cast_lift)
      + (1/5) · Σ_{X ∈ WUBRG} MSE(color_lift_X)

L_mlm = mean over masked positions of CE(token_logits, true_token)

loss  = L_reg + w_mlm · L_mlm
```

Per-card MSE terms are sample-weighted by the per-head weights from
*Low-n regularization*; cells where the slice denominator is zero
contribute zero loss.

The `1/5` color-lift normalization keeps the color-affinity family at
roughly one head's gradient pressure, matching how the play/draw split
keeps two heads at the original score family's pressure (each split
head sees half the data).

`w_mlm` (default 0.1, CLI flag `--mlm-weight`) is small enough that
the regression objective dominates training; the encoder isn't pushed
primarily toward token prediction. The MLM weight scales the *loss*,
not the gradient magnitude relative to per-card vs per-token signal —
the MLM head sees ~T loss contributions per card vs the regression
heads' 9, so a small `w_mlm` is what keeps the per-card aggregate
roughly balanced.

`train-encoder` performs a corpus consistency check at start: every card
name referenced in `cards-played.txt` must have a corresponding `.txt`
file in the corpus folder. If any card is missing, training fails
immediately with an error naming the missing cards (capped at a
reasonable display count, with the total reported) and points the user
at `python -m price_predictor convert` to rebuild the corpus. Training
does not proceed by silently dropping the missing cards — the user must
either rebuild the corpus or delete the offending `cards-played.txt`
lines.

Mana-cost resolution during pass 2 of aggregation reads the same
corpus. Cards whose converted text has no `mana cost:` line are treated
as colorless for deck-color computation; they still appear as label
targets, with all five `color_lift` cells carrying signal (no
diagonal-zero anchor).

Vocabulary freshness is *not* validated. Tokens introduced after the
last `build-vocab` run fall back to UNK silently. Keeping
`models/sealed/encoder/vocab.txt` in sync with the corpus is the user's
responsibility — re-run `python -m sealed build-vocab` after any
material corpus change.

The train/val split is at the *card* level (held-out cards, never shared
with the train set) and stratified by `score_play` quartile so each
split covers the full winning-influence range. Without stratification a
random per-card split risks under-representing the tails (the strongest
and weakest cards), which are the most informative to evaluate on.
Stratification on a single axis is sufficient in practice; the other
heads' ranges are wide enough across `score_play` quartiles that they
cover their tails too.

Masking is randomized per training step: each card's tokenized sequence
has ~15% of its non-special tokens replaced with `[MASK]` at the start
of the forward pass. The same card sees different masks across epochs.

The trained encoder checkpoint at `models/sealed/encoder/latest.pt`
becomes the default `--encoder-checkpoint` for `train-scorer` Phase A
and Phase B. The price-predictor pipeline is no longer a dependency of
the sealed encoder; it remains a separate product for predicting card
prices for users.

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
utility, with sealed-specific output defaults. `[MASK]` is reserved as
a special token during vocabulary construction and never collides with
a corpus-derived token. The vocabulary algorithm is task-agnostic
(just "scan converted card text → emit token vocab"), so sharing it
does not re-introduce a price-task dependency on the sealed encoder's
training data.

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
    [--mlm-weight F]             default 0.1
    [--mlm-mask-prob F]          default 0.15
```

Remaining knobs are hardcoded:

| Constant             | Value                          | Note                                                                  |
|----------------------|--------------------------------|-----------------------------------------------------------------------|
| `d_model`            | 256                            | matches price-predictor (= `d_token` in the architecture section)     |
| `ff_dim`             | 1024                           | matches price-predictor                                               |
| `max_seq_len`        | computed from corpus           | matches price-predictor; max card length rounded up to multiple of 8  |
| `loss`               | weighted sum (above)           | regression heads + MLM auxiliary, weights as specified                |
| `val_split`          | 0.2                            | 20% held-out cards                                                    |
| `val_stratification` | `score_play` quartiles         | each split covers the full winning-influence range                    |
| `seed`               | 42                             | matches price-predictor                                               |

Aggregation (cards-played.txt → per-card label map) runs inline at
train start; no separate command, no on-disk cache. Best checkpoint by
val loss saves to `models/sealed/encoder/{timestamp}.pt`; `latest.pt`
updates after each successful run.

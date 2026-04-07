# Goal

Train a sealed deck builder that takes a pool of cards opened from 6 boosters and selects an optimal 40-card deck, using
game outcomes as the training signal. This is a stepping stone toward a full MTG-playing AI.

# Card Representation

Each card is encoded by your pretrained price predictor transformer, producing a 512-dimensional vector (256 mean pool +
256 max pool over ~300 tokens of structured card text). The `name:` line is stripped from the card text before encoding
so that the embedding captures what the card does, not its name. This encoder is frozen initially and unfrozen in later
training stages.

At the pool level, each of the 90 entries is represented as the card embeddings plus a number of important features 
deterministically extractable from card text that are known to be difficult for transformer models to represent 
accurately, but are also important for judging card quality:

- card_embedding [512]
- is_land [1]
- mana cost:
   * number of pips of each color (colorless {C}) [6]
   * number of generic mana [1]
   * number of {X} pips [1]
   * mana value [1]
- card color (WUBRG or C) [6]
- count of mana produced (WUBRG + C) [6]
- power and toughness (for creatures): [2]
- starting loyalty [1]
- zero-padding reserved features (to make the vector size divisible by 8) [7]
- => 544 features per card

Card slots represent the input to the Set Transformer scorer. They all have the same number of features so that they
can be cleanly fed to the model.

## Card Pool Composition

- 84-90 cards (opened from 6 boosters, varies by set, 14-15 cards per booster. Contain lands, including basic lands)

# Overall Approach

The system is built in four sequential phases:

0. **Training Dataset Generation**
1. **Deck Scorer** — A Set Transformer trained via Bradley-Terry pairwise loss on game outcomes to score how good a
   deck is. This is the core model and the most training-intensive component.
2. **Search-Based Deck Builder** — At inference, a heuristic builder seeds a starting deck, and greedy hill-climbing
   guided by the scorer iteratively swaps cards to improve the deck. No ML needed for this step.
3. **One-Shot Policy Model (optional)** — Once the scorer is reliable, distill the search-based builder into a single
   forward-pass model that directly selects cards from a pool.

## Why This Approach

A previous attempt using RL with sequential card selection (PPO with per-step rewards) failed. The model could not learn
correct land-color coordination: it regressed to a "2 lands of each color" strategy because the sparse win/loss signal
made credit assignment across 40 sequential decisions intractable. Color balance is a 6D coupled optimization with a
moving target — every spell pick shifts the ideal distribution across all 6 color buckets simultaneously, and the model
could never get meaningful per-step gradient signal for this global property.

The current approach avoids this in two ways:
- **Basic lands are assigned deterministically**, not predicted. Land distribution is a near-deterministic function of
  the non-land cards selected — if you chose 12 red cards and 10 blue cards, the correct split is ~9 mountains and
  ~8 islands. Removing this from the model's decision space eliminates the regression-to-the-mean trap.
- **The scorer evaluates complete decks**, not partial sequences. One deck in, one score out — no credit assignment
  problem across sequential steps.

# Phase 0 — Training Dataset Generation

Producing training data is the most time-consuming part of the pipeline. The scorer learns entirely from pairwise game
outcomes, so this phase generates pools, builds deck variants, plays games, and stores the results.

Prerequisite: card embeddings and pool files must already exist (see Training Dataset Preparation below).

## Step 1 — Generate Pools

Simulate opening 6 boosters to create pools of ~84-90 cards. Generate thousands of pools. Use the pool generation
tooling described in the Training Dataset Preparation section.

## Step 2 — Build Deck Variants from Each Pool

For each pool, generate several decks of ~23 non-land cards (basic lands excluded, assigned deterministically):

- One from the **heuristic builder** (baseline)
- Several **small perturbations** of the baseline (swap 1-3 random cards with pool cards)
- A few **aggressive variants** (swap 5-8 cards, or rebuild in a different color pair)
- Optionally some **fully random decks** to anchor the low end of quality

Target: 10-20 variants per pool.

## Step 3 — Play Games

For each pool's deck variants, play games to generate pairwise outcomes. Two matchup types:

- **Cross-pool** (primary): Play a deck variant against a heuristic deck from a different pool. This simulates a real
  sealed event and is the actual use case.
- **Within-pool** (supplementary): Play variants from the same pool against each other. This teaches the scorer which
  build of a given pool is strongest.

Play 1-3 games per matchup (3 if noise is a concern). The game-playing AI handles both sides.

## Step 4 — Store Results

Each training example is simply:

```
(deck_A_card_ids, deck_B_card_ids, winner)
```

No win rates, no aggregation — just raw game outcomes.

## Step 5 — Iterate

As the scorer improves (in Phase 1), use it to generate better deck variants by running the search with the current
scorer instead of random perturbations. This creates a curriculum: early data distinguishes bad from decent, later data
distinguishes good from great. Re-run Phase 0 periodically with the improved scorer to produce higher-quality training
data.

## Data Volume

With ~3,000 games/hour (8 games in parallel, ~8 seconds per game), a reasonable initial dataset:
- 50+ pools, 10-20 variants per pool
- Round-robin within each pool's variants + cross-pool matchups against heuristic decks
- A few thousand games — roughly 1-2 hours of compute
- Scale up as needed; more data always helps since noise washes out in aggregate

# Phase 1 — Deck Scorer

## Architecture

The scorer uses a Set Transformer — a transformer variant designed for unordered sets rather than sequences.

Key differences from a standard transformer:
- **No positional encodings.** A deck is a set: {A, B, C} is identical to {C, A, B}. Dropping positional encodings
  makes the model inherently permutation-invariant.
- **Attention-based pooling.** Instead of mean-pooling the output tokens, a small set of learned query vectors
  ("inducing points" or "seed vectors") attend over all card representations. Different seed vectors can specialize
  in different aspects of deck quality — one might focus on removal density, another on mana curve, another on synergy
  clusters. Their outputs are concatenated into a fixed-size deck vector.

### Architecture Details

The model consists of:

1. **Optional input projection**: Linear layer mapping 544 → 512 or 256, giving the model a chance to blend the
   semantic (512-dim) and deterministic (32-dim) features into a unified space before attention.
2. **Self-attention layers**: Cards attend to each other, learning interactions like "this removal spell is more
   valuable because the rest of the deck is slow" or "these three cards form a synergy package."
3. **Pooling layer**: 4-8 learned seed vectors attend over the card representations, producing a fixed-size deck vector
   (seed vectors x model dimension, e.g. 4 x 512 = 2048).
4. **Scoring head**: A small MLP (2 hidden layers of 256-512 dims with ReLU) mapping the deck vector to a single
   scalar score.

### Starting Hyperparameters

- Layers: 2-4 (start with 2; pairwise card interactions likely sufficient, try 4 if validation loss stalls)
- Attention heads: 4-8 (each head can specialize in color alignment, curve distribution, synergy, removal density, etc.)
- d_model: 512 (or 256 if using input projection)
- d_ff: 1024-2048 (standard 2-4x model dimension heuristic)
- Pooling seed vectors: 4-8
- Total parameters: roughly 5-15M

Architecture details matter less than training data quality. A 2-layer, 4-head model with good data will outperform
a 6-layer, 16-head model with bad data. Get the data generation pipeline working well first, then tune architecture.

## Training Objective — Bradley-Terry Pairwise Loss

The scorer is trained using pairwise game outcomes, not absolute win rates.

**Why not win rates?** To get a reliable win rate for a single deck, you'd need 50-100 games against diverse opponents.
At ~3000 games/hour, scoring a single deck takes minutes. You'd need thousands of scored decks to train the evaluator.
The data generation becomes prohibitively slow.

**Pairwise outcomes fix this.** Every single game produces one training example. No aggregation needed, no statistical
estimation, no wasted data. The training objective is:

- The scorer assigns a score to each deck: `score_A = f(deck_A)`, `score_B = f(deck_B)`
- The probability that deck A beats deck B is modeled as `sigmoid(score_A - score_B)`
- If A won: loss = `-log(sigmoid(score_A - score_B))`
- If B won: loss = `-log(sigmoid(score_B - score_A))`
- This is standard binary cross-entropy where the logit is the score difference

The scores that emerge are not calibrated win rates, but they don't need to be — the search only needs them to rank
decks correctly.

**Noise from single games.** Individual MTG games are noisy — a good deck can lose to a bad deck due to random draws.
This is manageable:
- The model learns from thousands of games; noise washes out in aggregate (same principle as Elo ratings)
- Some noise is desirable — if the better deck always won, the function would be trivial
- Counter-deck non-transitivity (A beats B, B beats C, C beats A) is less severe in sealed than constructed because
  neither player chose their pool to counter anything; the model learns average-case strength, which is what matters
  when building blind
- If noise is still problematic, play each matchup 3 times and use majority outcome as the label

**Critical architectural detail:** Both decks in a pair are encoded by the **same shared encoder** (same Set Transformer,
same weights). The model learns one scoring function, not a comparison function.

## Training Loop

Training data comes from Phase 0.

For each training batch:

1. Sample a batch of game outcomes `(deck_A, deck_B, winner)`
2. Encode `deck_A` through the Set Transformer → `score_A`
3. Encode `deck_B` through the same Set Transformer (shared weights) → `score_B`
4. Compute Bradley-Terry loss (binary cross-entropy on score difference)
5. Backprop — gradients flow through the MLP, through the Set Transformer, and (when unfrozen) into the card embeddings

Batch composition: each batch should contain games from many different pools to prevent overfitting to pool-specific
patterns.

Validation: hold out some pools entirely. Generate decks and play games from those pools but never train on them.
Monitor whether the scorer correctly ranks held-out deck variants.

## Embedding Schedule

**Phase A — Frozen embeddings.** Train the Set Transformer and scoring MLP with card embeddings frozen (learning rate 0).
The model learns to work with the existing embedding space, extracting what pre-trained features predict deck quality.
This converges relatively fast because the embeddings already carry meaningful information.

**Phase B — Unfrozen with low learning rate.** Once validation loss plateaus, unfreeze the embeddings at 10-100x lower
learning rate than the rest of the network. At this point, gradients flowing into embeddings carry real signal about
what the scorer needs. The embeddings will shift to encode deckbuilding-relevant features — in particular,
**complementarity** rather than just similarity. You don't want 23 copies of the same effect; you want a curve,
removal, threats, and synergy. Fine-tuning lets the embeddings reorganize so that cards which work well together
are nearby, even if their text is very different.

Monitor embedding drift: track the average L2 distance of embeddings from their initial values. If they drift too far
too fast, lower the learning rate. If they barely move, increase it.

# Phase 2 — Search-Based Deck Builder (Inference)

Once the scorer is trained, deck building at inference is:

1. Receive a new pool of 84-90 cards
2. Heuristic builder produces a starting deck of ~23 non-land cards
3. Deterministically assign basic lands (proportional to pip demand from selected spells, floor of 2 sources per
   included color)
4. Score the deck with the trained scorer
5. For every possible swap (remove one non-land card from deck, add one from pool): score the resulting deck
6. Take the best-improving swap, apply it
7. Recompute lands after the swap
8. Repeat steps 5-7 until no swap improves the score

With ~23 cards in the deck and ~65 in the remaining pool, each iteration evaluates ~1,500 candidate swaps. A neural
forward pass takes microseconds, so many iterations complete well under a second.

## Non-Basic Land Handling

Booster pools contain non-basic lands (dual lands, utility lands, fetch lands, etc.) that are more valuable than basic
lands. These are treated as selectable cards alongside spells — the search can swap them in or out based on whether they
improve the scorer's output. A strong dual land that provides needed color fixing will score higher than a marginal spell
and be included automatically. Basic lands fill whatever land slots remain after non-basic land selection.

## Escaping Local Optima

Greedy hill-climbing may get stuck in local optima — for example, it cannot find a better blue-black build if the
starting deck is red-green, because every individual swap away from red-green makes the deck worse before it gets better.
If this is a problem, use simulated annealing or beam search instead of pure greedy search. The scorer supports any
search strategy since it evaluates complete decks.

# Phase 3 — One-Shot Policy Model (Optional)

Once the scorer is reliable, distill the search-based builder into a model that selects a deck from a pool in a single
forward pass.

## How It Works

1. Generate thousands of pools
2. For each pool, run the search-based builder (heuristic + scorer + hill-climbing) to produce the best deck found
3. Train a one-shot model to reproduce those selections via supervised learning

The one-shot model takes all ~85 card embeddings as input, runs them through a transformer, and outputs a probability
per card (include or not). Train with binary cross-entropy against the search-produced labels.

## Why Distill

- The search process evaluates hundreds of candidate decks; the one-shot model amortizes this into a single pass
- The model can generalize patterns the search finds repeatedly ("in pools with strong red and blue, build red-blue")
  as implicit rules rather than rediscovering them every time
- It can learn from non-greedy search labels (simulated annealing, beam search) that escape local optima

## Optional RL Fine-Tuning

After distillation, fine-tune the one-shot model using RL with the **scorer as the reward signal**. The model proposes
a deck, the scorer evaluates it, and gradients improve the policy. This can exceed the search-produced labels if the
model finds better solutions.

This is the RL approach that previously failed — but now it works because:
- The scorer provides a **dense per-deck reward** instead of sparse noisy game outcomes
- The model is initialized from distillation, not from scratch, so it never hits the "2 lands of each color" collapse
- Lands are deterministic, removing the hardest credit assignment problem

# Training Dataset Preparation

The training dataset is pre-generated before training begins. The preparation consists of two independent steps:

## Step 1 — Card Embedding Generation

The Python application scans cards-path and generates a 512-dimensional embedding vector for each card found, following
the process described in spec 006-card-script-parsing. Each embedding is stored as a .npz file named after the card
(e.g. Lightning-Bolt.npz) in the same cards-path folder. This step is skipped for cards that already have a
corresponding .npz file, making it safe to run incrementally when new cards are added or when the encoder is retrained.

## Step 2 — Pool Generation

The Python script invokes a forge-connector Java class that uses Forge's internal classes to generate a configurable
number of sealed pools, each consisting of 6 boosters from the same configurable set. The pools are written to a flat
text file in pools-path named pools.txt, one pool per line, with card names separated by semicolons. Duplicate card
names are allowed since a pool can contain multiple copies of the same card. Basic lands are not included in the
generated pools.

At training time, each pool is assembled by reading the .npz embedding file for each card in the pool.

The dataset preparation can be launched via the following commands:
```bash
python -m sealed encode-cards \
    --encoder-path [path] \
    --vocab-path [path] \
    --cards-path [path]

python -m sealed generate-pools \
    --set [set-code] \
    --size [n-pools] \
    --pools-path [path]
```

Defaults for encode-cards:
- **--encoder-path**: models/price-predictor/transformer/latest.pt
- **--vocab-path**: models/price-predictor/transformer/vocab.txt
- **--cards-path**: output/cardsfolder/

Defaults for generate-pools:
- **--set**: RVR
- **--size**: 10000
- **--pools-path**: output/sealed/pools/{set-code}/

# Evaluation Against External Baseline

Every N training iterations, freeze weights and run:

```
20 fresh pools
-> your scorer + search builds deck A
-> Forge's heuristic deck builder builds deck B from same pool
-> best-of-11 via Forge AI
-> log absolute win rate
~220 games, ~4 minutes
```

This tracks absolute quality independently of training data generation, catching the trap where the scorer learns to
rank training variants correctly but doesn't generalize to truly strong decks.

# Training Completion Criteria

Training is considered done when all of the following are stable across several consecutive evaluation checkpoints:

- Forge builder win rate -> plateaued (absolute quality peaked)
- Scorer validation loss -> converged
- Human spot check -> built decks look strategically coherent

# Expansion Path

Once the model is performing well on a single set:

Expand to multiple sets by including pools from different sets in training data. The architecture requires no changes —
the Set Transformer handles variable-size sets natively. Retrain or fine-tune on the expanded card pool.

# Longer Term

The card encoder and the understanding of card quality learned here feed directly into the next project — training a
model to actually play the game, where card evaluation is a prerequisite for good play decisions.

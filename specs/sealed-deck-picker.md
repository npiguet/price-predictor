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
- => 546 features per card

Card slots represent stage 1 of the model. The all have the same number of features so that they can be cleanly fed to
the transformer at stage 2.

## Card Pool Composition

- 84-90 cards (opened from 6 boosters, varies by set, 14-15 cards per booster. Contain lands, including basic lands)

# Model Architecture

A two-stage architecture:

- Stage 1 — Card encoder: Pretrained (from price predictor), reused for each card
    - [Structured Oracle text] → [Pretrained transformer] → [512-dim card embedding]
- Stage 2 — Pool-level transformer (trained from scratch)
    - [90 × 546 features] → [Transformer] → [90 scores] → deck selection

The pool transformer processes all 90 pool card embeddings in a single forward pass using self-attention, and outputs a
scalar score for each card.

The exact size of the transformer model will be the subject of experiments, but as a starting point:
- layers: 8
- heads: 8
- d_model: 546
- d_ff: 2048

# Deck Selection

Instead of picking cards one at a time over 40 sequential steps, the model scores all 90 pool cards in a single forward
pass and the deck is assembled from those scores:

1. The transformer outputs a score for each of the 90 pool cards
2. Sort cards by score descending
3. Walk down the sorted list, accepting cards until 23 non-land cards (spells) have been selected. Any non-basic lands
   that score higher than the lowest-accepted spell are also included.
4. Let x = the number of non-basic lands selected. Fill the remaining 17 - x land slots with basic lands, allocated
   across colors using the ideal mana distribution (proportional to pip demand from the selected spells, with a floor
   of 2 sources per color present).

The total deck is always 40 cards: 23 spells + x non-basic lands + (17 - x) basic lands.

## Why One-Shot Selection

The previous sequential approach (40 picks, one per step, PPO with per-step rewards) failed to learn color coordination.
The root cause is a credit assignment problem: evaluating a holistic set property (mana balance) requires reasoning about
all cards simultaneously, but a sequential model must decompose that into 40 independent per-step decisions. Spell/land
ratio (a 1D counting problem) was learned easily, but color balance (a 6D coupled optimization with a moving target) was
not — every spell pick shifts the ideal distribution across all 6 color buckets simultaneously, and the model could never
get meaningful per-step gradient signal for this global property.

The one-shot approach sidesteps the credit assignment problem entirely:
- The transformer sees all 90 cards simultaneously via self-attention, allowing holistic reasoning about card interactions
  and color synergies
- One forward pass produces one deck, one deck produces one score — direct gradient signal with no decomposition needed
- Mana balance is solved by construction through the deterministic basic land allocation

## Non-Basic Land Handling

Booster pools contain non-basic lands (dual lands, utility lands, fetch lands, etc.) that are more valuable than basic
lands. These must be evaluated alongside spells, not relegated to a separate land-allocation phase. The selection
procedure handles this naturally: non-basic lands compete with spells for inclusion based on their transformer scores.
A strong dual land that provides needed color fixing will score higher than a marginal spell and be included
automatically.

# Training Algorithm

With one-shot selection, the training signal is dramatically simpler than the sequential approach: one forward pass
produces one deck, one deck receives one score.

Two candidate training approaches:

## Option A: REINFORCE / Policy Gradient

Treat the card scores as a stochastic policy (e.g., via Plackett-Luce sampling for ordered selection without
replacement). Sample a deck, score it, compute the policy gradient. Standard variance reduction (baseline subtraction)
keeps gradients stable.

- Simple to implement
- Direct optimization of the scoring function
- Well-understood convergence properties

## Option B: Expert Iteration

1. Generate many random (or policy-sampled) decks from each pool
2. Score all decks with the evaluation function
3. Train the model to imitate the highest-scoring decks
4. Repeat — each iteration produces better decks to train on

- Avoids high-variance gradient estimates
- Naturally explores the deck space
- Can bootstrap from random decks with no pretrained policy

Either approach (or a hybrid) is viable. The choice will be informed by early experiments.

# Training Curriculum

## Stage 0 - Training Dataset Preparation

The training dataset is pre-generated before training begins. The preparation
consists of two independent steps that can be run separately:

### Step 1 - Card Embedding Generation
The Python application scans cards-path and generates a 512-dimensional embedding
vector for each card found, following the process described in spec
006-card-script-parsing. Each embedding is stored as a .npz file named after
the card (e.g. Lightning-Bolt.npz) in the same cards-path folder. This step is
skipped for cards that already have a corresponding .npz file, making it safe
to run incrementally when new cards are added or when the encoder is retrained.

### Step 2 - Pool Generation
The Python script invokes a forge-connector Java class that uses Forge's
internal classes to generate a configurable number of sealed pools, each
consisting of 6 boosters from the same configurable set. The pools are
written to a flat text file in pools-path named pools.txt, one pool per
line, with card names separated by semicolons. Duplicate card names are
allowed since a pool can contain multiple copies of the same card. Basic
lands are not included in the generated pools.

At training time, each pool is assembled by reading the .npz embedding file
for each card in the pool.

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

## Stage 1 - Heuristic Training

Train the model to produce decks that score well on heuristic evaluation functions — land count, mana curve, color
consistency — without needing Forge game simulations. This is cheap to run (pure Python, no JVM) and establishes a
baseline policy that builds structurally sound decks.

The heuristic scoring function evaluates the assembled 40-card deck (selected spells + non-basic lands + allocated
basics) and produces a scalar reward. The exact heuristic components will be defined in the feature spec for this stage.

Training advances to Stage 2 once the model consistently produces decks that pass all heuristic checks.

## Stage 2 — Forge Self-Play

The model builds decks from fresh pools. Each match is best-of-11 via Forge, with Forge AI playing both sides:
```
winner reward = games_won / 11    # 0.5 to 1.0
loser reward  = games_won / 11 - 1 # -0.5 to 0.0
```

The model builds both decks from the same pool, so every match produces two independent reward signals.

## Stage 3 — Encoder Fine-Tuning (Staged Unfreezing)

Triggered when Stage 2 win rate plateaus against the external benchmark. Unfreeze in order:

1. Projection layer (already trainable)
2. Top encoder layers at lr ~1e-6
3. Full encoder at lr ~1e-6

Pool transformer keeps its higher learning rate (~1e-4) throughout.

# Evaluation Against External Baseline

Every N training iterations, freeze weights and run:

```
20 fresh pools
→ your model builds deck A
→ Forge's deck builder builds deck B from same pool
→ best-of-11 via Forge AI
→ log absolute win rate
~220 games, ~4 minutes
```

This tracks absolute quality independently of self-play, catching the relativism trap where both self-play decks improve
relative to each other but not in absolute terms.

# Training Completion Criteria

Training is considered done when all of the following are stable across several consecutive evaluation checkpoints:

- Self-play win rate → ~50% (both decks consistently strong)
- Forge builder win rate → plateaued (absolute quality peaked)
- Heuristic scores → consistently high (structurally sound decks)
- Human spot check → built decks look strategically coherent

# Expansion Path

Once the model is performing well on a single set:

Expand to multiple sets by padding pools to the largest set's pool size.
The architecture requires no changes — just longer padding for smaller sets.
Retrain or fine-tune on the expanded card pool.

# Longer Term

The card encoder and the understanding of card quality learned here feed directly into the next project — training a
model to actually play the game, where card evaluation is a prerequisite for good play decisions.

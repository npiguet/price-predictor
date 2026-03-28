# Goal

Train a sealed deck builder that takes a pool of cards opened from 6 boosters and selects an optimal 40-card deck, using
game outcomes as the training signal. This is a stepping stone toward a full MTG-playing AI.

In order to keep the training cost manageable, this will be achieved by going through multiple training stages of 
increasing complexity.

1. Picking legal cards
2. Picking playable cards
3. Picking good cards
4. Fine-tuning

# Card Representation

Each card is encoded by your pretrained price predictor transformer, producing a 512-dimensional vector (256 mean pool +
256 max pool over ~300 tokens of structured Oracle text). This encoder is frozen initially and unfrozen in later
training stages.

At the pool level, each of the 96 entries (90 pool cards + 6 basic land slots) is represented as:

- card_embedding [512]
- picked_flag [1] (basic land slots: 0 when basic_land_count == 0, 1 otherwise)
- available_flag [1] (basic land slots: always 1)
- is_land [1]
- basic_land_count [1] (booster card slot: always 0)
- => 516 features per card

Card slots represent stage 1 of the model. The all have the same number of features so that they can be cleanly fed to
the transformer at stage 2.

## Card Pool Composition

- 84-90 cards (opened from 6 boosters, varies by set, 14-15 cards per booster. Contain lands, including basic lands)
- 6 basic land slots (one per color + colorless, always available, never masked out, counts how many times it was
  selected)
- 96 total entries

# Model Architecture

A two-stage architecture:

- Stage 1 — Card encoder: Pretrained (from price predictor), reused for each card
    - [Structured Oracle text] → [Pretrained transformer] → [512-dim card embedding]
- Stage 2 — Pool-level transformer (trained from scratch)
    - [95 × 516 features] → [Small transformer] → [95 logits] → [masked softmax] → [pick]

The pool transformer attends freely over all 95 cards at every step, including already-picked ones, so it can use the
current deck state as context for each new pick. A small projection layer (512 → 512) sits between the encoder and pool
transformer to allow adaptation without destabilizing the pretrained weights.

The exact size of the transformer model will be the subject of multiple experiments, but for the first version we'll 
set it at:
- layers: 8
- heads: 8
- d_model: 516
- d_ff: 2048

# Pick Phase

40 sequential steps, one card per step:

- At each step the model sees all 96 cards with their current flags and counts
- Already-picked nonland cards have available_flag = 0 and are masked out of selection
- Basic land slots are never masked — picking one increments its basic_land_count
- The selection mask is applied to logits after the transformer, not to attention, so picked cards remain visible as
  context
- After 40 picks the deck is complete — no separate land phase needed

# Training Algorithm

PPO (Proximal Policy Optimization) with experience replay:

- Handles the sequential 40-step decision process naturally
- Tolerates mild off-policy data from the replay buffer
- KL divergence monitored per episode to detect stale buffer entries

## Replay Buffer

Per episode stored:

| Feature            | Format          | Size        |
|--------------------|-----------------|-------------|
| pool embeddings:   | 95 × 512 floats | (~195 KB)   |
| actions:           | 40 integers     | (160 bytes) |
| log probabilities: | 40 floats       | (160 bytes) | 
| reward:            | 1 float         | (4 bytes)   |
|                    |                 |             |
| total per episode: |                 | ~195 KB     |
| buffer size:       | ~1000 episodes  | (~195 MB)   |

FIFO eviction with KL divergence monitoring to detect when episodes are too stale to be useful.


## Training Curriculum

### Stage 0 - Training Dataset Preparation

The training dataset is pre-generated before training begins. Sealed pools of
6 booster packs are generated using Forge's internal classes via a forge-connector
Java bridge. The dataset preparation process is:

1. The Python script invokes a forge-connector Java class that uses Forge's
   internal classes to generate a configurable number of sealed pools, each
   consisting of 6 boosters from the same configurable set. The pools are
   written to a flat text file in the pools-path folder named pools.txt,
   one pool per line, with card names separated by semicolons. Duplicate card
   names are allowed since a pool can contain multiple copies of the same card.

2. The Python application reads pools.txt and converts each pool into a matrix
   of card embeddings. Each card is looked up by name in cards-path and
   converted to a 512-dimensional embedding vector using the pretrained card
   encoder, following the process described in spec 006-card-script-parsing.
   The 5 basic land slots are appended to each pool at this stage, bringing
   the pool size to 95 entries. This means the card encoder never needs to
   run during training.

3. The converted dataset is stored as a pools.npz file (numpy compressed format)
   containing a single [N, 95, 516] float32 array, where N is the number of
   generated pools. The format loads efficiently into Python and converts to
   PyTorch tensors with zero overhead.

The dataset preparation can be launched via the following command:
```bash
python -m sealed prepare-dataset \
    --set [set-code] \
    --size [n-pools] \
    --encoder-path [path] \
    --vocab-path [path] \
    --cards-path [path] \
    --pools-path [path]
```

Defaults:
- **--set**: RVR
- **--size**: 10000
- **--encoder-path**: models/price-predictor/transformer/latest.pt
- **--vocab-path**: models/price-predictor/transformer/vocab.txt
- **--cards-path**: output/cardsfolder/
- **--pools-path**: output/sealed/pools/{set-code}/

### Stage 1 - Picking legal card (aka: Legal deck gate)
At each pick step, we verify that the model has selected a pool slot that has not already been chosen in the current 
episode. Basic land slots are exempt from this check since they can be picked any number of times.

During training, the card order within each pool is shuffled before each
episode to prevent the model from learning positional biases.

When an illegal pick is made, the episode terminates immediately.

The reward is calculated as:
```
reward = (current_run / best_run) × 2 - 1
```

Where:
- current_run: the number of legal picks made in this episode before
  termination (minimum 1, since the first pick is always legal)
- best_run: a high-water mark tracking the longest legal pick sequence
  ever achieved, initialized to 1 and never decreasing

When the model consistently reaches current_run = 40 over 100 consecutive episodes, training advances to stage 2.

### Stage 2 — Picking playable cards (aka: Heuristic gate)
Two instant checks computed from the picked deck:

1. Land count score: peaks at 16-18 lands, degrades outside that range
2. Mana pip matching: weighted by 1/cmc to penalize early color requirements more heavily

Decks failing either check receive a negative reward and are never submitted to Forge. This prevents wasting game budget
on uncastable decks. Gate pass rate is logged as a diagnostic — training moves to stage 2 once it approaches 100%.

### Stage 3 — Picking good cards (aka: Forge self-play)
Pool transformer and projection layer train freely. Model builds both decks, plays best-of-11, both decks receive
rewards. Replay buffer active.

Each match is best-of-11 via Forge, with Forge AI playing both sides:
```
winner reward = games_won / 11 # 0.5 to 1.0
loser reward = games_won / 11 - 1 # -0.5 to 0.0
```

Since the model builds both decks, every match produces two independent reward signals.

### Stage 4 — Encoder fine-tuning (staged unfreezing)
Triggered when stage 2 win rate plateaus against the external benchmark. Unfreeze in order:

1. Projection layer (already trainable)
2. Top encoder layers at lr ~1e-6
3. Full encoder at lr ~1e-6

Pool transformer keeps its higher learning rate (~1e-4) throughout.

## Evaluation Against External Baseline
Every 1000 training episodes, freeze weights and run:

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

## Training Completion Criteria
Training is considered done when all of the following are stable across several consecutive evaluation checkpoints:

- Self-play win rate → ~50% (both decks consistently strong)
- Forge builder win rate → plateaued (absolute quality peaked)
- Gate pass rate → ~100% (always builds legal decks)
- Human spot check → built decks look strategically coherent

# Expansion Path
Once the model is performing well on a single set:

Expand to multiple sets by padding pools to the largest set's pool size
The architecture requires no changes — just longer padding for smaller sets
Retrain or fine-tune on the expanded card pool

# Longer Term
The card encoder and the understanding of card quality learned here feed directly into the next project — training a
model to actually play the game, where card evaluation is a prerequisite for good play decisions.
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
256 max pool over ~300 tokens of structured card text). The `name:` line is stripped from the card text before encoding
so that the embedding captures what the card does, not its name. This encoder is frozen initially and unfrozen in later
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
current deck state as context for each new pick.

The exact size of the transformer model will be the subject of multiple experiments, but for the first version we'll 
set it at:
- layers: 8
- heads: 8
- d_model: 516
- d_ff: 2048

> **Implementation note**: `d_model=516` is not divisible by 8, so `n_heads=8` is invalid for
> `nn.TransformerEncoderLayer` (which requires `d_model % n_heads == 0`). The implementation
> uses `n_heads=4` (516 / 4 = 129). This should be corrected in a future spec revision — either
> change `n_heads` to 4, or adjust `d_model` to a value divisible by 8 (e.g. 512 with a small
> input projection, or 520).

# Pick Phase

40 sequential steps, one card per step:

- At each step the model sees all 96 cards with their current flags and counts
- Already-picked nonland cards have available_flag = 0 and are masked out of selection
- Basic land slots are never masked — picking one increments its basic_land_count
- The selection mask is applied to logits after the transformer, not to attention, so picked cards remain visible as
  context
- After 40 picks the deck is complete — no separate land phase needed

# Training Algorithm

Standard on-policy PPO (Proximal Policy Optimization):

- Handles the sequential 40-step decision process naturally
- Each batch of episodes is collected with the current policy, used for one gradient update, then discarded

## Episode Storage

Episodes are stored compactly using the pool's card list and per-step shuffle
seeds, allowing the exact state at each pick step to be reconstructed at
training time. No replay buffer is maintained.

Per episode stored:

| Feature           | Format                         | Size         |
|-------------------|--------------------------------|--------------|
| pool              | semicolon-separated card names | (~1.9 KB)    |
| shuffle_seeds     | 40 integers                    | (160 bytes)  |
| actions           | 40 integers                    | (160 bytes)  |
| log_probabilities | 40 floats                      | (160 bytes)  |
| reward            | 1 float                        | (4 bytes)    |
|                   |                                |              |
| total per episode |                                | ~2.4 KB      |

## Training Curriculum

### Stage 0 - Training Dataset Preparation

The training dataset is pre-generated before training begins. The preparation
consists of two independent steps that can be run separately:

#### Step 1 - Card Embedding Generation
The Python application scans cards-path and generates a 512-dimensional embedding
vector for each card found, following the process described in spec
006-card-script-parsing. Each embedding is stored as a .npz file named after
the card (e.g. Lightning-Bolt.npz) in the same cards-path folder. This step is
skipped for cards that already have a corresponding .npz file, making it safe
to run incrementally when new cards are added or when the encoder is retrained.

#### Step 2 - Pool Generation
The Python script invokes a forge-connector Java class that uses Forge's
internal classes to generate a configurable number of sealed pools, each
consisting of 6 boosters from the same configurable set. The pools are
written to a flat text file in pools-path named pools.txt, one pool per
line, with card names separated by semicolons. Duplicate card names are
allowed since a pool can contain multiple copies of the same card. Basic
lands are not included in the generated pools.

At training time, each pool is assembled by reading the .npz embedding file
for each card in the pool, then appending the 6 basic land embeddings looked
up by name from cards-path.

The dataset preparation can be launched via the following commands:
```bash
python -m sealed encode-cards \
    --encoder-path [path] \
    --vocab-path [path] \
    --cards-path [path] \
    --clean

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

### Stage 1 - Picking legal cards (aka: Legal Pick Gate)

This is the first stage of training proper. The pool transformer and PPO training
loop are initialized at the start of this stage.

Training proceeds as a sequence of episodes. At the start of each episode a pool
is sampled from the pre-generated dataset. The 6 basic land embeddings are appended 
at the end of the pool, bringing it to 96 entries. Empty slots (if any) are filled 
with zero vectors and placed before the basic lands. Before each pick step, the
non-basic-land portion of the pool is reshuffled to prevent the model from 
developing positional biases.

The model is asked to sequentially pick 40 cards from the pool. The episode runs
in two phases, controlled by how many non-land picks have been made so far.

**Phase 1** (fewer than 23 non-land picks made):
- Picking any land card (basic land slot, or a pool card whose `type:` line
  contains "land") terminates the episode immediately.
- Picking a duplicate non-land also terminates immediately.
- Non-land picks are recorded normally.

**Phase 2** (23 or more non-land picks made):
- Land picks are now allowed and recorded as actions. Basic land slots can be
  picked any number of times; non-basic pool lands can only be picked once.
- Non-land picks are still allowed (but penalised in the reward, see below).
- Duplicate picks still terminate the episode.
- The episode completes successfully when 40 total picks have been recorded.

The rationale: a 40-card sealed deck is ideally 23 spells followed by 17 lands.
Phase 1 forces the model to exhaust its spell picks before touching lands. Phase 2
rewards it for filling the remaining slots with lands rather than extra spells.

#### Reward function

    effective_run = n_total - max(n_spell - 23, 0)
    reward = (effective_run / best_run) × 2 - 1

Where:
- `n_total`: total picks recorded in the episode
- `n_spell`: non-land picks among those (lands never count toward n_spell)
- `effective_run`: `n_total` reduced by one for each non-land pick past 23;
  peaks at 40 for a perfect 23-spell + 17-land run
- `best_run`: high-water mark of `n_total` (raw pick count) across all episodes,
  initialised to 1 and never decreasing

Key properties:
- Any full 40-pick run yields `effective_run ≥ 23`, so `reward > 0` once
  `best_run ≤ 46` (which is always true).
- Maximum reward is only achieved with exactly 23 spells and 17 lands.
- Each non-land pick past 23 costs exactly as much as a missing land pick.
- Before phase 2 is reached (n_spell < 23), `effective_run = n_total = n_spell`
  and the formula reduces to the original `(n_spell / best_run) × 2 - 1`.

When `n_total = 40` in 100 consecutive episodes, training advances to stage 2.
The land/spell composition is not checked for advancement — any 40-pick run
counts, since the reward function already incentivises the right ratio.

#### Command Line

Training can be launched via the following command:
```bash
python -m sealed train \
    --stage 1 \
    --set [set-code] \
    --pools-path [path] \
    --cards-path [path] \
    --model-path [path]
```

Defaults:
- **--set**: RVR
- **--pools-path**: output/sealed/pools/{set-code}/
- **--cards-path**: output/cardsfolder/
- **--model-path**: models/sealed/stage{stage}/latest.pt

The --stage parameter controls which phase of training is executed, including
the reward function, termination conditions, and which model components are
trainable.

#### Model Checkpointing

The model is saved to model-path at the end of each training batch. A timestamped
checkpoint is also saved every 1000 episodes to the checkpoints/ subfolder of
model-path's parent folder, e.g. models/sealed/stage1/checkpoints/. This allows
training to be resumed from any checkpoint and makes it easy to compare
experiments across stages.

#### Sampling

A sample of the model's current picks can be generated at any time using:
```bash
python -m sealed sample \
    --set [set-code] \
    --pools-path [path] \
    --cards-path [path] \
    --model-path [path] \
    --n-samples [n]
```

Defaults:
- **--set**: RVR
- **--pools-path**: output/sealed/pools/{set-code}/
- **--cards-path**: output/cardsfolder/
- **--model-path**: models/sealed/stage1/latest.pt
- **--n-samples**: 10

This generates n deck selections from random pools and prints each pick sequence
as a human-readable list of card names, along with the number of legal picks made
before the first illegal pick (if any). A completed run of 40 legal picks is
reported as a success.

### Stage 2 — Picking playable cards (aka: Heuristic gate)
Two instant checks computed from the picked deck:

1. Land count score: peaks at 16-18 lands, degrades outside that range
2. Mana pip matching: weighted by 1/cmc to penalize early color requirements more heavily

Decks failing either check receive a negative reward and are never submitted to Forge. This prevents wasting game budget
on uncastable decks. Gate pass rate is logged as a diagnostic — training moves to stage 2 once it approaches 100%.

### Stage 3 — Picking good cards (aka: Forge self-play)
Pool transformer trains freely. Model builds both decks, plays best-of-11, both decks receive
rewards. Replay buffer active.

Each match is best-of-11 via Forge, with Forge AI playing both sides:
```
winner reward = games_won / 11 # 0.5 to 1.0
loser reward = games_won / 11 - 1 # -0.5 to 0.0
```

Since the model builds both decks, every match produces two independent reward signals.

### Stage 4 — Encoder fine-tuning (staged unfreezing)
Triggered when stage 2 win rate plateaus against the external benchmark. Unfreeze in order:

1. Top encoder layers at lr ~1e-6
2. Full encoder at lr ~1e-6

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
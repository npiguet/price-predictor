# Goal

Train a sealed deck builder that takes a pool of cards opened from 6 boosters and selects an optimal 40-card deck, using
game outcomes as the training signal. This is a stepping stone toward a full MTG-playing AI.

# Overall Approach

The system is built in five sequential phases:

0. **Training Dataset Generation** — Generate pools, build deck variants, play games, and store the results.
1. **Deck Scorer** — A Set Transformer trained via Bradley-Terry pairwise loss on game outcomes to score how good a
   deck is. This is the core model and the most training-intensive component.
2. **Search-Based Deck Builder** — At inference, a heuristic builder seeds a starting deck, and greedy hill-climbing
   guided by the scorer iteratively swaps cards to improve the deck. No ML needed for this step.
3. **Self-Play Refinement** — Iteratively improve the scorer by adding scorer-built decks to the training data,
   retraining, and evaluating against an external baseline until the results are satisfactory.
4. **One-Shot Policy Model (optional)** — Once the scorer is reliable, distill the search-based builder into a single
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

## Deterministic Feature Encoding (indices 512–543)

All 32 features are extracted from the converted card text file (see spec 006-card-script-parsing).

### is_land [1] — index 512

`1.0` if `"land"` is in the card's types, `0.0` otherwise. Derived from the parsed `types:` line.

### Mana cost [9] — indices 513–521

Extracted from the `mana cost:` line. Cards with no mana cost (lands, some special cards) get all zeros.

| Index | Feature             | Description                    |
|-------|---------------------|--------------------------------|
| 513   | white pips (W)      | count of {W} in mana cost      |
| 514   | blue pips (U)       | count of {U} in mana cost      |
| 515   | black pips (B)      | count of {B} in mana cost      |
| 516   | red pips (R)        | count of {R} in mana cost      |
| 517   | green pips (G)      | count of {G} in mana cost      |
| 518   | colorless pips (C)  | count of {C} in mana cost      |
| 519   | generic mana        | generic mana amount            |
| 520   | X pip count         | count of {X} in mana cost      |
| 521   | mana value          | total mana value               |

### Card color [6] — indices 522–527

Multi-hot encoding derived from mana cost pips. A card is a given color if it has at least one pip of that color.
Cards with no colored pips (lands, colorless artifacts, Eldrazi) are marked colorless. Cards with the `devoid`
keyword are colorless regardless of their mana cost (all WUBRG flags `0.0`, colorless flag `1.0`).

| Index | Feature    | Rule                                      |
|-------|------------|-------------------------------------------|
| 522   | is white   | `1.0` if W pips > 0 and not devoid        |
| 523   | is blue    | `1.0` if U pips > 0 and not devoid        |
| 524   | is black   | `1.0` if B pips > 0 and not devoid        |
| 525   | is red     | `1.0` if R pips > 0 and not devoid        |
| 526   | is green   | `1.0` if G pips > 0 and not devoid        |
| 527   | is colorless | `1.0` if no colored pips, or devoid, or no mana cost |

Note: this is color, not color identity. A card like `{2}{C}{C}` (colorless pips only, no W/U/B/R/G) is colorless.
Devoid is detected from `keyword` lines in the converted card text (e.g. `keyword[1]: devoid`).

### Mana produced [7] — indices 528–534

Whether and how much mana this card can produce. Parsed from `activated` ability lines in the converted card text
that contain `add` patterns (e.g. `activated[1]: {T}: add {R} or {W}.`). Only activated abilities are scanned —
triggered, static, and other ability types are ignored.

The encoding is 6 multi-hot color flags plus 1 mana count:

| Index | Feature         | Description                                     |
|-------|-----------------|-------------------------------------------------|
| 528   | produces W      | `1.0` if the card can produce white mana        |
| 529   | produces U      | `1.0` if the card can produce blue mana         |
| 530   | produces B      | `1.0` if the card can produce black mana        |
| 531   | produces R      | `1.0` if the card can produce red mana          |
| 532   | produces G      | `1.0` if the card can produce green mana        |
| 533   | produces C      | `1.0` if the card can produce colorless mana    |
| 534   | mana count      | total mana produced per activation (see below)  |

**Mana count rules:**

- Explicit symbols: `add {W}` → count 1, `add {U}{B}` → count 2, `add {R} or {W}` → count 1
- `add one mana of any color` / `add one mana of any type` → count 1, set all 5 color flags
- `add two mana of one color` / `add three mana in any combination` → count 2 or 3 respectively, set all 5 color flags
- Complex/game-dependent amounts (e.g. "add X mana", "add mana equal to...") → count 1 as default

Cards that produce no mana get all zeros.

**Simplification:** this does not attempt to model conditional production (e.g. "add one mana of any color a land
you control could produce"), mana from sacrifice costs, or mana from abilities that require specific conditions.
The parser handles the common patterns; edge cases get approximate values. Accuracy here is not critical — the
transformer embedding already encodes the full card text, so these features are supplemental hints, not the sole
signal.

### Power and toughness [2] — indices 535–536

From the `power toughness:` line. Non-creatures get `0.0` for both.

| Index | Feature   | Rule                                                 |
|-------|-----------|------------------------------------------------------|
| 535   | power     | Numeric value. `*` and `X` encode as `0.0`.          |
| 536   | toughness | Numeric value. `*` and `X` encode as `0.0`.          |

### Starting loyalty [1] — index 537

From the `loyalty:` line. `0.0` for non-planeswalkers or unparseable values.

### Zero-padding [6] — indices 538–543

All `0.0`. Reserved to make the feature vector length 544 (divisible by 8), which helps with memory alignment and
potential future feature additions.

Card slots represent the input to the Set Transformer scorer. They all have the same number of features so that they
can be cleanly fed to the model.

## Feature Normalization

The 32 deterministic features (indices 512–543) have varying scales — binary flags are 0/1, while mana value and
power/toughness can reach 15+. Without normalization, the larger-magnitude features dominate the Q/K dot products in
attention, forcing the model to spend capacity learning to downweight them instead of learning card interactions.

The 512 embedding dimensions (indices 0–511) come from the pretrained encoder and are expected to already be at a
reasonable scale. Only the 32 deterministic features are standardized (zero mean, unit variance).

**Pipeline:**
- The `.npz` files store **raw (unnormalized)** values. This keeps the encode-cards step incremental — new cards can
  be encoded without recomputing global statistics or re-encoding existing files.
- At training startup, the data loader reads all `.npz` files, computes per-feature mean and std across the corpus for
  indices 512–543, and normalizes in memory.
- The computed mean and std vectors (each of shape (32,)) are stored as non-trainable buffers on the model via
  `register_buffer`. This way they are part of `model.state_dict()` — they save/load with the model checkpoint, move
  to GPU with `.to(device)`, and can never get out of sync with the model that depends on them.
- At inference, the same buffers are used to normalize incoming feature vectors before scoring.

# Phase 0 — Training Dataset Generation

Producing training data is the most time-consuming part of the pipeline. The scorer learns entirely from pairwise game
outcomes, so this phase generates pools, builds deck variants, plays games, and stores the results.

The training dataset consists of the outcomes of a collection of 1v1 best-of-N matches between decks generated from
distinct card pools. N is configurable via the `--best-of` CLI flag (default 7, any positive odd integer).

## Card Embedding Generation

Before Phase 1 can run, each card must have a precomputed 544-dimensional feature vector. The Python
application scans cards-path and generates a feature vector for each card found. Each feature vector is stored as a
`.npz` file named after the card (e.g. `Lightning-Bolt.npz`) in the same cards-path folder. Cards that already have a
corresponding `.npz` file are skipped, making it safe to run incrementally when new cards are added or when the encoder
is retrained.

The `.npz` file contains a single array of 544 floats:
- **[0:512]** — the 512-dimensional embedding from the pretrained price predictor transformer, following the process
  described in spec 006-card-script-parsing (256 mean pool + 256 max pool, name line stripped)
- **[512:544]** — the 32 deterministic features parsed from the card's converted text file (see Card Representation
  above): is_land, mana cost fields, card color, mana produced, P/T, loyalty, and zero-padding

Storing both components together means training only needs a single `.npz` lookup per card name — no reparsing of card
text at training time.

```bash
python -m sealed encode-cards \
    --encoder-path [path] \
    --vocab-path [path] \
    --cards-path [path]
```

Defaults:
- **--encoder-path**: models/price-predictor/transformer/latest.pt
- **--vocab-path**: models/price-predictor/transformer/vocab.txt
- **--cards-path**: output/cardsfolder/

## Card Pool Composition

- 84-90 cards (opened from 6 boosters, varies by set, 14-15 cards per booster. Contain lands, including basic lands)

## Step 1 — Choose an expansion set

An MTG expansion set is selected at random from the list of sets for which the sealed format is available. This excludes
un-sets (joke sets) as well as aftermath-style sets. The selected set must provide "draft boosters" or "play boosters".

## Step 2 — Generate one pool of boosters for each player

For each player, generate a pool of cards using 6 boosters from the selected set. Each player gets different boosters,
so 12 distinct boosters are generated per match.

## Step 3 — Create a deck from each pool

A new 40-card deck is generated from each card pool. There are 4 distinct deck generation methods. One method is
selected at random for each deck, according to the following weights:

1. Use the standard Forge sealed deck generator (weight: 0.4)
2. Same as 1, but 3 type-matched swaps are made: each swap picks a random card from the deck's
   non-basics (spells or non-basic lands) and replaces it with a card of the same type from
   the remaining pool — spells swap with spells, non-basic lands swap with non-basic lands (weight: 0.3)
3. Same as 2 but 8 swaps (weight: 0.2)
4. 23 random spells (non-land cards) are picked from the pool; non-basic lands in the pool
   are excluded from picking (weight: 0.1)

Card picking follows the same constraints as physical card picks. A card pool may contain multiple copies of a card, but
each card instance can only be picked once.

For methods 2-4, an extra basic land rebalancing step is performed. All basic lands are removed from the
deck (non-basic lands are kept) and basic lands are added back using a pip-proportional algorithm:
 - basicLandsNeeded = 40 - spells - nonBasicLands
 - Slots distributed proportionally to the WUBRG mana pip counts of the **spells only**
   (non-basic lands have no mana cost and contribute no pips)
 - Note: colorless {C} pips are not handled. This is an accepted trade-off — very few sets/cards
   require {C}, not enough examples for the model to learn from anyway
 - Implementation note: `LimitedDeckBuilder.addLands()` is private and `SealedDeckBuilder` re-selects
   a subset of its input rather than treating all cards as already-chosen. The pip-proportional logic
   is therefore reimplemented directly in `DeckBuilder.rebalanceLands(spells, nonbasicLands)`.

## Step 4 — Play the game and record the result

The Forge AI then pits these 2 decks against each other in a best-of-N match. N is configurable via the
`--best-of` CLI flag on `match-outcomes`; the default is 7. Any positive odd integer is valid (Bo1, Bo3,
Bo7, Bo17, …). Longer matches produce less-noisy labels for Bradley-Terry training at the cost of more
games per match; the default of 7 trades game count for label reliability.

The outcome is recorded as a single line appended to `./output/sealed/match-outcomes.txt`.

The format of that line is:

    timestamp;run_id;set_code;method_A;method_B;deck_A_card_names;deck_B_card_names;games;play;duration_s

where:
 - **timestamp**: ISO 8601 UTC timestamp when the match was recorded (e.g. `2026-04-22T14:30:05Z`).
   Enables time-based filtering and correlation with external events (e.g. model training cutoffs).
 - **run_id**: a UUID generated once at supervisor start-up and attached to every match emitted
   by that supervisor invocation. Enables grouping or excising a specific run without
   timestamp-range guesswork (e.g. "drop everything from the buggy run on Tuesday").
 - **set_code**: the MTG set code from which both decks' pools were drawn (e.g. `RVR`, `MH3`).
   Both decks always come from the same set. Enables set-bucketed analysis (duplicate detection,
   per-set win-rate stats) without having to infer the set from card contents.
 - **method_A**: how deck A was built. One of the following tags:
   - `forge-best` — method 1 (Forge SealedDeckBuilder)
   - `forge-3sub` — method 2 (Forge + 3 swaps)
   - `forge-8sub` — method 3 (Forge + 8 swaps)
   - `random` — method 4 (23 random spells)
   - a user-supplied label — method 5 (scorer-built deck from the generated-decks file); only
     present in self-play matches (see Phase 3). The label is read out of the deck file's
     first column (set per-deck by `build-decks --label`, e.g. `gen-2`), so matches from
     different self-play generations can be distinguished — and a single generated-decks file
     can hold decks from multiple generations without collision.
 - **method_B**: same scheme as `method_A`, describing how deck B was built.
 - **deck_A_card_names**: a pipe-separated list of the names of the cards in deck A
 - **deck_B_card_names**: same, but for deck B
 - **games**: a 2- or 3-character string recording the winner of each game in sequence, using
   `A` for deck A and `B` for deck B. For example, `AA` means deck A swept 2-0, `ABA` means
   A won game 1, lost game 2, then won game 3. `wins_A` and `wins_B` can be recovered by
   counting characters. Length is between `ceil(N/2)` and `N` for a best-of-N match
   (e.g. 2-3 for Bo3, 4-7 for Bo7), depending on how decisively one side won.
 - **play**: a string of the same length as `games` recording which deck was on the play
   (i.e. went first) in each game. For example, `BAB` means deck B was on the play in games 1
   and 3, deck A in game 2. Game 1's play assignment is arbitrary; subsequent games follow
   the usual "loser of previous game chooses play or draw" rule. Useful for analyzing or
   controlling for the play-first bias, which is a known confounder in limited formats.
 - **duration_s**: total wall-clock duration of the match in whole seconds. Useful for
   detecting pathological matches (JVM hang, AI stall, infinite loop) and for correlating
   match length with outcome.

## Implementation

This process is implemented in Java in the forge-connector module, which already has access to Forge classes and the
necessary harnesses to list expansion sets, generate boosters, build sealed decks, and play games.

For consistency, this step is invoked from the Python CLI using `python -m sealed match-outcomes`. The Python CLI
creates a configurable number of Java worker processes (default 12) that all append game outcomes to the same file.
The match length is configurable via `--best-of` (default 7); it must be a positive odd integer, with no upper
bound. The supervisor propagates this value to each worker as a JVM system property.

Because the Forge AI is not very stable and tends to crash the JVM in various ways, the Python process monitors the
workers and automatically restarts any that die unexpectedly.

Additionally, Forge AI games tend to slow down over time within a long-running JVM (likely due to accumulated garbage
collection pressure or internal state growth). To mitigate this, the supervisor **recycles** workers: every 60 seconds
it terminates the longest-running worker process. The monitor thread automatically restarts it with a fresh JVM,
keeping throughput stable over long generation runs.

An example of this single-supervisor / multiple-worker pattern is available at `..\jumpstart-tierlist`.

# Phase 1 — Deck Scorer

## Scorer Input — Non-Land Cards Only

The scorer sees only spells and non-basic lands — basic lands are excluded. Since basic lands are assigned
deterministically from the selected spells (see Phase 0 Step 3 and Phase 2), they carry no information beyond what the
spell selection already provides. Including them would only inflate the sequence length with redundant tokens.

This means the scorer input is variable-length: typically ~20-29 cards per deck, depending on how many non-basic lands
the pool contained and the builder selected. The Set Transformer handles this natively — self-attention computes over
however many tokens are present, and the pooling seed vectors cross-attend over all cards regardless of count.

**Batching with variable lengths:** within a training batch, shorter decks are padded to the length of the longest deck
in that batch. A boolean attention mask of shape `(batch, max_cards)` marks real cards as `True` and padding as `False`.
Self-attention and seed-vector cross-attention both use this mask (setting pad positions to `-inf` before softmax) so
padding tokens contribute nothing to the output.

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

1. **Self-attention layers**: Cards attend to each other, learning interactions like "this removal spell is more
   valuable because the rest of the deck is slow" or "these three cards form a synergy package."
2. **Pooling layer**: 4-8 learned seed vectors attend over the card representations, producing a fixed-size deck vector
   (seed vectors x model dimension, e.g. 4 x 544 = 2176).
3. **Scoring head**: A small MLP (2 hidden layers of 256-512 dims with ReLU) mapping the deck vector to a single
   scalar score.

### Starting Hyperparameters

- Layers: 2-4 (start with 2; pairwise card interactions likely sufficient, try 4 if validation loss stalls)
- Attention heads: 4-8 (each head can specialize in color alignment, curve distribution, synergy, removal density, etc.)
- d_model: 544 (no input projection — the Q/K/V projections in each attention head already serve as
  per-head feature selection, letting different heads attend to different subspaces of the semantic
  and deterministic features)
- d_ff: 1088-2176 (standard 2-4x model dimension heuristic)
- Pooling seed vectors: 4-8
- Dropout: 0.0-0.2 (no dropout by default; raise toward 0.2 if validation loss diverges
  from training loss within the first 10 epochs)
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

### Validation Metrics

Three metrics are tracked to detect overfitting and measure progress:

1. **Validation Bradley-Terry loss** (BCE on held-out game pairs) — the primary signal. If training loss keeps dropping
   but validation loss rises, the model is overfitting.
2. **Validation prediction accuracy** — the fraction of held-out matchups where the higher-scored deck actually won.
   The theoretical ceiling is well below 100% because individual MTG games are noisy (the worse deck sometimes wins on
   draws alone), but this should climb and plateau.
3. **Forge baseline comparison** — the round-robin evaluation described under Phase 3 "Evaluation Against External
   Baseline." Per-pool paired win rate comparisons and aggregate win rates across a shared opponent field. This is the
   ground truth check that the scorer generalizes to actual deck quality, not just fitting the training distribution.

Note: Bradley-Terry scores have no fixed scale — all scores could drift up by 1000 and the loss would be unchanged
because only the *difference* between scores enters the sigmoid. Score magnitude is therefore not a useful metric.
If constraining it is desired for logging aesthetics, a small L2 penalty on the raw scores works, but it is not
necessary for correctness.

## Regularization

Dropout is applied at four sites in the scorer, all driven by the single `dropout` config field on `ScorerConfig`:
on the SAB (Self-Attention Block — the self-attention layers that let cards attend to each other) attention output
before the residual add+norm, on the SAB feed-forward output before the residual add+norm, on the PMA (Pooling by
Multihead Attention — the pooling layer where learned seed vectors attend over the card representations) attention
output before the residual add+norm, and inside the scoring MLP (Multi-Layer Perceptron — the small fully-connected
head that maps the pooled deck vector to a scalar score) after each ReLU activation. A single rate keeps the surface
simple; per-site rates can be added later if a sweep shows it is worth it.

Without regularization, the scorer overfits within a handful of epochs because card-level patterns memorize faster
than deck-level patterns generalize — the model latches onto specific card identities in the training set rather
than learning compositional structure. Dropout breaks those narrow paths by randomly silencing neurons at training
time, forcing the model to lean on broader, redundant signal that is more likely to transfer to unseen decks.

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
4. Score the deck with the trained scorer (input = spells + non-basic lands only; basic lands excluded)
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

`GreedyDeckBuilder` exposes simulated annealing via `--sa-temperature`, `--sa-cooling`, and `--sa-max-iterations`
flags on `build-decks`. With `temperature == 0` the algorithm is pure greedy; with `temperature > 0` swaps are
sampled from a softmax-temperature distribution over all candidates each iteration, and the best deck seen across
all iterations is returned.

`--restarts` controls how many independent searches run per pool. It accepts either a positive integer N (run N
searches from independent random 23-spell inits and keep the best deck across runs) or the literal `color-pairs`
(run one search per MTG two-color pair — WU, WB, WR, WG, UB, UR, UG, BR, BG, RG — with each search's initial
23 spells filtered to cards whose printed colors are a subset of the pair). The color filter under `color-pairs`
applies only to the seed deck; `spells_remaining` and `lands_remaining` are the unfiltered pool, so the search
is unconstrained after init. Pairs without 23 eligible on-color spells are skipped; if every pair is skipped on
a degenerate pool, the build falls back to one random restart so it always returns a deck.

See [`experiments/sa-deck-builder-tuning.md`](../experiments/sa-deck-builder-tuning.md) for empirical guidance
on choosing `temperature`, `cooling`, and the restart strategy.

# Phase 3 — Self-Play Refinement

Once the initial scorer (Phase 1) and search-based builder (Phase 2) are working, the training data can be improved
by adding scorer-built decks to the mix. This creates a curriculum: early data (Phase 0) distinguishes bad decks from
decent ones; self-play data distinguishes good decks from great ones.

Self-play refinement bridges the Python scorer and the Java game engine through a **generated-decks file** — a flat
text artifact that the Python side writes and the Java side reads. This avoids any runtime coupling between the two
languages during match generation.

## Generated Decks Preparation

Before running self-play match generation, the Python side pre-builds a corpus of scorer-guided decks:

1. **Generate pools from random sets.** Invoke `generate-pools` without `--set` to produce N pools (e.g. 10,000).
   When `--set` is omitted, each pool is generated from a randomly selected sealed-legal set (same eligible-set
   criteria as Phase 0 Step 1: sets with a draft booster template, excluding un-sets). The output always includes
   the set code as the first field on every line, regardless of whether `--set` was specified:

       SET_CODE;Card1|Card2|...|CardN

2. **Build scorer decks.** For each pool, build a deck using the scorer-guided greedy search (Phase 2), including
   deterministic basic land assignment. Write the results to a generated-decks file: one line per deck, formatted as:

       LABEL;SET_CODE;Card1|Card2|...|Card40

   Each line is a complete 40-card deck (spells + non-basic lands + basic lands), prefixed with the
   generation-method tag passed to `build-decks --label` (e.g. `gen-2`) and the source set code. The label
   identifies which scorer / search configuration produced this particular deck and is later used by
   `match-outcomes` as the `method_A` / `method_B` value when this deck is played.

```bash
# Step 1: Generate pools from random sets
python -m sealed generate-pools --size 10000

# Step 2: Build scorer decks from those pools and write generated-decks file
python -m sealed build-decks \
    --checkpoint models/sealed/scorer/best_*.pt \
    --pools-path output/sealed/pools/pools.txt \
    --label gen-2 \
    --output output/sealed/generated-decks.txt
```

## Self-Play Match Generation

Self-play match generation is triggered by passing `--generated-decks-path` to `match-outcomes`. When this argument
is absent, `match-outcomes` uses the unchanged Phase 0 behavior.

When `--generated-decks-path` is present, each match works as follows:

1. **Pick deck A**: select a random line from the generated-decks file. This gives a scorer-built 40-card deck and
   its set code.
2. **Roll for deck B's method**: choose one of 5 methods. Method 5 has the same relative weight as method 1 (the
   Forge standard builder):

   | Method | Weight | Description |
   |--------|--------|-------------|
   | 1      | 4      | Forge SealedDeckBuilder (standard) |
   | 2      | 3      | Forge builder + 3 type-matched swaps + land rebalance |
   | 3      | 2      | Forge builder + 8 type-matched swaps + land rebalance |
   | 4      | 1      | 23 random spells + land rebalance |
   | 5      | 4      | Random scorer-built deck from the generated-decks file |

3. **Build deck B**:
   - **Methods 1–4**: Generate a fresh pool from **deck A's set code**, then build deck B from that pool using the
     selected method. This ensures both decks come from the same set, matching Phase 0's same-set design.
   - **Method 5**: Pick a random line from the generated-decks file with the **same set code** as deck A.
4. **Play and record**: Play a best-of-N match (per `--best-of`, default 7) and append the result to
   `match-outcomes.txt` in the standard format (see Phase 0 Step 4). Deck A's method is the `LABEL` field
   read from the deck's line in the generated-decks file (set by `build-decks --label`). Deck B's method is
   whichever of `forge-best` / `forge-3sub` / `forge-8sub` / `random` was rolled, or — for method 5 — the
   `LABEL` of the second scorer deck sampled from the file.

The same-set constraint is critical: without it, set-level power differences (e.g. Modern Horizons vs a core set)
would dominate the training signal, drowning out the deck-building quality signal.

Because the label travels per-line inside the generated-decks file, multiple `build-decks` runs (e.g. one per
scorer generation, or per search configuration) can be concatenated into a single self-play file without losing
provenance: each match-outcomes line records the exact label of the deck it played.

```bash
python -m sealed match-outcomes \
    --generated-decks-path output/sealed/generated-decks.txt
```

## Self-Play Refinement Loop

Each iteration of the self-play loop:

1. Pre-build scorer decks (see Generated Decks Preparation above)
2. Generate self-play training data using `match-outcomes --generated-decks-path` — runs indefinitely, Ctrl-C to stop
3. Retrain the scorer from scratch on the full corpus (Phase 0 data + self-play data, all in `match-outcomes.txt`)
4. Run the external baseline evaluation (see below)
5. Inspect: Are scorer decks improving? Do they beat Forge? Do they look strategically coherent?
6. Repeat from step 1 with the improved scorer

## Evaluation Against External Baseline

The evaluation uses a round-robin design to compare the scorer's deck building against Forge's built-in SealedDeckBuilder.
Both builders receive the same pools, and all decks face a shared opponent field — this isolates builder quality from
pool quality.

### Design

1. **Generate N pools** (configurable, default 12) from a randomly selected sealed-legal set. If `--set` is specified
   on the CLI, all pools use that set instead.
2. **Build decks from each pool**: For each pool, build one deck using the scorer-guided greedy search (deck A_i) and
   one deck using Forge's SealedDeckBuilder (deck B_i). Both builders work from the same pool, so each has access to
   exactly the same cards. This produces N A-decks and N B-decks — 2N decks total.
3. **Play round-robin cross-group matches**: Every A deck plays every B deck in a best-of-K match (configurable K,
   default 3). That is N² matches total. A decks do not play other A decks, and B decks do not play other B decks —
   intra-group matches would mainly reflect pool quality differences rather than builder quality.
4. **Report results**: For each deck, compute its win rate across its N opponents. Compare the win rate of A_i (scorer)
   against B_i (Forge) built from the same pool — this is a paired comparison that controls for pool quality. Also
   report the aggregate win rate of all A decks vs the aggregate win rate of all B decks.

### Why Round-Robin

In the simplest evaluation, each scorer-built deck plays only the Forge deck from its own pool (N matches). A single
best-of-3 is extremely noisy — the worse deck frequently wins on draws alone. By giving each deck N opponents, the
per-deck win rate is averaged over a diverse field, dramatically reducing variance. With 12 pools, the round-robin
produces 144 matches (432 games at best-of-3) instead of 12, giving a much more stable signal.

The paired per-pool comparison (A_i win rate vs B_i win rate) is the most informative metric: pool quality cancels out,
so the delta directly measures builder quality. The aggregate win rates provide a high-level summary.

### Workflow

The evaluation workflow is orchestrated by Python, with Java processes handling Forge-dependent steps:

1. **Pool generation (Python → Java)**: Python invokes a Java process to generate N fresh pools from a randomly
   selected sealed-legal set (or a specific set if `--set` is provided).
2. **Deck building**:
   - **A decks (Python)**: For each pool, the Python script builds deck A_i using the scorer-guided greedy search.
   - **B decks (Python → Java)**: For each pool, Python invokes a Java command-line tool that builds a deck using
     Forge's SealedDeckBuilder and returns the 40-card deck list via stdout.
3. **Write validation matches files (Python)**: The N² match pairings (every A deck vs every B deck) are split into
   per-worker files upfront. Each line contains `{deck_A};{deck_B}` — both are pipe-separated lists of exactly 40
   card names. Each worker gets its own file, which simplifies result collection.
4. **Play matches (Java workers)**: Each worker reads its assigned matches file. For each match, it plays a best-of-K
   match between the two pre-built decks via the Forge AI and writes the outcome as `{wins_A};{wins_B}` to a companion
   outcomes file (`{input_file}-outcomes.txt`).
5. **Collect results (Python)**: After all workers complete, the Python script reads all per-worker outcome files and
   computes: per-deck win rates, per-pool A_i vs B_i comparison, and aggregate win rates for each builder group.

## Training Completion Criteria

Training is considered done when all of the following are stable across several consecutive evaluation checkpoints:

- Forge baseline comparison → scorer decks consistently outperform Forge decks from the same pool
- Scorer validation loss → converged
- Human spot check → built decks look strategically coherent

This tracks absolute quality independently of training data generation, catching the trap where the scorer learns to
rank training variants correctly but doesn't generalize to truly strong decks.

# Phase 4 — One-Shot Policy Model (Optional)

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

# Longer Term

The card encoder and the understanding of card quality learned here feed directly into the next project — training a
model to actually play the game, where card evaluation is a prerequisite for good play decisions.

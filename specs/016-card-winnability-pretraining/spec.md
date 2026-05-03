# Feature Specification: Card Winnability Pretraining for Sealed Encoder

**Feature Branch**: `016-card-winnability-pretraining`
**Created**: 2026-05-03
**Status**: Draft
**Input**: User description: "Create a new feature from the descriptions in .\specs\card-winnability-pretraining.md"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Per-game card-play data accumulates during self-play (Priority: P1)

While `python -m sealed match-outcomes` runs, every game produces one line in
`output/sealed/cards-played.txt` recording, for both sides, which cards
entered the battlefield or stack and which remained in the deck without ever
being played. This file is the raw signal from which per-card winnability
labels are derived; without it, the encoder has nothing to train on.

**Why this priority**: Card-play data collection is a precondition for every
other story in this feature. The encoder cannot be trained, the labels
cannot be aggregated, and the scorer cannot consume the new encoder until
this log file exists. It is also the only piece that has to run for an
extended period (sustained self-play) before the rest becomes useful.

**Independent Test**: Run `python -m sealed match-outcomes` for a handful of
matches, then verify that `output/sealed/cards-played.txt` contains one line
per game, that the line count equals the sum of game counts across the
match-outcomes lines written during the same run, and that game lines for a
given match appear contiguously and in game order.

**Acceptance Scenarios**:

1. **Given** a fresh sealed run with no `cards-played.txt`, **When**
   `match-outcomes` plays a Bo7 match that ends 4-0, **Then** four
   contiguous lines are appended to `cards-played.txt`, each with the same
   `set_code`, `method_A`, and `method_B` as the corresponding
   `match-outcomes.txt` row.
2. **Given** an in-progress `match-outcomes` supervisor, **When** one of its
   workers crashes mid-game, **Then** no partial line is written for the
   crashed game and the supervisor's restarted worker resumes appending
   complete game lines.
3. **Given** a card was in side A's deck but never entered the battlefield
   or stack during the game, **When** that game's line is written, **Then**
   the card appears in `cards_not_played_A` and not in `cards_played_A`.
4. **Given** a card entered the battlefield twice (e.g. flicker effect),
   **When** that game's line is written, **Then** the card appears once in
   `cards_played_X` (membership, not multiplicity).

---

### User Story 2 - Train a sealed encoder from scratch on the winnability target (Priority: P1)

The user invokes `python -m sealed train-encoder`, which reads
`cards-played.txt`, aggregates per-card win counts, computes a winnability
score per card, and trains a token-encoder + card-encoder stack from random
initialization to predict that score. After training, the encoder portion is
saved as `models/sealed/encoder/{timestamp}.pt` plus an updated
`latest.pt`. The regression head used during training is discarded — only
the encoder is preserved, since that is the artifact downstream tools
consume.

**Why this priority**: This is the headline feature. The goal is an encoder
shaped by sealed playability rather than market price, so that when it
replaces the price-predictor encoder in the sealed scorer it carries
per-card quality discrimination the price signal cannot.

**Independent Test**: Run `train-encoder` against a populated
`cards-played.txt` and verify that (a) `models/sealed/encoder/latest.pt`
exists and contains only the token-encoder and card-encoder weights, (b)
the validation set is disjoint from the training set at the card level, and
(c) the best checkpoint is selected by validation loss.

**Acceptance Scenarios**:

1. **Given** a populated `cards-played.txt` and a built sealed vocabulary,
   **When** the user runs `python -m sealed train-encoder`, **Then** training
   reads the cards-played corpus, computes the per-card winnability map
   inline at start, and proceeds without requiring a separate aggregation
   command or on-disk cache of the labels.
2. **Given** training completes successfully, **When** the saved
   `latest.pt` is loaded, **Then** it contains the token-encoder and
   card-encoder weights but no regression-head weights.
3. **Given** training is in progress, **When** the validation loss has not
   improved for `--patience` consecutive epochs, **Then** training stops
   early and the best-by-validation-loss checkpoint is preserved.
4. **Given** a card has zero `wins_when_in_deck` (i.e. it was never in a
   winning deck), **When** the per-card winnability map is built, **Then**
   that card is excluded from training.
5. **Given** the train/val split is built, **When** the user inspects which
   cards are in each side, **Then** every card in the validation set is
   absent from the training set, and both sides cover the full label range
   (stratified by winnability quartile).

---

### User Story 3 - Sealed scorer consumes the sealed-trained encoder (Priority: P1)

After the sealed encoder exists at `models/sealed/encoder/latest.pt`,
`train-scorer` and `encode-cards` use it as the default
`--encoder-checkpoint` source for both Phase A and Phase B. The user does
not have to pass any extra flag to switch over; running the existing scorer
pipeline picks up the new encoder automatically. The price-predictor
transformer is no longer in the sealed-scoring path — it remains a separate
product for predicting card prices.

**Why this priority**: Producing an encoder artifact has no impact on
sealed performance until the scorer pipeline actually consumes it. This
story closes the loop and is what makes the feature load-bearing.

**Independent Test**: Train a sealed encoder, then run `encode-cards`
followed by `train-scorer` with no `--encoder-checkpoint` flag, and verify
that the encoder weights flowing into the scorer come from
`models/sealed/encoder/latest.pt` rather than from
`models/price-predictor/transformer/latest.pt`.

**Acceptance Scenarios**:

1. **Given** `models/sealed/encoder/latest.pt` exists, **When** the user runs
   `python -m sealed encode-cards` with no `--encoder-checkpoint`, **Then**
   the resulting `.npz` files are produced from the sealed encoder.
2. **Given** `models/sealed/encoder/latest.pt` exists, **When** the user runs
   `python -m sealed train-scorer` with no `--encoder-checkpoint` (Phase A
   or Phase B), **Then** training uses the sealed encoder and the saved
   checkpoint records the sealed encoder path in its `config`.
3. **Given** the user explicitly passes
   `--encoder-checkpoint models/price-predictor/transformer/latest.pt` to
   `train-scorer`, **Then** training uses the price-predictor encoder
   instead, with no warning required.

---

### User Story 4 - Build a sealed-specific vocabulary (Priority: P2)

Before the sealed encoder can be trained, the user runs
`python -m sealed build-vocab` to scan the converted card corpus and emit
`models/sealed/encoder/vocab.txt`. The command is a thin sealed-side
wrapper around the existing price-predictor vocabulary builder, so the
algorithm is shared but the output file is sealed-specific.

**Why this priority**: Required before `train-encoder` can run. Lower
priority than P1 stories because the vocabulary algorithm is task-agnostic
and reuses logic that already exists for the price-predictor side.

**Independent Test**: Run `python -m sealed build-vocab` against
`output/cardsfolder/` and verify that `models/sealed/encoder/vocab.txt`
exists with one token per line, that known MTG terms (creature, enchantment,
flying) are each a single token, and that the file is independent from
`models/price-predictor/transformer/vocab.txt` (i.e. updating one does not
update the other).

**Acceptance Scenarios**:

1. **Given** a populated `output/cardsfolder/`, **When** the user runs
   `python -m sealed build-vocab`, **Then** `models/sealed/encoder/vocab.txt`
   is written with the expected per-line token format.
2. **Given** the user passes `--target-size N`, **Then** the resulting
   vocabulary contains approximately N tokens.
3. **Given** the price-predictor vocabulary already exists, **When** the
   sealed vocabulary is built, **Then** the price-predictor vocabulary file
   is not modified.

---

### User Story 5 - Tune low-n shrinkage for noisy labels (Priority: P3)

Cards with few in-deck observations have noisy raw winnability ratios. The
user passes `--shrinkage-k <F>` (default 20) to `train-encoder` to control
how aggressively low-observation labels are pulled toward 0.5 before
training. Higher `k` shrinks more; `k = 0` recovers raw ratios. This lets
the user trade off between fitting weakly-observed cards and avoiding the
bias their noisy labels would inject.

**Why this priority**: Useful for tuning runs but not required for the
feature to function. Default `k = 20` produces a usable model; the flag
exists so the user can experiment without code changes.

**Independent Test**: Run two `train-encoder` invocations differing only
in `--shrinkage-k` (e.g. `0` vs `20`) on the same corpus and verify that the
resulting per-card label maps differ for cards with few in-deck observations
but are nearly identical for cards with many.

**Acceptance Scenarios**:

1. **Given** a card with `wins_when_in_deck = 2` and `wins_when_played = 2`,
   **When** the label is computed with `k = 20`, **Then** the shrunk label
   is meaningfully below 1.0 (pulled toward 0.5) rather than the raw 1.0.
2. **Given** a card with `wins_when_in_deck = 1000`, **When** the label is
   computed with `k = 20`, **Then** the shrunk label is within a few
   thousandths of the raw ratio.

---

### Edge Cases

- **`cards-played.txt` and `match-outcomes.txt` line counts disagree**:
  the most likely cause is a crash between the two writes. Aggregation must
  not assume strict 1:1 alignment per match; it processes each
  `cards-played.txt` line independently and tolerates trailing
  unmatched-by-match-outcomes lines without aborting.
- **A card appears in `cards-played.txt` but never under any winning
  side's `cards_played_*` or `cards_not_played_*` columns**: its
  `wins_when_in_deck` is zero, so the divisor is undefined; the card is
  excluded from the per-card label map and is therefore not used in
  training. The encoder still produces an embedding for it at inference
  time, since the encoder is a function of token IDs, not of the
  training-set membership.
- **A card appears only on losing sides across the entire corpus**: same
  outcome as the previous case — it never contributes to either numerator
  or denominator, so it is excluded from the training label map.
- **`train-encoder` runs before `build-vocab`**: the vocabulary file does
  not exist, so training fails with a clear error pointing the user at
  `python -m sealed build-vocab`.
- **`train-encoder` runs before any `cards-played.txt` exists**: training
  fails with a clear error pointing the user at `python -m sealed
  match-outcomes`.
- **Downstream pipeline runs before the sealed encoder exists**:
  `train-scorer` and `encode-cards` default
  `--encoder-checkpoint` to the sealed encoder; if that file is absent
  and the user did not pass `--encoder-checkpoint` explicitly, the command
  fails with an error pointing the user at `train-encoder` or at the
  price-predictor encoder as an explicit fallback.
- **Card text whose token sequence exceeds `max_seq_len`**: matches the
  price-predictor convention — `max_seq_len` is computed from the corpus
  by rounding up the longest card to a multiple of 8, so no card overflows
  in normal operation.
- **The same card name appears more than once in a deck list**: the
  cards-played columns store each occurrence (multiplicity preserved in the
  file format), but a card "is played" in a game iff it entered the
  battlefield or stack at least once, so for label purposes it is counted
  once per game per side.
- **Stratification on a degenerate winnability distribution**: if all
  cards happen to fall in fewer than four distinct quartiles (small corpus
  or extreme distribution), stratification falls back to as many strata as
  there are distinct quantile bins, while still producing a card-level
  disjoint train/val split.

## Requirements *(mandatory)*

### Functional Requirements

#### Per-game data collection

- **FR-001**: The Java match worker MUST append one line per played game to
  `output/sealed/cards-played.txt`. Writes MUST be line-buffered so a
  worker crash mid-write does not produce a partial line.
- **FR-002**: The line format MUST be exactly nine semicolon-separated
  fields in order: `set_code`, `method_A`, `method_B`, `cards_played_A`,
  `cards_played_B`, `cards_not_played_A`, `cards_not_played_B`, `winner`,
  `starter`. The four card-list columns MUST be pipe-separated card names.
- **FR-003**: A card MUST be considered "played" iff it enters the
  battlefield or the stack at least once during the game, controlled by the
  side that owns it. Cards that remain in the library, hand, sideboard, or
  exile (without ever having been on the battlefield or stack) MUST NOT
  appear in `cards_played_X`.
- **FR-004**: For each game line, the union of `cards_played_X` and
  `cards_not_played_X` MUST equal the full deck of side X for that game,
  with multiplicities preserved.
- **FR-005**: The sequence of game lines for a single match MUST appear
  contiguously in `cards-played.txt`, in game order, and in the same
  relative position as the corresponding line in `match-outcomes.txt` (so
  the i-th match line corresponds to the i-th contiguous game block).
- **FR-006**: Per-game logging MUST be unconditional during
  `match-outcomes` runs. There is no opt-in flag.

#### Vocabulary build

- **FR-007**: A new subcommand `python -m sealed build-vocab` MUST scan
  the converted card corpus (default `output/cardsfolder/`) and emit a
  vocabulary file (default `models/sealed/encoder/vocab.txt`).
- **FR-008**: `build-vocab` MUST delegate the corpus scan and vocabulary
  fitting to the existing `price_predictor.application.build_vocabulary`
  utility. The sealed vocabulary file MUST be independent from the
  price-predictor vocabulary file; building one MUST NOT modify the other.
- **FR-009**: `build-vocab` MUST accept `--cards-folder`, `--vocab-path`,
  and `--target-size` flags, with defaults `output/cardsfolder/`,
  `models/sealed/encoder/vocab.txt`, and ~5000 respectively.

#### Label aggregation

- **FR-010**: At the start of every `train-encoder` run, the system MUST
  perform a single pass over `cards-played.txt` to compute, for every
  card name appearing in the file: `wins_when_played[c]` (count of games
  where the winning side played the card) and `wins_when_in_deck[c]`
  (count of games where the card was in the winning side's deck whether or
  not it was played). Losing-side games MUST be excluded from both counts.
- **FR-011**: The per-card winnability label MUST be computed as
  `(wins_when_played[c] + k/2) / (wins_when_in_deck[c] + k)`, where `k`
  is the value of `--shrinkage-k`. With `k = 0`, this reduces to the raw
  ratio.
- **FR-012**: Cards with `wins_when_in_deck[c] == 0` (and `k = 0`) MUST be
  excluded from the training label map. With `k > 0`, those cards have a
  shrunk label of 0.5 but MUST also be excluded, since their shrunk label
  carries no signal.
- **FR-013**: Aggregation MUST run inline at train start. The system MUST
  NOT persist the per-card label map to disk and MUST NOT expose a separate
  aggregation subcommand.

#### Encoder training

- **FR-014**: A new subcommand `python -m sealed train-encoder` MUST train
  a model composed of (a) a token encoder (learned token embedding table
  plus positional encoding, no cross-token mixing), (b) a card encoder (a
  stack of `--n-layers` transformer encoder blocks with self-attention and
  FFN, followed by a single pool layer), and (c) a regression head (linear
  projection to a scalar, followed by sigmoid).
- **FR-015**: The pool layer MUST produce the card vector by concatenating
  the output of two parallel operations over the contextualized token
  sequence: a multi-query attention pool with `--n-pool-queries` learned
  queries cross-attending to the tokens (output dims = `K * d_token`), and
  an element-wise max pool across the token sequence (output dims =
  `d_token`). The attention-pool half MUST be sized so the concatenated
  card vector has dimension `2 * d_token` to mirror the price-predictor's
  split (`d_token = 256`, `d_card = 512`).
- **FR-016**: The encoder and regression head MUST be trained jointly from
  random initialization. The system MUST NOT initialize any component from
  the price-predictor transformer, since the goal is an encoder shaped
  entirely by the winnability signal.
- **FR-017**: The training loss MUST be MSE between the regression head's
  output (after sigmoid) and the per-card shrunk winnability label.
- **FR-018**: The train/val split MUST be at the card level (a held-out
  card never appears in the training set) and stratified by winnability
  quartile so each split covers the full label range. The held-out
  fraction MUST be 20%.
- **FR-019**: The best checkpoint MUST be selected by validation loss.
  Early stopping MUST trigger after `--patience` consecutive epochs without
  a new best validation loss.
- **FR-020**: After training completes, the system MUST save the
  token-encoder and card-encoder weights to
  `models/sealed/encoder/{timestamp}.pt` and update
  `models/sealed/encoder/latest.pt` to point at (or be a copy of) the
  best-by-val-loss checkpoint. The regression head's weights MUST NOT be
  written to either file.
- **FR-021**: `train-encoder` MUST accept the following CLI flags with the
  documented defaults: `--cards-played-path` (default
  `output/sealed/cards-played.txt`), `--cards-folder` (default
  `output/cardsfolder/`), `--vocab-path` (default
  `models/sealed/encoder/vocab.txt`), `--model-output` (default
  `models/sealed/encoder/`), `--batch-size` (64), `--epochs` (100), `--lr`
  (1e-4), `--patience` (20), `--dropout` (0.1), `--n-layers` (6),
  `--n-heads` (4), `--n-pool-queries` (4), `--shrinkage-k` (20).
- **FR-022**: The following knobs MUST be hardcoded constants in
  `train-encoder`: `d_model = 256`, `ff_dim = 1024`, `loss = MSE`,
  `val_split = 0.2`, stratification = winnability quartiles, seed = 42.
  `max_seq_len` MUST be computed from the corpus at train start (longest
  card length rounded up to a multiple of 8).
- **FR-023**: `train-encoder` MUST fail with a clear error message
  pointing the user at the corrective command if (a) the vocabulary file
  is missing (point at `build-vocab`), (b) `cards-played.txt` is missing
  or empty (point at `match-outcomes`), or (c) the corpus folder is empty
  (point at `convert`).

#### Downstream integration

- **FR-024**: The default value of `--encoder-checkpoint` for
  `python -m sealed train-scorer` (Phase A and Phase B) MUST be
  `models/sealed/encoder/latest.pt`. The previous default
  (`models/price-predictor/transformer/latest.pt`) is replaced.
- **FR-025**: The default value of `--encoder-checkpoint` for
  `python -m sealed encode-cards` MUST be
  `models/sealed/encoder/latest.pt`. The previous default
  (`models/price-predictor/transformer/latest.pt`) is replaced.
- **FR-026**: When the default `--encoder-checkpoint` resolves to a file
  that does not exist, both `train-scorer` and `encode-cards` MUST fail
  with an error message that names the missing file and points the user
  at `python -m sealed train-encoder` (or at passing
  `--encoder-checkpoint` explicitly).
- **FR-027**: An explicitly passed `--encoder-checkpoint` MUST continue
  to work for both subcommands so the user retains the option to point at
  the price-predictor encoder or any other compatible checkpoint without
  code changes.

### Key Entities

- **`cards-played.txt`**: Append-only, line-buffered log of per-game
  card-play data, written automatically by every `match-outcomes` run.
  Lives at `output/sealed/cards-played.txt`. Rows align contiguously
  with `match-outcomes.txt` so a match's game lines appear in game order
  in the same position as the parent match line.
- **Per-card winnability map**: An in-memory dict from card name to
  `[0, 1]` shrunk-label, built at train start by a single pass over
  `cards-played.txt`. Cards with zero `wins_when_in_deck` are absent from
  the map. Not persisted to disk.
- **Sealed vocabulary**: A token list at
  `models/sealed/encoder/vocab.txt`, one token per line. Built from the
  converted card corpus by `python -m sealed build-vocab`. Independent
  from the price-predictor vocabulary; updating one does not update the
  other.
- **Sealed encoder checkpoint**: The token-encoder + card-encoder weights
  saved at `models/sealed/encoder/{timestamp}.pt` plus a `latest.pt`
  pointer (or copy). The regression head used during training is not part
  of this artifact. Default `--encoder-checkpoint` source for
  `train-scorer` and `encode-cards` after this feature ships.
- **Token encoder**: Learned token embedding table plus positional
  encoding, mapping a token ID and position to a `d_token`-dim vector.
  No cross-token mixing.
- **Card encoder**: A stack of `N` transformer encoder blocks
  (self-attention + FFN + residual) followed by a single pool layer that
  concatenates a multi-query attention pool with an element-wise max pool.
  Produces a single `d_card`-dim vector per card.
- **Regression head**: Linear-to-scalar projection with sigmoid, trained
  jointly with the encoder against the MSE-on-winnability loss and
  discarded after training. Not saved to disk.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Sustained `match-outcomes` runs accumulate per-game
  card-play data without manual intervention; the line count of
  `cards-played.txt` matches the sum of game counts in
  `match-outcomes.txt` for the same run.
- **SC-002**: A user with a populated `cards-played.txt` and a
  pre-existing converted corpus can produce a usable sealed encoder
  artifact end-to-end with three commands: `build-vocab`, `train-encoder`,
  and (optionally) `encode-cards`. No manual scripting between the steps
  is required.
- **SC-003**: After `train-encoder` finishes, the existing sealed scorer
  pipeline (`encode-cards` → `train-scorer` Phase A → `train-scorer`
  Phase B → `evaluate-scorer`) runs unmodified and produces a scorer
  whose feature inputs come from the sealed encoder rather than the
  price-predictor encoder.
- **SC-004**: When the deployment metric (`evaluate-scorer` win rate
  against `forge-best`) is compared between a sealed-encoder-based
  scorer and the prior price-predictor-based scorer on the same pool
  set, the comparison can be performed and reported using only
  documented CLI flags.
- **SC-005**: The `--shrinkage-k` flag changes the per-card label map in
  a way the user can verify by inspecting the labels of low-observation
  cards (e.g., a card with two in-deck games shifts visibly when `k`
  changes between 0 and 20) while leaving high-observation cards
  effectively unchanged.
- **SC-006**: Reverting to the price-predictor encoder for the sealed
  scoring pipeline is achievable in one step (passing
  `--encoder-checkpoint models/price-predictor/transformer/latest.pt`
  to `train-scorer` and `encode-cards`), with no code changes.

## Assumptions

- The price-predictor encoder is no longer a dependency of the sealed
  scoring pipeline, but the price-predictor product itself (predicting
  card prices for end users) remains live and is unaffected.
- The shared `price_predictor.application.build_vocabulary` utility is
  task-agnostic — it scans converted card text and emits a token vocab
  without consuming any price information — so reusing it for the sealed
  vocabulary does not re-introduce a price-task dependency.
- The Forge match worker can determine, for every game it plays, which
  cards entered the battlefield or stack at least once. The "is played"
  signal is observable from Forge's existing game-state APIs and does not
  require new instrumentation beyond what already exists for match
  outcomes.
- Bayesian shrinkage is the primary low-n regularization mechanism. The
  alternate sample-weighting approach is a future extension, not part of
  this feature.
- The `d_token = 256`, `d_card = 512` split mirroring the price-predictor
  is acceptable; the goal is to produce a drop-in replacement so
  downstream tools that expect `2 * d_model = 512` encoder features
  continue to work without modification.
- The encoder is trained from random init in every run. There is no
  resume capability for `train-encoder`; an interrupted run is restarted
  from scratch. (A `--resume` flag is a future extension if training
  durations grow long enough to justify it.)
- Training durations are expected to be short relative to the
  match-outcomes data-collection cost, so re-training the encoder from
  scratch whenever the corpus grows materially is acceptable.
- The `cards-played.txt` write path tolerates JVM crashes by relying on
  line-buffered output: a crash mid-line truncates the file at a line
  boundary or shortly after, and downstream readers tolerate a final
  partial line. The supervisor's existing crash-and-restart policy
  applies unchanged.

## Out of Scope

- **Reusing existing match-outcomes data**: This feature collects new
  per-game card-play data going forward. Reconstructing it for matches
  played before this feature shipped is not supported, since
  `match-outcomes.txt` does not record per-game card-play events.
- **Sample-weighting low-n regularization**: The Bayesian shrinkage flag
  `--shrinkage-k` is the only low-n knob exposed by this feature.
  Down-weighting low-n cards in the regression loss is a compatible
  extension but not part of this feature.
- **Encoder fine-tuning on top of the sealed-trained encoder**: Phase B
  fine-tuning (spec 015) continues to operate against whatever
  `--encoder-checkpoint` is supplied. After this feature ships the
  default of that flag is the sealed-trained encoder, but no new
  fine-tuning mechanism is introduced here.
- **Per-set or per-format label aggregation**: All winnability counts
  pool across every set in `cards-played.txt`. Set-conditioned
  winnability is a future extension.
- **Resume capability for `train-encoder`**: Interrupted training runs
  are restarted from scratch.
- **Encoder evaluation as a standalone task**: This feature does not add
  a separate `evaluate-encoder` command. The encoder's quality is
  measured indirectly via the downstream scorer's win rate against
  `forge-best`, using the existing `evaluate-scorer` command.
- **Auto-triggering re-training**: `train-encoder` is a manual user
  action. The system does not detect that `cards-played.txt` has grown
  and rebuild the encoder automatically.

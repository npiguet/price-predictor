# Feature Specification: Card Winnability Pretraining for Sealed Encoder

**Feature Branch**: `016-card-winnability-pretraining`
**Created**: 2026-05-03
**Status**: Draft
**Input**: User description: "Create a new feature from the descriptions in .\specs\card-winnability-pretraining.md"

## Clarifications

### Session 2026-05-10

- Q: What optimizer and gradient clipping strategy does `train-encoder` use? → A: AdamW with per-parameter-group max-norm 1.0 gradient clipping, matching the existing train-scorer convention in spec 015.
- Q: Is the "validation loss" used for best-checkpoint selection and `--patience` early stopping the regression-only loss (`L_reg`) or the full loss (`L_reg + mlm_weight · L_mlm`)? → A: Full validation loss (`L_reg + mlm_weight · L_mlm`) for both best-checkpoint selection and `--patience` early stopping.
- Q: How are FR-017a per-head sample weights normalized when accumulating the batch loss? → A: Per-head, per-batch sum-to-1 normalization (weighted average): each head's per-card MSE contributions are divided by that head's total sample weight in the batch, so loss magnitude is decoupled from how many high-weight cards land in a batch.
- Q: What learning-rate schedule does `train-encoder` use? → A: Linear warmup over the first 5% of total scheduled steps (`--epochs × batches_per_epoch`), then constant `--lr` for the remainder. No decay. Early stopping may end the run before the constant phase completes.
- Q: Does `cards-win-rates.txt` include a header row? → A: Yes, one header row with column names matching the FR-013a schema, followed by N data rows.

### Session 2026-05-03

- Q: With `--n-pool-queries K = 4` and `d_token = 256`, FR-015's "attention-pool output dims = K * d_token" and "card vector dimension = 2 * d_token" cannot both hold. What is the attention-pool half's actual output dimension? → A: Each of the K queries outputs `d_token / K` dims; K outputs concatenated = `d_token`; combined with the max-pool's `d_token` gives `d_card = 2 * d_token = 512`, independent of K. K controls per-query capacity, not `d_card`.
- Q: How are basic lands (Plains, Island, Swamp, Mountain, Forest, Wastes, snow basics) handled in `cards-played.txt` and label aggregation? → A: Excluded entirely. The Java worker filters basic lands at write time, so they appear in neither `cards_played_X` nor `cards_not_played_X`, and the label map never contains a basic-land entry.
- Q: What does `train-encoder` do when a card name in `cards-played.txt` has no corresponding text file in `output/cardsfolder/`? → A: Fail at train start with a clear error naming the missing cards and pointing the user at `python -m price_predictor convert` to rebuild the corpus. No training runs while any referenced card is missing.
- Q: Should `cards-played.txt` lines be joinable to `match-outcomes.txt` by an explicit key, or rely on positional alignment? → A: Prepend `timestamp;run_id;` to every line, mirroring `match-outcomes.txt`. Eleven fields total. `run_id` matches the parent match's `run_id`; positional offset within a single `(run_id, set_code, method_A, method_B)` group identifies which match the game belongs to.
- Q: Should `train-encoder` validate that `vocab.txt` covers the current corpus, or accept silent UNK fallback for tokens introduced after the last `build-vocab` run? → A: No check. UNK fallback is silent. Keeping vocab in sync with the corpus is the user's responsibility — re-run `build-vocab` after material corpus changes.
- Q: Does `train-encoder` strip the `name:` line from each converted card before tokenizing, matching `encode-cards`'s inference behavior? → A: Yes. The `name:` line MUST be stripped at training time, identical to `encode-cards`. This keeps train and inference inputs identically shaped and prevents the encoder from shortcutting on `name → label` (which would otherwise generalize poorly given the card-level train/val split).
- Q: How does the user inspect the per-card winnability label map to verify SC-005 (shrinkage effect)? → A: At train start, `train-encoder` writes the entire label map to `output/sealed/cards-win-rates.txt`, sorted by `shrunk_score_play` descending (per FR-013a). No flag, always emitted. The file is overwritten each run.

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
   contiguous lines are appended to `cards-played.txt`, each carrying the
   same `run_id`, `set_code`, `method_A`, and `method_B` as the
   corresponding `match-outcomes.txt` row, and each line's `timestamp`
   falls between the parent match line's `timestamp` and the next match
   line's `timestamp` (or the end of the file for the final match).
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

### User Story 2 - Train a sealed encoder from scratch on per-card play data (Priority: P1)

The user invokes `python -m sealed train-encoder`, which reads
`cards-played.txt`, aggregates per-card play counters across two
passes (primary + per-color), computes nine per-card regression labels
(net winning influence on the play and on the draw, played rate,
cast-lift, and a per-color affinity for each of WUBRG), and trains a
token-encoder + card-encoder stack from random initialization to
predict all nine jointly. A masked-token reconstruction head provides
auxiliary self-supervision over the same token stream during training.
After training, only the token-encoder and card-encoder weights are
saved as `models/sealed/encoder/{timestamp}.pt` plus an updated
`latest.pt`. The regression heads and the MLM head are discarded — only
the encoder is preserved, since that is the artifact downstream tools
consume.

**Why this priority**: This is the headline feature. The goal is an
encoder shaped by sealed playability across multiple distinct facets
(tempo vs reactive value, casting effect, cross-color affinity) rather
than by market price, so that when it replaces the price-predictor
encoder in the sealed scorer it carries per-card quality discrimination
the price signal cannot.

**Independent Test**: Run `train-encoder` against a populated
`cards-played.txt` and verify that (a) `models/sealed/encoder/latest.pt`
exists and contains only the token-encoder and card-encoder weights
(no regression heads, no MLM head), (b) the validation set is disjoint
from the training set at the card level, and (c) the best checkpoint
is selected by validation loss.

**Acceptance Scenarios**:

1. **Given** a populated `cards-played.txt` and a built sealed vocabulary,
   **When** the user runs `python -m sealed train-encoder`, **Then** training
   reads the cards-played corpus, computes the per-card label map inline
   at start (primary + play/draw counters and per-color slices), and
   proceeds without requiring a separate aggregation command or an
   on-disk cache of the labels.
2. **Given** training completes successfully, **When** the saved
   `latest.pt` is loaded, **Then** it contains the token-encoder and
   card-encoder weights but no regression-head weights and no MLM-head
   weights.
3. **Given** training is in progress, **When** the validation loss has
   not improved for `--patience` consecutive epochs, **Then** training
   stops early and the best-by-validation-loss checkpoint is preserved.
4. **Given** a card has zero in-deck observations across the entire
   `cards-played.txt`, **When** the per-card label map is built,
   **Then** that card is excluded from training entirely.
5. **Given** a card has nonzero in-deck observations but zero
   observations within a particular slice (e.g. it never appeared in a
   deck on the play, or it never appeared in a deck running color U),
   **When** training proceeds, **Then** that slice's cell contributes
   zero loss for that card while the card's other-slice cells continue
   to receive gradient.
6. **Given** the train/val split is built, **When** the user inspects
   which cards are in each side, **Then** every card in the validation
   set is absent from the training set, and both sides cover the full
   label range (stratified by `score_play` quartile).
7. **Given** training is in progress, **When** a forward pass runs,
   **Then** approximately `--mlm-mask-prob` fraction of each card's
   non-special tokens are replaced with the reserved `[MASK]` token
   for that pass, and cross-entropy loss is accumulated only at masked
   positions.

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
4. **Given** the sealed vocabulary file is written, **When** any consumer
   reads it, **Then** a reserved `[MASK]` token is present in the file and
   does not collide with any corpus-derived token.

---

### User Story 5 - Tune low-n shrinkage for noisy labels (Priority: P3)

Cards with few observations have noisy raw labels. The user passes
`--shrinkage-k <F>` (default 20) to `train-encoder` to control how
aggressively low-observation labels are pulled toward each head's
neutral point (0 for the signed heads, 0.5 for `played_rate`) before
training. Higher `k` shrinks more; `k = 0` recovers raw labels. The
same `k` also drives per-head sample-weighting, so cells whose slice
denominator is small contribute proportionally less to the loss.

**Why this priority**: Useful for tuning runs but not required for the
feature to function. Default `k = 20` produces a usable model; the flag
exists so the user can experiment without code changes.

**Independent Test**: Run two `train-encoder` invocations differing
only in `--shrinkage-k` (e.g. `0` vs `20`) on the same corpus and
verify that the resulting per-card label snapshot
(`output/sealed/cards-win-rates.txt`) differs for cards with few
observations but is nearly identical for cards with many.

**Acceptance Scenarios**:

1. **Given** a card with two in-deck observations both won-and-played,
   **When** the labels are computed with `k = 20`, **Then** the shrunk
   `score_play`/`score_draw`/`cast_lift`/`color_lift_X` cells are
   meaningfully closer to 0 (and `played_rate` closer to 0.5) than
   their raw counterparts.
2. **Given** a card with a thousand in-deck observations, **When** the
   labels are computed with `k = 20`, **Then** every shrunk label is
   within a few thousandths of its raw counterpart.
3. **Given** a card with a hundred in-deck observations of which only
   five appeared in a deck running color W, **When** training
   proceeds, **Then** that card's `color_lift_W` cell contributes
   meaningfully less loss per training step than its `color_lift_X`
   cells for colors it appears alongside more often.

---

### Edge Cases

- **`cards-played.txt` and `match-outcomes.txt` line counts disagree**:
  the most likely cause is a crash between the two writes. Aggregation must
  not assume strict 1:1 alignment per match; it processes each
  `cards-played.txt` line independently and tolerates trailing
  unmatched-by-match-outcomes lines without aborting.
- **A card has zero total in-deck observations across the entire
  `cards-played.txt`**: every label denominator is zero, so the card
  is excluded from the per-card label map entirely and is not used in
  training. The encoder still produces an embedding for it at inference
  time, since the encoder is a function of token IDs, not of training-set
  membership.
- **A card has nonzero total in-deck observations but zero observations
  within a particular slice** (e.g., the card never appeared in a deck
  on the play, or never appeared in a deck running color U): the
  card is included in the training label map; the affected slice's
  cell is empty and contributes zero loss for that head. The card's
  other heads continue to receive gradient.
- **`train-encoder` runs before `build-vocab`**: the vocabulary file does
  not exist, so training fails with a clear error pointing the user at
  `python -m sealed build-vocab`.
- **`train-encoder` runs before any `cards-played.txt` exists**: training
  fails with a clear error pointing the user at `python -m sealed
  match-outcomes`.
- **`cards-played.txt` references cards absent from `output/cardsfolder/`**
  (e.g., new sets played by Forge after the last `convert` run, or
  cards Forge has a play-side script for but no converted text):
  training logs a warning naming the missing cards (capped at a
  reasonable display count, with the total reported) and pointing the
  user at `python -m price_predictor convert`, then drops those cards
  from the label map / split / dataset and proceeds. Missing cards are
  reported but do not block the run.
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
  cards-played columns are sets of distinct names, not multisets. If at
  least one copy entered the battlefield or stack, the name appears once
  in `cards_played_X` and not at all in `cards_not_played_X`. If no copy
  was played, the name appears once in `cards_not_played_X` and not at
  all in `cards_played_X`.
- **Stratification on a degenerate `score_play` distribution**: if all
  cards happen to fall in fewer than four distinct quartiles (small
  corpus or extreme distribution), stratification falls back to as
  many strata as there are distinct quantile bins, while still
  producing a card-level disjoint train/val split.
- **A card with no `score_play` cell at all** (zero `@play`
  denominator) cannot be assigned a `score_play` quartile for
  stratification: the system MUST fall back to whichever non-empty
  signed-head cell the card carries, and to a single catch-all
  stratum if no signed-head cell is non-empty (per FR-018).

## Requirements *(mandatory)*

### Functional Requirements

#### Per-game data collection

- **FR-001**: The Java match worker MUST append one line per played game to
  `output/sealed/cards-played.txt`. Writes MUST be line-buffered so a
  worker crash mid-write does not produce a partial line.
- **FR-002**: The line format MUST be exactly eleven semicolon-separated
  fields in order: `timestamp`, `run_id`, `set_code`, `method_A`,
  `method_B`, `cards_played_A`, `cards_played_B`, `cards_not_played_A`,
  `cards_not_played_B`, `winner`, `starter`. `timestamp` MUST be ISO 8601
  UTC matching the format used by `match-outcomes.txt`. `run_id` MUST be
  the same UUID as the parent match's `run_id` in `match-outcomes.txt`.
  The four card-list columns MUST be pipe-separated card names.
- **FR-003**: A card MUST be considered "played" iff it enters the
  battlefield or the stack at least once during the game, controlled by the
  side that owns it. Cards that remain in the library, hand, sideboard, or
  exile (without ever having been on the battlefield or stack) MUST NOT
  appear in `cards_played_X`.
- **FR-004**: For each game line, the union of `cards_played_X` and
  `cards_not_played_X` MUST equal the set of distinct non-basic card
  names in side X's deck for that game (basic lands excluded per
  FR-004a). Each name appears in **at most one** of the two columns:
  `cards_played_X` lists names where at least one copy entered the
  battlefield or stack; `cards_not_played_X` lists names where no copy
  was played. Neither column ever contains duplicates, and a name in
  `cards_played_X` never also appears in `cards_not_played_X`.
- **FR-004a**: Basic lands MUST be excluded from both `cards_played_X`
  and `cards_not_played_X`. The Java match worker filters them out at
  write time, so basic lands never appear in `cards-played.txt` and
  never enter the per-card label map. "Basic land" means any card
  whose type line contains the supertype `Basic` (covers Plains,
  Island, Swamp, Mountain, Forest, Wastes, and snow-covered variants,
  plus any future basic printings).
- **FR-005**: The sequence of game lines for a single match MUST appear
  contiguously in `cards-played.txt`, in game order, and MUST share the
  parent match's `run_id`. Within a single `(run_id, set_code, method_A,
  method_B)` group, the i-th contiguous game block corresponds to the i-th
  matching line in `match-outcomes.txt`. Concurrent supervisors writing
  with distinct `run_id`s are joinable independently and never conflict.
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
- **FR-009a**: The emitted vocabulary file MUST contain a reserved
  `[MASK]` token used by the MLM auxiliary loss at training time. The
  reserved token MUST NOT collide with any corpus-derived token.

#### Label aggregation

- **FR-010**: At the start of every `train-encoder` run, the system
  MUST aggregate per-card play data from `cards-played.txt`: the
  primary and play/draw counters (FR-010a) and the per-color slices
  (FR-010b). (The implementation does this in a single streaming pass;
  the number of passes is not constrained — only the resulting
  counters are.)
- **FR-010a**: Aggregation MUST track, per card name, the four primary
  counters `wins_when_played`, `wins_when_in_deck`,
  `losses_when_played`, `losses_when_in_deck` over all games where the
  card appeared in *either* the winning or the losing side's deck,
  AND the play-restricted subsets `wins_when_played@play`,
  `wins_when_in_deck@play`, `losses_when_played@play`,
  `losses_when_in_deck@play` (counters restricted to games where the
  card's owner was the starter, as recorded in the `starter` field).
  The `@draw` counterparts MUST be derivable by subtraction from the
  primary and `@play` counters. The cast-lift counters
  (`wins_when_not_played`, `losses_when_not_played`) MUST be derivable
  from the primary counters by subtraction; they MUST NOT be tracked
  as independent variables.
- **FR-010b**: Aggregation MUST resolve each observed card's color
  identity by reading the `mana cost:` line of the card's converted
  text file under `--cards-folder`. WUBRG letters in the cost
  (including hybrid symbols like `{W/U}` and Phyrexian symbols like
  `{W/P}`) contribute; generic, colorless and `X` costs contribute
  nothing. Cards whose converted text has no `mana cost:` line
  contribute no colors. For each game, the deck's color set MUST be
  the union of the per-card colors over the deck's contents. For each
  side and each color present in that side's deck, aggregation MUST
  increment four per-color counters in parallel with the primary
  counters: `wins_when_played_with_X`, `wins_when_in_deck_with_X`,
  `losses_when_played_with_X`, `losses_when_in_deck_with_X`.
  Multi-color decks contribute to multiple slices.
- **FR-011**: The system MUST compute, per card, the following labels
  in both raw and shrunk form, where `k = --shrinkage-k`:

  - **`score_play`** (raw): `(wins_when_played@play - losses_when_played@play) / (wins_when_in_deck@play + losses_when_in_deck@play)`. Shrunk: numerator unchanged, denominator `+ k`. Range `[-1, +1]`. Neutral point 0.
  - **`score_draw`** (raw, shrunk): same form as `score_play` over the `@draw` slice.
  - **`played_rate`** (raw): `(wins_when_played + losses_when_played) / (wins_when_in_deck + losses_when_in_deck)`. Shrunk: numerator `+ k/2`, denominator `+ k`. Range `[0, 1]`. Neutral point 0.5.
  - **`cast_lift`** (raw): `wins_when_played / (wins_when_played + losses_when_played) - wins_when_not_played / (wins_when_not_played + losses_when_not_played)`. Shrunk: each rate gets `+ k/2` in its numerator and `+ k` in its denominator, then subtracted. Range `[-1, +1]`. Neutral point 0.
  - **`color_lift_X`** for each `X ∈ {W, U, B, R, G}` (raw): `(wins_when_played_with_X - losses_when_played_with_X) / (wins_when_in_deck_with_X + losses_when_in_deck_with_X) - (wins_when_played - losses_when_played) / (wins_when_in_deck + losses_when_in_deck)`. Shrunk: each conditional and overall score is computed with `+ k` in its denominator, then subtracted. Range `[-1, +1]`. Neutral point 0.

  With `k = 0`, every shrunk label reduces to its raw form (subject to
  divide-by-zero exclusions in FR-012).
- **FR-012**: Cards with zero total in-deck observations
  (`wins_when_in_deck + losses_when_in_deck == 0`) MUST be excluded
  from the training label map entirely. For cards that are included,
  cells whose slice denominator is zero (e.g.,
  `wins_when_in_deck@play + losses_when_in_deck@play == 0` for
  `score_play`, or `wins_when_in_deck_with_X + losses_when_in_deck_with_X == 0`
  for `color_lift_X`) MUST contribute zero loss for that head, not a
  zero label. The card's other heads continue to receive gradient.
- **FR-013**: Aggregation MUST run inline at train start. The system
  MUST NOT expose a separate aggregation subcommand. The per-card
  label map is not cached for cross-run reuse, but a human-readable
  inspection file is emitted (see FR-013a).
- **FR-013a**: After aggregation completes (and before training
  begins), `train-encoder` MUST write the entire per-card label map
  to `output/sealed/cards-win-rates.txt`. The file MUST start with
  one header row naming each column in the order given below,
  followed by one data row per card included in the training label
  map (i.e., excluding cards filtered out per FR-012), sorted by
  `shrunk_score_play` descending. The file MUST be semicolon-separated,
  with no trailing `;`. The header row and each data row MUST
  contain, in this order:

  ```
  card_name;
  wins_when_played; wins_when_in_deck; losses_when_played; losses_when_in_deck;
  raw_score_play; shrunk_score_play;
  raw_score_draw; shrunk_score_draw;
  raw_played_rate; shrunk_played_rate;
  raw_cast_lift; shrunk_cast_lift;
  raw_color_lift_W; shrunk_color_lift_W;
  raw_color_lift_U; shrunk_color_lift_U;
  raw_color_lift_B; shrunk_color_lift_B;
  raw_color_lift_R; shrunk_color_lift_R;
  raw_color_lift_G; shrunk_color_lift_G
  ```

  Cells whose slice denominator is zero MUST be written as the empty
  string in both raw and shrunk columns (the cell contributes no
  training signal and no neutral-point fill is meaningful). The file
  MUST be overwritten on every `train-encoder` run. The path is fixed
  and not configurable via a flag — its purpose is consistent
  debuggability of SC-005, not user-driven dataset export.

#### Encoder training

- **FR-014**: A new subcommand `python -m sealed train-encoder` MUST
  train a model composed of (a) a token encoder (learned token
  embedding table plus positional encoding, no cross-token mixing),
  (b) a card encoder (a stack of `--n-layers` transformer encoder
  blocks with self-attention and FFN, followed by a single pool
  layer), (c) five regression head families (`score_play`,
  `score_draw`, `played_rate`, `cast_lift` — each a linear
  projection to a scalar — plus `color_lift` — a single linear
  projection to five scalars, one per WUBRG color), and (d) a masked
  language modeling (MLM) head (a linear projection from each
  contextualized token vector to vocab-size logits).
- **FR-014a**: Before tokenizing, `train-encoder` MUST strip the
  `name:` line from each converted card file, matching the input
  transformation applied by `python -m sealed encode-cards` at
  inference time. This guarantees that the encoder is trained on the
  same input shape it sees at inference and prevents the model from
  shortcutting on `name → label` (which would generalize poorly across
  the card-level train/val split). Card-name identity is still
  required during aggregation to map labels to cards, but it MUST NOT
  enter the tokenized input.
- **FR-014b**: At each forward pass, approximately `--mlm-mask-prob`
  fraction of each card's non-special tokens MUST be replaced with the
  reserved `[MASK]` token (FR-009a) before the masked sequence enters
  the token encoder. Masking is randomized per training step; the same
  card sees different masks across epochs.
- **FR-015**: The pool layer MUST produce the card vector by
  concatenating the output of two parallel operations over the
  contextualized token sequence: a multi-query attention pool with
  `--n-pool-queries = K` learned queries cross-attending to the
  tokens, where each query outputs `d_token / K` dimensions and the K
  query outputs are concatenated to a single `d_token`-dim vector;
  and an element-wise max pool across the token sequence (output
  dims = `d_token`). The two halves are concatenated to produce a
  card vector of dimension `d_card = 2 * d_token` (with `d_token =
  256`, `d_card = 512`). `d_card` is independent of `K`; `K` controls
  per-query capacity, not the card-vector size. `d_token` MUST be
  divisible by `K`.
- **FR-015a**: The MLM head MUST read the contextualized token
  sequence (output of the transformer-layer stack, before the pool
  layer) and project each token vector to vocab-size logits via a
  single linear layer with shape `(d_token → vocab_size)`. Loss MUST
  be computed only at masked positions; unmasked positions MUST
  contribute zero MLM gradient.
- **FR-016**: The encoder, all regression heads, and the MLM head
  MUST be trained jointly from random initialization. The system MUST
  NOT initialize any component from the price-predictor transformer,
  since the goal is an encoder shaped entirely by the per-card
  play-data signals defined in this spec.
- **FR-017**: Each per-card output activation MUST match its label's
  range: `tanh` for `score_play`, `score_draw`, `cast_lift`, and each
  `color_lift_X`; `sigmoid` for `played_rate`. The training loss MUST
  be:

  ```
  L_reg  = MSE(score_play) + MSE(score_draw) + MSE(played_rate) + MSE(cast_lift)
         + (1/5) · Σ_{X ∈ WUBRG} MSE(color_lift_X)
  L_mlm  = mean over masked positions of CE(token_logits, true_token)
  loss   = L_reg + (--mlm-weight) · L_mlm
  ```

  Per-card MSE terms MUST be sample-weighted by the per-head weights
  defined in FR-017a; cells where the slice denominator is zero MUST
  contribute zero loss. Within each training batch, each head's
  per-card weighted MSE contributions MUST be normalized to sum to
  one (i.e., divided by that head's total sample weight in the batch
  before being summed into `L_reg`), producing a weighted average per
  head. This decouples loss magnitude from how many high-weight
  cards happen to land in a given batch and keeps `--lr` and
  `--mlm-weight` stable across batches with varying weight totals.
  If a head's total sample weight in a batch is zero (no card
  contributes to that head), that head's term in `L_reg` is zero
  for that batch.
- **FR-017a**: Per-head sample weights MUST be:

  - `weight_score_play  = n_in_deck@play / (n_in_deck@play + k)`
  - `weight_score_draw  = n_in_deck@draw / (n_in_deck@draw + k)`
  - `weight_played_rate = n_in_deck      / (n_in_deck      + k)`
  - `weight_cast_lift   = m / (m + k)`, where `m = min(played_total, n_in_deck − played_total)`
  - `weight_color_lift_X = n_in_deck_with_X / (n_in_deck_with_X + k)`

  with `k = --shrinkage-k`, `n_in_deck_<slice>` defined as the sum of
  the corresponding wins-in-deck and losses-in-deck counters over the
  slice, and `played_total = wins_when_played + losses_when_played`.
- **FR-018**: The train/val split MUST be at the card level (a
  held-out card never appears in the training set) and stratified by
  `score_play` quartile so each split covers the full
  winning-influence range. The held-out fraction MUST be 20%. Cards
  whose `score_play` cell is empty (zero `@play` denominator) MUST be
  assigned to a stratum based on whichever non-empty signed-head cell
  the card carries, falling back to a single catch-all stratum if
  none.
- **FR-019**: The best checkpoint MUST be selected by full
  validation loss (`L_reg + (--mlm-weight) · L_mlm`, evaluated on the
  held-out card-level val set per FR-017). Early stopping MUST
  trigger after `--patience` consecutive epochs without a new best
  full validation loss. The MLM term is included in both signals;
  the auxiliary objective is treated as part of the encoder's
  optimization target, not as a discardable regularizer.
- **FR-020**: After training completes, the system MUST save the
  token-encoder and card-encoder weights to
  `models/sealed/encoder/{timestamp}.pt` and update
  `models/sealed/encoder/latest.pt` to point at (or be a copy of) the
  best-by-val-loss checkpoint. The regression heads' weights and the
  MLM head's weights MUST NOT be written to either file.
- **FR-021**: `train-encoder` MUST accept the following CLI flags
  with the documented defaults: `--cards-played-path` (default
  `output/sealed/cards-played.txt`), `--cards-folder` (default
  `output/cardsfolder/`), `--vocab-path` (default
  `models/sealed/encoder/vocab.txt`), `--model-output` (default
  `models/sealed/encoder/`), `--batch-size` (64), `--epochs` (100),
  `--lr` (1e-4), `--patience` (20), `--dropout` (0.1), `--n-layers`
  (6), `--n-heads` (4), `--n-pool-queries` (4), `--shrinkage-k` (20),
  `--mlm-weight` (0.1), `--mlm-mask-prob` (0.15).
- **FR-022**: The following knobs MUST be hardcoded constants in
  `train-encoder`: `d_model = 256`, `ff_dim = 1024`, loss formula =
  weighted sum per FR-017, `val_split = 0.2`, stratification =
  `score_play` quartiles, seed = 42, optimizer = AdamW with
  per-parameter-group max-norm 1.0 gradient clipping (matching the
  train-scorer convention in spec 015), learning-rate schedule =
  linear warmup over the first 5% of `--epochs × batches_per_epoch`
  scheduled steps followed by constant `--lr` for the remainder (no
  decay). `max_seq_len` MUST be computed from the corpus at train
  start (longest card length rounded up to a multiple of 8).
- **FR-023**: `train-encoder` MUST fail with a clear error message
  pointing the user at the corrective command if (a) the vocabulary
  file is missing or does not contain a reserved `[MASK]` token
  (point at `build-vocab`), (b) `cards-played.txt` is missing or
  empty (point at `match-outcomes`), or (c) the corpus folder is
  empty (point at `convert`).
- **FR-023d**: After aggregation, `train-encoder` MUST check
  every observed card name against the corpus folder. Card names with
  no corresponding `.txt` MUST be *reported* — a warning naming the
  missing cards (capped at a reasonable display count, with the total
  reported) and pointing the user at `python -m price_predictor
  convert` — and then dropped from the label map, the train/val split,
  and the dataset. Missing cards MUST NOT block the run; training
  proceeds with the remaining cards.

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
  Lives at `output/sealed/cards-played.txt`. Each line carries the parent
  match's `timestamp` and `run_id`, so a single supervisor's lines are
  joinable to `match-outcomes.txt` by `run_id`, and a match's game lines
  appear contiguously in game order within its `run_id` group.
- **Per-card label map**: An in-memory dict from card name to a tuple
  of nine scalar labels (`score_play`, `score_draw`, `played_rate`,
  `cast_lift`, and `color_lift_X` for each of WUBRG), built at train
  start by aggregating `cards-played.txt`. Cards with zero total
  in-deck observations are absent from the map. Cells whose slice
  denominator is zero are present-but-empty and contribute zero loss
  during training. Not cached across runs.
- **`cards-win-rates.txt`**: Human-readable snapshot of the per-card
  label map, written at train start to
  `output/sealed/cards-win-rates.txt`. Begins with one header row
  naming each column, followed by one data row per card included in
  training, sorted by `shrunk_score_play` descending.
  Semicolon-separated; per-card columns are `card_name`, the four
  primary counters, and raw + shrunk pairs for each of the nine
  regression labels (see FR-013a for the full schema). Overwritten
  on every `train-encoder` run. Lets the user verify SC-005 by
  diffing two runs with different `--shrinkage-k` values.
- **Sealed vocabulary**: A token list at
  `models/sealed/encoder/vocab.txt`, one token per line, including a
  reserved `[MASK]` token used by the MLM auxiliary loss. Built from
  the converted card corpus by `python -m sealed build-vocab`.
  Independent from the price-predictor vocabulary; updating one does
  not update the other.
- **Sealed encoder checkpoint**: The token-encoder + card-encoder
  weights saved at `models/sealed/encoder/{timestamp}.pt` plus a
  `latest.pt` pointer (or copy). The regression heads and the MLM
  head used during training are not part of this artifact. Default
  `--encoder-checkpoint` source for `train-scorer` and `encode-cards`
  after this feature ships.
- **Token encoder**: Learned token embedding table plus positional
  encoding, mapping a token ID and position to a `d_token`-dim vector.
  No cross-token mixing. Vocabulary includes the reserved `[MASK]`
  token used during training.
- **Card encoder**: A stack of `N` transformer encoder blocks
  (self-attention + FFN + residual) followed by a single pool layer
  that concatenates a multi-query attention pool with an element-wise
  max pool. Produces a single `d_card`-dim vector per card. The
  contextualized token sequence (output of the transformer-layer
  stack, before the pool) is also consumed by the MLM head during
  training.
- **Regression heads**: Five training-only output families projecting
  the card vector to scalar (or 5-vector for `color_lift`) targets,
  each with a range-matched activation (`tanh` for the four signed
  families, `sigmoid` for `played_rate`). Trained jointly with the
  encoder against the weighted MSE loss in FR-017 and discarded after
  training. Not saved to disk.
- **MLM head**: A single linear projection from each contextualized
  token vector (shape `d_token`) to vocab-size logits, training-only.
  Operates over masked positions in the input token sequence; loss is
  cross-entropy against the original (pre-mask) token. Discarded
  after training. Not saved to disk.

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
- **SC-005**: The `--shrinkage-k` flag changes the per-card label map
  in a way the user can verify by inspecting
  `output/sealed/cards-win-rates.txt` produced at the start of two
  `train-encoder` runs differing only in `--shrinkage-k` (e.g., `0` vs
  `20`): for every regression head, low-observation cards (e.g., a
  card with two in-deck games) shift visibly between runs while
  high-observation cards' shrunk labels remain within a few
  thousandths of their raw counterparts.
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
- Bayesian shrinkage and per-head sample weighting (both driven by the
  same `--shrinkage-k`) together comprise the low-n regularization
  mechanism. Shrinkage pulls noisy labels toward each head's neutral
  point; sample weighting reduces the loss contribution of cells whose
  slice denominator is small. Both are always-on; no separate flags
  toggle them.
- The `d_token = 256`, `d_card = 512` split mirroring the price-predictor
  is acceptable; the goal is to produce a drop-in replacement so
  downstream tools that expect `2 * d_model = 512` encoder features
  continue to work without modification.
- The encoder is trained from random init in every run. There is no
  resume capability for `train-encoder`; an interrupted run is restarted
  from scratch. (A `--resume` flag is a future extension if training
  durations grow long enough to justify it.)
- Vocab freshness is the user's responsibility. `train-encoder` does not
  validate that `vocab.txt` covers the current corpus or
  `cards-played.txt`; tokens introduced after the last `build-vocab` run
  fall back to UNK silently. Rerun `python -m sealed build-vocab` after
  any material corpus change.
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
- **Tied embeddings between the input token table and the MLM head**:
  The MLM head uses its own `Linear(d_token, vocab_size)` projection
  rather than reusing the input embedding matrix transposed. Tying is
  a compatible parameter-saving extension but not part of this
  feature.
- **Per-head shrinkage tuning**: A single `--shrinkage-k` drives
  shrinkage and sample weighting for every head. Per-head `k` values
  are a compatible extension but not part of this feature.
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

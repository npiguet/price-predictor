# Feature Specification: Stage 2 Training — Heuristic Gate

**Feature Branch**: `013-stage2-heuristic-gate`
**Created**: 2026-03-31
**Status**: Draft
**Input**: User description: "The second stage of sealed deck picking follow the section 2 of the Training curriculum described in sealed-deck-picker.md"

## Clarifications

### Session 2026-03-31

- Q: When Stage 2 initializes from a Stage 1 checkpoint via `--init-from`, what state carries over? → A: Model weights only — fresh optimizer, episode_count = 0.
- Q: For multi-face cards (transform, split, adventure), how should pip counting handle their mana costs? → A: All faces — count pips from every `mana cost:` line on the card.

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Launch Stage 2 Training (Priority: P1)

A researcher who has completed Stage 1 training (legal pick gate) wants to advance the model to learn playable mana
bases. They invoke a single command that loads the Stage 1 checkpoint, switches to the heuristic mana-score reward
function, and begins running episodes where the model is graded on how well its land choices match the mana
requirements of its spells.

**Why this priority**: This is the core deliverable — the Stage 2 training loop with the new reward function. Without
it, none of the other stories are meaningful.

**Independent Test**: Can be fully tested by running the train command with `--stage 2` against a small dataset and
observing that episodes complete all 40 picks, the heuristic mana score is computed, and the model checkpoint is
written to disk.

**Acceptance Scenarios**:

1. **Given** a completed Stage 1 checkpoint exists and card embeddings and pools are present, **When** the user runs
   `python -m sealed train --stage 2`, **Then** the Stage 1 checkpoint is loaded as the starting point and training
   begins with the heuristic mana-score reward.

2. **Given** Stage 2 training is running, **When** an episode completes all 40 picks without a duplicate, **Then** the
   heuristic mana score is computed from the full deck and the same reward value is assigned uniformly to all 40 steps.

3. **Given** Stage 2 training is running, **When** the model re-selects an already-picked booster card before
   completing 40 picks, **Then** the episode terminates immediately and the Stage 1 per-step reward scheme is applied
   with `best_run = 40` (each prior pick receives +1 or -1 based on spell/land budgets; the duplicate pick receives
   -1 advantage).

4. **Given** Stage 2 training is running, **When** a batch of 32 episodes all complete 40 picks without duplicates
   and all achieve a mana score above 0.90, **Then** Stage 2 is reported as complete and training halts.

5. **Given** Stage 2 training is running, **When** a batch completes but not all 32 episodes meet the completion
   criteria, **Then** training continues with the next batch.

---

### User Story 2 — Heuristic Mana Score Computation (Priority: P1)

The system needs to evaluate how well a completed 40-card deck's land base matches the mana requirements of its spells.
This scoring mechanism is the heart of Stage 2 and must correctly count pips, compute the ideal mana distribution, count
actual sources, and produce the final score.

**Why this priority**: The mana score is the reward signal that drives all of Stage 2 training. If it is incorrect,
training will learn the wrong thing. Co-equal with Story 1 since neither works without the other.

**Independent Test**: Can be tested independently by constructing known decks with predictable mana distributions and
verifying the score matches hand-calculated expected values.

**Acceptance Scenarios**:

1. **Given** a deck with 23 spells all costing only {W} mana and 17 Plains, **When** the mana score is computed,
   **Then** the score is 1.0 (perfect match).

2. **Given** a deck with spells requiring {W} and {U} mana in a 2:1 ratio and an appropriately distributed land base
   of 17 lands, **When** the mana score is computed, **Then** the ideal distribution reflects the 2:1 pip ratio plus
   the 2-source minimum per color, and the score reflects how closely actual sources match ideal.

3. **Given** a deck with 15 lands instead of 17, **When** the mana score is computed, **Then** the score is penalised
   by the land-count deviation (|15 - 17| = 2 contributes to the L1 error denominator).

4. **Given** a deck containing a dual land that produces both {W} and {U}, **When** actual sources are counted,
   **Then** the dual land contributes +1 to both the W and U actual source counts.

5. **Given** a deck containing spells with Phyrexian mana costs like {W/P}, **When** pips are counted, **Then**
   Phyrexian pips contribute +0.5 to their color.

6. **Given** a deck containing spells with hybrid mana costs like {G/R}, **When** pips are counted, **Then**
   hybrid pips contribute +0.5 to each of their two colors.

7. **Given** a deck containing spells with generic mana costs like {2} or {X}, **When** pips are counted, **Then**
   generic mana is ignored entirely.

8. **Given** a deck with spells requiring {C} (colorless) mana, **When** pips are counted, **Then** {C} is tracked
   as a sixth color alongside W/U/B/R/G.

---

### User Story 3 — Inspect Stage 2 Sample Picks (Priority: P2)

A researcher wants to see what the model is picking during Stage 2 training and how well the mana base matches the
spell requirements, to get a qualitative sense of training progress.

**Why this priority**: Diagnostic visibility is important for understanding model behaviour during training, but it
does not block the training loop itself.

**Independent Test**: Can be tested independently by running the sample command with `--stage 2` after any Stage 2
checkpoint exists, and verifying it prints human-readable pick sequences alongside mana analysis.

**Acceptance Scenarios**:

1. **Given** a Stage 2 model checkpoint exists, **When** the user runs `python -m sealed sample --stage 2`,
   **Then** the command prints N deck selections from random pools.

2. **Given** a sample deck is printed, **When** the output is displayed, **Then** it shows the 40 picks, the ideal
   vs actual mana source distribution per color, and the heuristic mana score.

---

### User Story 4 — Resume Stage 2 from Checkpoint (Priority: P2)

A researcher whose Stage 2 training run was interrupted wants to resume from the most recent checkpoint.

**Why this priority**: Training runs can take hours. Resuming is essential for practical use. This story follows the
same pattern established in Stage 1.

**Independent Test**: Can be tested by running Stage 2 training briefly, stopping it, then re-launching and verifying
training resumes from the saved state.

**Acceptance Scenarios**:

1. **Given** a Stage 2 checkpoint exists at the configured model path, **When** the user runs
   `python -m sealed train --stage 2` again, **Then** training resumes from the checkpoint, preserving episode count.

2. **Given** Stage 2 training is running, **When** each training batch completes, **Then** the latest model state is
   saved to the `latest.pt` file at the model path.

3. **Given** Stage 2 training is running, **When** 1000 episodes have elapsed, **Then** a timestamped checkpoint is
   saved to the `checkpoints/` subfolder of the model path's parent directory.

---

### Edge Cases

- What happens when a deck contains zero spells (all 40 picks are lands)? The pip count is zero for all colors, ideal
  distribution is zero for all colors, and the score reflects only the land-count penalty (|40 - 17| = 23).
- What happens when a deck contains zero lands (all 40 picks are spells)? Actual sources are zero for all colors,
  and the L1 error is the full ideal distribution plus the land-count penalty (|0 - 17| = 17).
- What happens when a deck uses only colorless spells ({C} pips or generic-only costs)? The ideal distribution
  allocates all 17 sources to colorless, and the score reflects how well the land base provides colorless mana.
- What happens when the Stage 1 checkpoint is missing or invalid? The system reports an error and does not begin
  training.
- What happens when a card's Oracle text has no mana cost (e.g. lands in the non-land card list due to data issues)?
  Cards with no mana cost contribute zero pips to the count.
- What happens when the model makes a duplicate pick on pick 1 (the very first pick)? The episode terminates with
  zero completed picks and the duplicate pick receives -1 advantage. No mana score is computed.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST load a Stage 1 checkpoint as the starting point for Stage 2 training via the `--init-from`
  parameter. Only the pool transformer model weights are loaded; the optimizer state and training state (episode count)
  are NOT carried over — Stage 2 begins with a fresh optimizer and episode_count = 0.
- **FR-002**: System MUST keep the card encoder frozen during Stage 2 training (only the pool transformer is trained).
- **FR-003**: System MUST terminate an episode immediately if the model re-selects an already-picked booster card
  (duplicate pick). In this case, the Stage 1 per-step reward scheme is applied with `best_run = 40`: each prior
  pick receives +1 or -1 based on spell/land budgets (spell < 23 → +1, spell ≥ 23 → -1; land < 17 → +1,
  land ≥ 17 → -1), and the duplicate pick receives -1 advantage.
- **FR-004**: System MUST compute the heuristic mana score at the end of each episode that completes all 40 picks
  without a duplicate.
- **FR-005**: For episodes that complete all 40 picks, system MUST assign the mana-score reward uniformly to all
  40 steps (reflecting that every pick contributed equally to the deck's mana base quality).
- **FR-006**: System MUST count mana pips from every `mana cost:` line of each non-land card in the deck (see spec
  006-card-script-parsing for the card file format). For multi-face cards (transform, split, adventure), pips from
  all faces are counted. Each pip symbol is counted independently: single-color pips ({W}, {U}, {B}, {R}, {G})
  count +1.0 to that color, {C} counts +1.0 to colorless (tracked as a sixth color), Phyrexian pips ({W/P}, etc.)
  count +0.5 to their color, hybrid pips ({G/R}, etc.) count +0.5 to each of their two colors, and generic mana
  ({1}, {2}, {X}, etc.) is ignored. A card with `mana cost: {W}{W}{G}` contributes 2.0 white and 1.0 green.
- **FR-007**: System MUST compute the ideal mana source distribution targeting 17 total sources, with a mandatory
  minimum of 2 sources per color present in the deck and the remaining sources distributed proportionally to pip counts.
- **FR-008**: System MUST count actual mana sources by examining each land's `activated[N]: {T}: Add ...` lines in
  the card file (see spec 006-card-script-parsing). Each such ability contributes +1 to `actual_c` for each color
  symbol that appears in its "Add" clause. Dual lands (e.g., `{T}: Add {W} or {U}`) count as one source for each
  of their colors.
- **FR-009**: System MUST compute the score as `max(0.0, 1.0 - (l1_error + |n_lands - 17|) / 17.0)` where `l1_error`
  is the sum of absolute differences between actual and ideal source counts per color.
- **FR-010**: System MUST convert the score to the reward range [-1, 1] via `reward = 2 * score - 1`.
- **FR-011**: System MUST apply standard PPO advantage normalization across all steps in the batch.
- **FR-012**: System MUST declare Stage 2 complete when all 32 episodes in a batch complete 40 picks without
  duplicates and all achieve score > 0.90.
- **FR-013**: System MUST save the model to `model-path` after each training batch and save a timestamped checkpoint
  every 1000 episodes to the `checkpoints/` subfolder.
- **FR-014**: System MUST support resuming Stage 2 training from a previously saved checkpoint.
- **FR-015**: System MUST provide a sample command (`python -m sealed sample --stage 2`) that displays deck picks
  alongside ideal vs actual mana source distributions and the heuristic score.
- **FR-016**: System MUST reshuffle the non-basic-land portion of the pool before each pick step, consistent with
  Stage 1 behaviour.

### Key Entities

- **Pip Count**: Per-color tally of mana requirements across all non-land cards in a deck. Tracks W, U, B, R, G,
  and C (colorless) as six independent dimensions.
- **Ideal Source Distribution**: The target number of mana sources per color, computed from pip counts with a
  2-source-per-color floor and proportional allocation of the remaining sources out of 17 total.
- **Actual Source Count**: Per-color tally of mana-producing lands in the deck, derived from each land's
  "{T}: Add ..." abilities.
- **Mana Score**: A value in [0.0, 1.0] measuring how closely the actual source distribution matches the ideal,
  penalising both distribution mismatch and land-count deviation from 17.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The heuristic mana score computation produces correct results for all pip types (single, Phyrexian,
  hybrid, generic, colorless) as verified by unit tests with hand-calculated expected values.
- **SC-002**: A model trained through Stage 2 consistently produces decks where the land base reasonably matches
  the mana requirements of the spells, as evidenced by batch-wide score > 0.90 at completion.
- **SC-003**: Stage 2 training converges — the mean batch score trends upward over time and eventually meets the
  completion criterion (all 32 episodes scoring > 0.90 in a single batch).
- **SC-004**: The sample command output clearly shows per-color ideal vs actual source distributions, enabling
  a researcher to visually assess mana base quality.
- **SC-005**: Training can be interrupted and resumed without loss of progress — episode count and model state
  are preserved across restarts.

## Assumptions

- A completed Stage 1 checkpoint is available before Stage 2 begins.
- Card files follow the format defined in spec 006-card-script-parsing: mana costs are on the `mana cost:` line
  using standard notation ({W}, {U}, {B}, {R}, {G}, {C}, {W/P}, {G/R}, {1}, {2}, {X}, etc.), and land mana
  production is on `activated[N]: {T}: Add ...` lines.
- Land cards can be identified from their `types:` line (containing "land") and their mana-producing abilities
  are reliably present as activated abilities in the card file. Basic lands always have an implicit mana ability
  generated by the card converter (see spec 006 FR-020).
- The existing pool assembly, shuffling, PPO training infrastructure, and duplicate-pick termination from Stage 1
  are reused. Stage 2 adds the heuristic mana-score reward for completed episodes and changes the sample output
  format.
- Checkpointing follows the same conventions established in Stage 1 (latest.pt + timestamped checkpoints every
  1000 episodes).

# Feature Specification: Stage 1 Training — Legal Pick Gate

**Feature Branch**: `012-sealed-stage1-training`
**Created**: 2026-03-28
**Status**: Draft
**Input**: User description: "New operations are added to the sealed python module to start the first steps of training. The stage 1 training feature is described in details in specs/sealed-deck-picker.md. This feature (and its plan an implementation) aims to implement what is described in that file as Stage 1 of the Training Curriculum."

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Launch Stage 1 Training (Priority: P1)

A researcher with a pre-generated card embeddings folder and pools file wants to initialize and run the Stage 1
training loop. They invoke a single command that sets up the pool transformer, PPO training loop, and begins running
episodes against the sealed pool dataset.

**Why this priority**: This is the core deliverable — without a working training loop there is nothing else to build on.
All other stories assume this one is working.

**Independent Test**: Can be fully tested by running the train command against a small dataset and observing that
episodes execute, rewards are computed, and the model checkpoint is written to disk.

**Acceptance Scenarios**:

1. **Given** card embeddings and a pools file are present, **When** the user runs `python -m sealed train --stage 1`,
   **Then** the training loop begins, episodes execute sequentially, and the model is saved at the configured path.

2. **Given** training is running at curriculum level `best_run`, **When** the model makes an illegal pick
   before completing `best_run` picks, **Then** the episode terminates immediately and the episode-level reward
   is computed as `(effective_run / best_run) × 2 - 1` where `effective_run = n_total - max(n_spell - 23, 0)`.

3. **Given** training is running at curriculum level `best_run`, **When** the model completes `best_run` picks
   without any illegal selection, **Then** the episode is recorded as successful with `n_total = best_run`.

4. **Given** a training batch of 32 episodes all succeeded at the current `best_run` level, **When** `best_run`
   was already 40, **Then** the training loop reports completion of Stage 1 and halts.

5. **Given** a training batch of 32 episodes all succeeded at the current `best_run` level, **When** `best_run`
   is less than 40, **Then** `best_run` advances by 1 and subsequent episodes are limited to the new level.

---

### User Story 2 — Resume Training from Checkpoint (Priority: P2)

A researcher whose training run was interrupted (by a crash, deliberate stop, or machine restart) wants to continue
from the most recent checkpoint rather than starting over.

**Why this priority**: Training runs can take hours. Resuming from a checkpoint is essential for practical use and
prevents wasted compute.

**Independent Test**: Can be tested by running training for a short time, stopping it, then re-launching with the same
model-path and verifying that training resumes from the saved state rather than from scratch.

**Acceptance Scenarios**:

1. **Given** a model checkpoint exists at the configured path, **When** the user runs `python -m sealed train --stage 1`
   again, **Then** training resumes from the checkpoint, preserving `best_run` and `episode_count`.

2. **Given** training is running, **When** 1000 episodes have elapsed, **Then** a timestamped checkpoint is saved to
   the `checkpoints/` subfolder of the model path's parent directory.

3. **Given** training is running, **When** each training batch completes, **Then** the latest model state is saved to
   the `latest.pt` file at the model path.

---

### User Story 3 — Inspect Current Model Picks (Priority: P3)

A researcher wants to inspect what the model is currently picking from real pools to get a qualitative sense of
training progress without having to read raw metrics.

**Why this priority**: Diagnostic visibility is important for a training experiment, but it is not blocking. It can be
added after the training loop is proven to work.

**Independent Test**: Can be tested independently by running the sample command after any checkpoint exists, and
verifying that it prints human-readable pick sequences.

**Acceptance Scenarios**:

1. **Given** a trained or partially trained model checkpoint exists, **When** the user runs
   `python -m sealed sample`, **Then** the command prints N pick sequences drawn from random pools, each showing the
   selected card names in pick order.

2. **Given** a sample is generated where the model made an illegal pick before completing 40 picks, **Then** the output
   also reports how many legal picks were made before the first illegal pick.

3. **Given** a sample is generated where the model completed all 40 picks legally, **Then** the run is reported as a
   success with 40/40 legal picks.

---

### Edge Cases

- What happens when the pools file is empty or contains fewer entries than one full pool? → Training aborts at startup with a clear error message.
- What happens when a card named in a pool has no corresponding embedding file in cards-path? → Training aborts immediately with an error message identifying the missing card(s).
- What happens when the model path does not exist yet (first run, no checkpoint)? → The model-path directory tree is created automatically.
- What happens when training is interrupted mid-episode (e.g. keyboard interrupt between pick steps)?
- What happens when `best_run` is 1 and the episode terminates on the very first pick? (0 legal picks: `effective_run = 0`, reward = −1.0. If the pick is legal: reward = 1.0. The model must learn to make a single legal pick before the curriculum advances.)

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The sealed module MUST expose a `train` subcommand accepting `--stage`, `--set`, `--pools-path`,
  `--cards-path`, `--model-path`, and `--batch-size` arguments, with defaults matching the Stage 0 conventions
  (`--batch-size` defaults to 32).
- **FR-002**: When `--stage 1` is passed, the system MUST validate at startup that pools.txt exists and contains at
  least one valid pool entry, aborting with a clear error message if not. The system then initializes a pool-level
  transformer model from scratch when no checkpoint exists, or resumes from the checkpoint at model-path when one is
  present. The model-path directory tree is created automatically if it does not exist. On resume, the sequential
  pool iteration always restarts from pool 0; pool position is not persisted.
- **FR-003**: Each training episode MUST draw the next pool from the dataset sequentially, looping back to the first
  pool once all have been used. The episode then appends the 6 basic land slot embeddings, fills any empty slots with
  zero vectors, and reshuffles the non-basic-land portion of the pool before each pick step.
- **FR-004**: Each of the 96 pool slots MUST be represented as a 520-dimensional feature vector: a 512-dimensional
  card embedding, a `pick_count` (0/1 for booster cards, 0..N for basic land slots), an `available_flag` (cleared
  after first pick for booster cards; always 1 for basic land slots), an `is_land` flag, and 5 reserved padding
  dimensions (always 0).
- **FR-005**: During each pick step, the model MUST sample from the unmasked distribution over all 96 slots. No
  selection mask is applied to the logits — the model is free to pick any slot, including already-picked ones. The
  `available_flag = 0` on picked slots serves as an input feature (context for the transformer) but does NOT block
  selection. This is intentional: Stage 1 teaches the model to avoid illegal picks through the reward signal, not by
  making them mechanically impossible.
- **FR-006**: When the model selects an already-picked non-basic-land slot, the episode MUST terminate immediately.
- **FR-007**: Each episode MUST run for at most `best_run` picks. The episode-level reward MUST be computed as
  `(effective_run / best_run) × 2 - 1`, where `effective_run = n_total - max(n_spell - 23, 0)`. Per-step rewards
  are +1 for each legal pick and −1 for each non-land pick made in Phase 2 (n_spell ≥ 23). Per-step rewards are
  batch-normalised (subtract batch mean, divide by batch std) before computing PPO advantages. An entropy bonus
  with coefficient 0.01 is added to the PPO objective to encourage exploration. PPO clip ε = 0.2.
- **FR-008**: *(removed — replay buffer replaced by standard on-policy PPO)*
- **FR-009**: *(removed — KL divergence monitoring for stale buffer entries no longer applicable)*
- **FR-016**: After each training batch completes, the system MUST print one summary line to stdout containing: total
  episode count, the `current_run` value for each episode in the batch, the current `best_run`, and the mean reward
  across the batch.
- **FR-010**: The system MUST save the model state to model-path at the end of each training batch (every
  `--batch-size` episodes), and MUST save a timestamped checkpoint to the `checkpoints/` subfolder of the model
  path's parent every 1000 episodes.
- **FR-011**: `best_run` MUST advance by 1 after any training batch in which all 32 episodes completed `best_run`
  picks without an illegal pick. The training loop MUST halt and report Stage 1 completion when `best_run` reaches
  40 and a full batch succeeds at that level.
- **FR-012**: The pool transformer MUST use the architecture specified in the sealed-deck-picker design: 8 transformer
  layers, 8 attention heads, model dimension 520, feed-forward dimension 2048.
- *(FR-013 intentionally removed during spec refinement — numbering preserved for traceability)*
- **FR-014**: The sealed module MUST expose a `sample` subcommand accepting `--set`, `--pools-path`, `--cards-path`,
  `--model-path`, and `--n-samples` arguments, with sensible defaults.
- **FR-015**: The `sample` command MUST load the specified checkpoint, run N pick sequences from randomly chosen pools,
  and print each sequence as an ordered list of card names plus the count of legal picks before the first illegal pick
  (or a success report if all 40 picks were legal).

### Key Entities

- **Pool**: A list of ~84–90 card names from the pre-generated dataset, assembled at runtime by loading each card's
  embedding file. Augmented with 6 basic land slots and zero-padded to 96 entries.
- **Episode**: A single 40-step pick sequence over one pool. Stores the pool card names, the shuffle seed used at each
  pick step, the actions taken, the log-probabilities of those actions, and the final reward.
- **best_run**: The current curriculum level — the maximum number of picks allowed per episode. Starts at 1 and
  advances by 1 each time a full batch (32 episodes) completes with every episode reaching `best_run` picks without
  an illegal pick. Used as the denominator in the reward function and persisted across training restarts.
- **Model Checkpoint**: The serialized state of the pool transformer, saved periodically to allow
  resumption and comparison across training stages.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Training can be launched from the command line with a single command and no code changes required.
- **SC-002**: A fresh training run from scratch reliably progresses: the model's `best_run` curriculum level
  increases over time, demonstrating that the model is learning to make more consecutive legal picks.
- **SC-003**: The training loop halts automatically and reports Stage 1 completion once a full batch of 32 episodes
  all succeed at `best_run = 40`, with no manual intervention needed.
- **SC-004**: An interrupted training run can be resumed from the last saved checkpoint, with `best_run` and
  `episode_count` preserved, losing at most one batch of progress.
- **SC-005**: The `sample` command produces human-readable pick sequences that allow qualitative assessment of model
  behavior at any stage of training.
- **SC-006**: Timestamped checkpoints saved every 1000 episodes allow rollback to any prior training state.

## Clarifications

### Session 2026-03-28 (continued)

- Q: How many episodes constitute one training batch (PPO update frequency)? → A: Fixed episode count per batch, configurable via `--batch-size` (default 32 episodes).
- Q: What training progress output is shown on the console? → A: One summary line per batch showing episode count, per-episode current_run values, best_run, and mean batch reward.
- Q: When does `best_run` advance? → A: After any batch of 32 episodes in which every episode completed `best_run` picks without an illegal pick. If even one episode fails, `best_run` stays at its current level.
- Q: What advancement threshold was considered and rejected? → A: A percentage-based threshold (e.g. 95% success rate over a rolling window) was considered but deferred. The full-batch requirement will be evaluated first; if training stalls at a particular level, the threshold can be relaxed based on training logs.
- Q: What happens when a card named in a pool has no corresponding embedding file in cards-path? → A: Abort immediately with a clear error message identifying the missing card(s).
- Q: How are pools sampled from the dataset across episodes? → A: Sequential pass through the dataset, looping back to the start when all pools have been used.
- Q: Is the sequential pool position persisted in the checkpoint and restored on resume? → A: No — pool position is not persisted; resume always restarts the sequential iteration from pool 0.
- Q: What happens when pools.txt is empty or contains no valid pool entries? → A: Abort at startup with a clear error message.
- Q: What happens when the model-path directory does not exist on first run? → A: Create the directory tree automatically.

## Assumptions

- Card embeddings (`.npz` files) for all cards referenced in `pools.txt` are already present in cards-path (generated
  by the Stage 0 `encode-cards` step).
- The `pools.txt` file is already present in pools-path (generated by the Stage 0 `generate-pools` step).
- The pretrained card encoder is frozen during Stage 1; only the pool transformer is trained.
- The `sample` command is read-only — it does not modify the model or replay buffer.
- Default paths: `--set RVR`, `--pools-path output/sealed/pools/{set-code}/`, `--cards-path output/cardsfolder/`,
  `--model-path models/sealed/stage1/latest.pt`, `--n-samples 10`.

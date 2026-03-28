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

2. **Given** training is running, **When** the model makes an illegal pick (selecting a slot already chosen in the
   current episode), **Then** the episode terminates immediately and the reward is computed as
   `(current_run / best_run) × 2 - 1` where `current_run` is the number of legal picks made before termination.

3. **Given** training is running, **When** the model completes all 40 picks without any illegal selection,
   **Then** the episode is recorded as successful and `current_run` equals 40.

4. **Given** training is running and the model achieves 40 legal picks in 100 consecutive episodes, **Then** the
   training loop reports completion of Stage 1 and halts.

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
   again, **Then** training resumes from the checkpoint, preserving `best_run` and replay buffer state.

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

- What happens when the pools file is empty or contains fewer entries than one full pool?
- What happens when a card named in a pool has no corresponding embedding file in cards-path? → Training aborts immediately with an error message identifying the missing card(s).
- What happens when the model path does not exist yet (first run, no checkpoint)?
- What happens when training is interrupted mid-episode (e.g. keyboard interrupt between pick steps)?
- What happens when `best_run` is 1 and the episode terminates on the very first pick? (Reward is `(1/1)×2-1 = 1.0` — intentional per spec.)

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The sealed module MUST expose a `train` subcommand accepting `--stage`, `--set`, `--pools-path`,
  `--cards-path`, `--model-path`, and `--batch-size` arguments, with defaults matching the Stage 0 conventions
  (`--batch-size` defaults to 32).
- **FR-002**: When `--stage 1` is passed, the system MUST initialize a pool-level transformer model from scratch when
  no checkpoint exists, or resume from the checkpoint at model-path when one is present.
- **FR-003**: Each training episode MUST draw the next pool from the dataset sequentially, looping back to the first
  pool once all have been used. The episode then appends the 6 basic land slot embeddings, fills any empty slots with
  zero vectors, and reshuffles the non-basic-land portion of the pool before each pick step.
- **FR-004**: Each of the 96 pool slots MUST be represented as a 516-dimensional feature vector: a 512-dimensional
  card embedding, a `picked_flag`, an `available_flag`, an `is_land` flag, and a `basic_land_count` value.
- **FR-005**: During each pick step, a selection mask MUST be applied to the output logits that prevents re-selection
  of already-picked non-basic-land slots. Basic land slots MUST remain selectable at every step.
- **FR-006**: When the model selects an already-picked non-basic-land slot, the episode MUST terminate immediately.
- **FR-007**: The reward for each episode MUST be computed as `(current_run / best_run) × 2 - 1`, where `current_run`
  is the number of legal picks made (minimum 1) and `best_run` is the high-water mark of all prior runs, initialized
  to 1 and never decremented.
- **FR-008**: The training loop MUST maintain a replay buffer of up to ~1000 episodes stored compactly as: pool card
  names, per-step shuffle seeds, actions taken, log-probabilities of those actions, and episode reward. FIFO eviction
  MUST apply when the buffer is full.
- **FR-009**: The system MUST monitor KL divergence on replay buffer entries to detect stale off-policy episodes.
  When stale entries are detected, the system MUST log a warning to stdout and continue training without modifying
  the buffer.
- **FR-016**: After each training batch completes, the system MUST print one summary line to stdout containing: total
  episode count, the `current_run` value for each episode in the batch, the current `best_run`, and the mean reward
  across the batch.
- **FR-010**: The system MUST save the model state to model-path at the end of each training batch (every
  `--batch-size` episodes), and MUST save a timestamped checkpoint to the `checkpoints/` subfolder of the model
  path's parent every 1000 episodes.
- **FR-011**: The training loop MUST halt and report Stage 1 completion when the model achieves `current_run = 40` in
  100 consecutive episodes.
- **FR-012**: The pool transformer MUST use the architecture specified in the sealed-deck-picker design: 8 transformer
  layers, 8 attention heads, model dimension 516, feed-forward dimension 2048.
- **FR-013**: A linear projection layer (input dimension 512, output dimension 512) MUST sit between the frozen
  pretrained card encoder and the pool transformer input.
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
- **Replay Buffer**: A capped FIFO queue of episodes stored in compact form, used to sample training batches off-policy.
- **best_run**: A high-water-mark counter tracking the longest legal pick run ever achieved. Initialized to 1, never
  decremented. Used as the denominator in the reward function and persisted across training restarts.
- **Model Checkpoint**: The serialized state of the pool transformer and projection layer, saved periodically to allow
  resumption and comparison across training stages.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Training can be launched from the command line with a single command and no code changes required.
- **SC-002**: A fresh training run from scratch reliably progresses: the model's `best_run` counter increases over
  time, demonstrating that the model is learning to make more consecutive legal picks.
- **SC-003**: The training loop halts automatically and reports Stage 1 completion once the model achieves 40
  consecutive legal picks in 100 consecutive episodes, with no manual intervention needed.
- **SC-004**: An interrupted training run can be resumed from the last saved checkpoint, with `best_run` and replay
  buffer preserved, losing at most one batch of progress.
- **SC-005**: The `sample` command produces human-readable pick sequences that allow qualitative assessment of model
  behavior at any stage of training.
- **SC-006**: Timestamped checkpoints saved every 1000 episodes allow rollback to any prior training state.

## Clarifications

### Session 2026-03-28

- Q: How many episodes constitute one training batch (PPO update frequency)? → A: Fixed episode count per batch, configurable via `--batch-size` (default 32 episodes).
- Q: What training progress output is shown on the console? → A: One summary line per batch showing episode count, per-episode current_run values, best_run, and mean batch reward.
- Q: What action is taken when replay buffer entries are detected as stale via KL divergence? → A: Log a warning to stdout; no further action, training continues normally.
- Q: What happens when a card named in a pool has no corresponding embedding file in cards-path? → A: Abort immediately with a clear error message identifying the missing card(s).
- Q: How are pools sampled from the dataset across episodes? → A: Sequential pass through the dataset, looping back to the start when all pools have been used.

## Assumptions

- Card embeddings (`.npz` files) for all cards referenced in `pools.txt` are already present in cards-path (generated
  by the Stage 0 `encode-cards` step).
- The `pools.txt` file is already present in pools-path (generated by the Stage 0 `generate-pools` step).
- The pretrained card encoder is frozen during Stage 1; only the projection layer and pool transformer are trained.
- The `sample` command is read-only — it does not modify the model or replay buffer.
- Default paths: `--set RVR`, `--pools-path output/sealed/pools/{set-code}/`, `--cards-path output/cardsfolder/`,
  `--model-path models/sealed/stage1/latest.pt`, `--n-samples 10`.

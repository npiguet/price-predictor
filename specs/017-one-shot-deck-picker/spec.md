# Feature Specification: One-Shot Sealed Deck Picker

**Feature Branch**: `017-one-shot-deck-picker`
**Created**: 2026-05-19
**Status**: Draft
**Input**: User description: "I need to implement the feature that is described in ./specs/2026-05-19-one-shot-deck-picker.md"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Train a one-shot picker against a frozen scorer (Priority: P1)

A sealed-ML practitioner has a trained deck scorer and a corpus of pre-generated sealed pools. They want to train a new policy model — the "picker" — that, given a sealed pool, emits a full 23-spell deck in a single forward pass. The picker is trained with REINFORCE: the frozen scorer's evaluation of the picker's sampled decks is the reward signal, and the picker's policy is pushed toward decks the scorer rates highly. Training proceeds in epochs over the pool file, with a held-out validation slice driving best-checkpoint selection and early stopping. The training run produces a checkpoint in `models/sealed/picker/` that downstream tools can load.

**Why this priority**: Without a trained picker checkpoint, the one-shot inference path delivers nothing. The training pipeline is the load-bearing artifact-producer; everything else in this feature consumes its output.

**Independent Test**: Run the new `train-picker` subcommand against an existing scorer checkpoint and a pre-generated pools file. After a small number of epochs, verify that a `latest.pt` checkpoint exists in `models/sealed/picker/`, that validation reward improved from epoch 0 to the final epoch, and that the checkpoint contains the picker config plus training metadata.

**Acceptance Scenarios**:

1. **Given** a frozen scorer checkpoint and a pools file with at least the validation fraction + a few training pools, **When** the practitioner runs `train-picker` with default flags, **Then** the run completes at least one epoch, writes `latest.pt` and a run-stamped `best_{timestamp}.pt` to `models/sealed/picker/`, and reports per-epoch training loss decomposition (policy / entropy / aux) plus validation reward.
2. **Given** a training run in progress that produced one or more checkpoints, **When** the practitioner interrupts the run and re-runs `train-picker --resume <checkpoint>`, **Then** training continues from the saved epoch counter, best-validation-reward metadata, and optimizer state, and architecture flags passed alongside `--resume` are rejected with a clear error.
3. **Given** a pools file but no available scorer at the default path and no explicit `--scorer-checkpoint`, **When** the practitioner runs `train-picker`, **Then** the run fails fast with a message directing them to train a scorer first or pass `--scorer-checkpoint` explicitly.
4. **Given** an `.npz` embedding cache whose width disagrees with a checkpoint passed via `--resume` or `--picker-checkpoint`, **When** the practitioner runs `train-picker`, **Then** the run fails fast at startup with a clear width-mismatch error rather than a downstream torch shape error.

---

### User Story 2 - Use a trained picker to build decks from pools (Priority: P2)

A sealed-ML practitioner has a trained picker checkpoint and a pools file. They want to produce a `generated-decks.txt` file containing one 40-card deck per pool, built deterministically by the picker (one forward pass per pool, plus the basic-land fill). The resulting file is in the same format that `build-decks` produces and is therefore a drop-in input for `match-outcomes --side-a-decks` / `--side-b-decks` and any other downstream consumer of generated decks. This is the operational counterpart to `build-decks` — the same role, with the picker replacing the simulated-annealing search.

**Why this priority**: Inference is what makes the picker useful in self-play and evaluation. Without it the trained checkpoint has no production path, but it depends on User Story 1 (a trained picker must exist).

**Independent Test**: Given a trained picker checkpoint and a pools file, run the new `pick-decks` subcommand and verify that the output `generated-decks.txt` has one line per input pool, each line conforms to the `LABEL;SET_CODE;Card1|...|Card40` format, the deck contains exactly 40 cards, exactly 23 are nonland (spells), and the remaining cards include any nonbasic lands the picker selected plus basic lands distributed by the existing manabase heuristic.

**Acceptance Scenarios**:

1. **Given** a picker checkpoint and a pools file with N pools, **When** the practitioner runs `pick-decks --pools-path <pools> --label <tag>`, **Then** the output file contains exactly N lines, each with the practitioner-supplied label, the source pool's set code, and 40 card names.
2. **Given** a pool from which the picker would select zero nonbasic lands, **When** `pick-decks` processes that pool, **Then** the output line contains 23 spells plus 17 basic lands totaling 40 cards.
3. **Given** a pool that contains several nonbasic lands the picker ranks above some of the spells, **When** `pick-decks` processes that pool, **Then** the picker selects spells in ranked order until the 23-spell quota is met, takes any nonbasic lands encountered along the way, and the basic-land fill brings the total to exactly 40 cards.
4. **Given** an existing output file and `--resume`, **When** `pick-decks` is re-run on the same pools file, **Then** pools already represented in the output are skipped and new picks are appended.
5. **Given** a `generated-decks.txt` produced by `pick-decks`, **When** the practitioner runs `match-outcomes --side-a-decks <that-file>`, **Then** the supervisor accepts the file without modification and tags resulting matches with the picker label as `method_A`.

---

### User Story 3 - Monitor training quality and detect reward hacking (Priority: P3)

While a picker is training, the sealed-ML practitioner needs visibility into whether the picker is actually getting better (rather than overfitting to scorer blind spots). At the end of each epoch the training loop reports validation reward on a fixed held-out pool slice. It also reports cheap reward-hacking audits: an agreement metric between the training scorer and a second auditor scorer on the picker's validation decks, plus distributional summaries (color count, CMC histogram, creature count, type balance) on those decks for comparison against the established distribution of human/Forge-built decks. If the audits raise alarms, the practitioner has the information needed to decide whether to lower the learning rate, raise the entropy coefficient, or stop the run.

**Why this priority**: Without these audits a training run can converge to a high-reward-but-bad-decks failure mode (reward hacking) that only surfaces in the expensive end-of-training Forge validation. The in-training audits are cheap (one extra scorer forward per validation epoch, plus free numpy aggregations) and provide early-warning signal. Marked P3 because the system can technically produce a picker without them; they are quality-of-life infrastructure for the training loop.

**Independent Test**: Run `train-picker` with both the training scorer and an alternate auditor scorer (e.g., gen4-512 as training, gen3-256 as auditor). Verify that each per-epoch log line includes validation reward, cross-scorer rank correlation on the validation decks, and distributional summary statistics. Verify that an established baseline correlation can be pre-computed once on the existing match-outcomes corpus and that per-epoch correlations are reported alongside it.

**Acceptance Scenarios**:

1. **Given** a training run with an auditor-scorer configured alongside the training scorer, **When** the run completes an epoch, **Then** the per-epoch log includes mean validation reward, rank correlation between training-scorer and auditor-scorer scores on the validation decks, and distributional summaries (color count, CMC histogram, creature count, type balance).
2. **Given** the existing match-outcomes corpus, **When** the practitioner runs a one-off baseline computation, **Then** the system produces a reference rank correlation between the training scorer and the auditor scorer over that corpus, suitable for comparison against the per-epoch correlations.
3. **Given** a training run whose per-epoch auditor correlation drifts significantly below the corpus baseline, **When** the practitioner inspects the logs, **Then** the deviation is visible from the per-epoch numbers alone, without requiring a separate analysis pass.

---

### Edge Cases

- **All sampled decks in a pool receive an identical reward.** The per-pool baseline equals every reward, the advantage is zero, and the gradient contribution from that pool is zero. Training continues on the other pools in the batch. This is the degenerate-reward-landscape failure mode the cold-start sanity check is designed to detect before a full training run.
- **Validation reward stalls.** Early stopping fires after `--patience` epochs without validation improvement. Best-checkpoint metadata still points at the last improving epoch, and the run exits cleanly with that checkpoint as the artifact.
- **`--n-heads` does not divide `--d-model` (resolved from the scorer width).** The run fails fast at startup with a clear divisibility error rather than producing an invalid model and crashing during the first forward pass.
- **Scorer width changes between training and inference.** A picker trained against an `.npz` cache of width W cannot consume a cache of width W' ≠ W. The inference path rejects mismatched caches at load time with a clear error.

## Clarifications

### Session 2026-05-19

- Q: How should the auditor scorer (FR-030 cross-scorer audit) be configured? → A: Add an off-by-default `--auditor-scorer-checkpoint <path>` flag to `train-picker`; when present, enables the cross-scorer audit.
- Q: How should the baseline cross-scorer correlation (FR-031) be surfaced? → A: Out-of-band manual procedure / ad-hoc script — no CLI surface (same treatment as the cold-start sanity check).
- Q: What is the best-checkpoint artifact convention? → A: Save a run-stamped `best_{timestamp}.pt` (overwritten on each new val-reward best) alongside a per-epoch `latest.pt`. No per-epoch snapshot files. (Refined from the initial "best.pt + {timestamp}.pt + latest.pt" three-file scheme to two files, matching the scorer/encoder convention of best + latest only.)
- Q: How should the FR-032 distributional summaries be reported in the per-epoch log? → A: Mean / median summaries — across-validation-decks mean of color count, creature count, type-balance ratios, plus a condensed CMC histogram (bins for CMC≤2, 3, 4, 5, 6+).
- Q: Should the picker's random seed be configurable or hardcoded? → A: Hardcoded `seed=42` (no CLI flag), matching the `train-encoder` convention.

## Requirements *(mandatory)*

### Functional Requirements

#### Architecture and inference

- **FR-001**: The picker MUST accept a sealed pool as a tensor of per-card embeddings drawn from the same `.npz` embedding cache that the scorer consumes, and MUST produce a vector of N real-valued logits — one per pool card — in a single forward pass.
- **FR-002**: The picker's internal width MUST default to the embedding width derived from the configured scorer checkpoint. When the user requests a different internal width, the picker MUST insert a single linear projection from the embedding width to the requested internal width ahead of its transformer stack; when the requested width equals the embedding width, no projection layer is inserted.
- **FR-003**: The picker's transformer trunk MUST treat the pool as an unordered set: there is no positional encoding, and the architecture MUST be consistent with the project's existing set-input transformer primitives so that scorer and picker can share an encoder family.
- **FR-004**: The picker MUST produce one logit per pool card via a shared per-card head applied to each token output. Basic lands are not part of the pool input and MUST NOT be scored by the picker.
- **FR-005**: The picker MUST include an auxiliary head that produces a single scalar prediction of the per-pool mean reward (the average scorer score across decks sampled at this pool). The auxiliary head MUST always be present in the model; its loss-term coefficient is what is varied or zeroed.
- **FR-006**: Deterministic inference MUST proceed by sorting pool cards by logit (descending) and walking the sorted order: each spell encountered is taken until the 23-spell quota is met, and each nonbasic land encountered before that point is also taken. The walk halts as soon as the 23-spell quota is filled.
- **FR-007**: Basic lands MUST be added after the pick-decomposition walk by the existing manabase heuristic, computing `40 − len(chosen)` basic lands distributed by the mana-pip histogram of the chosen spells. Lands in `chosen` contribute no pips and are silently skipped by the histogram.
- **FR-008**: Detection of "is this pool card a nonbasic land" MUST use the existing land-flag slot already present in the deterministic-feature block of each card embedding, with semantics identical to the existing greedy deck builder's pool partitioning logic.

#### Training pipeline

- **FR-009**: The system MUST provide a new `train-picker` subcommand that trains a picker from random initialization using REINFORCE with a per-pool empirical-mean baseline, scoring sampled decks against a frozen scorer.
- **FR-010**: The training loop MUST stream pools from a user-provided pre-generated pools file. One epoch is one shuffled pass through the training portion of the file (the file minus the held-out validation slice). The pool-source path MUST be a required flag.
- **FR-011**: For each training step the loop MUST: draw a batch of pools, run one picker forward per pool, sample multiple decks per pool from the picker's distribution, score every sampled deck with the frozen scorer in one batched forward, compute the configured objective's loss (REINFORCE baseline/advantage by default; reward-ranked top-k per FR-039) plus the entropy + auxiliary terms, and step AdamW on the picker parameters only. The scorer and the encoder used to produce the `.npz` cache MUST remain frozen throughout training.
- **FR-012**: Deck sampling MUST be sequential categorical sampling without replacement over the full pool under softmax-of-logits at the configured temperature, halting when the 23-spell quota is met. Each sampled "deck" passed to the scorer is the chosen spells + nonbasic lands only — basic lands MUST NOT be scored. (Basic lands are added deterministically by FR-007 only when materializing the final 40-card deck for output; the scorer was trained on the chosen-card representation, identical to the input `GreedyDeckBuilder` feeds it, so adding basics to the scoring input would be off-distribution noise.)
- **FR-013**: The sampling loop MUST be implemented on GPU in a batched fashion: each loop iteration is one vectorized categorical-sample call (plus a mask update) across the full `(batch_size × samples_per_pool, N)` probability tensor. Per-sample Python loops dispatching individual GPU calls are forbidden.
- **FR-014**: The policy-gradient loss MUST use the Plackett-Luce log-probability of the sampled deck under the current picker distribution: the sum, across pick steps, of `logit_picked − logsumexp(remaining_logits)`. Backpropagation MUST flow through `log_prob`, the entropy term, and the auxiliary head's prediction; the advantage MUST NOT have gradient flow.
- **FR-015**: The auxiliary head MUST be trained against the per-pool mean reward (the same quantity used as the policy-gradient baseline) with mean-squared error. The target MUST be detached so the auxiliary loss does not flow back into the reward computation.
- **FR-016**: The entropy bonus MUST follow a val-reward-driven schedule: the coefficient is held constant at its configured initial value until validation reward has improved monotonically for a configured number of consecutive epochs, after which the coefficient is multiplied by a decay factor at the end of every subsequent epoch in which validation reward fails to improve on its previous best. Decay tracks validation-reward plateaus, not wall-clock or step count.
- **FR-017**: AdamW MUST be used as the optimizer with per-parameter-group gradient-norm clipping at the configured cap.
- **FR-018**: The training loop MUST shuffle the training pool slice at the start of each epoch. The validation slice MUST be the first `--val-fraction` of the file, excluded from training shuffles, and reused identically across epochs. The random seed governing weight initialization, pool shuffles, deck sampling, and the train/val split MUST be hardcoded to `42` (no CLI flag), matching the `train-encoder` convention.
- **FR-019**: At each validation point the system MUST run deterministic inference (FR-006) on the entire validation slice, score the resulting chosen-card decks (the FR-006 output — spells + nonbasic lands, before the FR-007 basic-land fill) with the frozen training scorer, and report mean reward. Best-checkpoint selection MUST use this metric. Validation runs `--evals-per-epoch` times per epoch (default 1 = once at epoch end) at evenly spaced step intervals; an interval between two consecutive validations is a "mini-epoch." Both `latest.pt` and `best_*.pt` (FR-037) MUST be updated at every validation point, not only at epoch boundaries.
- **FR-020**: Early stopping MUST fire after `--patience` validation points (mini-epochs) without improvement in validation reward. With `--evals-per-epoch 1` a mini-epoch equals an epoch, so patience is denominated in epochs.

#### CLI flags and modes

- **FR-021**: The `train-picker` subcommand MUST expose flags for: scorer checkpoint path, auditor-scorer checkpoint path (off by default; when set, enables the FR-030 cross-scorer audit), embedding cache path, pools file path (required), picker internal width (derived by default), number of attention layers, number of attention heads, feed-forward dimension (computed from width by default), dropout, auxiliary-loss weight, batch size, samples per pool, sampling temperature, entropy coefficient initial value, entropy decay-after epoch count, learning rate, gradient-norm cap, max epochs, validation fraction, early-stopping patience, resume checkpoint, prior-picker bootstrap checkpoint, and KL coefficient against the prior picker.
- **FR-022**: The system MUST support resuming a stopped training run from a checkpoint, restoring picker weights, optimizer state, epoch counter, and best-validation-reward metadata. Architecture flags MUST be forbidden when resuming; architecture MUST be inherited from the checkpoint.
- **FR-023**: The system MUST support bootstrapping a fresh training run from another picker checkpoint's weights only, discarding optimizer state, epoch counter, and validation metadata. Architecture flags MUST be forbidden in this mode; architecture MUST be inherited from the checkpoint.
- **FR-024**: The resume mode and the prior-picker-bootstrap mode MUST be mutually exclusive.
- **FR-025**: The KL coefficient against the prior picker MUST default to zero, disabling the penalty for a from-random-init REINFORCE run. Non-zero values MUST require a prior-picker bootstrap checkpoint to be provided; that bootstrap checkpoint MUST be the reference distribution for the KL penalty.
- **FR-026**: The system MUST provide a new `pick-decks` subcommand that reads a pools file, runs deterministic picker inference once per pool, fills basic lands, and writes an output file in the same `LABEL;SET_CODE;Card1|...|Card40` format produced by the existing greedy deck builder.
- **FR-027**: The `pick-decks` subcommand MUST require a generation-method label flag, which MUST be written verbatim as the `LABEL` field of every output line so the resulting decks are usable as `method_A` / `method_B` in downstream self-play.
- **FR-028**: The `pick-decks` subcommand MUST support append-and-skip resume semantics matching the existing greedy-builder `--resume`.

#### Validation and reward-hacking audits

- **FR-029**: The per-epoch training log MUST include the loss decomposition (policy / entropy / auxiliary) and the validation reward.
- **FR-030**: When the practitioner passes `--auditor-scorer-checkpoint` to `train-picker`, the system MUST score the validation decks with the auditor and report rank correlation between training-scorer and auditor scores on those decks each epoch. When the flag is omitted, the audit MUST be skipped (no auditor forward, no correlation line).
- **FR-031**: The baseline rank correlation between two scorers over the existing match-outcomes corpus (the reference value the per-epoch correlations are compared against) is computed by a documented manual procedure / ad-hoc script. No CLI subcommand or flag is required, mirroring the cold-start sanity check.
- **FR-032**: The per-epoch training log MUST include distributional summaries of the validation decks: the across-validation-decks mean of color count, creature count, and type-balance ratios, plus a condensed CMC histogram with bins for CMC≤2, 3, 4, 5, and 6+ (five bins total, reported as counts or fractions per bin).

#### Failure modes

- **FR-033**: At startup, the system MUST fail fast with a clear error if the picker's resolved internal width is not divisible by the number of attention heads.
- **FR-034**: At startup, the system MUST fail fast with a clear error if the `.npz` embedding cache width disagrees with the width carried by a checkpoint loaded via resume or prior-picker-bootstrap.
- **FR-035**: At inference, the system MUST fail fast with a clear error if the picker checkpoint's input width disagrees with the embedding cache width.
- **FR-036**: At startup, when no scorer checkpoint is provided and the default scorer path does not exist, the run MUST fail fast with a message directing the user to train a scorer first or pass `--scorer-checkpoint` explicitly.

#### Artifact layout

- **FR-037**: Trained picker checkpoints MUST be saved to `models/sealed/picker/` as two files: a `latest.pt` overwritten with the most recent checkpoint each epoch (the resume point), and a `best_{timestamp}.pt` (where `{timestamp}` is fixed at training-run start) overwritten in place whenever a checkpoint sets a new validation-reward best. No per-epoch snapshot files are written. The run-stamped best name keeps each run's best checkpoint distinct from other runs', and lets downstream tooling (Forge end-of-training validation) locate the best checkpoint by filename without inspecting metadata. The end-of-training Forge validation accordingly compares the best (`best_{timestamp}.pt`) and the final (`latest.pt`) checkpoint rather than an arbitrary top-K of per-epoch snapshots.
- **FR-038**: Each picker checkpoint MUST contain: picker weights only (no scorer or encoder weights), the picker config including the input width inherited from the training scorer, the current epoch counter, the best validation reward, and training metadata.

#### Training objective

- **FR-039**: The system MUST support a selectable training objective: the default `reinforce` (REINFORCE-with-baseline per FR-009 / FR-011 / FR-014) or a reward-ranked `topk` objective that, per pool, selects the `k` highest-reward sampled decks (by frozen-scorer reward) and trains the policy by maximum likelihood on their Plackett-Luce log-probabilities, with no baseline/advantage term. The entropy bonus (FR-016) and the auxiliary head (FR-015) MUST be retained under both objectives — only the policy-loss term differs. The objective and `k` MUST be configurable and resumable, and MUST NOT be architecture-locked, so a run started under one objective may be continued or warm-started under the other (e.g. a `reinforce` checkpoint resumed under `topk`). `k` MUST satisfy `1 ≤ k < samples_per_pool`; `k = samples_per_pool` (no selection pressure) MUST fail fast. The `topk` objective is scale-invariant (it depends only on the ranking of the per-pool rewards, not their magnitudes) and carries no negative-gradient term, which addresses the weak, high-variance gradient REINFORCE produces when within-pool reward variance is small.

### Key Entities

- **Sealed pool**: An unordered set of cards from one MTG set, sized ~60–90, drawn from booster packs. Each card is represented in training and inference by its row in the `.npz` embedding cache (encoder-pooled text vector concatenated with deterministic game features). Basic lands are not part of the pool.
- **Pool corpus**: A pre-generated file containing many sealed pools, one pool per line, set-code-prefixed. Consumed by `train-picker`, `pick-decks`, and the existing greedy builder. The first slice is the validation set; the rest is the training set.
- **Picker**: The new policy model — a transformer over a pool that emits one logit per pool card. Held in `models/sealed/picker/`.
- **Scorer (frozen, training reward)**: An existing trained deck scorer. Its evaluations of sampled decks are the picker's training reward signal. Held in `models/sealed/scorer/` and not modified by this feature.
- **Auditor scorer (optional)**: A second trained deck scorer of a different generation or architecture, used at validation time to detect reward hacking via rank-correlation disagreement on the picker's validation decks.
- **Generated decks file**: A file produced by `pick-decks` (and also by the existing greedy builder) containing one 40-card deck per line, each tagged with a label identifying its build method. Consumed by `match-outcomes` for self-play.
- **Auxiliary pool-quality head**: A second output of the picker model that predicts the per-pool mean sampled-deck reward. Used only at training time as a representation-pressure regularizer on the shared trunk; discarded at inference.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A practitioner can produce a 40-card sealed deck from any pool in the corpus using a single picker forward pass (plus basic-land fill), with no iterative search at inference. Wall-clock time per deck on a single GPU is at least an order of magnitude lower than the existing search-based deck builder using the same scorer.
- **SC-002**: A from-scratch picker training run on a single GPU completes within a few hours on a corpus of 100k+ pools and produces a checkpoint that the inference subcommand loads without modification.
- **SC-003**: Validation reward on the held-out pool slice increases monotonically over the run's first several epochs and plateaus before the early-stopping patience window expires, demonstrating that REINFORCE training is lifting off.
- **SC-004**: For the best checkpoint, end-to-end win rate against the strongest Forge-built side across at least 200 best-of-7 matches is measurably above 50%, and the win-rate ranking of the top-K candidate checkpoints is consistent with their validation-reward ranking.
- **SC-005**: Generated-decks files produced by the picker subcommand are accepted unmodified by the existing self-play match supervisor as side-A or side-B deck sources, and the picker's label appears verbatim as the corresponding method tag in the resulting match-outcomes rows.
- **SC-006**: The per-epoch cross-scorer rank correlation reported alongside validation reward stays within a stable band of the reference baseline correlation across the training run; deviations are visible from the per-epoch log alone without offline analysis.
- **SC-007**: Resuming a training run from a saved checkpoint yields per-epoch validation rewards that continue the trajectory of the original run rather than restarting from a cold state.

## Assumptions

- The `.npz` embedding cache is already populated for the cards in every pool the picker will see during training and inference. Re-running the existing `encode-cards` is a prerequisite that this feature does not duplicate.
- A trained scorer checkpoint is available in `models/sealed/scorer/`. This feature does not retrain or modify scorers; it consumes them frozen.
- A pre-generated pools file of sufficient size (the spec assumes 100k+ pools as a baseline) is available. Pool generation is the existing `generate-pools` subcommand's responsibility, not this feature's.
- The existing manabase heuristic (`compute_basic_lands`) is reused unchanged. It already accepts a list of chosen cards and computes a basic-land distribution from their mana-pip histogram; lands in the chosen list contribute no pips.
- The existing `is_land_embedding` partition logic on deterministic features is reused unchanged for nonbasic-land detection in both training-time sampling and inference-time decomposition.
- The cold-start sanity check (running random-init picker samples to verify the scorer's within-pool reward std at random init) is documented as a one-off manual procedure, not built as a CLI subcommand. The full training run is gated on a practitioner having confirmed the check passes.
- The baseline cross-scorer rank correlation over the existing match-outcomes corpus (the FR-030 reference value) is computed by an ad-hoc script run once per (training-scorer, auditor-scorer) pair, not built as a CLI subcommand. The resulting baseline number is supplied to the practitioner as context for interpreting the per-epoch correlations.
- The end-of-training Forge validation (~200 best-of-7 matches against the strongest Forge-built opponent) is run via the existing `match-outcomes` infrastructure with the picker's `generated-decks.txt` as the side-A source. No new validation CLI surface is required.
- The picker is trained against fixed card embeddings (frozen encoder). A future Phase B that jointly fine-tunes the encoder with the picker is explicitly out of scope.
- Reward hacking is anticipated as a real risk. The in-training audits (cross-scorer rank correlation, distributional summaries) are the cheap early-warning layer; the end-of-training Forge validation is the definitive guard. If audits alert, the response (raise entropy, restart from earlier checkpoint, etc.) is a practitioner judgment call rather than an automated training-loop policy.
- If REINFORCE-from-random-init fails to lift off in initial attempts, the documented contingency (SA-warmstart + KL-regularized REINFORCE) is a separate-spec decision, not a flag flip within this feature. The CLI surface includes the prior-picker-bootstrap and KL-coefficient flags because they are generally useful for any prior-picker continuation, not because adoption of the contingency is authorized by this spec.

## Out of Scope

- **Phase B picker fine-tuning** — jointly training the picker alongside the underlying encoder. Analogous to the existing scorer Phase B; a future spec.
- **Actor-critic baseline** using the auxiliary head as the policy-gradient baseline instead of the empirical-mean baseline. Recorded for a future spec round once basic lift-off is confirmed.
- **SA-warmstart supervised pretraining** and any contingency-plan pipeline that would produce a checkpoint suitable for the prior-picker-bootstrap mode. The CLI surface accommodates such a checkpoint as input, but generating one is not authorized here.
- **Picker integration into `match-outcomes` weighted rolls**. The natural integration is via the existing `generated-decks.txt` file format, which `match-outcomes` already consumes; no changes to the match-outcomes subcommand itself are required.
- **Multi-pool batching at inference**. The single-pool inference path is the operationally relevant one (the mobile-class deployment target). Bulk evaluation uses the natural batch dimension across pools but does not need new CLI surface.
- **A `probe-picker` CLI subcommand** for the cold-start sanity check. The check runs a handful of times across the project's lifetime and is documented as a manual procedure.

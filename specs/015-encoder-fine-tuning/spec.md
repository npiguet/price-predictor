# Feature Specification: Encoder Fine-Tuning (Phase B) for Sealed Scorer

**Feature Branch**: `015-encoder-fine-tuning`
**Created**: 2026-04-29
**Status**: Draft
**Input**: User description: "a new feature described by the specification contained in .\specs\encoder-fine-tuning.md"

## Clarifications

### Session 2026-04-29

- Q: When resuming a Phase A checkpoint into a Phase B run (or vice versa), what should happen to the saved optimizer state given the parameter-group mismatch? → A: Reject the invocation. `--resume` is phase-locked — the resumed checkpoint's phase must match the current run's phase, with no override flag.
- Q: How should a fresh Phase B run obtain its starting scorer weights now that `--resume` is phase-locked? → A: Add a new `--scorer-checkpoint <ckpt>` flag to `train-scorer` that loads scorer weights only (no optimizer state, epoch counter, or `best_val_accuracy`). Phase B kickoff is `train-scorer --scorer-checkpoint <phaseA>.pt --encoder-checkpoint <pp>.pt --embedding-lr 1e-7`. Mutually exclusive with `--resume`.
- Q: How is the `embedding_drift` reference batch constructed? → A: All unique cards in the very first Phase B training batch, captured during that step's forward pass before the optimizer step. No separate sampling code; the captured vectors become the step-0 baseline reused for the rest of the run.
- Q: What should `encode-cards --scorer-checkpoint <ckpt>` do when the supplied checkpoint contains no `encoder.state_dict` (i.e. it is a Phase A checkpoint)? → A: Reject the invocation with a clear error message that points the user at `--encoder-checkpoint` for non-Phase-B sources. No silent fallback or no-op.
- Q: What is the validation cadence, and therefore the granularity of `--patience` early stopping? → A: Once per epoch (end of epoch). `val_acc`, `embedding_drift`, and encoder gradient norm are all logged at end of every epoch; `--patience` counts epochs without a new peak `val_acc`.
- Q: How should `encode-cards --scorer-checkpoint` behave when `--clean` is not passed and some `.npz` files already exist (mixed-cache risk)? → A: Leave the existing idempotent behavior unchanged — skip files that already exist. The user is expected to operate against either a fully-populated or fully-empty cache; partial caches don't occur in practice and are not worth a guard.
- Q: How should `train-scorer --resume <oldPhaseA>.pt` behave when the resumed checkpoint carries Adam optimizer state (pre-AdamW switch in FR-005a)? → A: Don't address it. Phase A will be retrained from scratch after this feature ships; no pre-feature Phase A checkpoints need to remain resumable.
- Q: Should Phase A and Phase B share the same train/val split so `val_acc` is directly comparable across phases? → A: Yes. Both phases derive their train/val split deterministically from the corpus (fixed seed), so any two `train-scorer` invocations on the same `match-outcomes.txt` produce the same split.
- Q: Is the "encoder gradient norm" logged in FR-012 a single combined norm or split per sub-component? → A: Single combined L2 norm across the whole encoder parameter group (token embedding table + SAB layers + output projection).
- Q: Which CLI flags belong in the checkpoint's `config` dict (FR-009/FR-010)? → A: All `train-scorer` CLI flags that affect training — architecture, optimizer, data, and schedule — so the checkpoint is self-describing for reproducibility. The override-on-resume rule lets the user change runtime knobs between resumes.
- Q: When `--scorer-checkpoint <phaseA>.pt` bootstraps a fresh Phase B run, how should scorer architecture flags (`--n-layers`, `--n-heads`, `--n-seeds`, `--d-ff`, `--mlp-hidden`) be resolved against the checkpoint's stored `config`? → A: Inherit architecture from the checkpoint's `config` dict; reject any CLI architecture flag as a conflict with a clear error.
- Q: Should `train-scorer --embedding-lr <nonzero>` be allowed when neither `--scorer-checkpoint` nor `--resume` is supplied (i.e. Phase B from a randomly-initialized scorer)? → A: Reject the invocation with a clear error; Phase B always requires either `--scorer-checkpoint` (fresh bootstrap from Phase A) or `--resume` (continuing an existing Phase B run).
- Q: Does the `encode-cards` mutual-exclusivity rule between `--encoder-checkpoint` and `--scorer-checkpoint` trigger on the default `--encoder-checkpoint` value, or only when the user explicitly passes both? → A: Only an explicitly passed `--encoder-checkpoint` conflicts with `--scorer-checkpoint`; the default value applied when the user did not pass the flag does not trigger the error (mirrors the FR-004 carve-out on `train-scorer`).
- Q: Should Phase A also adopt max-norm 1.0 gradient clipping after this feature ships, or should Phase A's existing clipping behavior be left unchanged? → A: Phase A also uses max-norm 1.0 clipping (single parameter group), so both phases share the same clipping policy and Phase A vs Phase B `val_acc` remains an apples-to-apples comparison.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Train Phase B with encoder fine-tuning (Priority: P1)

After Phase A scorer training plateaus on validation accuracy, the user starts a
second `train-scorer` invocation that bootstraps scorer weights from the Phase A
checkpoint via `--scorer-checkpoint` and fine-tunes the price-predictor encoder
jointly with the scorer. The encoder shifts from "what predicts a card's market
price" toward "what predicts deck quality in sealed", which is the underlying
goal of the embedding schedule referenced in the sealed deck picker spec.

**Why this priority**: This is the feature. Every other story exists to support
or verify this run. Without it, the prior `--unfreeze-embeddings` flag (which
only updates a decoupled lookup table) remains the only fine-tuning option, and
it cannot produce shared-feature updates or transfer to unseen cards.

**Independent Test**: Run Phase A to early-stop, then run a Phase B invocation
with `--scorer-checkpoint <best_phaseA>.pt --encoder-checkpoint <pp>.pt
--embedding-lr 1e-7`. Verify the resulting checkpoint contains both
`scorer.state_dict` and `encoder.state_dict`, that training completes within
the `--patience` early-stopping window, and that encoder gradient norms and
`embedding_drift` are non-zero (i.e. the encoder is actually in the training
graph).

**Acceptance Scenarios**:

1. **Given** a Phase A scorer checkpoint and a price-predictor encoder
   checkpoint, **When** the user runs `train-scorer --scorer-checkpoint <phaseA> --encoder-checkpoint <pp> --embedding-lr 1e-7`,
   **Then** training proceeds with both parameter groups active starting
   from epoch 0 with a fresh AdamW state, the saved `best_*.pt` contains
   an `encoder.state_dict` key, and per-step encoder gradient norms are
   logged.
2. **Given** a Phase B run already in progress that was interrupted, **When**
   the user runs `train-scorer --resume <phaseB_checkpoint>` (no
   `--encoder-checkpoint` argument), **Then** training resumes with encoder
   weights loaded from the Phase B checkpoint itself rather than from the
   price-predictor file.
3. **Given** the user runs `train-scorer` with `--embedding-lr 0` (the
   default), **Then** the encoder is excluded from the training graph (Phase A
   semantics) and the resulting checkpoint contains no `encoder.state_dict`
   key.
4. **Given** the user runs `train-scorer --resume <phaseB_checkpoint>
   --encoder-checkpoint <pp>` (explicitly passing both flags), **Then** the
   command rejects the invocation with an error explaining that
   `--encoder-checkpoint` conflicts with the encoder weights already present
   in the resumed Phase B checkpoint.

---

### User Story 2 - Refresh cached card embeddings from a Phase B scorer (Priority: P1)

After Phase B finishes, the user re-runs `encode-cards` so that every
`.npz` file under `output/cardsfolder/` reflects the fine-tuned encoder
weights. Downstream tools (`build-decks`, `evaluate-scorer`,
`match-outcomes`) consume `.npz` files directly and never invoke the
encoder, so the embeddings on disk must be refreshed for the fine-tuning to
have any downstream effect.

**Why this priority**: Without this step, Phase B improvements are invisible
to every other tool. The fine-tuned encoder lives only inside the scorer
checkpoint until the cache is refreshed.

**Independent Test**: Pass `encode-cards --scorer-checkpoint <best_phaseB>.pt
--clean` and verify that all `.npz` files are rewritten, have the expected
`(2 * d_model,)` `float32` shape under the `"embedding"` key, and that the
new vectors differ from the pre-Phase-B vectors for at least the cards that
appeared in the match-outcomes corpus.

**Acceptance Scenarios**:

1. **Given** a Phase B scorer checkpoint, **When** the user runs `encode-cards
   --scorer-checkpoint <ckpt> --clean`, **Then** every card under
   `output/cardsfolder/` has a regenerated `.npz` and the embeddings differ
   from the pre-Phase-B baseline.
2. **Given** the user passes both `--encoder-checkpoint` and
   `--scorer-checkpoint` to `encode-cards`, **Then** the command rejects the
   invocation with an error explaining the two flags are mutually exclusive.
3. **Given** a card in `output/cardsfolder/` was never seen during Phase B
   training, **When** `encode-cards --scorer-checkpoint <ckpt>` runs,
   **Then** that card still receives a refreshed `.npz` file produced by
   running the fine-tuned encoder on its text.

---

### User Story 3 - Verify Phase B against Phase A on the deployment metric (Priority: P2)

The user runs `evaluate-scorer` twice — once with the Phase A `best_*.pt` and
once with the Phase B `best_*.pt` (with refreshed `.npz` files) — and compares
match-win rate against `forge-best`. Validation accuracy alone is not
authoritative because experiments under `experiments/gen2-initial-training.md`
show several scorer-side interventions hitting the same val_acc ceiling; the
deployment metric is the gate for whether to keep the Phase B checkpoint.

**Why this priority**: This is how the user decides whether Phase B was
worth running. It's lower priority than P1 because it reuses an existing
evaluation tool and the comparison itself is mechanical once two checkpoints
exist.

**Independent Test**: Evaluate both checkpoints against the same set of pools
(by passing `--set <SET>`) and confirm that the win rate, match counts, and
per-checkpoint statistics are reported in a way that supports a direct
comparison.

**Acceptance Scenarios**:

1. **Given** Phase A and Phase B checkpoints both exist, **When** the user
   runs `evaluate-scorer --set <SET>` against each and the embeddings cache
   matches the active checkpoint, **Then** the results identify the
   higher-win-rate checkpoint.
2. **Given** Phase B regresses on win rate compared with Phase A, **When**
   the user reverts to using the Phase A checkpoint and its corresponding
   `.npz` cache, **Then** all downstream tools function as they did before
   Phase B was attempted.

---

### User Story 4 - Detect runaway encoder drift early (Priority: P2)

While Phase B trains, the user watches the `embedding_drift` metric (mean L2
distance between current post-encoder card vectors on a fixed reference batch
and their step-0 values). Drift greater than 1.0 in the first three epochs
signals that the encoder is moving too fast and the run is unlikely to
produce a useful checkpoint.

**Why this priority**: Distinct from val_acc, drift catches the
catastrophic-forgetting failure mode where the encoder shifts so far that it
loses its general card-text understanding before validation accuracy reflects
it. Without this signal, the user only finds out after eval-scorer.

**Acceptance Scenarios**:

1. **Given** Phase B training is in progress, **When** drift on the reference
   batch exceeds 1.0 within the first three epochs, **Then** the user can
   abort the run and restart from the Phase A checkpoint with a lower
   `--embedding-lr`.
2. **Given** Phase B training is in progress, **When** the encoder gradient
   norm spikes for a single step, **Then** the per-parameter-group gradient
   clipping at max-norm 1.0 caps that step's movement before the optimizer
   applies it.

---

### Edge Cases

- **Resuming a Phase B checkpoint with `--encoder-checkpoint` explicitly
  passed**: the system must reject the invocation with an error rather than
  silently preferring one source of encoder weights over the other. The
  combination almost always indicates user mistake — the resumed Phase B
  checkpoint already carries fine-tuned encoder weights, so an explicit
  `--encoder-checkpoint` would discard them. The error makes the intent
  conflict visible instead of silently doing the right thing.
- **Cross-phase `--resume`** (e.g. `--resume <phaseA> --embedding-lr 1e-7`,
  or `--resume <phaseB> --embedding-lr 0`): rejected per FR-004. The
  resumed checkpoint's phase (presence of `encoder.state_dict`) must match
  the current run's phase (`--embedding-lr`). To start a fresh Phase B
  run from a Phase A checkpoint, use `--scorer-checkpoint` instead of
  `--resume`.
- **Bootstrapping Phase B without `--encoder-checkpoint`**: a fresh Phase
  B run via `--scorer-checkpoint <phaseA>` carries scorer weights only;
  encoder weights come from `--encoder-checkpoint` (default points at
  `models/price-predictor/transformer/latest.pt`) so the encoder is
  consistent with the `.npz` cache Phase A trained against.
- **Combining `--resume` and `--scorer-checkpoint`**: rejected as
  mutually exclusive. `--resume` is for same-phase full-state restoration;
  `--scorer-checkpoint` is for fresh-run weight initialization.
- **Architecture flags passed alongside `--scorer-checkpoint`**: rejected
  with a clear error per FR-003a. The bootstrap path constructs the
  scorer from the checkpoint's stored `config` so it matches the saved
  `scorer.state_dict`; any CLI architecture flag on the same invocation
  is a conflict.
- **Phase B without `--scorer-checkpoint` or `--resume`**: rejected per
  FR-004a. `train-scorer --embedding-lr <nonzero>` requires one of the
  two bootstrap paths; running Phase B against a freshly-initialized
  scorer is never a valid workflow.
- **Resuming with a different `--embedding-lr` than the original run**:
  permitted within the same phase (cross-phase resume is already
  rejected); the CLI flag overrides the value stored in `config` so the
  user can tune the learning rate between resumes.
- **A card text the encoder never saw during Phase B**: `encode-cards
  --scorer-checkpoint` must run the fine-tuned encoder on it normally; no
  fallback to the price-predictor encoder.
- **`encode-cards --scorer-checkpoint <phaseA>.pt`**: rejected with a
  clear error per FR-014. A Phase A checkpoint contains no
  `encoder.state_dict`, so there is no fine-tuned encoder to extract;
  the user should use `--encoder-checkpoint` for price-predictor or
  Phase A sources.
- **Cards appearing many times in one batch**: per-batch encoder caching must
  collapse duplicate references into one forward pass while still letting
  autograd accumulate gradients from every reference.
- **Mid-batch cache lifetime**: the encoder cache must clear between training
  steps so each step builds a fresh computation graph.
- **`--embedding-lr` specified but `--encoder-checkpoint` missing on a fresh
  Phase B run started via `--scorer-checkpoint`**: the system must either
  default to the price-predictor latest checkpoint or fail clearly;
  silently leaving the encoder uninitialized is not acceptable.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: `train-scorer` MUST accept an `--embedding-lr` flag where `0`
  keeps the encoder out of the training graph (Phase A) and any non-zero
  value puts the encoder in the training graph (Phase B).
- **FR-002**: `train-scorer` MUST remove the existing boolean
  `--unfreeze-embeddings` flag; encoder participation is fully determined by
  `--embedding-lr`.
- **FR-003**: `train-scorer` MUST accept an `--encoder-checkpoint` flag,
  defaulting to `models/price-predictor/transformer/latest.pt`, used to load
  encoder weights into the encoder parameter group when starting a fresh
  Phase B run via `--scorer-checkpoint`. The flag has no effect on Phase A
  runs (which consume `.npz` embeddings rather than running the encoder)
  and is forbidden on a Phase B `--resume` (see FR-004).
- **FR-003a**: `train-scorer` MUST accept a `--scorer-checkpoint <ckpt>`
  flag (no default) that loads scorer weights only from the given
  checkpoint; any `optimizer.state_dict`, `epoch`, `best_val_accuracy`,
  and `encoder.state_dict` present in the file MUST be ignored. This is
  the bootstrap mechanism for a fresh Phase B run from a Phase A
  checkpoint. The flag MUST be mutually exclusive with `--resume`;
  passing both MUST be rejected with a clear error. Scorer architecture
  flags (`--n-layers`, `--n-heads`, `--n-seeds`, `--d-ff`,
  `--mlp-hidden`) MUST be inherited from the loaded checkpoint's
  `config` dict so the constructed scorer matches the saved
  `scorer.state_dict`; passing any architecture flag on the CLI
  alongside `--scorer-checkpoint` MUST be rejected with a clear error
  identifying the conflicting flag and directing the user to omit it.
- **FR-004**: `--resume` is phase-locked: the resumed checkpoint's phase
  MUST match the current run's phase, where the current run's phase is
  determined by `--embedding-lr` (`0` = Phase A, non-zero = Phase B) and
  the resumed checkpoint's phase is determined by the presence or absence
  of `encoder.state_dict`. Cross-phase resume MUST be rejected with a
  clear error; no override flag exists. When resuming a Phase B
  checkpoint, encoder weights MUST be loaded from the resumed checkpoint.
  If the user explicitly passes `--encoder-checkpoint` on a Phase B
  resume, the system MUST reject the invocation with a clear error
  explaining that the flag conflicts with the resumed checkpoint's
  embedded encoder weights. (The default value of `--encoder-checkpoint`
  applied when the user did not pass the flag does not trigger the
  error.)
- **FR-004a**: A Phase B invocation (`--embedding-lr` non-zero) MUST be
  paired with either `--scorer-checkpoint` (fresh bootstrap from a
  Phase A checkpoint) or `--resume` (continuing an existing Phase B
  run). A Phase B invocation that supplies neither MUST be rejected
  with a clear error, since fine-tuning the encoder against a
  randomly-initialized scorer would push the encoder in arbitrary
  directions during the first epochs and is never a valid workflow.
- **FR-005**: During Phase B training, the system MUST construct a single
  AdamW optimizer with two parameter groups: a scorer group at `--lr` and an
  encoder group (token embedding table + encoder SAB layers + output
  projection) at `--embedding-lr`.
- **FR-005a**: Phase A training MUST also use AdamW (with a single parameter
  group at `--lr`), replacing the pre-existing Adam optimizer. Both phases
  share the same optimizer family so that resuming a Phase A checkpoint
  into a Phase B run does not require an optimizer-class swap.
- **FR-006**: The training step MUST run the encoder forward pass on each
  card's tokenized text, concatenate the encoder output with the
  deterministic feature vector parsed from the same card text, feed the
  per-card vectors through the scorer's SAB stack and PMA pooling and the
  scoring MLP, and backpropagate the Bradley-Terry pairwise loss through the
  entire graph (scorer → encoder → token embedding table) when
  `--embedding-lr > 0`.
- **FR-007**: Within each Phase B training step, the system MUST cache the
  encoder output for each unique card and reuse it for all references to
  that card in the same batch, while still letting autograd accumulate
  gradients from every reference into the shared encoder parameters. The
  cache MUST be cleared between batches.
- **FR-008**: The system MUST apply max-norm gradient clipping at 1.0
  per parameter group during both Phase A and Phase B training. In
  Phase A there is a single parameter group (the scorer); in Phase B
  the clip is applied independently to the scorer group and the
  encoder group. Sharing the clipping policy across phases keeps
  Phase A and Phase B `val_acc` directly comparable (per FR-011a).
- **FR-009**: Phase B checkpoints saved by `train-scorer` MUST contain
  `scorer.state_dict`, `encoder.state_dict`, `optimizer.state_dict`,
  `epoch`, `best_val_accuracy`, and the `config` dict of CLI flag values
  used to produce the checkpoint. Phase A checkpoints MUST NOT contain
  `encoder.state_dict`. The `config` dict MUST include every
  `train-scorer` CLI flag that affects training — architecture flags
  (`--n-layers`, `--n-heads`, `--n-seeds`, `--d-ff`, `--mlp-hidden`),
  optimizer flags (`--lr`, `--embedding-lr`), data flags, and schedule
  flags (`--patience`, `--batch-size`, etc.) — so the checkpoint is
  self-describing for reproducibility.
- **FR-009a**: Phase B `train-scorer` MUST write its `best_*.pt` and
  `latest_*.pt` files under filenames distinct from Phase A's, so a Phase B
  run does not overwrite the Phase A checkpoints in the same
  `--checkpoint-dir`. The Phase B `best_*.pt` filename MUST encode the
  active `--embedding-lr`, since two Phase B runs that differ only in
  `--embedding-lr` would otherwise collide. Reverting Phase B in favor of
  Phase A is then a matter of pointing downstream tools back at the Phase
  A files, which survived the Phase B run intact.
- **FR-010**: When resuming, CLI flags supplied on the resuming invocation
  MUST override the `config` dict stored in the resumed checkpoint for the
  current run.
- **FR-011**: Validation runs once per epoch at end of epoch. `--patience`
  MUST drive early stopping in both phases: training stops after this many
  consecutive epochs without a new peak `val_acc`, and the best checkpoint
  to date is preserved as `best_*.pt`. Default value: 5.
- **FR-011a**: The train/val split MUST be derived deterministically from
  the corpus (fixed seed, independent of invocation) so that any two
  `train-scorer` invocations on the same `match-outcomes.txt` produce the
  same split. This makes Phase A and Phase B `val_acc` directly comparable
  across the two checkpoints.
- **FR-012**: At the end of every epoch, `train-scorer` MUST log `val_acc`,
  `embedding_drift`, and the encoder gradient norm. `embedding_drift` is
  the mean L2 distance between current post-encoder card vectors on a
  fixed reference batch and their step-0 values, recorded in the
  `embedding_drifts` field on `TrainingMetrics`. The reference batch is
  the set of unique cards present in the very first Phase B training
  batch, captured during that batch's forward pass before the first
  optimizer step; the captured vectors are the step-0 baseline reused for
  every subsequent drift computation in the run. The encoder gradient
  norm is a single combined L2 norm across the entire encoder parameter
  group (token embedding table + encoder SAB layers + output projection),
  **measured pre-clip** (the value returned by `clip_grad_norm_` before
  it scales the gradients down to max-norm 1.0); post-clip would be
  bounded at 1.0 by construction and carry no diagnostic signal.
- **FR-013**: `encode-cards` MUST accept an `--encoder-checkpoint` flag
  (default `models/price-predictor/transformer/latest.pt`) and a
  `--scorer-checkpoint` flag (no default). The two flags MUST be mutually
  exclusive; explicitly passing both MUST be rejected with a clear error.
  The default value of `--encoder-checkpoint` applied when the user did
  not pass the flag does not trigger the error, so
  `encode-cards --scorer-checkpoint <ckpt>` is a valid invocation
  (mirrors the FR-004 carve-out on `train-scorer`).
- **FR-014**: When `encode-cards --scorer-checkpoint <ckpt>` runs, the
  system MUST extract encoder weights from the scorer checkpoint's
  `state_dict` and use those weights to produce `.npz` files with the same
  structure and shape as `.npz` files produced via `--encoder-checkpoint`.
  If the supplied scorer checkpoint contains no `encoder.state_dict` key
  (i.e. it is a Phase A checkpoint), the system MUST reject the
  invocation with a clear error directing the user to use
  `--encoder-checkpoint` instead for non-Phase-B sources.
- **FR-015**: `encode-cards` MUST produce a refreshed `.npz` file for every
  card under `output/cardsfolder/`, including cards never seen during Phase
  B training, by invoking the fine-tuned encoder on each card's text.
- **FR-016**: Every new or changed CLI flag introduced by this feature
  (`--embedding-lr`, `--encoder-checkpoint`, `--scorer-checkpoint`, and
  `--patience` on `train-scorer`; `--encoder-checkpoint` and
  `--scorer-checkpoint` on `encode-cards`; plus the removal of
  `--unfreeze-embeddings`) MUST be documented in the corresponding
  subcommand's `--help` output. Each flag's help text MUST state its
  purpose, its default value (or "no default" when none applies), and any
  mutual-exclusivity or phase-activation semantics that are not obvious from
  the flag name alone.

### Key Entities

- **Phase A checkpoint**: A scorer checkpoint produced with `--embedding-lr 0`.
  Contains `scorer.state_dict` only and acts as the resumption source for a
  fresh Phase B run.
- **Phase B checkpoint**: A scorer checkpoint produced with non-zero
  `--embedding-lr`. Contains both `scorer.state_dict` and
  `encoder.state_dict`; the presence of `encoder.state_dict` is the
  authoritative signal that the checkpoint is from Phase B.
- **Encoder parameter group**: The token embedding table, the encoder SAB
  layers, and the output projection — the price-predictor encoder
  components that participate in Phase B fine-tuning.
- **Reference batch (drift metric)**: The set of unique cards in the very
  first Phase B training batch, with their post-encoder vectors captured
  during that batch's forward pass before the first optimizer step. Reused
  across the run as the step-0 baseline for `embedding_drift`.
- **`.npz` embedding cache**: One file per card under `output/cardsfolder/`,
  each holding a `float32` array of shape `(2 * d_model,)` under key
  `"embedding"`. Refreshed by `encode-cards` after Phase B; consumed by
  `build-decks`, `evaluate-scorer`, and `match-outcomes`.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A Phase B run starting from a Phase A checkpoint completes
  end-to-end (training → checkpoint → `encode-cards` refresh →
  `evaluate-scorer`) using only documented CLI flags, with no manual
  intervention between steps.
- **SC-002**: Phase B match-win rate against `forge-best`, measured on a
  held-out set of pools via `evaluate-scorer`, can be compared head-to-head
  against the Phase A baseline using the same pool set, and the comparison
  decides whether to keep the Phase B checkpoint.
- **SC-003**: When Phase B regresses on the deployment metric, reverting to
  the Phase A checkpoint and its `.npz` cache restores the prior behavior
  with no required code changes.
- **SC-004**: After Phase B finishes and `encode-cards` runs, every card in
  `output/cardsfolder/` has a refreshed `.npz` file and downstream tools
  (`build-decks`, `evaluate-scorer`, `match-outcomes`) operate without
  modification on the new cache.
- **SC-005**: Encoder drift on the reference batch is observable at every
  validation interval during Phase B, allowing a runaway run to be
  identified within the first three epochs.

## Assumptions

- The price-predictor `models/price-predictor/transformer/latest.pt` is the
  encoder checkpoint that produced the existing `.npz` cache the Phase A
  scorer trained against; this consistency is required for a fresh Phase B
  bootstrap to be meaningful.
- The match-outcomes corpus and Bradley-Terry pairwise loss are the same
  ones used in Phase A; Phase B only changes which parameters receive
  gradient updates, not the loss function or training data format.
- A single AdamW optimizer with two parameter groups is acceptable; no
  separate optimizer state is needed for the encoder group.
- The deterministic feature parser is treated as a constant function for
  gradient purposes (its outputs participate in the forward pass but receive
  no gradient).
- Tokenization is non-differentiable; gradients stop at the token embedding
  lookup, which is itself a parameter and does receive updates.
- Re-caching `.npz` files via `encode-cards` after Phase B is acceptable
  even though it touches every card in the corpus, because the alternative
  (running the encoder at inference time) would push encoder weights and
  tokenizer state into every downstream tool.
- The user runs `encode-cards` against either a fully-populated or
  fully-empty `output/cardsfolder/` cache. Partial caches are not a
  supported state, so `encode-cards --scorer-checkpoint` retains the
  existing idempotent behavior of skipping files that already exist; the
  user passes `--clean` when they intend a full refresh after Phase B.
- Phase A is retrained from scratch after this feature ships. Pre-feature
  Phase A checkpoints (which carry Adam optimizer state, see FR-005a)
  are not expected to be resumable, so no migration path or guard is
  required for the Adam → AdamW switch.

## Out of Scope

- **Label-noise reduction.** Phase B operates on per-card features, not
  labels, so it cannot move a label-noise-bound `val_acc` ceiling.
  Interventions on labels themselves — repeated matchups, longer formats,
  fixed-opponent training — are not part of this feature.
- **Mid-training phase switching.** Phase A and Phase B are two separate
  `train-scorer` invocations chained via `--scorer-checkpoint`, not a
  single training loop that flips a switch when Phase A plateaus.
- **Inference-time encoding.** Downstream tools continue to consume `.npz`
  files directly; they do not load the encoder at inference time.
- **Embedding cache staleness detection.** A future enhancement may embed a
  checkpoint-hash reference into `.npz` files so downstream tools can detect
  a mismatch with the active scorer; this feature does not include that
  mechanism.

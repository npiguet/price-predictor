# Feature Specification: Draft Agent — Online Self-Play GRPO Trainer (Generation 3)

**Feature Branch**: `021-draft-online-grpo`  
**Created**: 2026-08-04  
**Status**: Draft  
**Input**: User description: "spec out a new feature (in a new branch) for the new online trainer"

## Overview

An online, critic-free, GRPO-style RL fine-tuning capability for the draft agent.
Per round it generates a small batch of fresh self-play drafts from the *current*
policy, takes **one pass** over them with a single-term advantage-weighted
policy-gradient update, discards the batch, and regenerates from the updated
policy. There is no learned critic, no GAE, no KL anchor, and no entropy bonus. A
single in-process command owns the whole loop. In-run steering uses per-round
diagnostics plus a live **anchor-margin** progress read; cross-generation
promotion reuses the existing gen-2 yardstick unchanged.

**Generation convention** (agent lineage): gen-0 = Forge AI; gen-1 = offline
imitation+critic agent; gen-2 = offline RL agent; gen-3 = the agent this feature
trains (a base checkpoint fine-tuned online). The base checkpoint (gen-1 or gen-2)
is an operator input.

Rationale, the gen-2 post-mortem, and the design trade-offs behind these choices
live in `experiments/2026-06-10-draft-agent-gen2-design.md` and
`experiments/2026-06-15-draft-agent-gen3-online-grpo-design.md` — they are not
repeated here.

## Clarifications

### Session 2026-08-04

- Q: How is the streaming generate→train→repeat loop driven? → A: A single dedicated in-process online-training command owns the whole loop — it invokes the existing generation path in-process each round, applies the one-pass GRPO update, and holds the optimiser state, LR schedule, and checkpoint continuity in memory across rounds.

### Session 2026-08-05

- Q: Where is the streaming self-play corpus written on disk? → A: Appended to the shared `output/draft/drafts.jsonl` (the same file gen-1 / live-play use); sample-mode online rollouts mix into the canonical corpus by design.
- Q: Where are gen-3 checkpoints written? → A: The shared `models/draft/agent/` with a shared `latest.pt`, exactly like gen-1/gen-2; `latest.pt` tracks the in-progress gen-3 during a run (tools defaulting to `latest.pt` pick up the in-progress agent).

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Apply one online GRPO update from a fresh self-play batch (Priority: P1)

Given the current-round checkpoint and a small batch of self-play drafts that this
checkpoint generated in sample mode, the system applies one pass of a critic-free
GRPO policy-gradient update — advantage-weighted policy gradient only, each learner
seat's advantage being its pod-relative leave-one-out deck-quality score
standardised over the batch — and produces the next checkpoint, logging per-round
exploration and movement diagnostics.

**Why this priority**: The core update rule; the standalone MVP (runnable once on a
single batch). Everything else feeds or steers it.

**Independent Test**: Provide a checkpoint and one small batch it generated; run a
single online update; confirm it produces a well-formed next checkpoint (loadable,
gen-1 format + gen-3 RL metadata) warm-started from the input, applies the
single-term policy update only to learner-seat picks, and logs the round's
exploration diagnostics (perplexity / off-argmax rate) and movement (mean logπ,
KL-to-previous-round).

**Acceptance Scenarios**:

1. **Given** a checkpoint and a small on-policy batch it generated in sample mode at temperature `T`, **When** the operator applies one online GRPO update, **Then** the system computes each learner seat's pod-relative leave-one-out deck-quality reward, standardises those rewards over the batch to get advantages, takes one pass of the advantage-weighted policy-gradient update over the learner-seat picks only, and writes the next checkpoint.
2. **Given** the update, **When** the loss is formed, **Then** it is the single term `−A·logπ_T(a|s)` averaged over learner picks — with no critic/value term, GAE, KL anchor, or entropy bonus — and the policy log-probs are evaluated at the same temperature `T` the batch was sampled at.
3. **Given** the batch, **When** advantages are computed, **Then** learner seats whose draft produced a failed deck build are excluded from the reward, the pod mean, and the gradient, without aborting the round.
4. **Given** the round, **When** it completes, **Then** the system logs the exploration diagnostics (policy entropy / perplexity `exp(H)` and off-argmax sampling rate over learner picks) and movement diagnostics (mean logπ of taken actions and KL-to-previous-round), and carries the model's critic head unchanged (neither trained nor used).
5. **Given** an invalid startup configuration (missing base checkpoint, architecture/embedding-width mismatch, missing learner agent, or a missing/non-positive rollout temperature), **When** the operator starts training, **Then** it exits nonzero with a clear message before any update.

---

### User Story 2 - Run the streaming self-play loop with a live progress read (Priority: P1)

A single command runs the loop: generate ~10 fresh drafts from the current policy,
apply one online update (US1), discard the batch, regenerate from the updated
policy, repeat — while logging a live **anchor margin** (mean deck-quality of the
gen-3 seats minus that of a frozen gen-1 reference held in the self-play mix, over
a recent window) as the progress signal.

**Why this priority**: The update (US1) delivers value only when run in the loop
with a live steering signal; co-P1 with US1.

**Independent Test**: Run the loop for a modest number of rounds against a fixed
mix that co-seats gen-3 learners with a frozen gen-1 anchor and a Forge/random-bot
field; confirm each round regenerates from the just-updated checkpoint (nothing is
shown twice), the streaming corpus accumulates on disk, and a per-agent mean
deck-quality read yields an anchor margin (gen-3 minus frozen gen-1) trackable over
rounds.

**Acceptance Scenarios**:

1. **Given** the current checkpoint and a fixed generation mix (gen-3 learner seats + a frozen gen-1 anchor + a Forge/random-bot field), **When** the loop runs a round, **Then** it generates a fresh small batch in sample mode with the *current* policy piloting the learner seats, applies exactly one pass over it (US1), and drafts the next round with the updated policy — driving a single resident Forge worker (started once, kept alive across rounds) and never re-showing a batch.
2. **Given** an accumulating self-play corpus, **When** the operator reads progress, **Then** they obtain the per-agent mean deck-quality score over a recent window and compute the **anchor margin** = mean(gen-3 seats) − mean(frozen gen-1 seats).
3. **Given** the loop's per-round diagnostics, **When** exploration decays (perplexity / off-argmax rate drifting below the target band at fixed `T`), **Then** it is visible in the logs so the operator can raise `T`.
4. **Given** the run, **When** it proceeds, **Then** the reference anchor in the mix stays fixed for the whole run (never swapped to a later generation or to "previous round").
5. **Given** the loop running in one process, **When** it advances from one round to the next, **Then** the optimiser state and learning-rate schedule continue in-process across rounds (a single warmup at the start, then constant) rather than resetting each round; periodic checkpoints are written only for snapshotting/recovery, not to carry per-round training state.

---

### User Story 3 - Decide when to pause and whether to promote (Priority: P2)

The promotion decision uses the existing gen-2 cross-generation yardstick unchanged
(no new code): one greedy (deterministic) self-play data-generation pass with a
single fixed agent mix that co-seats the candidate, its base generation, Forge, and
a random-bot minority into shared pods over many drafts, then the existing
per-agent deck-composition analysis to read each generation's mean deck quality.

**Why this priority**: Required to make a promotion decision, but introduces no new
code (reuses the gen-2 yardstick).

**Independent Test**: Take a gen-3 candidate and its base, run the greedy fixed-mix
generation pass followed by the per-agent analysis, and confirm the report gives a
per-agent mean deck-quality score on one shared absolute scale.

**Acceptance Scenarios**:

1. **Given** a plateau in the live anchor margin, **When** the operator decides to check strength, **Then** they pause the loop and run the greedy fixed-mix yardstick (candidate + base + Forge + random-bot minority, randomly co-seated) — the plateau is not an automatic stop or promotion.
2. **Given** the yardstick report, **When** the operator inspects the per-agent means, **Then** they make a manual promotion judgment (the system does not auto-promote).
3. **Given** the yardstick, **When** it is run, **Then** it uses only the existing greedy data-generation and per-agent deck-composition-analysis commands (no new evaluation engine).

---

### Edge Cases

- **Exploration collapse mid-run**: entropy/perplexity decays below the target band at fixed `T`. The per-round diagnostics surface it; the run does not auto-abort.
- **Failed builds among learner seats**: seats with an empty deck / null score drop out of reward, pod mean, and gradient without aborting the round.
- **A mix draw with zero learner seats**: because seats are sampled independently, a pod can come up learner-free → the sampling is constrained (resampled, or one seat forced to the learner) so every played draft has ≥1 learner seat; a learner-free draft carries no on-policy data and is never generated.
- **A pod with only one non-failed seat total**: the leave-one-out baseline (the mean of the *other* non-failed seats, any agent) has no other seats to average → that seat's reward is undefined and it is excluded.
- **A whole round's advantages are (near) zero** (every seat scores identically, or one non-failed seat per pod): the standardised advantage is undefined or degenerate → the round is a safe no-op, not a divide-by-zero.
- **Architecture / embedding-width mismatch**: a base checkpoint whose architecture disagrees with the embedding cache fails fast with a clear message, not a low-level shape error.
- **No learner agent specified**: rejected at startup.
- **Trailing partial line in the streaming corpus** (a generator crash mid-write): tolerated by the reader, as in the existing corpus format.
- **Resident worker crash mid-run**: the single Forge draft worker JVM crashes (long games can crash Forge, though drafts play none) → it is restarted (paying startup once more) and the run continues; the round in flight is regenerated, not lost to a corrupt batch.

## Requirements *(mandatory)*

### Functional Requirements

**Online training command & inputs**

- **FR-001**: The system MUST provide an online GRPO training command that, per round, consumes a small fresh self-play batch generated by the current policy and applies one pass of a critic-free advantage-weighted policy-gradient update, producing the next checkpoint.
- **FR-002**: The command MUST require **exactly one learner agent** — a mix label bound to a warm-start checkpoint (gen-1 or gen-2). That checkpoint warm-starts the policy at round 0; each subsequent round continues from the previous round's weights (held in memory). There is no persistent frozen reference inside the trainer (no KL anchor).
- **FR-003**: Each seat kind in the generation mix is named by a **label** in one of three categories: (a) a **Forge built-in** (pure or partly-random Forge AI — unbound); (b) a **frozen agent** — a label bound to a checkpoint (repeatable; untrained; e.g. the gen-1 anchor); or (c) the **single learner agent** (FR-002) — its label piloted by the live in-training policy (FR-012). The learner label MUST appear in the mix and MUST NOT also be bound as a frozen agent. The anchor (FR-015/FR-021) MUST be a frozen-agent label present in the mix (not the learner, not a Forge built-in); with exactly one frozen agent bound it defaults to that agent. Only learner-label seats feed the policy gradient; every other seat is an opponent / anchor / critic-free coverage only. Additionally, every generated draft MUST contain **at least one learner seat** — because the per-seat mix is sampled independently, a learner-free pod is possible by chance, so the sampling MUST be constrained (resample the pod, or force one seat to the learner) so such a pod is never played. A draft with no learner seat carries no on-policy picks and is useless for training.
- **FR-004**: The command MUST require the rollout temperature `T` (positive, no default); all policy distributions used in the update (log-probs, entropy) MUST be evaluated at that same `T`.
- **FR-005**: The command MUST reuse the live-play generation mechanism (Forge draft-worker protocol, pick service, typed-token state reconstruction) to produce each round's batch, and MUST keep a **single Forge draft worker resident across rounds** — started once per run and driven each round — so Forge's ~20 s per-process startup is amortised over the whole run, not paid per round. A single worker suffices because draft generation plays no games (game-playing, not drafting, is Forge's slow path). It MUST NOT spawn a fresh one-shot generation subprocess per round, and MUST NOT write new draft-driving logic beyond managing worker lifetime. If the resident worker crashes it MUST be restarted (paying startup once more) without aborting the run. Generation MUST also build each seat's drafted pool into a deck (via a configurable build method — greedy SA builder or picker) and score it with a **required frozen scorer checkpoint** to produce each seat's `deck_score` (the reward, FR-008), exactly as live-play generation does: the command MUST require the scorer checkpoint, MUST accept the build method (default **greedy**, matching the gen-1 corpus and the yardstick), and MUST accept a picker checkpoint used only when the build method is picker.
- **FR-006**: The command MUST expose exactly three primary tuning knobs — learning rate, temperature `T`, and drafts-per-round — and MUST NOT expose a critic weight, GAE-λ, KL coefficient, or entropy coefficient. Batch size, gradient-clip norm, one-pass-per-round, and the learner label are fixed-and-forget.

**Online GRPO update contract**

- **FR-007**: For each learner seat and pick, the system MUST reconstruct the typed-token state exactly as the gen-1 loader builds it for that `(seat, pack, pick)`, and the action MUST be the card recorded as taken at that pick.
- **FR-008**: The reward for a learner seat MUST be its pod-relative leave-one-out deck-quality score, where a seat's score is the frozen scorer's `deck_score` on that seat's built deck (FR-005): the seat's score minus the mean score of the other non-failed seats in its pod. The reward is terminal (γ=1); the same scalar applies to all of that seat's picks.
- **FR-009**: The advantage MUST be the batch-standardised pod-relative reward — the round's learner-seat rewards centred and scaled to unit variance over the round — with no learned critic and no GAE.
- **FR-010**: The training loss MUST be the single term `−A·logπ_T(a|s)` averaged over learner-seat picks, where `logπ_T` is the trained policy's temperature-`T` log-probability of the taken action. No critic/value term, GAE, KL anchor, or entropy bonus is present.
- **FR-011**: Each round MUST take exactly one pass over its fresh batch; no batch is ever reused across rounds.
- **FR-012**: On-policy correctness MUST be guaranteed by construction: each round the learner (gen-3) seats are piloted by the current in-training policy held in the worker's pick service, so the batch is generated by the checkpoint immediately before the pass that updates it; after the update the new weights MUST be pushed into that pick service so the next round is drafted by the updated policy (the frozen-anchor pick service is never updated). The trainer MUST NOT rely on a behaviour-anomaly reject or a stored corpus-checkpoint pairing.

**Diagnostics, logging & live progress**

Every round MUST emit its diagnostics to **stdout** so that progress, lack of
progress, and any collapse are all diagnosable from the run log alone — without
attaching a debugger, loading a checkpoint, or running an external analysis. The
diagnostics MUST at minimum cover four axes each round: **reward signal**,
**exploration**, **policy movement**, and **absolute progress (anchor margin)**.

- **FR-013**: At startup the system MUST echo the resolved run configuration and the results of the FR-024 validation to stdout: learner label + warm-start checkpoint, generation index, frozen/anchor labels, scorer checkpoint + build method (+ picker checkpoint when picker), rollout temperature `T`, learning rate, drafts-per-round, batch size, device, and the embedding-cache/checkpoint width. (Mirrors the gen-2 trainer's startup echo.)
- **FR-014**: Each round MUST log a **reward-signal** line: the round's learner-seat pod-relative reward distribution (mean, std) and the standardised-advantage distribution — its spread and the fraction of near-zero advantages (e.g. `|A|<0.1`) — so a round in which the reward fails to discriminate picks (no learning signal — the "nothing learned" failure) is visible as advantages collapsing toward zero.
- **FR-015**: Each round MUST log an **exploration** line over the learner picks: policy entropy, perplexity `exp(H)`, and the off-argmax sampling rate (fraction of picks whose sampled card ≠ the argmax card). Target band (guideline): perplexity ≈ 2–3 / off-argmax ≈ 25–40%. A drift of these toward perplexity → 1 / off-argmax → 0 / entropy → 0 is the readable signature of exploration collapse.
- **FR-016**: Each round MUST log a **movement** line: the mean log-probability of taken actions, the mean policy-gradient loss term for the round, and the KL-to-previous-round policy (logged, unpenalised) — a large per-round KL signals the step was too large for the round size.
- **FR-017**: Each round MUST compute and log the **anchor margin** live — mean deck-quality of the gen-3 (learner) seats minus mean deck-quality of the frozen-reference seats, over a sliding recent-drafts window — alongside the raw component means (learner, frozen gen-1 anchor, and Forge field) and the window size / draft count backing them. This is the round-over-round absolute-progress signal and MUST NOT require pausing training or a separate command to read.
- **FR-018**: Each round MUST log a single consolidated **round-summary line** (round index, drafts and learner-pick counts this round, generation + training wall-clock for the round, and the headline figures from FR-014–FR-017) so the whole run's trajectory is scannable from one column of stdout. The detailed per-axis lines (FR-014–FR-017) accompany it.
- **FR-019**: At run end or on operator interrupt the system MUST log a **final summary**: rounds completed, total drafts generated, the latest checkpoint path, and the best anchor margin observed and the round it occurred at.
- **FR-020**: The streaming self-play corpus MUST also accumulate on disk in the unchanged corpus schema, **appended to the shared `output/draft/drafts.jsonl`** (the same corpus file gen-1 / live-play use), so the same anchor margin (and full deck-composition detail) can be recomputed post-hoc by the existing per-agent deck-composition analysis. The live FR-017 log is the in-loop signal; the on-disk corpus is the authoritative/detailed read. (Sample-mode online rollouts mixing into the canonical corpus is accepted — clarified 2026-08-05.)
- **FR-021**: The frozen reference used for the anchor margin MUST stay fixed for the whole run (never swapped to a later generation or to "previous round").

**Failure handling & integrity**

- **FR-022**: Learner seats with a failed build MUST be excluded from the reward, the pod mean, and the gradient, without aborting the round; the round-summary line MUST report how many seats/picks were dropped.
- **FR-023**: A round whose standardised advantages are undefined or degenerate (fewer than two surviving learner rewards, or zero-variance learner rewards) MUST be a safe no-op for that round rather than a divide-by-zero or a crash, and MUST be logged as a skipped/no-signal round. (A learner seat whose pod has no other non-failed seat has an undefined leave-one-out reward and is excluded per FR-022, not counted here.)
- **FR-024**: At startup the system MUST validate that the learner's warm-start checkpoint exists and its architecture matches the embedding-cache width, that exactly one learner agent is specified, that the required scorer checkpoint exists (and the picker checkpoint exists when the build method is picker), that the rollout temperature is supplied and positive, and that the agent-label wiring is consistent (FR-003: every mix label is a known built-in / frozen / learner label; the learner label is in the mix and unbound; the anchor is a frozen label in the mix, or defaults to the sole frozen agent); failures exit nonzero with a clear message.

**Optimisation & checkpoints**

- **FR-025**: Optimisation MUST reuse the gen-1/gen-2 trainer conventions (adaptive optimiser with per-group gradient-norm clipping, linear-warmup-then-constant learning rate, fixed batch size within the 8 GB VRAM budget). The optimiser state and learning-rate schedule MUST be continuous across rounds (a single warmup at the start of the run, then constant), not reset per round. The per-round gradient norm (pre-clip) MUST be included in the round's movement diagnostics (FR-016).
- **FR-026**: The online loop MUST NOT use a held-out-loss best-checkpoint / early-stop guard. It MUST persist a "latest" checkpoint each round plus periodic timestamped snapshots; checkpoint selection and stopping are the operator's judgment, driven by the live anchor margin (FR-017) and the external yardstick (FR-030).
- **FR-027**: The next checkpoint MUST be written in the gen-1 agent checkpoint format (trunk + policy + critic + recency/context tables, config, round/epoch counters) plus gen-3 RL metadata: the generation index, the base-checkpoint identity, the algorithm tag (`online-grpo`), and the RL hyper-parameters (learning rate, rollout temperature, drafts-per-round). The critic head is carried unchanged; no critic, GAE, KL, or entropy hyper-parameters are stored. Encoder weights are not trained or stored (Phase A).
- **FR-028**: Checkpoints MUST be written to the existing agent checkpoint location — the shared `models/draft/agent/` — producing both timestamped files and a shared `latest.pt` pointer, consistent with gen-1/gen-2. `latest.pt` is rewritten every round, so during a run it tracks the in-progress gen-3 (tools defaulting to `latest.pt` pick up the in-progress agent — accepted, clarified 2026-08-05).

**Loop orchestration**

- **FR-029**: The streaming generate→train→discard→regenerate loop MUST be driven by a single dedicated in-process online-training command that owns the whole loop: at startup it launches the resident Forge draft worker (FR-005) once; each round it drives that worker to produce the fresh batch, applies the one-pass GRPO update (FR-001, FR-010, FR-011), pushes the updated weights into the learner pick service (FR-012), then drives the next round. The optimiser state, learning-rate schedule, resident worker, and checkpoint continuity (FR-005, FR-025, FR-028) are held in-process across rounds. The command runs for many rounds until the operator stops it (no automatic stop — FR-026).

**Cross-generation evaluation & promotion** *(the gen-2 yardstick reused — no new code)*

- **FR-030**: Comparing generations MUST be done as in gen-2: a single greedy (deterministic) self-play data-generation pass using the existing command, with one fixed agent mix holding every generation being compared (the candidate, its base, Forge, and the random-bot minority), each randomly co-seated into shared pods over a large pod count; the metric is the per-agent mean deck-quality score on the frozen scorer's raw absolute scale, read from the existing per-agent deck-composition analysis. Promotion is a manual operator judgment; the system does not auto-promote.

### Key Entities *(include if data involved)*

- **Base checkpoint**: the agent (gen-1 or gen-2) that warm-starts gen-3 at round 0. Carries policy + critic + recency/context tables and config; the critic head is carried through gen-3 unchanged.
- **Current/next checkpoint (round k → k+1)**: each round warm-starts from the current checkpoint and writes the next; there is no separate frozen reference inside the trainer.
- **Fresh self-play batch**: a small per-round set of draft records generated by the current checkpoint in sample mode at temperature `T`, against a fixed mix (gen-3 learners + a frozen gen-1 anchor + a Forge/random-bot field). Consumed once, then discarded.
- **Learner seat**: a seat whose mix label is the learner label — the single generation being trained. Its picks and batch-standardised pod-relative reward drive the policy update.
- **Frozen anchor seat**: a fixed-strength reference seat (frozen gen-1, plus the Forge bots) held in the generation mix; supplies opponents and the anchor-margin baseline. Never enters the gradient and never changes generation during a run.
- **Per-pick training quantity**: for each learner pick — state, action (taken card), temperature-`T` policy log-prob, and the seat's shared batch-standardised advantage.
- **Anchor margin**: mean deck-quality of gen-3 seats minus mean deck-quality of the frozen reference seats over a recent window.
- **Cross-generation yardstick run**: the gen-2 greedy fixed-mix, randomly co-seated evaluation producing each agent's mean deck-quality score for the promotion decision.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: On the cross-generation yardstick, a promoted gen-3 candidate achieves a higher mean deck-quality score than its base generation by more than the measured run-to-run noise band.
- **SC-002**: On the same yardstick, the promoted generation's mean deck-quality score also exceeds Forge's (gen-0).
- **SC-003**: Every round emits to stdout a reward-signal line (reward mean/std + advantage spread and near-zero fraction), an exploration line (entropy / perplexity / off-argmax rate), a movement line (mean logπ, policy-loss term, gradient norm, KL-to-previous-round), a live anchor-margin line (margin + raw component means + window size), and a consolidated round-summary line — so the four diagnostic axes are all present in the log for every round.
- **SC-004**: From the stdout log alone (no external tool, no checkpoint load, no training pause) an operator can distinguish, per round: (a) healthy progress — anchor margin rising; (b) no progress — anchor margin flat and/or advantages collapsing toward zero; and (c) collapse — perplexity → 1 / off-argmax → 0 / entropy → 0. In particular a policy collapse cannot occur silently.
- **SC-005**: The live anchor margin logged each round tracks the direction of the eventual greedy yardstick (sample-mode and argmax deck-quality move together), so the in-loop signal is a valid stand-in for the external check between yardsticks.
- **SC-006**: 100% of runs with an invalid startup configuration (missing/architecture-mismatched base checkpoint, missing learner agent, or a missing/non-positive rollout temperature) fail fast before any update, with a clear message.
- **SC-007**: Learner seats with failed builds are excluded from reward, pod mean, and gradient in 100% of cases, their presence never aborts an otherwise valid round, and the drop count appears in the round-summary line.
- **SC-008**: No round reuses a batch (each round trains on a batch generated by that round's current checkpoint), verifiable from the loop's per-round logging.

## Assumptions

- **Reward unit**: the pod-relative leave-one-out deck-quality score (the frozen scorer scalar) is both the reward and the promotion metric, unchanged from gen-2.
- **Advantage baseline is the group mean, not a critic**: the pod-relative leave-one-out reward is the RLOO/group baseline; batch standardisation is a variance-reducing rescale on top, not a second learned baseline.
- **On-policy by construction**: each round generates fresh from the current checkpoint and immediately trains it, so the corpus is on-policy without provenance bookkeeping. No corpus schema change and no behaviour-anomaly reject.
- **Exploration comes only from `T`**: sample-mode generation at temperature `T` is the sole exploration lever; no entropy bonus (a small fixed one may be reconsidered later if entropy crashes, never a decaying schedule).
- **Anchor margin is computed live in-loop and also recomputable externally**: the loop computes and prints the sliding-window anchor margin each round (FR-017) from the deck-quality scores it already has; the same figure is recomputable post-hoc by running the existing per-agent deck-composition analysis over the accumulated corpus. The in-loop read adds only arithmetic over scores already in hand — no new analysis engine.
- **Unchanged components**: model architecture, typed-token state, recency scheme, embedding cache, and corpus schema are reused from gen-1 / gen-2 / live-play. The live-play generation *mechanism* is reused (sample mode for rounds, greedy for the yardstick); the only change is worker lifetime — the online loop keeps a single Forge worker resident across rounds (FR-005), while the greedy yardstick still uses the one-shot command.
- **Single worker suffices**: draft generation plays no games, so one resident Forge worker keeps up with the ~10-draft rounds; parallel workers (Forge's answer to slow game-playing) are unnecessary here.
- **Phase A only**: the encoder is frozen; no joint encoder fine-tuning and no picker/scorer changes.
- **Yardstick is inherited**: the cross-generation comparison (US3) is the gen-2 greedy fixed-mix procedure over existing commands; this feature adds no evaluation code.
- **Base-generation choice is an operator input**: gen-1 or gen-2 as the round-0 base, supplied at run time.

## Dependencies

- The offline gen-1 draft agent (two-headed policy + critic) and its checkpoint format; optionally a gen-2 checkpoint as an alternate base.
- The live-play self-play data generation that lets a trained agent pilot live seats and emit the corpus (typed-token state reconstruction + online state tracker), reused unchanged for both the sample-mode rounds and the greedy yardstick.
- The frozen deck scorer that produces each seat's deck-quality score (the reward source), plus the deck builder (greedy SA builder or picker) that turns each drafted pool into the deck the scorer scores — both reused from the sealed pipeline as live-play generation already does.
- The existing per-agent deck-composition analysis tooling (for the live anchor margin and the yardstick).

## Out of Scope

- The gen-2 offline recipe (multi-epoch reuse of a frozen corpus, learned critic + GAE, KL anchor, entropy bonus, and any val-keyed coefficient decay schedule).
- Automated, unattended multi-generation promotion (promotion stays a manual operator judgment on the yardstick).
- Any corpus schema change or stored corpus-checkpoint provenance.
- In-draft live use of a critic to pick (the policy alone picks during rollouts; there is no critic in the gen-3 loss).
- PPO/clipped-ratio and vine/pack-2-forking advantage estimation.
- Joint encoder fine-tuning, picker/scorer changes.
- Shipping the agent into the Forge client (a separate deployment concern).

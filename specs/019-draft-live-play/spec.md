# Feature Specification: Draft agent — live Forge integration

**Feature Branch**: `019-draft-live-play`  
**Created**: 2026-06-10  
**Status**: Draft  
**Input**: User description: "Use the existing specification in ./specs/2026-06-04-draft-agent-live-play.md to create a new feature"

**Source**: This specification is derived from the normative design note
[`specs/2026-06-04-draft-agent-live-play.md`](../2026-06-04-draft-agent-live-play.md),
which captures the operator-facing behavior, the worker↔supervisor pick protocol,
and the self-play corpus this feature emits. That note remains the authority on
the communication protocol; this document restates the capability as prioritized,
testable user stories and requirements for planning.

## User Scenarios & Testing *(mandatory)*

The actor throughout is the **operator** — the researcher running the draft
data-generation pipeline to produce training data and measure agent strength.

### User Story 1 - Pilot a trained agent as a live draft seat (Priority: P1)

The operator binds a trained draft-agent checkpoint to a draft "agent label" and
runs the existing draft data-generation command. Forge runs a real eight-seat
pod; whenever a seat assigned that label must pick, the agent's policy chooses the
card. Any subset of the eight seats can be agent-piloted; the rest stay on the
Forge AI. Every completed draft is labeled (a deck is built and scored for each
seat) and appended to the existing draft corpus, producing a **self-play corpus**
that can train the next agent generation.

**Why this priority**: This is the feature's reason to exist. Until a trained
agent can pilot a live seat and have its drafts recorded, there is no way to
produce self-play data or to put the agent into a live pod at all. On its own it
is a complete, valuable MVP: the operator gets a usable self-play corpus.

**Independent Test**: Run the generation command with one agent label bound to a
checkpoint and a mix that assigns that label to at least one seat. Confirm the
output corpus gains completed-draft records in which the agent-piloted seats carry
the agent label and each seat has a built, scored deck, and that those seats' picks
are the agent's own choices.

**Acceptance Scenarios**:

1. **Given** a trained agent checkpoint bound to a label that the seat-assignment mix can select, **When** the operator runs the generation command for N drafts, **Then** completed drafts are appended to the corpus with agent-piloted seats carrying that label and each seat having a built+scored deck.
2. **Given** a run with no agent labels (Forge labels only), **When** the operator runs the command, **Then** the behavior and output are identical to the prior Forge-only generation (no regression).
3. **Given** a fault prevents an agent seat from making its genuine pick (protocol desync, policy error, or no usable action), **When** that draft is in progress, **Then** the entire draft is abandoned and never written to the corpus, an error is logged prominently, the run continues toward N, and no substitute pick is ever recorded.
4. **Given** the worker process crashes mid-draft, **When** the supervisor detects it, **Then** the in-flight draft is discarded, the worker is restarted, and the run continues toward N.

---

### User Story 2 - Measure agent strength against Forge in the same pod (Priority: P2)

The operator runs a **mixed pod** in which some seats are agent-piloted and the
rest are Forge-piloted, all drafting from the same physical boosters. Because the
finished-draft records score every seat's built deck on the same scale and pin the
shared booster geometry, the operator can compare the agent seats' deck scores
against the Forge seats' within each pod — a like-for-like strength measurement
where set and pool luck cancel out.

**Why this priority**: The second stated purpose of the feature. It depends on
US1 producing records, but adds distinct value: a head-to-head agent-vs-Forge
scoreboard. Deferring it still leaves US1 fully usable as a corpus generator.

**Independent Test**: Run a pod whose mix contains both agent and Forge labels.
Confirm each completed-draft record contains seats of both kinds drawn from the
same boosters, each with a comparable deck score, so an agent-vs-Forge score
delta can be computed per draft.

**Acceptance Scenarios**:

1. **Given** a mix containing both agent and Forge labels, **When** a draft completes, **Then** its record contains agent-piloted and Forge-piloted seats from the same boosters, each carrying a deck score on the same scale.
2. **Given** a corpus of such mixed-pod drafts, **When** the operator aggregates per-draft agent-minus-Forge score deltas, **Then** every draft contributes a within-pod comparison with no need for cross-pod normalization.

---

### User Story 3 - Configure rollouts: rival checkpoints and pick determinism (Priority: P3)

The operator tunes how rollouts are produced: binding **multiple** labels to
**different** checkpoints in one run (e.g. champion vs challenger), choosing
between the agent's strongest line and diversified temperature-sampled picks, and
fixing a seed so a sampled run is reproducible.

**Why this priority**: These are power-user controls that broaden how the corpus
is generated and how checkpoints are compared. They are valuable but optional —
US1 with default settings already yields a clean self-play corpus.

**Independent Test**: Run with two labels bound to two checkpoints and confirm each
checkpoint's seats are recorded under its own label. Separately, run twice with
temperature sampling and a fixed seed and confirm the agent seats' picks are
identical across the two runs.

**Acceptance Scenarios**:

1. **Given** two labels each bound to a different checkpoint and a mix that can select both, **When** a draft runs, **Then** seats piloted by each checkpoint are recorded under their respective labels.
2. **Given** the strongest-line pick mode, **When** an agent seat picks, **Then** it always selects the agent's highest-probability legal card.
3. **Given** the sampled pick mode with a fixed seed, **When** the run is repeated with the same inputs, **Then** the agent seats make identical picks.

---

### Edge Cases

- **Un-embeddable card in the pack**: a card the agent has no representation for is dropped from the choices considered, matching how the agent was trained — this is normal, not a fault.
- **No usable action**: if *every* legal card in a pack is un-embeddable (not expected with a complete card set), the pick cannot be made genuinely, so the draft is abandoned (US1 scenario 3) rather than guessed.
- **Protocol desync**: a pick answer that does not match the outstanding request (wrong draft/seat/pick, or a card not in the held pack) abandons the draft rather than being repaired with a substitute.
- **Supervisor disappears**: if the worker loses contact while awaiting a pick, it abandons the current draft instead of hanging the pod indefinitely.
- **Persistent deterministic fault**: if a fault recurs on every draft, the run makes visibly no progress (errors surfaced prominently) so the operator investigates, rather than silently filling the corpus with degraded data.
- **Geometry mismatch**: a checkpoint whose expected pack size or pod size disagrees with the live draft fails fast at startup rather than producing malformed picks.
- **Unknown label**: a seat-assignment label that is neither a Forge built-in nor bound to a checkpoint fails fast at startup.
- **Resume**: re-running against an existing corpus counts drafts already present toward the requested total.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Operators MUST be able to bind a seat-assignment label to a trained agent checkpoint so that seats assigned that label are piloted by the agent's policy in a live Forge draft.
- **FR-002**: The system MUST allow any subset of pod seats to be agent-piloted within a single draft, with all remaining seats piloted by the Forge AI (including its random-override variants).
- **FR-003**: For an agent-piloted seat, the recorded pick MUST be the policy's genuine choice given that seat's full reconstructed draft context; no Forge, first-card, or random substitute pick is ever recorded.
- **FR-004**: The draft context presented to the agent during live play MUST be identical to the context the offline training pipeline reconstructs for the same point in a draft, so the agent behaves online exactly as it was trained.
- **FR-005**: The system MUST support two pick modes — the agent's highest-probability legal card (default) and a temperature-controlled sampled pick — and MUST make sampled picks reproducible when given a seed.
- **FR-006**: On any fault that prevents an agent seat from making its genuine pick, the system MUST abandon the entire in-flight draft, omit it from the corpus, log the error prominently, and continue toward the requested draft count without counting the abandoned draft.
- **FR-007**: A protocol or policy fault MUST never hang the pod or crash the run; the draft is abandoned and generation proceeds.
- **FR-008**: With no agent labels in the mix, the command MUST behave identically to the prior Forge-only generation and produce the same record format (backward compatible).
- **FR-009**: Every completed draft MUST be labeled (a deck built and scored for each seat) and appended to the existing corpus with no schema change, regardless of which seats were agent-piloted.
- **FR-010**: The system MUST support binding multiple distinct labels to different checkpoints in one run so checkpoints can be pitted against each other.
- **FR-011**: A seat-assignment label that is neither a Forge built-in nor bound to a checkpoint MUST cause a fast, clear startup failure.
- **FR-012**: A checkpoint whose expected draft geometry (pack size, pod size) disagrees with the live draft MUST cause a fast, clear failure rather than silent misbehavior.
- **FR-013**: The operator MUST be able to observe run progress — target count, completed count, ETA, and which seats each draft agent-piloted — and a prominent error whenever a draft is abandoned.
- **FR-014**: The system MUST support resuming a run, counting drafts already present in the corpus toward the requested total.
- **FR-015**: The feature MUST NOT produce new model artifacts; the only persistent outputs are the appended corpus and a diagnostic log.

### Key Entities

- **Trained draft agent**: a frozen checkpoint whose policy chooses a card for an agent-piloted seat. Only the pick-making (policy) behavior is used for selection.
- **Agent mix**: the per-seat categorical assignment of labels (Forge built-ins and/or agent labels) that decides, per draft, which seats are agent-piloted.
- **Live pick exchange**: a single request/response interaction for one agent-seat pick — the pack currently in hand plus draft position goes out, the chosen card comes back. At most one is outstanding at a time.
- **Reconstructed draft context**: the agent seat's accumulated knowledge (its picked pool, what it has passed, what opponents are known to have taken, and per-card recency), rebuilt incrementally from the packs the seat has been shown.
- **Draft record (corpus entry)**: one completed draft — its seats (each with an agent label, a built deck, and a deck score) and its boosters (pinning the shared geometry). Schema is unchanged from the prior generation.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: In a single command invocation, an operator can generate a corpus of N completed self-play drafts, each containing at least one agent-piloted seat, with no manual intervention beyond starting the run.
- **SC-002**: 100% of recorded agent-seat picks are the agent's genuine choices — no completed draft containing a substituted pick is ever written to the corpus.
- **SC-003**: For every draft position, the card the agent picks live is the same card it would pick offline for the identical situation, verified by a position-by-position equivalence check that passes 100%.
- **SC-004**: Running with no agent labels reproduces the prior generation's output exactly (zero behavioral or format regression).
- **SC-005**: A single fault aborts only the affected draft; the run still reaches the requested completed-draft count, with no hangs and no partial records left in the corpus.
- **SC-006**: In every mixed-pod draft, agent and Forge seats are scored on the same scale from the same boosters, so an agent-vs-Forge strength delta is computable per draft without cross-pod normalization.
- **SC-007**: With a fixed seed and the sampled pick mode, two runs over the same inputs produce identical agent-seat picks.

## Assumptions & Dependencies

- **Trained agent available**: at least one draft-agent checkpoint already exists (produced by the offline draft-agent training feature); this feature consumes it and does not train.
- **Deck labeling reused**: the existing frozen deck builder and scorer label every seat's pool identically regardless of who piloted it; agent seats are scored on the same scale as Forge seats.
- **Corpus format frozen**: the draft corpus schema is unchanged; existing readers and downstream training consume the new records as-is.
- **One worker per run**: a single Forge worker drives the pod; the supervisor restarts it on crash. Parallel multi-worker throughput is out of scope.
- **Strict single-pick synchrony**: at most one agent pick is awaited at any moment, so picks are handled one at a time in order (no concurrent or out-of-order picks).
- **Out of scope**: reinforcement-learning fine-tuning from these rollouts, an automated multi-generation self-play loop, using the agent's value/critic judgment for in-draft decisions, and any change to the agent architecture, the card representation cache, or the offline training command.

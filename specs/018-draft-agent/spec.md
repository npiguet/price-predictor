# Feature Specification: Draft agent — imitation policy + critic (generation 1)

**Feature Branch**: `018-draft-agent`  
**Created**: 2026-05-31  
**Status**: Draft  
**Input**: Normative spec `specs/2026-05-28-draft-agent.md`; design rationale `experiments/2026-05-30-draft-agent-design.md`

## Overview

A generation-1 draft agent: given the state of an MTG booster draft, it picks the
next card (an imitation policy) and predicts the final deck's quality (a critic).
Both heads share one transformer body and are trained offline from a corpus of
Forge-generated drafts. There is no reinforcement learning and no live integration
into Forge in this generation — those arrive in generation 2. This generation
delivers the data-generation pipeline, the state representation, the two-headed
model, the training pipeline, and offline evaluation only.

The objective the whole stack optimises is win rate **when the Forge AI pilots the
deck the agent drafts** — Forge's piloting tendencies are the target distribution
by design, not a bias to correct.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Generate a labeled draft corpus (Priority: P1)

A researcher runs a single command to produce a corpus of complete Forge drafts.
Forge's draft AI fills all eight seats (some seats optionally degraded with random
picks), and for every seat the system builds a 40-card deck from the seat's full
drafted pool and scores it with the frozen sealed scorer. Each finished draft is appended as one
self-contained JSON record to an append-only file.

**Why this priority**: Training has no input without this corpus. It is the
foundation of the feature and is independently valuable as a reusable dataset.

**Independent Test**: Run `generate-draft-data --n-drafts N` and confirm the output
file gains N parseable records, each containing per-seat agents, per-seat decks with
scores, and the full per-booster pick transcript; readers can reconstruct any seat's
state at any pick from the record alone.

**Acceptance Scenarios**:

1. **Given** a frozen scorer and picker checkpoint and a populated `.npz` card cache,
   **When** the researcher runs `generate-draft-data --n-drafts 10`, **Then** the
   output file contains 10 self-contained JSON records, one per line, each with
   `pod_size` seats and `pod_size × packs` fully-drained boosters.
2. **Given** a seat whose drafted pool cannot be built into a legal deck, **When**
   the record is written, **Then** that seat has `deck = []` and `deck_score = null`,
   and the run continues.
3. **Given** a long Forge AI draft crashes the worker JVM mid-run, **When** the
   supervisor detects the crash, **Then** it restarts a fresh worker and continues
   toward `--n-drafts` without aborting the whole run.
4. **Given** an existing output file from a prior run, **When** the researcher runs
   with `--resume`, **Then** new records are appended and the run counts pre-existing
   drafts toward `--n-drafts` rather than overwriting.

---

### User Story 2 - Train the two-headed draft agent (Priority: P2)

A researcher trains the policy and critic jointly on a recorded corpus with one
command. Each `(draft, seat, pack, pick)` decision becomes a training example. The
policy head learns to imitate the picks of whitelisted (competent) agents; the
critic head regresses every seat's final pod-relative reward. Training reports its
loss decomposition and validation metrics per epoch and saves the best checkpoint.

**Why this priority**: This produces the actual model artefact, but it depends on a
corpus from User Story 1 already existing.

**Independent Test**: Run `train-draft-agent` against a small recorded corpus and
confirm it produces a checkpoint, logs the per-epoch loss decomposition plus
validation imitation top-1/top-3 accuracy and critic MSE sliced by pack number, and
selects a best checkpoint by validation loss.

**Acceptance Scenarios**:

1. **Given** a recorded corpus, **When** the researcher runs `train-draft-agent`,
   **Then** the system trains both heads, writes `{timestamp}.pt` and `latest.pt`
   under `models/draft/agent/`, and each per-epoch log line reports the imitation/
   critic loss split and the validation metrics.
2. **Given** `--imitation-weight 0`, **When** training runs, **Then** only the critic
   is optimised; given `--critic-weight 0`, only the policy is optimised.
3. **Given** `--imitation-agents forge-full`, **When** training runs, **Then** only
   seats whose agent is `forge-full` contribute imitation (policy) gradients, while
   the critic still trains on all seats (excluding failed builds).
4. **Given** a chosen `--d-model` not divisible by `--n-heads`, **When** the command
   starts, **Then** it fails fast at startup with a clear error rather than a runtime
   shape mismatch.
5. **Given** a stopped run, **When** the researcher passes `--resume <checkpoint>`,
   **Then** weights, optimiser state, epoch counter, and best-validation score are
   restored and architecture-defining flags are rejected.

---

### User Story 3 - Validate the picker as a label-builder (Priority: P3)

Before committing to a large corpus run, a researcher runs a one-off diagnostic that
decides whether the fast one-shot picker can be trusted to build the deck used for
each seat's label, or whether the slower stochastic SA builder is required. The
diagnostic reports a rank-correlation gating number against a self-consistency
reference ceiling.

**Why this priority**: A supporting diagnostic that de-risks label quality, but the
pipeline can run with either builder, so it is not on the critical path.

**Independent Test**: Run the diagnostic over a few hundred drafted pools and confirm
it reports the Spearman rank correlation of picker-built vs SA-built deck scores, the
distribution of their score gap, and the SA-vs-SA reference correlation.

**Acceptance Scenarios**:

1. **Given** a set of drafted pools, **When** the diagnostic runs, **Then**
   it builds each pool both ways, scores both with the frozen scorer, and reports the
   picker-vs-SA Spearman correlation plus the SA-vs-SA reference ceiling.
2. **Given** the picker tracks SA roughly as well as SA tracks itself, **Then** the
   researcher proceeds with `--build-method picker`; **given** the picker is
   materially below the reference or shows a composition-dependent gap, **Then** the
   researcher uses `--build-method greedy`.

---

### Edge Cases

- **Failed deck build**: a seat whose pool yields no legal deck is recorded with
  `deck = []` / `deck_score = null` and is excluded from critic training and from the
  pod mean used to compute pod-relative reward.
- **Trailing partial line**: a JVM crash mid-write can leave a partial final line;
  readers tolerate and skip it.
- **Unparseable worker output**: worker stdout lines that lack the event sentinel or
  whose JSON suffix fails to parse are silently skipped; Forge's incidental stdout is
  ignored and worker diagnostics go to a stderr log.
- **Supervisor crash mid-record**: the in-flight draft is lost (acceptable at
  data-gen scale); completed appended records are unaffected.
- **Missing card embeddings**: corpus cards with no `.npz` entry under the cards path
  are warned (up to 20 names plus the total) and their picks are dropped; the run is
  not blocked.
- **The wheel**: in an 8-seat pod a pack returns 8 picks after first sighting; cards
  missing on return (less the seat's own pick) move from `PASSED` to `TAKEN`, and the
  survivors re-enter the pack.
- **Pack-end flush**: when a pack is exhausted, all remaining `PASSED` instances flush
  to `TAKEN`.
- **Chaos draft**: set code is recorded per booster (not per draft), so mixed-set
  drafts are representable.

## Requirements *(mandatory)*

### Functional Requirements

#### Package & artefacts

- **FR-001**: The feature MUST live in a top-level `draft` package laid out in
  hexagonal style (`domain` / `application` / `infrastructure`) with entry point
  `python -m draft <subcommand>`.
- **FR-002**: `draft` MUST import from `sealed` (scorer, picker, greedy builder,
  card-embedding-layout helpers) and `price_predictor` (tokenizer); neither `sealed`
  nor `price_predictor` may import from `draft`.
- **FR-003**: Model artefacts MUST be written under `models/draft/agent/` and data
  under `output/draft/`.

#### Draft-event corpus generation (`generate-draft-data`)

- **FR-004**: The system MUST provide a `generate-draft-data` subcommand that
  generates `--n-drafts` (required) complete drafts via a supervised Java draft worker
  running Forge's draft AI for all pod seats.
- **FR-005**: The supervisor MUST generate one fresh `run_id` (UUID) at startup and
  stamp it on every record it appends, so a bad batch can be located and filtered out
  after the fact.
- **FR-006**: Each pod seat's agent MUST be sampled independently from the
  `--agent-mix` categorical weight distribution (default
  `forge-full:6,forge-r30:1,forge-r100:1`), producing pod compositions that vary
  draft-to-draft. Built-in agents: `forge-full` (Forge draft AI), `forge-r30` /
  `forge-r100` (30% / 100% of that seat's picks replaced by uniform-random legal
  picks).
- **FR-007**: For each completed draft the supervisor MUST build a 40-card deck from
  each seat's full drafted pool using `--build-method` (default `picker`; alternative
  `greedy`), score the non-basic subset with the frozen scorer
  (`--scorer-checkpoint`, default `models/sealed/scorer/latest.pt`), and append one
  complete JSON record.
- **FR-008**: When `--build-method picker`, the picker (`--picker-checkpoint`,
  default `models/sealed/picker/latest.pt`) MUST be fed each seat's full drafted pool
  with no padding or resizing, producing the 23-spell deck via the existing spell-quota walk,
  with basics filled by the existing basic-land computation.
- **FR-009**: `--set` MUST restrict all drafts to one set code; when omitted, each
  draft independently selects a random sealed-legal set.
- **FR-010**: Worker→supervisor transport MUST be sentinel-prefixed stdout: the
  worker emits each completed draft as one flushed line beginning with
  `<<DRAFT-EVENT-JSON>>` followed by the compact, newline-free JSON transcript
  (boosters + per-seat agents, without deck/score); the supervisor filters for the
  sentinel and defensively parses the suffix, skipping anything that fails to parse.
- **FR-011**: The supervisor MUST restart a crashed worker JVM and continue; worker
  stderr MUST be piped to a log file and Forge's incidental stdout ignored.
- **FR-012**: `--resume` MUST append to existing output and count pre-existing drafts
  toward `--n-drafts`; output defaults to `output/draft/drafts.jsonl`, is created if
  missing, and is append-only.

#### Draft-event file format

- **FR-013**: The corpus MUST be a JSONL file with one self-contained JSON record per
  line, with fields `draft_id`, `run_id`, `timestamp` (ISO 8601 UTC), `seats`, and
  `boosters`; card names are Forge canonical names; readers MUST tolerate a trailing
  partial line.
- **FR-014**: Each `seats[i]` MUST carry `agent` (free-form identifier), `deck`
  (40-card list including basics, or `[]` on failed build), and `deck_score` (scorer
  scalar, or `null` on failed build).
- **FR-015**: Each `boosters[k]` MUST carry `set_code` (per-booster) and `picks`
  (cards in pick-order, fully drained so `len(picks) == pack_size`); the multiset of
  `picks` is the booster's initial contents.
- **FR-016**: The record MUST be reconstructable without external state via the fixed
  conventions: `seats` length = `pod_size`; `boosters` ordered pack-1 first then
  pack-2 then pack-3, one per opening seat in seat order, so for `boosters[k]`,
  `pack_number = floor(k / pod_size) + 1` and `opening_seat = k mod pod_size`; within
  a booster the pick at position `j` was made by seat
  `(opening_seat + j · dir_p) mod pod_size` with `dir_p = +1` for packs 1 & 3 and
  `−1` for pack 2. Pod size, pack count, and pack size are derived from `len(seats)`,
  `len(boosters)/len(seats)`, and `len(boosters[0].picks)`.

#### Draft state representation

- **FR-017**: A single decision MUST be represented as one seat at one pick, encoded
  as a typed token sequence of one `CONTEXT` token plus card tokens for everything the
  seat has observed, laid out as `[CONTEXT] [POOL…] [PACK…] [PASSED…] [TAKEN…]`.
- **FR-018**: Every observed card instance MUST be in exactly one of four mutually
  exclusive token types at a time:
  - `POOL` — cards the seat drafted, kept as a multiset, accumulating across all
    packs;
  - `PACK` — cards currently in front of the seat (the legal actions this pick),
    deduped to distinct card names, reset each pick/pack;
  - `PASSED` — cards seen and passed this pack whose fate is not yet observed, one
    token per instance, emptied at every pack boundary;
  - `TAKEN` — cards the seat saw that ended up in an opponent's pool (known via wheel
    diff or pack-end flush), one token per instance, accumulating across the draft.
- **FR-019**: `PASSED → TAKEN` transitions MUST occur at two times: (a) wheel diff —
  on a pack's return, cards missing (less the seat's own pick) become `TAKEN` and
  survivors re-enter `PACK`; (b) pack-end flush — when a pack is exhausted, all
  remaining `PASSED` instances flush to `TAKEN`.
- **FR-020**: Each card token MUST be the card's `.npz` vector concatenated with a
  4-dim type one-hot and two learned recency embeddings (`packs_ago`, `pick_ago`).
  The one-hot's four positions correspond one-to-one to the FR-018 types
  (`POOL`, `PACK`, `PASSED`, `TAKEN`), with exactly one position set per token —
  this one-hot is the **sole differentiator of multiset membership**: two tokens
  for the same card name in different sets share the `.npz` block and differ only
  in these four dimensions. Because there is no input projection (FR-025), those
  dimensions persist through the residual stream and the first transformer layer's
  query/key/value projections are where the per-type interpretation is learned; no
  separate learned type table is used.
- **FR-021**: `packs_ago ∈ {0,1,2}` MUST measure packs since the card was last in the
  seat's pack (`0` = this pack / wheel-capable). `pick_ago ∈ {0,…,P−1}` MUST measure
  picks since the card was last in the seat's pack prior to the current pick (`0` if
  never before), and MUST freeze at its end-of-pack value once `packs_ago ≥ 1`.
- **FR-022**: The `CONTEXT` token MUST be the sum of two learned `d_model`-wide
  embeddings — `pack_number ∈ {1,2,3}` and `pick_number ∈ {1,…,P}` — with no card
  embedding and no seat/set/agent identity.
- **FR-023**: Token order within each type group MUST NOT be significant (no
  positional encoding); sequences MUST be padded per batch to the longest sequence,
  with padding positions masked from attention and every head.

#### Model architecture

- **FR-024**: The model MUST be a set transformer over the assembled sequence with a
  trunk of `n_layers` Set Attention Blocks where all tokens attend to all tokens.
- **FR-025**: By default there MUST be no input projection — the draft features are
  concatenated onto the card embedding and `d_model = embedding_dim + 4 +
  d(packs_ago) + d(pick_ago)`. A non-default `--d-model` MUST insert a single linear
  map from the concatenated width to `d_model`.
- **FR-026**: `d_model` MUST be divisible by `n_heads`, validated fast at startup.
- **FR-027**: The policy head MUST apply a shared `Linear(d_model, 1)` to each `PACK`
  token output, producing one logit per pack card, masked-softmax over `PACK`
  positions only (`argmax` at inference).
- **FR-028**: The critic head MUST apply a `Linear(d_model, 1)` to the `CONTEXT`
  token output, producing one scalar predicted final pod-relative reward.
- **FR-029**: Card embeddings MUST be frozen (Phase A only; no joint encoder
  fine-tuning).

#### Training (`train-draft-agent`)

- **FR-030**: The system MUST provide a `train-draft-agent` subcommand that trains
  policy and critic jointly in one pass over the recorded corpus, with each
  `(draft, seat, pack, pick)` as one example (up to `pod_size × pack_size` per draft).
- **FR-031**: For each example the loader MUST reconstruct `PACK`, `POOL`, `PASSED`,
  and `TAKEN` token sets and recency features from the JSON record using the FR-016
  geometry, with the imitation target = the card actually taken at that pick.
- **FR-032**: The critic target MUST be the leave-one-out pod-relative reward
  `seats[s].deck_score − mean({seats[j].deck_score : j ≠ s})`, shared by all of a
  seat's states (Monte-Carlo regression); failed-build seats are excluded from both
  the mean and critic training.
- **FR-033**: The loss MUST be
  `imitation_weight · CE(policy, taken)` over imitation-whitelisted seats only, plus
  `critic_weight · MSE(critic, reward)` over all (non-failed) seats. The imitation
  whitelist is set by `--imitation-agents` (default `forge-full`) and does not affect
  the critic.
- **FR-034**: Optimisation MUST use AdamW with per-parameter-group max-norm gradient
  clipping (`--max-grad-norm`, default 1.0) and a linear LR warmup over the first
  `--warmup-frac` (default 0.05) of scheduled steps, then constant `--lr`
  (default `3e-4`).
- **FR-035**: The train/validation split MUST be draft-disjoint — all picks of a
  `draft_id` go entirely to one side; the first `--val-fraction` (default 0.2) of
  distinct draft IDs form the held-out set with `random_seed = 42`.
- **FR-036**: States MUST be padded per batch (length bucketing permitted); the best
  checkpoint MUST be selected by validation loss `L`.
- **FR-037**: Each per-epoch log MUST report the loss decomposition plus validation
  imitation top-1 / top-3 accuracy and critic MSE sliced by `pack_number`.
- **FR-038**: Corpus cards with no `.npz` under `--cards-path` MUST be warned (up to
  20 names plus the total) and their picks dropped, without blocking the run.
- **FR-039**: Training MUST support `--resume` (restores weights, optimiser, epoch
  counter, best-val; architecture flags forbidden) and `--checkpoint` (bootstrap a
  fresh run from weights only; architecture flags forbidden); the two are mutually
  exclusive. `--epochs` (default 100) caps epochs and `--patience` (default 10)
  early-stops on no validation improvement.

#### Checkpoints

- **FR-040**: Each checkpoint MUST contain `model_state_dict` (trunk + policy head +
  critic head + recency and context embedding tables; type is a one-hot, not a learned
  table), `config` (architecture: `d_model`, `n_layers`, `n_heads`, `ff_dim`,
  `dropout`, derived `embedding_dim`, and booster size `P`), and `epoch`,
  `best_val_loss`, plus training metadata. No encoder weights are embedded.
- **FR-041**: Checkpoints MUST be written as `{timestamp}.pt` and `latest.pt` under
  `models/draft/agent/`.

#### Builder validation diagnostic

- **FR-042**: A one-off diagnostic script (not a CLI subcommand) MUST, over a few
  hundred drafted pools, build each pool with both the picker and the SA
  greedy builder, score both with the frozen scorer, and report the picker-vs-SA
  Spearman rank correlation, the distribution (median + spread) of the SA−picker
  score gap, and the SA-vs-SA reference correlation across independent SA restarts.

### Key Entities

- **Draft record**: one completed draft; groups `draft_id`, `run_id`, `timestamp`,
  the per-seat results, and the full booster transcript. The unit of the corpus and
  of the draft-disjoint split.
- **Seat**: one drafter in the pod; carries an `agent` identifier, a built 40-card
  `deck`, and a `deck_score`. Source of imitation targets (when whitelisted) and of
  the critic's pod-relative reward.
- **Booster**: one opened pack; carries a `set_code` and the full pick-ordered list of
  its cards. The geometry that lets any seat's observation history be reconstructed.
- **Draft state**: a typed token sequence (`CONTEXT` + `POOL`/`PACK`/`PASSED`/`TAKEN`
  card tokens) representing one seat at one pick — the model input.
- **Recency features**: per-token `packs_ago` and `pick_ago` learned embeddings
  encoding observation staleness and the wheel signal.
- **Draft agent checkpoint**: the trained two-headed model plus its architecture
  config and training metadata.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: `generate-draft-data --n-drafts N` produces exactly N additional
  complete records (modulo records lost to supervisor crashes), each independently
  parseable and self-contained.
- **SC-002**: Every generated record's geometry round-trips: any seat's pool, pack,
  passed, and taken sets at any pick can be reconstructed from the record alone using
  the documented conventions.
- **SC-003**: A worker JVM crash during generation does not abort the run — the
  supervisor restarts and the run reaches its target draft count.
- **SC-004**: `train-draft-agent` on a recorded corpus runs to completion, emits a
  best checkpoint under `models/draft/agent/`, and the checkpoint reloads and produces
  picks and critic scalars on held-out states.
- **SC-005**: Per-epoch validation logging reports imitation top-1 and top-3 accuracy
  and critic MSE sliced by pack number, enabling ranking of model variants on a shared
  held-out set.
- **SC-006**: A misconfigured architecture (`d_model` not divisible by `n_heads`, or
  architecture flags supplied with `--resume`/`--checkpoint`) fails fast at startup
  with a clear message rather than a runtime error.
- **SC-007**: The builder validation diagnostic outputs a single gating Spearman
  correlation and its self-consistency reference ceiling, sufficient to decide
  `picker` vs `greedy` build method before a large corpus run.

## Assumptions

- Pod size is 8 and there are 3 packs per draft, matching standard Forge booster
  draft. Pack size `P` is **set-dependent** (typically 12–15 cards) — it is not a
  fixed constant. The loader derives pod size, pack count, and pack size from each
  record (`len(seats)`, `len(boosters)/len(seats)`, `len(boosters[0].picks)`; FR-016)
  rather than hardcoding them, and `P`-dependent table sizes (`pick_number`,
  `pick_ago`) are recorded in the checkpoint config (FR-040), so other pod/pack sizes
  are representable. A single draft's boosters are assumed to share one pack size.
- The frozen sealed scorer and one-shot picker checkpoints exist at their default
  paths and were produced by an `encode-cards` run whose `.npz` width matches; width
  mismatches fail fast per the existing scorer/picker contracts.
- Reward is regressed in raw scorer-score space (no temperature calibration in
  generation 1); calibration to win-probability is a generation-2 concern.
- Generation 1 ships data generation, training, and offline evaluation only — there
  is no inference/play-out CLI and no live Forge seat.

## Out of Scope

- Live integration of the policy as a Forge draft seat (self-play) — generation 2.
- RL fine-tuning of the policy (actor-critic, GAE, REINFORCE/PPO) — generation 2.
- Surgical pack-2 forking / vine-style advantage estimation — generation 2.
- Self-play data regeneration across generations — generation 2.
- An auxiliary opponent-archetype prediction head — generation 2.
- Encoder fine-tuning alongside the drafter (Phase B analogue).
- Picker fine-tuning on variable / smaller pools (only relevant if the builder
  validation rejects the picker; the SA fallback otherwise suffices).
- An inference / play-out CLI.

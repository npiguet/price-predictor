# Feature Specification: Self-Play Refinement

**Feature Branch**: `014-self-play-refinement`
**Created**: 2026-04-20
**Status**: Draft
**Input**: User description: "Create a new feature based on 'phase 3' of the training curriculum as described in sealed-deck-picker.md"

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Random-Set Pool Generation (Priority: P1)

As a practitioner, I want to generate sealed pools from randomly selected sets so that my
training data and evaluation cover the full range of sealed-legal expansions, not just one
hardcoded set.

**Why this priority**: Every downstream step (deck building, match generation, evaluation)
depends on pools that carry their set code. This is the foundational format change.

**Independent Test**: Run `generate-pools` without `--set` and verify the output contains pools
from multiple sets, each line prefixed with its set code.

**Acceptance Scenarios**:

1. **Given** `generate-pools` is invoked without `--set`, **When** pool generation completes,
   **Then** each line in `pools.txt` begins with a set code followed by a semicolon and
   pipe-separated card names, and multiple distinct set codes appear across lines.
2. **Given** `generate-pools` is invoked with `--set MH3`, **When** pool generation completes,
   **Then** every line in `pools.txt` begins with `MH3;` and the card names follow.
3. **Given** `generate-pools` is invoked without `--set`, **When** a set is randomly selected,
   **Then** only sets with a draft booster template and not of type "funny" (un-sets) are eligible.

---

### User Story 2 — Build Scorer Decks from Pools (Priority: P2)

As a practitioner, I want to feed pools through the scorer-guided greedy deck builder and
produce a generated-decks file so that the Java match generator can use scorer-built decks
without needing access to the Python model at runtime.

**Why this priority**: The generated-decks file is the bridge between the Python scorer and
the Java game engine. Without it, self-play match generation cannot work.

**Independent Test**: Run `build-decks` with a trained scorer checkpoint and a pools file,
verify the output file has one 40-card deck per input pool with correct set code prefixes.

**Acceptance Scenarios**:

1. **Given** a pools file with N pools (each with a set code prefix) and a trained scorer
   checkpoint, **When** `build-decks` is run, **Then** the output generated-decks file
   contains exactly N lines, each formatted as `SET_CODE;Card1|Card2|...|Card40` with
   exactly 40 pipe-separated card names.
2. **Given** a pool with cards from set MH3, **When** the scorer builds a deck, **Then** the
   output line begins with `MH3;` and the deck includes spells, non-basic lands, and basic
   lands summing to 40 cards.
3. **Given** a pool where some card names cannot be resolved to embeddings, **When** the
   scorer builds a deck, **Then** unresolvable cards are skipped and the deck is still
   built from the remaining pool (provided enough cards remain).

---

### User Story 3 — Self-Play Match Generation (Priority: P3)

As a practitioner, I want to run `match-outcomes` with a `--generated-decks-path` argument
so that each match pits a scorer-built deck against an opponent built by one of five methods,
all enforcing same-set pairing.

**Why this priority**: This is the core self-play capability — the reason the feature exists.
It depends on both random-set pools (P1) and generated decks (P2).

**Independent Test**: Run `match-outcomes --generated-decks-path <file>` and verify that new
match outcome lines are appended to `match-outcomes.txt` in the standard format.

**Acceptance Scenarios**:

1. **Given** a generated-decks file with decks from multiple sets, **When** `match-outcomes`
   runs with `--generated-decks-path`, **Then** each match's deck A is taken from the
   generated-decks file and deck B is built using one of the 5 methods.
2. **Given** deck A has set code MH3, **When** deck B is built using methods 1–4, **Then**
   deck B's pool is generated from set MH3 (same set as deck A).
3. **Given** deck A has set code MH3, **When** method 5 is selected for deck B, **Then**
   deck B is a random line from the generated-decks file that also has set code MH3.
4. **Given** `match-outcomes` is invoked without `--generated-decks-path`, **When** match
   generation runs, **Then** behavior is identical to the existing Phase 0 flow (no
   regressions).
5. **Given** the 5 deck-building methods, **When** many matches are generated, **Then**
   methods are selected with relative weights 4:3:2:1:4 (method 5 has the same weight as
   method 1).

---

### User Story 4 — Random-Set Evaluation (Priority: P4)

As a practitioner, I want `evaluate-scorer` to use a randomly selected set by default (instead
of hardcoded RVR) so that evaluation reflects the scorer's performance across the full range
of sealed formats.

**Why this priority**: This is an independent improvement to the evaluation pipeline. It does
not block self-play match generation but makes evaluation results more representative.

**Independent Test**: Run `evaluate-scorer` without `--set` and verify that the generated
pools come from a randomly selected set. Run with `--set RVR` and verify all pools are RVR.

**Acceptance Scenarios**:

1. **Given** `evaluate-scorer` is run without `--set`, **When** pools are generated, **Then**
   a random sealed-legal set is selected and all N pools come from that set.
2. **Given** `evaluate-scorer` is run with `--set BLB`, **When** pools are generated, **Then**
   all N pools come from set BLB.

---

### Edge Cases

- What happens when a generated-decks file contains no other deck with the same set code as
  deck A when method 5 is selected? With 10,000 decks across ~215 eligible sets, this is
  statistically negligible and not handled — accepted trade-off.
- What happens when `build-decks` encounters a pool with fewer than 23 non-land cards that
  have valid embeddings? The greedy builder should return the best deck possible from the
  available cards, same as its current behavior with undersized pools.
- What happens when `match-outcomes` workers crash during self-play generation? Same crash
  recovery as Phase 0: the Python supervisor restarts crashed workers, and the Java worker
  resumes from the last completed match.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: `generate-pools` MUST support omitting `--set`, in which case each pool is
  generated from a randomly selected sealed-legal set.
- **FR-002**: `generate-pools` output MUST always include the set code as the first
  semicolon-delimited field on every line, regardless of whether `--set` was specified.
  Format: `SET_CODE;Card1|Card2|...|CardN`.
- **FR-003**: Random set selection MUST use the same eligibility criteria as Phase 0 match
  generation: sets with a draft booster template, excluding un-sets (funny type).
- **FR-004**: A new `build-decks` subcommand MUST read a pools file, build a scorer-guided
  greedy deck for each pool (including deterministic basic land assignment), and write a
  generated-decks file.
- **FR-005**: The generated-decks file format MUST be one line per deck:
  `SET_CODE;Card1|Card2|...|Card40` — a complete 40-card deck with its source set code.
- **FR-006**: `match-outcomes` MUST accept an optional `--generated-decks-path` argument.
  When absent, behavior MUST be identical to the existing Phase 0 flow.
- **FR-007**: When `--generated-decks-path` is present, deck A MUST be a random line from
  the generated-decks file, and deck B MUST be built using one of 5 weighted methods.
- **FR-008**: Deck B method selection MUST use relative weights 4:3:2:1:4 for methods 1
  through 5 respectively.
- **FR-009**: For methods 1–4, deck B's pool MUST be generated from the same set as deck A
  (same-set constraint).
- **FR-010**: For method 5, deck B MUST be a random line from the generated-decks file with
  the same set code as deck A.
- **FR-011**: Self-play match outcomes MUST be appended to the same `match-outcomes.txt` file
  in the same format as Phase 0 outcomes.
- **FR-012**: `evaluate-scorer` MUST use a randomly selected sealed-legal set by default
  instead of a hardcoded set.
- **FR-013**: `evaluate-scorer` MUST accept an optional `--set` argument to override the
  random set selection.

### Key Entities

- **Pools file** (`pools.txt`): one pool per line; format `SET_CODE;Card1|Card2|...|CardN`.
  Output of `generate-pools`. Contains non-basic-land card names from 6 boosters.
- **Generated-decks file**: one deck per line; format `SET_CODE;Card1|Card2|...|Card40`.
  Output of `build-decks`. Contains complete 40-card decks (spells + non-basic lands + basic
  lands) built by the scorer-guided greedy builder.
- **Match-outcomes file** (`match-outcomes.txt`): append-only; format
  `deck_A_cards;deck_B_cards;wins_A;wins_B`. Training data consumed by `train-scorer`.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A full self-play refinement loop (generate pools → build decks → generate
  matches → retrain scorer) can be completed end-to-end using only CLI commands without
  manual file editing or ad-hoc scripts.
- **SC-002**: Self-play match generation produces valid match outcome lines at a throughput
  comparable to Phase 0 match generation (same order of magnitude, measured in matches/hour).
- **SC-003**: Existing Phase 0 workflows (`match-outcomes` without `--generated-decks-path`,
  `generate-pools` with `--set`) continue to work without regressions.
- **SC-004**: After one iteration of the self-play loop, the retrained scorer's evaluation
  win rate against Forge is at least as good as the pre-self-play scorer (no regression from
  adding self-play data).

## Assumptions

- A trained scorer checkpoint from Phase 1 is available before running `build-decks`.
- Card embeddings (`.npz` files) exist for all cards in the generated pools.
- The forge-connector JAR is built and the Forge sibling checkout is available.
- The ~215 sealed-legal sets produce sufficient coverage (average ~46 pools per set at
  N=10,000) that method 5's same-set filter always finds at least one other deck.

## Dependencies

- Spec 013 (sealed deck scorer) — trained scorer model and `GreedyDeckBuilder`.
- Spec 011/012 (sealed dataset / training data) — `match-outcomes` supervisor,
  `MatchWorkerMain`, `DeckBuilder` Java classes.
- Spec 006 (card script parsing) — converted card text files for `build-decks`.
- The `generate-pools` / `PoolMain` infrastructure from the existing sealed module.

# Feature Specification: Draft agent game-played evaluation

**Feature Branch**: `022-draft-game-evaluation`
**Created**: 2026-08-09
**Status**: Draft
**Input**: User description: "Play Forge matches between decks drafted in the same pod of an existing draft corpus, and record the outcomes in the sealed match-outcome format so the existing win-rate tally reports per-agent results"

Derived from the root spec [`../2026-08-09-draft-game-evaluation.md`](../2026-08-09-draft-game-evaluation.md).
Rationale and sizing arithmetic live in
[`../../experiments/2026-08-09-draft-agent-gen4-online-grpo.md`](../../experiments/2026-08-09-draft-agent-gen4-online-grpo.md).

## Clarifications

### Session 2026-08-09

- Q: Matches complete out of draw order under concurrency, so what does the seed guarantee?
  → A: Nothing, so the reproducibility criterion was dropped. The seed option itself was
  then dropped too, once sampling moved into the workers: it reached nothing that samples,
  and a single seed shared across every worker would have made them all draw the same
  sequence.
- Q: What does the command report while running and when it ends? → A: A periodic progress
  line, and a summary on exit.
- Q: What shape should the progress output take? → A: The sealed `match-outcomes`
  supervisor's, adopted unchanged — a 60-second status line carrying elapsed time, matches
  completed, matches per minute and live workers, plus its worker-lifecycle lines.
- Q: How does work reach the workers, given Java cannot read `drafts.jsonl` without a JSON
  dependency and a second copy of the booster geometry? → A: Python projects the corpus once
  into a flat seat table and the workers sample from it autonomously, as
  `GeneratedDecksIndex` already does for `match-outcomes`. A consequence is that no count of
  failed pairings exists to report: a worker whose match fails just draws another.
- Q: What is the worker-count default, given sealed uses 12 for `match-outcomes` and 4 for
  `evaluate-scorer`? → A: 12, following `match-outcomes`, the long-running supervisor this
  most resembles.
- Q: What method tag should a Forge-reference seat carry once its deck is rebuilt by Forge's
  own builder? → A: `forge-native`. A distinct label is required, since the tally groups by
  that column alone and a shared tag would average the rebuilt and recorded decks into one
  win rate.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Measure an agent by games won (Priority: P1)

An operator has a draft corpus in which several agents drafted alongside each other in the
same pods. Every current measure of those agents is a function of the deck scorer. The
operator wants an independent measure: which agent's decks actually win games.

They point the command at the corpus, ask for a number of matches, and wait. The command
draws a pod at random, draws two seats from it, plays a match between those two decks, and
appends a row. When it finishes, the operator runs the existing win-rate tally over the
output file and reads per-agent win rates and a head-to-head matrix.

**Why this priority**: This is the whole feature. Without it there is no game-played
measure at all, and every other story is a refinement of how the sample is chosen.

**Independent Test**: Run the command against a corpus containing at least two agent labels
with a small pairing count, then run the tally over the output. Delivers per-agent win
rates that no existing tool can produce.

**Acceptance Scenarios**:

1. **Given** a corpus whose pods contain two or more distinct agent labels, **When** the
   command runs with a pairing count of N, **Then** the output file gains up to N rows, one
   per completed match, in the sealed match-outcome format.
2. **Given** a completed output file, **When** the existing win-rate tally is run over it
   with no modification to that tally, **Then** it reports per-agent win rate and a
   head-to-head matrix keyed by the agent labels.
3. **Given** any output row, **When** its two decks are traced back to the corpus, **Then**
   both were drafted by seats of the same draft record.
4. **Given** an output row, **When** its method columns are read, **Then** they hold the two
   seats' agent labels.

---

### User Story 2 - Evaluate one corpus among many (Priority: P2)

The default corpus file is append-only and shared, so it accumulates records from every
generation that ever wrote to it. An operator evaluating one generation restricts sampling
to that generation's run identifiers, and the command confirms at startup which records are
in scope.

**Why this priority**: Without it, an operator with a shared corpus can only evaluate
everything at once, which mixes generations into one measurement. It is a filter over a
working feature, so it is not required for the first useful run.

**Independent Test**: Run against a corpus holding two run identifiers, restricted to one,
and confirm every output row's decks trace to records carrying that identifier.

**Acceptance Scenarios**:

1. **Given** a corpus with records from two run identifiers, **When** the command runs
   restricted to one of them, **Then** no output row's decks come from a record carrying
   the other.
2. **Given** the same corpus, **When** the command runs with no restriction, **Then**
   records from both identifiers are eligible to be drawn.
3. **Given** either invocation, **When** the command starts, **Then** it echoes which run
   identifiers are in scope, or states that the whole corpus is.

---

### User Story 3 - Accumulate matches over a long session (Priority: P3)

Matches are slow and the operator does not know in advance how many they want. They start
the command with no pairing count, let it run, and interrupt it when the tally looks stable
enough. Later they start it again to add more.

**Why this priority**: A convenience over the bounded run. The same result is reachable by
invoking the bounded form repeatedly.

**Independent Test**: Start the command with no pairing count, interrupt it after some
matches, and confirm the output file is complete and readable, then re-invoke and confirm
rows are added rather than replaced.

**Acceptance Scenarios**:

1. **Given** no pairing count, **When** the command runs, **Then** it continues drawing and
   playing until interrupted.
2. **Given** an interrupt during a match, **When** the command exits, **Then** every match
   completed before the interrupt is present in the output and the file parses.
3. **Given** an existing output file, **When** the command is invoked again against it,
   **Then** the new rows are appended and the existing rows are unchanged.

---

### User Story 4 - Compare Forge's builder against the recorded one (Priority: P3)

The Forge reference seats drafted their cards but their decks were assembled by this
project's builder. An operator wants to know how much of the reference's result is drafting
and how much is building, so they divert a share of those seats to Forge's own sealed deck
builder, working from the same drafted cards. Both variants then appear as separate agents
in the tally, and pods holding one of each pit the two builders against each other directly.

**Why this priority**: An add-on to a working measurement. It is inert at its default, and
the feature delivers its primary value without it.

**Independent Test**: Run with the fraction at a half and confirm the tally reports both the
recorded and the rebuilt variant as separate rows with comparable sample sizes.

**Acceptance Scenarios**:

1. **Given** a corpus containing Forge reference seats, **When** the command runs with the
   fraction at zero, **Then** no seat is diverted and every row carries the label recorded in
   the corpus.
2. **Given** the same corpus, **When** the command runs with the fraction at one, **Then**
   every Forge reference seat plays a deck built by Forge's own builder and no row carries
   the recorded reference label.
3. **Given** a fraction strictly between zero and one, **When** the tally is run, **Then**
   the recorded and rebuilt variants appear as two distinct rows.
4. **Given** a fraction strictly between zero and one, **When** the same seat appears in
   several recorded matches, **Then** it carries the same label in all of them.
5. **Given** a pod holding one diverted and one undiverted reference seat, **When** that pair
   is drawn, **Then** it is not treated as a mirror and the match is played.

---

### Edge Cases

- A drawn pod's seats all carry one agent label and mirrors are excluded: the pod is
  skipped and another is drawn, so the resampling terminates.
- Every pod in the corpus is single-label and mirrors are excluded: no pairing can ever be
  drawn, so the run fails validation at startup rather than looping.
- The same pairing is drawn twice: both matches are played and both rows are written.
  Repeats are permitted by design.
- A match fails, its worker dies, or its worker is recycled mid-match: no row is written,
  the worker is restarted if it died, and it draws another pairing. Nothing is lost, because
  no pairing was reserved for it.
- A match hangs: the recycle bounds how long it can occupy a worker, so the run does not
  stall on it.
- Every worker dies repeatedly and no match completes: the status line shows the match count
  flat and the live-worker count low, which is how the operator sees it.
- The corpus file ends in a partial line: the partial record is ignored, consistent with
  existing readers of that file.
- An interrupt arrives before any match has completed: nothing is appended and the run
  reports an interrupt.
- A pairing count larger than the number of distinct pairs available: the run still
  completes, drawing repeats, because sampling is with replacement.
- A non-zero native fraction on a corpus holding no Forge reference seats: nothing is
  diverted and the run proceeds, since the fraction selects from an empty population.
- A native fraction of 1 on a pod whose only reference seats are all diverted: that pod
  yields no `forge-full` pairings at all, only `forge-native` ones.
- Forge's builder produces a deck from a diverted seat's pool that differs in size from the
  recorded 40: the built deck is played as Forge built it, since Forge's builder defines
  what a legal sealed deck is here.

## Requirements *(mandatory)*

### Functional Requirements

Sampling

- **FR-001**: The command MUST draw a pairing by selecting a draft record uniformly at
  random, then two distinct seats of that record uniformly at random.
- **FR-002**: The command MUST NOT pair seats from different draft records.
- **FR-003**: The command MUST exclude pairs whose two seats carry the same agent label by
  default, and MUST provide a flag that retains them.
- **FR-004**: When mirror pairs are excluded, the command MUST discard a drawn mirror and
  draw again.
- **FR-005**: When mirror pairs are excluded, the command MUST skip records whose seats all
  carry one agent label, so that resampling terminates.
- **FR-006**: The command MUST sample with replacement, permitting the same pairing to be
  drawn more than once within a run and across runs.
- **FR-007**: The command MUST accept an optional pairing count and stop once that many
  matches have been recorded by this run. Absent, it MUST keep playing until interrupted.
Match execution

- **FR-008**: The command MUST play each pairing as one best-of-N match, where N is an
  operator-supplied positive odd integer defaulting to 3.
- **FR-009**: The command MUST accept a worker count controlling how many matches are
  played concurrently, defaulting to 12.
- **FR-010**: A pairing whose match fails MUST write no row, and the run MUST continue with
  further pairings. A worker that dies MUST be restarted, and workers MUST be recycled, both
  as the sealed match-generation supervisor does. A match cut short by a recycle MUST be
  treated as any other failure.
- **FR-011**: The command MUST append each match's row as that match completes, so that an
  interrupted run retains every match already played.

Input

- **FR-012**: The command MUST read a draft corpus whose path defaults to the standard
  drafts file, in the format specified by the draft-agent spec.
- **FR-013**: The command MUST accept a repeatable run-identifier filter, and MUST consider
  the whole corpus in scope when none is supplied.
- **FR-014**: The command MUST echo at startup which run identifiers are in scope, or state
  that the whole corpus is.
- **FR-015**: The command MUST take each match's set from the draft record the two decks
  were drafted in.
- **FR-016**: The command MUST tolerate a trailing partial line in the corpus, ignoring the
  incomplete record, as existing readers of that file do.

Output

- **FR-017**: The command MUST append to an output file, whose path is operator-supplied
  with a default, in the sealed match-outcome format specified by the sealed deck-picker
  spec.
- **FR-018**: The command MUST write one row per match.
- **FR-019**: The command MUST write the two seats' agent labels into the row's two method
  columns.
- **FR-020**: The command MUST write every other column with the meaning the sealed
  match-outcome format gives it, so that existing readers of that format need no change.

Operator surface

- **FR-021**: The command MUST validate, before playing any game, that the corpus exists
  and parses, that at least one pairing survives the run-identifier and mirror filters, and
  that the best-of value is a positive odd integer.
- **FR-022**: The command MUST exit 0 when the requested matches have been recorded or the
  run was cleanly interrupted, 2 on a validation failure, missing file, or bad flag, and
  130 when interrupted before the first match completed.
- **FR-023**: The command MUST report progress the way the sealed match-generation
  supervisor does: a status line every 60 seconds carrying elapsed seconds, matches
  completed, matches per minute, and the count of live workers against the configured
  count, in that order and in the same shape.
- **FR-024**: The command MUST emit the same worker-lifecycle lines that supervisor emits:
  one on starting the workers, one when a worker exits and is restarted, and one on
  shutdown.
- **FR-025**: The command MUST print a summary when it ends, including when it ends by
  interrupt, reporting matches played, elapsed time, and the output path.

Forge-native decks

- **FR-026**: The command MUST accept a fraction between 0 and 1, defaulting to 0, that
  controls what share of Forge reference seats are diverted to Forge's own deck builder.
- **FR-027**: The command MUST select seats labelled `forge-full` independently with that
  probability, deciding once per seat as the seat table is written, so a seat is diverted in
  every pairing it appears in or in none.
- **FR-028**: A diverted seat's deck MUST be built by Forge's own sealed deck builder from
  that seat's drafted pool, reconstructed from the draft record. The deck recorded in the
  corpus MUST NOT be used for that seat.
- **FR-029**: A diverted seat MUST carry the label `forge-native` wherever a label is used:
  in the output row's method column, in mirror exclusion, and hence in the tally.
- **FR-030**: At a fraction of 0 the command MUST behave exactly as it does without the
  option, diverting no seat and reconstructing no pool.

### Key Entities

- **Draft record**: One pod's draft, holding a run identifier, a set, and one seat per
  player. Read-only input; the unit of random selection.
- **Seat**: One player's position in a record, carrying an agent label, a built deck, and
  the cards it drafted. Two distinct seats of one record form a pairing. A seat is either
  undiverted, playing its recorded deck under its recorded label, or diverted, playing a
  deck Forge builds from its drafted pool under the label `forge-native`. Which one is
  decided once, when the corpus is loaded.
- **Pairing**: Two seats of the same record selected to play, each contributing either a
  deck or a pool to be built. Not persisted in its own right; it becomes one output row.
- **Match outcome row**: One completed best-of-N match, in the sealed match-outcome format,
  with the two agent labels in the method columns.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The existing win-rate tally reads the output file and reports per-agent win
  rates and a head-to-head matrix, with no change to that tally.
- **SC-002**: Every recorded match is between two decks drafted in the same pod, in 100% of
  rows.
- **SC-003**: An interrupted run leaves an output file that parses completely, and a later
  run appends to it without altering earlier rows.
- **SC-004**: An operator can measure a new agent against the references it drafted
  alongside using only this command and the existing tally, with no further tooling.
- **SC-005**: Matches that fail never abort the run or corrupt the output; the run
  continues and records the pairings that do complete.
- **SC-006**: An operator can tell from the command's own output how fast matches are
  accumulating and how many workers are alive, so a run that has stopped producing is
  visible without inspecting the output file.
- **SC-007**: With the native fraction between 0 and 1, the tally reports the recorded and
  the Forge-built variants of the reference agent as two separate rows, so the effect of the
  builder is readable independently of the drafting.

## Assumptions

- The corpus is produced by the existing draft-data generation command and needs no
  migration. Boundary validation is therefore parse-level: FR-012 and FR-016 require the
  corpus to parse and a trailing partial line to be tolerated, and FR-021 fails the run
  before any game if nothing survives filtering. Semantic record validation — empty decks,
  absent scores, pods spanning sets — is deliberately absent, having been measured as
  non-occurring across the corpus rather than assumed.
- Both decks of a pairing are 40 cards including basic lands, as the corpus records them.
  A diverted seat's deck is instead whatever Forge's builder produces from its pool.
- A seat's drafted pool is reconstructible from its draft record alone, which the draft
  package already relies on; the corpus stores no pool of its own.
- The label `forge-full` identifies the Forge reference seats. It is the only label the
  native fraction selects from.
- Which side plays first in each game of a match is decided by the match rules as Forge
  implements them, and is recorded in the row's play column. This feature does not control
  it.
- The worker count defaults to 12, following the sealed match-generation supervisor. The
  sealed evaluation path this feature reuses defaults to 4; the higher value is chosen
  because this command is long-running like the supervisor, not bounded like the
  evaluation.
- Matches are played by the same Forge match execution path the sealed evaluation uses;
  this feature adds sampling and recording, not game logic.
- The output file is a distinct file from the sealed match-outcome corpus, so draft games
  never enter scorer training data.

## Dependencies

- An existing draft corpus containing at least two distinct agent labels within single
  pods.
- The Forge match execution path used by the sealed evaluation command.
- The existing win-rate tally script, which is read-only for this feature and must not be
  modified.

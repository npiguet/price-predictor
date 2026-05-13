# Feature Specification: Sealed Training Data Generation

**Feature Branch**: `012-sealed-training-data`  
**Created**: 2026-04-08  
**Status**: Draft  
**Input**: User description: "Phase 0 — Training dataset generation for sealed deck scorer, as described in 2026-03-28-sealed-deck-picker.md"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Generate Match Outcome Data (Priority: P1)

As a model trainer, I want to generate a large dataset of sealed-format game outcomes so that the deck scorer (Phase 1) has sufficient training signal to learn which decks are strong.

The user runs a single command that spawns multiple parallel workers. Each worker independently and repeatedly: picks a random sealed-legal expansion set, generates two booster pools (6 boosters each), builds a deck from each pool, plays a best-of-3 match via the Forge AI, and appends the outcome to a shared results file. The process runs indefinitely until the user stops it.

**Why this priority**: Without match outcome data, no subsequent phase of the sealed deck project can proceed. This is the foundational data generation step.

**Independent Test**: Can be fully tested by running the command, letting it produce a few dozen outcomes, and verifying the output file contains well-formed records with valid card names and plausible win counts.

**Acceptance Scenarios**:

1. **Given** the system is properly configured with Forge available, **When** the user runs the match outcome generation command, **Then** workers begin producing match outcome records and appending them to the output file.
2. **Given** multiple workers are running, **When** a worker crashes or the JVM terminates unexpectedly, **Then** the supervisor automatically restarts that worker without affecting other workers or corrupting the output file.
3. **Given** the process has been running, **When** the user inspects the output file, **Then** each line contains exactly four semicolon-separated fields: deck A card names (pipe-separated), deck B card names (pipe-separated), wins for A (0-2), and wins for B (0-2), and wins_A + wins_B equals 2 or 3.
4. **Given** workers are running, **When** the user interrupts the supervisor (e.g. Ctrl+C), **Then** the supervisor terminates all worker processes it started and exits cleanly, leaving no orphaned worker processes behind.

---

### User Story 2 - Configurable Parallelism (Priority: P2)

As a user with varying hardware, I want to control how many worker processes run in parallel so that I can maximize throughput without overloading my machine.

**Why this priority**: Different machines have different CPU/memory capacities. The default of 12 workers may be too many or too few depending on the hardware.

**Independent Test**: Run the command with an explicit worker count (e.g. 2), verify exactly that many worker processes are spawned.

**Acceptance Scenarios**:

1. **Given** the user specifies a worker count, **When** the generation command starts, **Then** exactly that many worker processes are created.
2. **Given** no worker count is specified, **When** the generation command starts, **Then** 12 workers are created by default.

---

### User Story 3 - Varied Deck Quality via Multiple Construction Methods (Priority: P2)

As a model trainer, I want the generated data to contain decks of varying quality levels — from competent to mediocre to random — so that the scorer learns to differentiate across the full spectrum of deck quality rather than only distinguishing between similarly-strong decks.

**Why this priority**: A scorer trained only on well-built decks would not learn to distinguish bad decks from terrible ones. Diversity in deck quality creates a richer training signal.

**Independent Test**: Collect a sample of generated decks and verify that the four deck construction methods appear at approximately their expected proportions.

**Acceptance Scenarios**:

1. **Given** a deck is being constructed, **When** the construction method is selected, **Then** it is chosen randomly with the following approximate weights: standard Forge builder (40%), Forge builder with 3-card swap (30%), Forge builder with 8-card swap (20%), random card selection (10%).
2. **Given** a deck is built with methods 2-4, **When** basic lands are assigned, **Then** basic lands are removed (non-basic lands kept), and the deck is filled to exactly 40 cards with basic lands distributed proportional to the WUBRG mana pip demands of the spells (non-basic lands contribute no pips).

---

### User Story 4 - Expansion Set Diversity (Priority: P3)

As a model trainer, I want matches to use a wide variety of sealed-legal expansion sets so that the scorer generalizes across different card pools and set mechanics rather than overfitting to a single set.

**Why this priority**: A model trained on one set would not generalize. Set diversity is important but follows naturally from random selection.

**Independent Test**: Run generation for several hundred outcomes and verify that multiple distinct sets appear in the data.

**Acceptance Scenarios**:

1. **Given** the system is generating matches, **When** an expansion set is selected, **Then** it is chosen at random from all sets that provide "draft boosters" or "play boosters", excluding un-sets (joke sets) and aftermath-style sets.
2. **Given** a set has been selected, **When** booster pools are generated, **Then** each of the two players receives cards from 6 independently generated boosters of that set (12 boosters total per match).

---

### Edge Cases

- What happens when Forge does not support a particular set's sealed format? The set is automatically excluded — only sets for which forge-connector can generate sealed boosters are in the selection pool.
- What happens when a game hangs or takes excessively long? In practice, excessively long games cause the worker JVM to crash, which is handled by the supervisor's automatic restart mechanism.
- What happens when workers slow down over extended runs? Forge AI games tend to slow down within long-running JVMs. The supervisor periodically recycles the longest-running worker by terminating it; the monitor thread automatically restarts it with a fresh JVM.
- What happens when the output file grows very large? The append-only flat file format must remain performant for concurrent writes from many workers. Each worker writes one complete line atomically per match.
- ~~What happens when a deck construction method produces fewer than 23 non-land cards?~~ Not applicable — 6 boosters always produce enough non-land cards for any construction method. No special handling required.

## Clarifications

### Session 2026-04-09

- Q: Should the supervisor provide progress feedback during long runs? → A: Supervisor periodically prints a status line (total matches completed, generation rate, alive workers).
- Q: Can a match pit players from different expansion sets? → A: No. Both players always use the same set, matching real sealed play rules.
- Q: What happens when the process is restarted after a previous run? → A: Always append to existing file, accumulating data across runs.
- Q: What if a pool has fewer than 23 non-land cards? → A: Not possible. 6 boosters always produce enough non-land cards; no special handling needed.
- Q: How are duplicate card copies represented in the output? → A: Repeat the card name once per copy (e.g., `Lightning Strike|Lightning Strike|...`).
- Q: Should the CLI accept a target match count to stop automatically? → A: No. The process always runs indefinitely until the user stops it manually.
- Q: Are deck construction method weights configurable via CLI or hardcoded? → A: Hardcoded constants in source code.
- Q: What is the worker process architecture? → A: Python supervisor spawns Java (forge-connector) worker subprocesses.
- Q: How does colorless mana factor into land rebalancing? → A: Colorless {C} pips are not handled by the land rebalancing algorithm (Forge's built-in `LimitedDeckBuilder.addLands()` only supports WUBRG). Cards requiring {C} may be slightly disadvantaged, but this affects very few sets and cards — not enough examples for the model to learn from anyway. Accepted trade-off.
- Q: How are ineligible sets (un-sets, aftermath) identified for exclusion? → A: Data-driven from Forge: a set is eligible if forge-connector can generate sealed boosters for it.
- Q: How frequently should the supervisor print status? → A: Every 60 seconds.
- Q: Should the output format include the expansion set name as an additional field? → A: No. Keep the 4-field format (`deck_A;deck_B;wins_A;wins_B`); set info is not needed in the output.
- Q: Should the output record the deck construction method ID used for each deck? → A: No. Method distribution correctness is verified via code review and unit tests only; no runtime tracking needed.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST generate sealed-format match outcomes by selecting a random sealed-legal expansion set, producing two 6-booster pools **from that same set**, constructing a deck from each pool, playing a best-of-3 match, and recording the result.
- **FR-002**: System MUST support four deck construction methods selected randomly per deck with the following hardcoded weights:
  1. Standard Forge sealed deck generator (default weight 40%)
  2. Same as method 1, but N=3 random cards are swapped out of the deck and replaced by type-matched cards from the remaining pool: spells swap with spells, non-basic lands swap with non-basic lands (default weight 30%)
  3. Same as method 2 but N=8 swaps (default weight 20%)
  4. 23 random **spells** (non-land cards) picked from the pool; no non-basic lands included (default weight 10%)
- **FR-003**: For deck construction methods 2-4, the system MUST rebalance basic lands after card swaps: remove all basic lands (keeping non-basic lands), then add basic lands to reach exactly 40 cards with distribution proportional to the WUBRG mana pip counts of the selected nonland cards. Colorless {C} pips are not handled (accepted trade-off — too few affected cards for the model to learn from). Note: `LimitedDeckBuilder.addLands()` is private and cannot be called externally; the pip-proportional logic is reimplemented directly in `DeckBuilder.rebalanceLands()`.
- **FR-004**: System MUST record each match outcome as a single line in the format: `deck_A_card_names;deck_B_card_names;wins_A;wins_B` where card names are pipe-separated (duplicate copies repeat the name, e.g. `Lightning Strike|Lightning Strike|...`) and wins sum to 2 or 3.
- **FR-005**: System MUST run multiple worker processes in parallel (configurable, default 12), all appending to the same output file.
- **FR-006**: System MUST monitor worker processes and automatically restart any worker that dies unexpectedly, without affecting other workers or corrupting existing output data.
- **FR-007**: When the supervisor receives an interrupt or termination signal (e.g. Ctrl+C), it MUST terminate all worker processes it started and exit cleanly, leaving no orphaned workers behind.
- **FR-008**: System MUST exclude ineligible sets from the expansion set selection pool. A set is eligible if and only if forge-connector can successfully generate sealed boosters for it. This implicitly excludes un-sets, aftermath-style sets, and any other sets Forge does not support for sealed play.
- **FR-009**: System MUST be invocable via the command `python -m sealed match-outcomes`.
- **FR-010**: Each card instance in a pool MUST be selectable at most once (matching physical sealed rules), even if multiple copies of that card exist in the pool.
- **FR-011**: System MUST write output by appending to `./output/sealed/match-outcomes.txt`, creating the directory structure if it does not exist. If the file already exists from a previous run, new outcomes are appended (data accumulates across runs).
- **FR-012**: The supervisor MUST print a status summary to the terminal every 60 seconds, including total matches completed, current generation rate, and number of alive workers.
- **FR-013**: The supervisor MUST recycle the longest-running worker every 60 seconds (at each status interval) by terminating it. The monitor thread automatically restarts it with a fresh JVM. This prevents throughput degradation caused by long-running Forge JVMs accumulating internal state.

### Key Entities

- **Expansion Set**: An MTG expansion that supports sealed play (provides draft or play boosters). Excludes un-sets and aftermath-style sets.
- **Booster Pool**: A collection of 84-90 cards generated from 6 boosters of a single expansion set. Each player in a match receives their own distinct pool.
- **Deck**: A 40-card selection from a booster pool, consisting of approximately 23 non-land cards plus basic lands distributed according to the deck's mana requirements.
- **Match Outcome**: The result of a best-of-3 game between two decks, recording the card composition of each deck and the number of games won by each side.
- **Worker**: An independent Java subprocess (running forge-connector) that generates and plays matches. Multiple workers run in parallel under a Python supervisor that restarts crashed workers.
- **Supervisor**: A Python process (`python -m sealed match-outcomes`) that spawns and monitors Java worker subprocesses, handles restarts, prints status, and responds to termination signals.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The system can sustain continuous match generation at a rate of at least 500 matches per hour on a machine with 12 workers.
- **SC-002**: Worker crashes are detected and the crashed worker is restarted within 10 seconds, with no data loss or corruption in the output file.
- **SC-003**: Over a sample of 1000+ generated matches, each of the four deck construction methods appears within 5 percentage points of its target weight (40/30/20/10).
- **SC-004**: Over a sample of 1000+ generated matches, at least 10 distinct expansion sets are represented in the data.
- **SC-005**: 100% of output lines conform to the specified format: four semicolon-separated fields, pipe-separated card names in the first two fields, integer win counts in the last two fields summing to 2 or 3.
- **SC-006**: The system can generate a dataset of at least 100,000 match outcomes without manual intervention (beyond initial startup).

## Assumptions

- The Forge game engine and its sealed deck builder are available and accessible from the forge-connector module.
- The Forge AI can play games autonomously without user interaction.
- The forge-connector module already has the necessary harnesses to list expansion sets and generate boosters. Deck building and game playing capabilities are added by this feature.
- Card names in the output use the canonical Forge card naming convention.
- Workers appending single lines to the output file is safe for concurrent writes on the target operating system.
- An example of the supervisor/worker pattern used here is available in the jumpstart-tierlist project (`../jumpstart-tierlist`).

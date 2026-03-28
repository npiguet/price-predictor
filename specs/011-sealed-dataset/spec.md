# Feature Specification: Sealed Dataset Preparation

**Feature Branch**: `011-sealed-dataset`
**Created**: 2026-03-28
**Status**: Draft
**Parent Spec**: [`specs/sealed-deck-picker.md`](../sealed-deck-picker.md) — this feature implements Stage 0 (Training Dataset Preparation) of the Training Curriculum defined there
**Input**: User description: "A new python command line is created to prepare a training dataset for a future sealed deck picker feature. The sealed deck feature is described in details in specs/sealed-deck-picker.md. This feature (and its plan an implementation) aims to implement what is described in that file as Stage 0 of the Training Curriculum."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Encode Card Embeddings (Priority: P1)

A researcher preparing training data runs the `encode-cards` command against a folder of Forge card scripts. The command reads each card's structured Oracle text, generates a 512-dimensional vector representation using the pretrained price predictor encoder, and writes a named embedding file alongside each card script. Cards that already have an embedding file are silently skipped, so the command is safe to re-run after adding new cards.

**Why this priority**: Card embeddings are the foundational input for both pool assembly and model training — nothing else in Stage 0 or beyond can proceed without them.

**Independent Test**: Can be fully tested by pointing the command at a small folder of card scripts and verifying that a named embedding file is produced for each card, then re-running and confirming no files are overwritten and the run completes near-instantly.

**Acceptance Scenarios**:

1. **Given** a folder containing card scripts with no existing embedding files, **When** the researcher runs `encode-cards`, **Then** every card script has a corresponding embedding file in the same folder when the command finishes.
2. **Given** a folder where all cards already have embedding files, **When** the researcher runs `encode-cards` again, **Then** no existing files are modified and the command completes without re-processing any card.
3. **Given** a folder where some cards have embedding files and some do not, **When** the researcher runs `encode-cards`, **Then** only the cards missing embedding files are processed, and all cards have embedding files when the command finishes.
4. **Given** an invalid encoder model path, **When** the researcher runs `encode-cards`, **Then** the command reports a clear error message and exits without creating any partial output.

---

### User Story 2 - Generate Sealed Pools (Priority: P2)

A researcher runs the `generate-pools` command, specifying a Magic: The Gathering set code and a desired number of pools. The command uses Forge's internal booster-generation logic to simulate opening 6 boosters per pool, collects the resulting card names, and writes all pools to a single text file — one pool per line, card names separated by semicolons. Basic lands that appear in the booster results are filtered out before writing, since they are represented separately at training time via fixed basic land slots rather than as pool entries.

**Why this priority**: The pool dataset drives what sealed positions the model trains on. Without pools, Stage 1 training cannot begin. Card embeddings must exist first (P1), but pool generation is otherwise independent.

**Independent Test**: Can be fully tested by generating a small batch of pools for a known set and verifying the output file contains the expected number of lines, each with a realistic set-appropriate card count, no basic lands, and no malformed entries.

**Acceptance Scenarios**:

1. **Given** a valid set code and a requested pool count, **When** the researcher runs `generate-pools`, **Then** the output file contains exactly the requested number of lines and each line is a semicolon-separated list of card names.
2. **Given** the generated pools file, **When** any line is inspected, **Then** no basic land names appear in that line.
3. **Given** a valid set code, **When** the researcher runs `generate-pools`, **Then** each pool contains between 84 and 90 card names, consistent with 6 booster packs for that set.
4. **Given** an invalid or unrecognized set code, **When** the researcher runs `generate-pools`, **Then** the command reports a clear error message and produces no output file.
5. **Given** an existing pools file at the target path, **When** the researcher runs `generate-pools`, **Then** the existing file is overwritten with fresh pools (not appended to).

---

### User Story 3 - Incremental Encoding After Encoder Retrain (Priority: P3)

After retraining the price predictor encoder, a researcher deletes stale embedding files and re-runs `encode-cards`. Because the command skips any card that already has an embedding file, only the deleted (stale) entries are re-processed — leaving any intentionally-kept embeddings untouched.

**Why this priority**: This is a workflow convenience and correctness concern for encoder iteration cycles. It does not add new functionality; it documents the expected incremental behavior already covered by the P1 story.

**Independent Test**: Can be tested by deleting a subset of embedding files and confirming that only those cards are re-encoded on the next run.

**Acceptance Scenarios**:

1. **Given** a folder where a subset of embedding files has been manually deleted, **When** the researcher runs `encode-cards`, **Then** only the cards missing embedding files are re-encoded, and all previously existing files are left unchanged.

---

### Edge Cases

- If the output directory for pools does not exist, it is created automatically (see Assumptions).
- MTG card names never contain semicolons, so the semicolon pool separator cannot produce malformed lines.
- If `encode-cards` is interrupted mid-run, no partial embedding file is left on disk (atomic write). A re-run will process only the cards whose embedding files were not yet fully written.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The `encode-cards` command MUST scan a specified folder of card scripts and generate a 512-dimensional embedding for each card found.
- **FR-002**: The `encode-cards` command MUST write each embedding as a file in the same cards folder, using the same filename as the card script but with a `.npz` extension (e.g., `Lightning-Bolt.txt` → `Lightning-Bolt.npz`).
- **FR-003**: The `encode-cards` command MUST skip any card whose embedding file already exists in the cards folder, making repeated runs safe and incremental.
- **FR-003a**: The `encode-cards` command MUST write each embedding atomically (write to a temporary file, then rename to final name on success), so that an interrupted run leaves no partial files on disk and a re-run recovers correctly.
- **FR-004**: The `encode-cards` command MUST accept command-line arguments for: the encoder model path, the vocabulary file path, and the cards folder path — with documented defaults matching the project's standard model layout.
- **FR-005**: The `generate-pools` command MUST generate a configurable number of sealed pools, each consisting of cards drawn from 6 boosters of a specified set, using Forge's booster generation logic.
- **FR-006**: The `generate-pools` command MUST write all pools to a single flat text file, one pool per line, with card names separated by semicolons.
- **FR-007**: The `generate-pools` command MUST filter out any basic lands that Forge's booster generator includes in a pool, so that no basic land names appear in the pools text file.
- **FR-008**: The `generate-pools` command MUST accept command-line arguments for: the set code, the number of pools to generate, and the output directory path — with documented defaults (set: RVR, count: 10,000, path: output/sealed/pools/{set-code}/).
- **FR-009**: The two commands (`encode-cards` and `generate-pools`) MUST be independently executable — neither requires the other to have run first in the same invocation.
- **FR-010**: Both commands MUST report meaningful progress to the terminal so the researcher knows the operation is proceeding (e.g., a count of cards processed or pools generated).
- **FR-011**: Both commands MUST report clear, actionable error messages when given invalid inputs (nonexistent paths, unrecognized set codes, etc.) and exit with a non-zero status code.

### Key Entities

- **Card Script**: A Forge-format text file representing a single MTG card, including its Oracle text, mana cost, type line, and other structured attributes. Each card script is the source of truth for embedding generation.
- **Card Embedding**: A 512-dimensional numerical vector derived from a card script's structured text. Stored as a named file per card and used at training time to represent that card in a pool.
- **Sealed Pool**: A list of 84–90 card names representing the non-land cards opened from 6 boosters of a single MTG set. Stored as one line in the pools text file.
- **Pool Dataset**: The complete collection of generated sealed pools stored in a flat text file. Consumed by the training system to assemble input tensors for each training episode.
- **Basic Land Embedding**: An embedding file for each of the 5 basic land types (Plains, Island, Swamp, Mountain, Forest) plus colorless. Stored in the cards folder alongside non-land card embeddings. Used at training time to append to each pool, but never included in the pool text file.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Running `encode-cards` on a fresh cards folder of 30,000+ card scripts completes without errors and produces one embedding file per card.
- **SC-002**: Re-running `encode-cards` on a fully-encoded cards folder produces no changes and reports zero cards processed, confirming that skipping already-encoded cards works correctly.
- **SC-003**: Running `generate-pools` with a count of 10,000 produces a pools file containing exactly 10,000 lines, each with 84–90 card names and no basic lands.
- **SC-004**: 100% of generated pool lines can be successfully loaded by the training system — meaning every card name in every pool has a corresponding embedding file in the cards folder (given that `encode-cards` has been run for the same set).
- **SC-005**: Both commands report per-item progress updates so a researcher can estimate completion without external tooling.

## Clarifications

### Session 2026-03-28

- Q: What happens when `encode-cards` is interrupted mid-run — are partial embedding files left on disk, and will a re-run recover correctly? → A: Atomic writes (temp file → rename on success); no partial files left on disk, re-run recovers cleanly.

## Assumptions

- All structural constants used in this spec (embedding dimensions, pool size bounds, basic land slot count, command signatures and defaults) are defined in [`specs/sealed-deck-picker.md`](../sealed-deck-picker.md). If those values change, this spec must be reviewed for consistency.
- This feature introduces a new top-level module (`sealed`) invoked as `python -m sealed`. It is separate from the existing `python -m price_predictor` module — they share no entry point and may evolve independently.
- The pretrained price predictor encoder and vocabulary file are available at their default paths before `encode-cards` is run. This feature does not retrain or modify the encoder.
- The Forge connector exists (from feature 002) but does not yet implement pool generation. Adding the pool generation logic to the connector is in scope for this feature, alongside the `generate-pools` CLI command that invokes it.
- Card names in the Forge card scripts match the names used in pool lines — no name normalization is needed between the two commands.
- Basic land embedding files (Plains, Island, Swamp, Mountain, Forest, and Wastes or a colorless placeholder) are created by `encode-cards` as a natural byproduct of scanning the cards folder, since basic land scripts reside there.
- Overwriting an existing pools file (rather than appending) is the correct behavior for `generate-pools`, since a re-run implies a fresh dataset is wanted.
- The output directory for pools is created automatically if it does not exist.

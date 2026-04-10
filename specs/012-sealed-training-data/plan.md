# Implementation Plan: Sealed Training Data Generation

**Branch**: `012-sealed-training-data` | **Date**: 2026-04-09 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/012-sealed-training-data/spec.md`

## Summary

Generate a large dataset of sealed-format match outcomes for training a deck scorer. A Python supervisor (`python -m sealed match-outcomes`) spawns configurable Java worker subprocesses (forge-connector), each of which independently picks a random sealed-legal set, generates two 6-booster pools, constructs decks using four weighted methods, plays best-of-3 matches via Forge AI, and appends results to a shared flat file. The supervisor monitors workers, restarts crashes, reports status every 60 seconds, and handles clean shutdown.

## Technical Context

**Language/Version**: Python 3.14+ (supervisor), Java 17+ (forge-connector worker)
**Primary Dependencies**: Python stdlib only (subprocess, signal, time, pathlib); Java: forge-game 2.0.10-SNAPSHOT (already in forge-connector pom.xml)
**Storage**: Append-only flat text file at `./output/sealed/match-outcomes.txt`
**Testing**: pytest (Python supervisor unit tests); JUnit 5 (Java worker unit tests)
**Target Platform**: Local workstation (Windows, same as existing project)
**Project Type**: CLI tool (supervisor + worker subprocess farm)
**Performance Goals**: ≥500 matches/hour with 12 workers (SC-001)
**Constraints**: Worker crash recovery within 10 seconds (SC-002); no new Python pip dependencies
**Scale/Scope**: Target 100,000+ match outcomes (SC-006); 12 parallel workers default

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Fast Automated Tests | ✅ Pass | Python unit tests for supervisor logic (process monitoring, signal handling, status reporting), deck construction methods, land rebalancing. Java unit tests for deck building, game orchestration, output formatting. Integration tests are inherently slow (Forge AI games) and will be clearly separated. |
| II. Simplicity First | ✅ Pass | No new Python dependencies. Supervisor follows proven pattern from jumpstart-tierlist. Java worker extends existing forge-connector module. No speculative abstractions. |
| III. Data Integrity | ✅ Pass | Output format validated (4 semicolon-separated fields, wins sum to 2 or 3). Atomic line-level appends. Worker crashes cannot corrupt existing data (append-only file). |
| IV. DDD & Separation | ✅ Pass | Python supervisor in `sealed/application/` and `sealed/infrastructure/`. Java match logic in forge-connector domain classes. Clear separation: Python orchestrates processes, Java plays games. |
| V. Forge Interoperability | ✅ Pass | Extends existing forge-connector JAR with new Java classes for deck building and game playing. No new interop mechanism. |
| VI. Documentation | ✅ Pass | README update required: new `match-outcomes` command, workflow description, output format documentation. |

**Gate result: PASS. No violations.**

### Post-Design Re-Check

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Fast Automated Tests | ✅ Pass | `DeckBuilderTest` (land rebalancing, method selection), `MatchResultWriterTest` (format validation), `test_match_outcomes.py` (supervisor logic — mock subprocess). Integration tests (actual Forge games) separated via JUnit `@Tag("integration")`. |
| II. Simplicity First | ✅ Pass | No new Python dependencies. Java classes are thin wrappers around Forge APIs. Supervisor follows proven jumpstart-tierlist pattern. No speculative abstractions. |
| III. Data Integrity | ✅ Pass | Output format enforced by `MatchResultWriter` (validates wins sum). Atomic single-line appends. Worker crashes cannot corrupt existing file content. |
| IV. DDD & Separation | ✅ Pass | Python: `MatchOutcomeSupervisor` (application) orchestrates `MatchWorkerConnector` (infrastructure). Java: `MatchGenerator` (application-level orchestration), `DeckBuilder`/`GamePlayer` (domain logic), `MatchResultWriter`/`MatchWorkerMain` (infrastructure). |
| V. Forge Interoperability | ✅ Pass | Extends forge-connector JAR. Adds `forge-gui` and `forge-ai` dependencies (system scope). Uses standard Forge APIs (`SealedDeckBuilder`, `Match`, `LobbyPlayerAi`). |
| VI. Documentation | ✅ Pass | `quickstart.md` covers the new command. CLI contract documents all arguments and output format. README update tracked as a task. |

**Post-design gate result: PASS.**

## Project Structure

### Documentation (this feature)

```text
specs/012-sealed-training-data/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/
│   └── cli.md           # CLI subcommand contract
└── tasks.md             # Phase 2 output (/speckit.tasks — not yet created)
```

### Source Code

```text
# Python supervisor (extends existing sealed module)
src/sealed/
├── application/
│   └── match_outcomes.py            # MatchOutcomeSupervisor: spawn, monitor, restart workers
└── infrastructure/
    ├── cli.py                       # Extended: add match-outcomes subcommand
    └── match_worker_connector.py    # MatchWorkerConnector: subprocess call to MatchWorkerMain

# Java worker (extends existing forge-connector module)
forge-connector/src/main/java/com/pricepredictor/connector/
├── MatchWorkerMain.java             # Worker entry point: init Forge, loop match generation
├── MatchGenerator.java              # Core logic: pick set, generate pools, build decks, play match
├── DeckBuilder.java                 # Deck construction methods 1-4, land rebalancing
├── GamePlayer.java                  # Match/RegisteredPlayer/LobbyPlayerAi orchestration
├── MatchResult.java                 # Value object: deckA, deckB, winsA, winsB
├── MatchResultWriter.java           # Append one result line to output file
└── GuiHeadless.java                 # IGuiBase stub for headless Forge (used by rewritten ForgeEnvironmentInitializer)

# Modified Java files
forge-connector/src/main/java/com/pricepredictor/connector/
└── ForgeEnvironmentInitializer.java # Rewritten: FModel.initialize() + GuiHeadless (replaces manual StaticData setup)

# Tests
tests/unit/sealed/application/test_match_outcomes.py          # Supervisor unit tests
tests/unit/sealed/infrastructure/test_match_worker_connector.py
forge-connector/src/test/java/com/pricepredictor/connector/DeckBuilderTest.java
forge-connector/src/test/java/com/pricepredictor/connector/MatchResultWriterTest.java
forge-connector/src/test/java/com/pricepredictor/connector/MatchGeneratorTest.java
```

**Structure Decision**: Extends the existing `src/sealed/` Python package (from feature 011) with a new `match-outcomes` subcommand. Java worker classes are added to the existing `forge-connector` module. This mirrors the existing pattern where Python orchestrates and Java does the heavy lifting via subprocess.

## Design

### MatchWorkerMain (`forge-connector/.../MatchWorkerMain.java`)

Worker entry point. Initializes Forge in headless mode, then loops indefinitely generating matches.

```java
public class MatchWorkerMain {
    public static void main(String[] args) {
        // 1. Initialize Forge via shared initializer
        //    ForgeEnvironmentInitializer.initialize()
        // 2. Create MatchGenerator + MatchResultWriter
        // 3. Loop forever:
        //    MatchResult result = generator.generateMatch();
        //    writer.write(result);
    }
}
```

### MatchGenerator (`forge-connector/.../MatchGenerator.java`)

Core orchestration: one call = one complete match.

```java
public class MatchGenerator {
    // Dependencies: DeckBuilder, GamePlayer, list of eligible set codes

    public MatchResult generateMatch() {
        // 1. Pick random set from eligible sets
        // 2. Generate 2 pools (6 boosters each) via PoolGenerator
        // 3. Build deck from each pool via DeckBuilder (random method)
        // 4. Play best-of-3 via GamePlayer
        // 5. Return MatchResult
    }
}
```

### DeckBuilder (`forge-connector/.../DeckBuilder.java`)

Implements the four deck construction methods.

```java
public class DeckBuilder {
    private static final double[] METHOD_WEIGHTS = {0.4, 0.3, 0.2, 0.1};

    public Deck buildDeck(List<PaperCard> pool) {
        int method = selectMethod();
        return switch (method) {
            case 1 -> buildStandard(pool);
            case 2 -> buildWithSwaps(pool, 3);
            case 3 -> buildWithSwaps(pool, 8);
            case 4 -> buildRandom(pool);
        };
    }

    private Deck buildStandard(List<PaperCard> pool) {
        // SealedDeckBuilder(pool).buildDeck()
    }

    private Deck buildWithSwaps(List<PaperCard> pool, int swapCount) {
        // 1. buildStandard(pool); partition result into spellsInDeck, nonbasicLandsInDeck
        // 2. Remaining pool split into spellsRemaining, nonbasicLandsRemaining
        // 3. N type-matched swaps: randomly pick any deck card; if spell → replace from
        //    spellsRemaining; if non-basic land → replace from nonbasicLandsRemaining;
        //    skip swap if no matching replacement available
        // 4. rebalanceLands(chosenSpells, chosenNonbasics)
    }

    private Deck buildRandom(List<PaperCard> pool) {
        // 1. Filter pool to spells only (!isLand()); non-basic lands excluded
        // 2. Random-pick 23 spells
        // 3. rebalanceLands(chosenSpells, List.of())
    }

    Deck rebalanceLands(List<PaperCard> spells, List<PaperCard> nonbasicLands) {
        // LimitedDeckBuilder.addLands() is private; SealedDeckBuilder re-selects a
        // subset of its input rather than using all cards — produces land-heavy decks.
        // Reimplemented pip-proportional logic:
        //
        // 1. Count WUBRG mana pips from spells only (non-basic lands have no mana cost)
        // 2. basicLandsNeeded = 40 - spells.size() - nonbasicLands.size()
        // 3. Distribute basicLandsNeeded slots proportionally to pip counts
        //    using greedy-proportional method (WUBRG order, last color absorbs remainder)
        // 4. Source PaperCard land objects via FModel.getMagicDb().getCommonCards().getCard()
        //    using any set from the spell pool that has basic lands (fallback: "GRN")
        // 5. Build a new Deck: spells + nonbasicLands + basicLands
        // Note: {C} pips not handled — accepted trade-off (too few affected cards)
    }

    private String findBasicLandSet(List<PaperCard> cards) {
        // Return the edition code of the first card whose set has basic lands;
        // fall back to "GRN" (Guilds of Ravnica) if none found.
    }
}
```

### GamePlayer (`forge-connector/.../GamePlayer.java`)

Wraps Forge's Match API to play a best-of-3.

```java
public class GamePlayer {
    public int[] playMatch(Deck deckA, Deck deckB) {
        // 1. Create RegisteredPlayer for each deck with LobbyPlayerAi
        // 2. Create GameRules(GameType.Sealed), setGamesPerMatch(3)
        // 3. Create Match
        // 4. Loop: createGame(), startGame(), track wins
        //    until match.isMatchOver()
        // 5. Return [winsA, winsB]
    }
}
```

### MatchResultWriter (`forge-connector/.../MatchResultWriter.java`)

Atomic line append to the shared output file.

```java
public class MatchResultWriter {
    private final Path outputFile;

    public void write(MatchResult result) {
        // Format: deckA_cards;deckB_cards;winsA;winsB
        // Open file in APPEND mode, write line, close
        // One line per call — atomic for concurrent workers
    }
}
```

### MatchOutcomeSupervisor (`sealed/application/match_outcomes.py`)

Python supervisor managing Java worker processes.

```python
class MatchOutcomeSupervisor:
    def __init__(self, worker_count: int, output_path: Path) -> None: ...

    def run(self) -> None:
        # 1. Ensure output directory exists
        # 2. Register signal handlers (SIGINT, SIGTERM)
        # 3. Spawn worker_count monitor threads
        # 4. Run status reporter loop (every 60s)

    def _monitor_worker(self, worker_id: int) -> None:
        # Loop: start worker, waitFor, restart if not shutting down

    def _start_worker(self, worker_id: int) -> subprocess.Popen:
        # subprocess.Popen(["java", "-Xmx1200m", "-cp", ..., "MatchWorkerMain"])

    def _report_status(self) -> None:
        # Count lines in output file, compute rate, count alive workers
        # Print: "[Ns] M matches | R/min | W/W workers alive"

    def shutdown(self) -> None:
        # Set shutdown event, terminate all workers, wait for monitor threads
```

### CLI Extension (`sealed/infrastructure/cli.py`)

Add `match-outcomes` subcommand to the existing parser.

```python
# New subparser: match-outcomes
match_parser = subparsers.add_parser("match-outcomes", ...)
match_parser.add_argument("--workers", type=int, default=12)
```

## Complexity Tracking

> No constitution violations — table not required.

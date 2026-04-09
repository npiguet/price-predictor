# Research: Sealed Training Data Generation

## R1: Forge Initialization for Game Playing

**Decision**: Rewrite `ForgeEnvironmentInitializer` to use the `FModel.initialize()` + `GuiHeadless` pattern (same as jumpstart-tierlist), replacing the current manual `StaticData` setup. This single initializer is then used by all forge-connector entry points (pool generation, card conversion, and the new match worker).

**Rationale**: Game playing requires `FModel.initialize()`, which internally sets up `StaticData`, AI subsystems, and the game rules engine. Rather than maintaining two initialization paths, rewrite the existing one to use `FModel`. The extra subsystems loaded by `FModel` are irrelevant in practice — the overhead is negligible and having a single initialization path reduces code duplication. A `GuiHeadless` class (adapted from jumpstart-tierlist) is needed to satisfy the `IGuiBase` interface in headless mode.

**Alternatives considered**:
- Adding a second initializer for game playing only — rejected to avoid code duplication; one initialization path is simpler to maintain.
- Using the current `ForgeEnvironmentInitializer` as-is for game playing — rejected because `Match.startGame()` relies on `FModel` internals that the manual `StaticData` setup doesn't provide.

## R2: Additional Forge Dependencies

**Decision**: Add `forge-gui`, `forge-ai`, and `forge-gui/target/dependency/*` to the classpath for worker processes. The pom.xml gains `forge-gui` and `forge-ai` as system-scope dependencies.

**Rationale**: The feature requires classes from three Forge modules not currently in the forge-connector classpath:
- `forge-gui`: `SealedDeckBuilder` (`forge.gamemodes.limited`), `FModel` (`forge.model`), `GuiBase`/`IGuiBase` (`forge.gui.interfaces`)
- `forge-ai`: `LobbyPlayerAi` (`forge.ai`)
- `forge-gui` transitive dependencies (in `forge-gui/target/dependency/`)

The existing `forge-game` and `forge-core` are still needed. The `forge-gui` dependency directory contains all Forge transitive dependencies including `forge-ai`'s, so adding `forge-gui/target/dependency/*` to the runtime classpath covers all transitives.

**Alternatives considered**:
- Reimplementing deck building and game playing without Forge's built-in classes — rejected as needlessly complex and fragile.

## R3: Sealed Deck Building via Forge API

**Decision**: Use `SealedDeckBuilder` from `forge.gamemodes.limited` for deck construction method 1. For methods 2-4, post-process the result (or build from scratch for method 4).

**Rationale**: `SealedDeckBuilder` is Forge's standard sealed deck AI. It:
1. Takes a `List<PaperCard>` (the pool)
2. Automatically chooses colors based on card rankings
3. Returns a `Deck` with a 40-card main deck including basic lands

Usage pattern (from `SealedCardPoolGenerator`):
```java
new SealedDeckBuilder(pool.toFlatList()).buildDeck()
```

For methods 2-3, the post-processing swaps nonland cards between deck and remaining pool, then rebalances lands. For method 4, cards are randomly selected from the pool (excluding basic lands) until 23 nonland cards are chosen, then lands are added.

**Alternatives considered**:
- Using `BoosterDeckBuilder` (draft variant) — rejected because `SealedDeckBuilder` has sealed-specific color selection logic.

## R4: Game Playing via Forge API

**Decision**: Use `Match` + `RegisteredPlayer` + `LobbyPlayerAi` pattern from jumpstart-tierlist, with `GameType.Sealed` and `gamesPerMatch=3` for best-of-3.

**Rationale**: The jumpstart-tierlist project provides a proven pattern for headless AI game play:
```java
var players = List.of(
    new RegisteredPlayer(deck).setPlayer(new LobbyPlayerAi("p1", null)),
    new RegisteredPlayer(deck).setPlayer(new LobbyPlayerAi("p2", null))
);
var rules = new GameRules(GameType.Sealed);
rules.setGamesPerMatch(3);
var match = new Match(rules, players, "description");
```

Games are played via `match.createGame()` + `match.startGame(game)` (blocks until done). Results come from `game.getOutcome().getWinningLobbyPlayer()`. For best-of-3, iterate: create game, start game, check if match is over (`match.isMatchOver()`), repeat.

**Alternatives considered**:
- `GameSimulator` — rejected, it's for AI decision tree exploration, not full game simulation.

## R5: Supervisor/Worker Architecture

**Decision**: Python supervisor spawns Java worker subprocesses. Follow the jumpstart-tierlist `SupervisorApp` pattern, translated to Python.

**Rationale**: The jumpstart-tierlist demonstrates this exact pattern in Java:
- Each worker is an independent JVM process
- A monitor thread per worker calls `process.waitFor()`, restarts on non-zero exit
- `AtomicBoolean shuttingDown` flag prevents restart during shutdown
- Shutdown hook kills all processes on Ctrl+C

Translated to Python:
- `subprocess.Popen` to launch workers
- One `threading.Thread` per worker for monitoring
- `threading.Event` for shutdown signaling
- `signal.signal(signal.SIGINT, ...)` / `signal.signal(signal.SIGTERM, ...)` for clean shutdown
- `process.terminate()` + `process.kill()` fallback for cleanup

**Alternatives considered**:
- `multiprocessing.Pool` — rejected because workers are Java subprocesses, not Python functions.
- Single Java process with thread-per-worker — rejected because Forge JVM crashes kill all threads; separate processes isolate failures.

## R6: Concurrent File Writes

**Decision**: Each Java worker independently opens the output file in append mode and writes one complete line per match. No file locking needed.

**Rationale**: On Windows (the target platform), appending a short line (<~4KB) to a file opened with append mode is atomic at the OS level when the write fits in a single filesystem buffer. Each output line contains two 40-card decks (~1-2KB total), well within atomic write limits. The jumpstart-tierlist project uses this same pattern (`Files.newBufferedWriter` with `APPEND`) for multi-process output without corruption.

Each worker opens the file, writes one line, and flushes after each match. Workers don't keep the file open between matches — open-write-close per line ensures no buffering issues.

**Alternatives considered**:
- Python supervisor collects results via stdout pipes — rejected because it adds complexity and creates a bottleneck at the supervisor.
- File locking — rejected as unnecessary for atomic single-line appends.

## R7: Land Rebalancing Algorithm

**Decision**: Use Forge's built-in `LimitedDeckBuilder.addLands()` algorithm. Only **basic lands** are removed before rebalancing; non-basic lands stay in the deck.

**Rationale**: Forge's algorithm (`LimitedDeckBuilder`, lines 337-395) already implements exactly the behavior the spec needs:
1. Guarantee minimum 2 basic lands per required color
2. Distribute remaining slots proportionally to mana pip counts
3. Accounts for non-basic lands already in the deck

For methods 2-4, the rebalancing flow is: remove basic lands from the deck, keep non-basic lands, then invoke Forge's land allocation on the remaining cards to fill back to 40.

**Colorless {C} limitation**: Forge's algorithm only supports WUBRG (5 colors) — it does not allocate Wastes for colorless {C} pips. This is an accepted trade-off: very few sets contain cards requiring {C} (mainly Battle for Zendikar block), so the training data will have too few examples for the model to learn anything meaningful about colorless mana anyway. Cards with {C} costs may be slightly disadvantaged in the training data.

**Alternatives considered**:
- Custom land allocation algorithm with {C}/Wastes support — rejected because the added complexity is not justified by the negligible number of affected cards.

## R8: Eligible Set Selection

**Decision**: Build the eligible set list at worker startup by filtering with two criteria:
1. Set has a draft booster template: `edition.getBoosterTemplate("Draft") != null`
2. Set is not an un-set: `edition.getType() != CardEdition.Type.FUNNY`

**Rationale**: `CardEdition` stores booster templates in a `Map<String, SealedTemplate>` keyed by type (`"Draft"`, `"Collector"`, `"Set"`). Checking specifically for `"Draft"` excludes collector and set boosters (which are not used for sealed play) as well as small-booster products that only exist as non-draft variants. Play boosters (the newer format replacing draft boosters) are stored under the `"Draft"` key in Forge's metadata, so this check covers both.

Un-sets are excluded via `CardEdition.getType() != Type.FUNNY`. Forge classifies Unglued, Unhinged, Unstable, Unfinity, etc. as `Type.FUNNY`.

Both checks use Forge APIs — no manual blocklist needed:
- `StaticData.instance().getEditions()` iterates all editions
- `CardEdition.getBoosterTemplate("Draft")` returns the draft/play booster template or null
- `CardEdition.getType()` returns the edition type enum

The filtered list is computed once at startup and a random set is picked from it for each match.

**Alternatives considered**:
- Filtering by booster card count (14-15) — rejected in favor of the more semantically correct `"Draft"` key check, since collector boosters can also have 15 cards.
- Relying solely on `getBoosters()` existence — rejected because that storage includes all booster types (draft, collector, set), not just draft/play.
- Maintaining a hardcoded blocklist — rejected because the Forge API provides all necessary metadata programmatically.

## R9: Worker Entry Point Design

**Decision**: Create a new `MatchWorkerMain` Java class as the worker entry point. It initializes Forge via `FModel`, then loops indefinitely: pick set, generate pools, build decks, play match, write result.

**Rationale**: The worker is a new CLI entry point separate from `PoolMain` and `ConvertMain`. It needs:
- Forge full initialization (`GuiHeadless` + `FModel.initialize()`)
- A `GuiHeadless` class (copied/adapted from jumpstart-tierlist)
- Access to `SealedDeckBuilder`, `Match`, `RegisteredPlayer`, `LobbyPlayerAi`
- File output in append mode

The worker runs indefinitely (no iteration count). The Python supervisor is responsible for termination.

**Alternatives considered**:
- Extending `PoolMain` to also play games — rejected because separation of concerns; pool generation and match outcome generation have different lifecycles and dependencies.

# `forge-connector` — Java Maven module

Java 17+ module at `forge-connector/` holding the CLI workers the Python side runs against Forge. Besides the JDK and the sibling Forge checkout, its own code imports only `minlog` and the Guava, commons-lang3 and jupnp types Forge's API hands it; the `jar-with-dependencies` assembly bundles those with the rest of the Forge runtime dependencies the pom declares (`lombok` is `provided`). No JSON library is among them: the `json` package hand-writes every JSON document the workers emit and reads the one flat object they receive. Note the worker classpath is only the five JARs `build_forge_classpath` assembles — Forge's own transitive dependencies are **not** on it unless `include_dependency_glob` is set, so a new third-party dependency has to be declared here and bundled.

The fat JAR is built with `mvn install -DskipTests` → `target/forge-connector-1.0.0-SNAPSHOT-jar-with-dependencies.jar`. Main classes invoked by Python subprocess:
- `ConvertMain` — `python -m price_predictor convert`
- `PoolMain` — `python -m sealed generate-pools`
- `MatchWorkerMain` — one per worker in `python -m sealed match-outcomes`
- `ValidationWorkerMain` — `python -m sealed evaluate-scorer`
- `DeckBuilderMain` — used during scorer evaluation to build decks from pools
- `DraftWorkerMain` — one per worker in `python -m draft generate-draft-data` (drives Forge's `BoosterDraft`/`LimitedPlayerAI` for all pod seats, streams one transcript per draft)
- `DraftGameWorkerMain` — one per worker in `python -m draft play-draft-games` (loads a seat table, draws its own same-pod pairings, plays best-of-N matches, appends match-outcome rows)

These workers import `forge-game` / `forge-core` from the sibling `../forge` checkout (classpath assembled by `build_forge_classpath` in `src/price_predictor/infrastructure/forge_classpath.py`). `ForgeEnvironmentInitializer` bootstraps Forge's static state in each JVM. It pins the four Forge user preferences that change a game or a conversion (mulligan rule London, filtered hands off, performance mode off, language `en-US`) through `FModel.initialize`'s preference hook, so the machine's Forge profile never decides them (`pinnedPreferences()` returns them; the worker's status line carries them as the engine holds them, `PinnedForgePreferences.inEffect()`); it refuses to start when Forge's custom card or token directory holds a script the variant staging did not copy there; a second `initialize` with different arguments throws.

No worker has a per-game timeout, and none should get one. A game that does not finish in reasonable time, whether long or hung, ends with its JVM: the worker crashes, or `ForgeWorkerPool` kills it (its longest-running worker every status interval, under the stop protocol too), and the supervisor restarts it. Why the workers are short-lived by design, and what that kill passes over: `src/effects/docs/collection-runs.md` § Short-lived workers.

`MatchWorkerMain` also stops gracefully at a game end (`WorkerStop`), when the file `-Dworker.stop.file` names exists or once `-Dworker.recycle.seconds` have passed, after the game's recording is closed and its game-end marker written. Its stop line, its other printed lines and what the supervisor does are the contract `specs/025-ability-effect-model-gen3/contracts/cli.md` § Match worker stop protocol: the supervisor parses them, so they do not drift.

Every main runs under `ToolMain.guard` or `ToolMain.run`: `ToolMain.exit` and `badConfig` return a `ToolMain.Exit` (an `Error`) for the caller to throw, and the guard turns it into the exit status, and any other `Throwable` (an `Error` out of a game, a main's `IllegalStateException`) is printed to stderr and ends the process with `ExitCode.FAILURE`, so the JVM never lingers on Forge's non-daemon threads (`ToolMain.guardedStatus`); an exit path is testable by catching it (`ToolMainExitTest` checks every main). A tool main reads its `--flag value` arguments through `ToolMain.args`, which refuses with `BAD_CONFIG` any argument the main does not declare, so a Python caller passes exactly the flags its main declares; no tool main has a default output path.

## Engine hooks

The effect collectors reach the patched engine's hooks by name, through `PatchHooks`, the inventory of `Hook`s (a `ListenerSetter`, a `StaticAccessor` or an `InstanceAccessor`, each looked up by `HookReflection`). The connector compiles against the branch's installed Forge, not a stock one: `ReplaceCardOperation` calls `Phase.rewriteCommands`, `Card.rewriteCommands`, `copyCommandsFrom` and `TriggerHandler.getPendingDelayedTriggers` directly, so a Forge without those fails the connector build, while a hook only `PatchHooks` names shows up as absent at runtime. `PatchHooks.report()`, printed at worker startup, names each hook absent from the classpath's Forge and each listener callback or argument its interface lacks; the worker's status line (`WorkerStatus`) carries the same gaps and the mode as JSON, with the builds the JVM loaded (`LoadedBuilds`: the `Forge-Build-Commit` and `Forge-Build-Dirty` of the jar `forge.game.Game` came from, and the connector JAR's `Forge-Commit`), the one line the supervisor reads counts from and judges the engine by (`src/effects/docs/collection-runs.md` § The workers' engine). Forge widens a listener by adding `void` default methods or same-name overloads with appended arguments, which an unaware proxy answers with nothing. `PatchHooksTest` fails on a listener method Forge declares that no stand-in restates (there is no allow-list), and `theSiblingForgeCheckoutIsThePatchedOne` fails, rather than skips, on a sibling checkout without the hooks, where every hook-needing test and golden comparison would otherwise skip. What each one says, and what the tests check of a listener, is in `src/effects/docs/collection-internals.md` § The hook inventory.

## Build

```bash
# Fat JAR required by convert / generate-pools / match-outcomes / generate-draft-data
cd forge-connector && mvn install -DskipTests
```

Build with `install`, never `package`, and rebuild `../forge` with `mvn install -DskipTests` first whenever Forge changed: the connector compiles against Forge's installed artifacts, so `package` alone can bundle stale Forge code. The build writes the `../forge` checkout's HEAD (`git -C ${forge.dir} rev-parse HEAD`, `maven-antrun-plugin` at `initialize`) into the fat JAR's manifest as `Forge-Commit`, which every collecting command compares with its converted tree's conversion and refuses when it is absent or differs (`src/effects/docs/collection-runs.md` § The conversion the keys join against); it names the Forge the JAR links only when that build order was followed, so the workers also print the commit their loaded Forge jars were built at (Forge's own `pom.xml` writes `Forge-Build-Commit` into every jar it builds) and a collection whose workers name no commit or another one is refused. The collecting commands also compare the `../forge` checkout's current HEAD (where the workers read card scripts from) with it, so a checkout moved after the build is refused. Installing and smoke-running the fresh JAR needs no go-ahead, except while collection workers are running, because they restart from the installed JARs.

## Files the workers write

The workers write append-only corpora whose formats are documented beside their readers:

- `ConvertMain` — converted cards and their provenance sidecars: `src/price_predictor/CLAUDE.md`, `src/effects/docs/text-encoding.md` § Provenance sidecars.
- `PoolMain`, `MatchWorkerMain` — pools, match outcomes, cards played: `src/sealed/CLAUDE.md` § Corpus file formats.
- `MatchWorkerMain` under `--effect-records` — effect-record shards and dropped-event counts: `src/effects/docs/record-format.md`.
- The effect-record collectors under `connector/effects/` — before changing a collector, a hook or a cap, read `src/effects/docs/collection-internals.md`; before changing a fork or a directed rule, read `src/effects/docs/forks.md`. Neither loads by itself.
- `DraftWorkerMain`, `DraftGameWorkerMain` — draft transcripts and draft games: `src/draft/CLAUDE.md` § Corpus file formats.

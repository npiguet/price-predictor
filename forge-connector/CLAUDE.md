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

These workers import `forge-game` / `forge-core` from the sibling `../forge` checkout (classpath assembled by `build_forge_classpath` in `src/price_predictor/infrastructure/forge_jvm.py`). `ForgeEnvironmentInitializer` bootstraps Forge's static state in each JVM.

No worker has a per-game timeout, and none should get one. A game that does not finish in reasonable time, whether long or hung, ends with its JVM: the worker crashes, or `ForgeWorkerPool` kills it (its longest-running worker every status interval, under the stop protocol too), and the supervisor restarts it. Why the workers are short-lived by design, and what that kill passes over: `src/effects/docs/collection-runs.md` § Short-lived workers.

`MatchWorkerMain` also stops gracefully at a game end (`WorkerStop`), when the file `-Dworker.stop.file` names exists or once `-Dworker.recycle.seconds` have passed, after the game's recording is closed and its game-end marker written. Its stop line, its other printed lines and what the supervisor does are the contract `specs/025-ability-effect-model-gen3/contracts/cli.md` § Match worker stop protocol: the supervisor parses them, so they do not drift.

Every main runs under `ToolMain.guard` or `ToolMain.run`: `ToolMain.exit` and `badConfig` return a `ToolMain.Exit` (an `Error`) for the caller to throw, and the guard turns it into the exit status, so an exit path is testable by catching it (`ToolMainExitTest` checks every main).

## Engine hooks

The effect collectors reach the patched engine's hooks by name, through `PatchHooks`, and compile against stock Forge. Forge widens a listener by adding `void` default methods or same-name overloads with appended arguments, which an unaware proxy answers with nothing. Every listener method, argument and accessor the branch offers is read: `PatchHooksTest` fails on a listener method Forge declares that no stand-in restates (there is no allow-list), and `theSiblingForgeCheckoutIsThePatchedOne` fails, rather than skips, on a sibling checkout without the hooks, where every hook-needing test and golden comparison would otherwise skip. What each one says is in `src/effects/docs/collection-internals.md` § The hook inventory.

## Build

```bash
# Fat JAR required by convert / generate-pools / match-outcomes / generate-draft-data
cd forge-connector && mvn install -DskipTests
```

Build with `install`, never `package`, and rebuild `../forge` with `mvn install -DskipTests` first whenever Forge changed: the connector compiles against Forge's installed artifacts, so `package` alone can bundle stale Forge code. Installing and smoke-running the fresh JAR needs no go-ahead, except while collection workers are running, because they restart from the installed JARs.

## Files the workers write

The workers write append-only corpora whose formats are documented beside their readers:

- `ConvertMain` — converted cards and their provenance sidecars: `src/price_predictor/CLAUDE.md`, `src/effects/docs/text-encoding.md` § Provenance sidecars.
- `PoolMain`, `MatchWorkerMain` — pools, match outcomes, cards played: `src/sealed/CLAUDE.md` § Corpus file formats.
- `MatchWorkerMain` under `--effect-records` — effect-record shards and dropped-event counts: `src/effects/docs/record-format.md`.
- The effect-record collectors under `connector/effects/` — before changing a collector, a hook or a cap, read `src/effects/docs/collection-internals.md`; before changing a fork or a directed rule, read `src/effects/docs/forks.md`. Neither loads by itself.
- `DraftWorkerMain`, `DraftGameWorkerMain` — draft transcripts and draft games: `src/draft/CLAUDE.md` § Corpus file formats.

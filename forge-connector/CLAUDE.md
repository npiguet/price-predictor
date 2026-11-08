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

## Build

```bash
# Fat JAR required by convert / generate-pools / match-outcomes / generate-draft-data
cd forge-connector && mvn install -DskipTests
```

Build with `install`, never `package`, and rebuild `../forge` with `mvn install -DskipTests` first whenever Forge changed.

## Files the workers write

The workers write append-only corpora whose formats are documented beside their readers:

- `ConvertMain` — converted cards and their provenance sidecars: `src/price_predictor/CLAUDE.md`, `src/effects/CLAUDE.md` § File formats.
- `PoolMain`, `MatchWorkerMain` — pools, match outcomes, cards played: `src/sealed/CLAUDE.md` § Corpus file formats.
- `MatchWorkerMain` under `--effect-records` — effect-record shards and dropped-event counts: `src/effects/CLAUDE.md` § File formats.
- `DraftWorkerMain`, `DraftGameWorkerMain` — draft transcripts and draft games: `src/draft/CLAUDE.md` § Corpus file formats.

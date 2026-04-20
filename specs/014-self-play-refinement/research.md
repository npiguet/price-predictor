# Research: Self-Play Refinement

**Feature**: 014-self-play-refinement
**Date**: 2026-04-20

## Codebase Survey

### Overlapping Domain Vocabulary

| Existing Concept | Location | Decision | Notes |
|---|---|---|---|
| `GreedyDeckBuilder` | `src/sealed/domain/greedy_deck_builder.py:15` | **Reuse** | Already builds 23-card nonland decks from pools using scorer. `build-decks` will call this directly. |
| `compute_basic_lands()` | `src/sealed/domain/manabase.py:20` | **Reuse** | Deterministic land assignment from nonland card texts. Already used in `evaluate_scorer._build_a_decks()`. |
| `SetTransformerScorer` | `src/sealed/domain/scorer_model.py:66` | **Reuse** | The scorer model. `build-decks` loads a checkpoint and feeds it to `GreedyDeckBuilder`. |
| `ScorerConfig` | `src/sealed/domain/scorer_model.py:13` | **Reuse** | Immutable config dataclass stored in checkpoints. |
| `ConvertedCardLocator` | `src/sealed/infrastructure/converted_card_locator.py:27` | **Reuse** | Card name → embedding/text file lookup with sanitization. Already used in evaluate_scorer. |
| `LoadedScorerCheckpoint` | `src/sealed/infrastructure/scorer_store.py:19` | **Reuse** | Typed checkpoint record. `ScorerStore.load_checkpoint()` returns this. |
| `PoolConnector` | `src/sealed/infrastructure/pool_connector.py:10` | **Extend** | Needs to support omitting `--set` so Java can do random set selection. |
| `MatchWorkerConnector` | `src/sealed/infrastructure/match_worker_connector.py:15` | **Extend** | Needs to pass `--generated-decks-path` system property to Java when in self-play mode. |
| `MatchOutcomeSupervisor` | `src/sealed/application/match_outcomes.py:15` | **Extend** | Needs optional `generated_decks_path` parameter, forwarded to connector. |
| `EvaluateScorerUseCase` | `src/sealed/application/evaluate_scorer.py:81` | **Extend** | `_generate_pools()` at line 137 hardcodes `"RVR"` — needs `set_code` parameter. |
| `_build_a_decks()` | `src/sealed/application/evaluate_scorer.py:191` | **Reuse as template** | Module-level function that builds scorer decks from pools. The new `build-decks` subcommand will reuse the same pattern (load embeddings → `GreedyDeckBuilder.build()` → `compute_basic_lands()`). |
| `_parse_pools()` | `src/sealed/application/evaluate_scorer.py:182` | **Extend** | Currently parses `Card1|Card2|...` per line. Needs to handle the new `SET_CODE;Card1|Card2|...` format. |
| `MatchGenerator` (Java) | `forge-connector/.../MatchGenerator.java:27` | **Extend** | Needs a new `generateSelfPlayMatch()` method that takes deck A from a generated-decks file, rolls for method 1–5, and builds deck B accordingly. |
| `DeckBuilder` (Java) | `forge-connector/.../DeckBuilder.java:38` | **Reuse** | Methods 1–4 already exist (`buildStandard`, `buildWithSwaps(3)`, `buildWithSwaps(8)`, `buildRandom`). Method 5 is pure file lookup, not a Java deck build. |
| `PoolGenerator` (Java) | `forge-connector/.../PoolGenerator.java:18` | **Reuse** | Generates pools for a given set code. Used by both `PoolMain` and `MatchGenerator.generatePool()`. |

No parallel concepts introduced. All new functionality extends or composes existing classes.

### Adjacent Prior Art

#### Pool generation pipeline
- **Python**: `cli.py:266-292` → `GeneratePoolsUseCase.execute()` → `PoolConnector.generate()` → Java `PoolMain`
- **Java**: `PoolMain.java` accepts `--set`, `--size`, `--pools-path`. Writes `pools.txt` with pipe-separated card names, one pool per line.
- **Current format**: `Card1|Card2|...|CardN` (no set code prefix)
- **New format**: `SET_CODE;Card1|Card2|...|CardN` (set code always present)
- **Change needed**: `PoolMain.java` must prepend `setCode + ";"` to each line. When `--set` is omitted, `PoolMain` must select a random set per pool using `MatchGenerator.computeEligibleSets()`.

#### Match generation pipeline
- **Python**: `cli.py:358-375` → `MatchOutcomeSupervisor.run()` → `MatchWorkerConnector.start()` → Java `MatchWorkerMain`
- **Java**: `MatchWorkerMain.java` reads `-Doutput.file`, computes eligible sets, creates `MatchGenerator`, loops `generateMatch()`.
- **Current behavior**: Each match picks a random set, generates two pools, builds decks via weighted methods 1–4, plays best-of-3.
- **New behavior**: When `-Dgenerated.decks.file` system property is set, the worker enters self-play mode:
  - Loads the generated-decks file into memory (indexed by set code).
  - Each match: pick random deck A from file → roll method 1–5 → build deck B → play → write result.
  - Methods 1–4: generate a fresh pool from deck A's set code, build via `DeckBuilder`.
  - Method 5: pick another deck from the file with the same set code (exclude deck A).

#### Evaluation pipeline
- **Python**: `cli.py:332-355` → `EvaluateScorerUseCase.execute()` → `PoolConnector`, `EvaluationConnector`
- **Hardcoded set**: `evaluate_scorer.py:137` — `PoolConnector().generate("RVR", n_pools, pools_path)`
- **Change needed**: Accept optional `set_code` in config. When `None`, pick a random set. Random set selection must happen in Python (not Java) because `evaluate_scorer` needs a single set for all N pools.

#### Scorer deck building (the `_build_a_decks` pattern)
- **Location**: `evaluate_scorer.py:191-221`
- **Pattern**: For each pool → load embeddings via `ConvertedCardLocator` → `GreedyDeckBuilder(model, embeddings).build(names)` → `compute_basic_lands(nonland_texts)` → concatenate to 40-card deck.
- **Reuse**: The new `build-decks` subcommand will extract this into `BuildDecksUseCase` with the same logic, operating on a pools file and writing a generated-decks file.

#### Forge JVM helpers
- `build_jvm_command()`, `build_forge_classpath()`, `run_forge_worker()`, `kill_process_tree()` — all in `price_predictor/infrastructure/forge_jvm.py`.
- Used consistently by `PoolConnector`, `MatchWorkerConnector`, `EvaluationConnector`.

### Convention Alignment

**Sibling module to mirror**: The `sealed` package — all four stories add to this package.

| Convention | Pattern | Source |
|---|---|---|
| CLI registration | `_build_X_parser(subparsers)` function, `set_defaults(func=run_X)`, arguments via `add_argument` or `add_dataclass_arg` | `cli.py:66-205` |
| Application layer | Simple use cases: class with `execute()` method taking inline args. Complex ones: `execute(config: XConfig)` with a `@dataclass` config. | `generate_pools.py`, `evaluate_scorer.py` |
| Connector layer | Synchronous: `run_forge_worker()` for single-shot. Async: `build_jvm_command()` + `subprocess.Popen` for long-running workers. | `pool_connector.py`, `match_worker_connector.py` |
| Java main classes | System properties for workers (`-Dkey=value`), CLI args for tools (`--flag value`). | `MatchWorkerMain.java`, `PoolMain.java` |
| Test style | `unittest.mock.MagicMock` for connectors, `FakeProcess` for subprocess mocks, `tmp_path` for file I/O, class-per-behavior. | `test_generate_pools.py`, `test_match_outcomes.py` |
| File formats | Pipe-separated card names within a field, semicolons between fields. | `MatchResultWriter.java`, `PoolMain.java` |

**Deviation**: None anticipated. All new code follows existing patterns.

### Third-Instance Check

| Sub-problem | Instance 1 | Instance 2 | Action |
|---|---|---|---|
| Pool parsing | `evaluate_scorer._parse_pools()` (line 182) — `Card1\|Card2\|...` per line | `match_data_loader.parse_match_outcome()` (line 102) — semicolons between fields, pipes within | **Not a third instance**. The new `build-decks` will parse the new `SET_CODE;Card1\|Card2\|...` format. `_parse_pools()` must be updated to handle the set code prefix, but no new parser is needed — it's the same function, extended. |
| Random set selection | `MatchGenerator.java:75` — Java-side, picks from `eligibleSets` | `PoolMain.java` — currently no random selection | Feature 014 adds random selection to `PoolMain.java` (reusing `MatchGenerator.computeEligibleSets()`) and to `evaluate_scorer.py` (Python-side). The Java selection reuses existing `computeEligibleSets()`. The Python selection for evaluate-scorer is a new call site but the logic is trivial (one random choice from a list). **Not a pattern warranting extraction** — the two call sites serve different purposes (per-pool vs per-evaluation-run). |
| Deck building from pools | `_build_a_decks()` in `evaluate_scorer.py:191` | (none) | Feature 014 adds a second call site in `BuildDecksUseCase`. This is the first duplication — extract shared logic if a third appears. For now, the new use case can call the same helper or inline the same pattern. |

No third instances found. No shared abstraction extraction needed.

## Design Decisions

### Decision 1: Pool file format change (set code prefix)

**Decision**: Always prefix pool lines with `SET_CODE;`, even when `--set` is specified.

**Rationale**: A uniform format means downstream consumers (build-decks, match-outcomes) never need to know whether the pools came from a fixed set or random selection. The set code is always the first semicolon-delimited field.

**Alternatives considered**:
- Separate metadata file with set codes: rejected — adds file management complexity for no benefit.
- Set code only when random: rejected — inconsistent format complicates parsing.

### Decision 2: Random set selection location

**Decision**: For `generate-pools`, random selection happens in Java (`PoolMain`), per pool. For `evaluate-scorer`, random selection happens in Python, once per evaluation run (all pools from the same set).

**Rationale**: `generate-pools` needs per-pool randomness across many sets (FR-001). `evaluate-scorer` needs a single set for all pools because the round-robin design requires same-set decks for fair comparison (the existing design in `sealed-deck-picker.md` section "Evaluation Against External Baseline" specifies "N pools from a randomly selected set").

**Alternatives considered**:
- All random selection in Python: rejected — would require N separate `PoolConnector.generate()` calls (one per set), each spinning up a JVM. A single JVM selecting random sets per pool is far more efficient.
- All random selection in Java: rejected — evaluate-scorer needs Python-side control to generate all pools from the same set.

### Decision 3: Self-play match generation architecture

**Decision**: Extend `MatchWorkerMain` to accept an optional `-Dgenerated.decks.file` system property. When set, the worker loads the generated-decks file and uses self-play match generation. When absent, behavior is unchanged (Phase 0).

**Rationale**: Keeps the worker model consistent — Python supervisor spawns Java workers, workers loop generating matches. The generated-decks file is a simple flat text artifact that the Java worker reads at startup. No runtime Python↔Java coupling.

**Alternatives considered**:
- New `SelfPlayWorkerMain` class: rejected — duplicates the worker lifecycle, Forge init, and output writing. The only difference is match generation logic.
- Python-side match orchestration: rejected — would require Python to call Java for each individual match, adding latency and complexity.

### Decision 4: `build-decks` as a Python-only subcommand

**Decision**: `build-decks` runs entirely in Python. It loads the scorer checkpoint, reads pools, builds decks using `GreedyDeckBuilder` + `compute_basic_lands()`, and writes the generated-decks file. No Java involved.

**Rationale**: All components needed are already in Python: the scorer model, the greedy builder, the manabase calculator, and the card embedding locator. The Java side only needs to *read* generated decks, not build them.

**Alternatives considered**:
- Java-side deck building using scorer: rejected — would require serving the scorer as an API or embedding Python in Java. The spec explicitly designs the generated-decks file as the Python→Java bridge.

### Decision 5: Eligible set selection for evaluate-scorer

**Decision**: Add an optional `set_code` field to `EvaluateScorerConfig`. When `None`, Python calls `PoolMain` with a randomly selected set code. The set list is obtained by parsing `AllPrintings.json` using the same criteria as Java (`hasDraftBoosterTemplate && type != "funny"`).

**Rationale**: Python needs the set code before calling `PoolConnector` (to pass `--set` to Java). Getting the eligible set list in Python avoids a separate Java round-trip just for set enumeration.

**Alternatives considered**:
- Have Java select the set and report it back: rejected — `PoolConnector.generate()` is a fire-and-forget call; adding a response channel for the chosen set code adds unnecessary complexity.
- Hardcode a list of eligible sets in Python: rejected — the list changes with Forge updates. Parsing `AllPrintings.json` keeps it in sync.

**Implementation detail**: A shared `eligible_sealed_sets()` function in `sealed/infrastructure/` can parse `AllPrintings.json` once and return the list. This is used by evaluate-scorer. The generate-pools Java path uses `MatchGenerator.computeEligibleSets()` directly.

### Decision 6: Method 5 deck selection (re-roll on same deck)

**Decision**: When method 5 selects deck B = deck A (same line), re-roll. With 10,000 decks across ~215 sets, the probability of having only 1 deck per set is negligible. No special handling for the edge case.

**Rationale**: Spec explicitly calls this out as an accepted trade-off. Mirror matches produce pure RNG outcomes that add noise to the training signal.

### Decision 7: Pool file format — no backward compatibility

**Decision**: The new `SET_CODE;Card1|Card2|...` format is a breaking change to `pools.txt`. Old pool files without set code prefixes are not supported. `_parse_pools()` will expect the new format only.

**Rationale**: Pool files are cheap to regenerate and mostly unused — no existing workflows depend on cached pool files. Adding format-detection logic for a format nobody relies on violates Principle II (Simplicity First).

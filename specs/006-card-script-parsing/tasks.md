# Tasks: Card Script Conversion

**Input**: Design documents from `/specs/006-card-script-parsing/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, quickstart.md

**Tests**: Per the project constitution (Principle I: Fast Automated Tests), all features MUST include automated tests. Test tasks are MANDATORY in every task list.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Update forge-connector build configuration for card script conversion

- [ ] T001 Update pom.xml to add forge-game dependency (provided scope), maven-shade-plugin with mainClass, and dependency-copy plugin in forge-connector/pom.xml

  **Details**: Add `forge.game:forge-game:2.0.10-SNAPSHOT` with `<scope>provided</scope>` (compile-time only, excluded from fat JAR). Add `forge.game:forge-core:2.0.10-SNAPSHOT` with `<scope>provided</scope>`. Configure `maven-shade-plugin` to produce `forge-connector-1.0.0-SNAPSHOT-jar-with-dependencies.jar` with `mainClass` set to `com.pricepredictor.connector.ConvertMain`. Exclude all `forge.*` group artifacts from the shaded JAR. Existing surefire plugin config stays unchanged. Verify build compiles with `mvn compile`.

---

## Phase 2: Foundational (Data Model Classes)

**Purpose**: Create the shared data model classes used by all conversion logic

**CRITICAL**: No user story work can begin until this phase is complete

- [ ] T002 [P] Create AbilityType enum in forge-connector/src/main/java/com/pricepredictor/connector/AbilityType.java

  **Details**: Enum with 10 values: `KEYWORD_PASSIVE`, `KEYWORD_ACTIVE`, `ACTIVATED`, `TRIGGERED`, `STATIC`, `REPLACEMENT`, `SPELL`, `PLANESWALKER`, `OPTION`, `TEXT`. Each value stores: `outputPrefix` (String, e.g. `"keyword"`, `"activated"`), `actionable` (boolean). Provide `getOutputPrefix()` and `isActionable()` methods. See data-model.md AbilityType table for complete mapping.

- [ ] T003 [P] Create AbilityLine record in forge-connector/src/main/java/com/pricepredictor/connector/AbilityLine.java

  **Details**: Java record (or immutable class) with fields: `AbilityType type`, `String description`, `Integer actionNumber` (nullable). Validation: description must not be null/empty; actionNumber must be >= 1 when present; actionNumber must be null for non-actionable types (check `type.isActionable()`). Provide a `formatLine()` method that returns the formatted output string (e.g., `"keyword: flying"`, `"activated[1]: {T}: add {G}."`). See data-model.md AbilityLine section.

- [ ] T004 [P] Create ConvertedCard record in forge-connector/src/main/java/com/pricepredictor/connector/ConvertedCard.java

  **Details**: Immutable class with fields: `String name` (required), `String manaCost` (nullable), `String types` (required), `String powerToughness` (nullable), `String loyalty` (nullable), `String colors` (nullable), `String text` (nullable), `List<AbilityLine> abilities` (required, may be empty). Validation: name and types must not be null/empty. See data-model.md ConvertedCard section.

**Checkpoint**: Data model classes compiled — conversion implementation can begin

---

## Phase 3: User Story 1 — Convert Single Card Script (Priority: P1) MVP

**Goal**: Convert a single Forge card script into LLM-friendly text format with correct properties, keywords, activated/triggered/static/replacement abilities, action counters, and text casing.

**Independent Test**: Provide any basic card script (vanilla creature, creature with keywords, creature with activated abilities) and compare converted output against Oracle text.

### Tests for User Story 1

- [ ] T005 [P] [US1] Create KeywordClassifierTest in forge-connector/src/test/java/com/pricepredictor/connector/KeywordClassifierTest.java

  **Details**: Test passive keyword classification (flying, trample, deathtouch, vigilance, first strike, double strike, haste, lifelink, reach, flash, indestructible, menace, hexproof, defender → returns false for `isActivatable`). Test activatable keyword classification (kicker, cycling, equip, ninjutsu, evoke, flashback, morph, bestow, dash, unearth, madness, foretell, escape, crew → returns true). Test unknown keywords default to passive. Use JUnit 5 `@ParameterizedTest` where appropriate. See research.md R-006 for full keyword lists.

- [ ] T006 [P] [US1] Create CostTranslationTest in forge-connector/src/test/java/com/pricepredictor/connector/CostTranslationTest.java

  **Details**: Test Forge's `Cost` class integration for key cost patterns: `T` → `{T}`, `Q` → `{Q}`, `2 W W` → `{2}{W}{W}`, `Sac<1/CARDNAME>` → contains `Sacrifice CARDNAME`, `Discard<1/Card>` → contains `Discard`, `PayLife<3>` → contains `Pay 3 life`, `PayEnergy<2>` → contains `{E}{E}`. Test `Cost.abilityToString()` produces human-readable output. Test multiple cost parts joined with `, `. See data-model.md Cost Translation table.

### Implementation for User Story 1

- [ ] T007 [P] [US1] Create KeywordClassifier in forge-connector/src/main/java/com/pricepredictor/connector/KeywordClassifier.java

  **Details**: Static utility class with method `boolean isActivatable(String keywordName)`. Maintains a `Set<String>` of activatable keyword names (lowercase): kicker, cycling, equip, ninjutsu, evoke, emerge, channel, flashback, morph, megamorph, bestow, dash, unearth, replicate, madness, foretell, suspend, mutate, level up, escape, encore, overload, retrace, disturb, boast, craft, prototype, prowl, spectacle, surge, entwine, buyback, crew, reconfigure, adapt, monstrosity, scavenge, embalm, eternalize, outlast, transfigure, transmute, forecast, fortify, reinforce, bloodrush. Returns true if keyword name (lowercased) is in the set, false otherwise. See research.md R-006.

- [ ] T008 [US1] Create CardScriptConverter in forge-connector/src/main/java/com/pricepredictor/connector/CardScriptConverter.java

  **Details**: Core conversion class. Method: `ConvertedCard convertFace(ICardFace face, Map<String, String> svars)`. Implementation per data-model.md conversion pipeline:

  1. **Top-level properties**: Extract name (lowercase), manaCost via `ManaCost.getSimpleString()` (null if `no cost`), types (lowercase, space-separated supertypes+types+subtypes), powerToughness (`P/T` format if creature), loyalty (if planeswalker), colors (only if explicit override via `getColor()`), text from `getNonAbilityText()`.
  2. **Keywords** (`face.getKeywords()`): For each K: string, use `Keyword.getInstance()` to parse, then `KeywordClassifier.isActivatable()` to classify. Passive → `KEYWORD_PASSIVE` line with `KeywordInstance.getTitle()`. Activatable → `KEYWORD_ACTIVE` line with title (increment action counter). Skip special K: patterns for now (saga/class/etbCounter/plaintext handled in US2).
  3. **Abilities** (`face.getAbilities()`): For each A: string, parse via `FileSection.parseToMap()`. Detect `AB$` prefix → `ACTIVATED` line. Cost: `new Cost(costParam, true).abilityToString()`. Description: `SpellDescription$` param. Format: `"cost: description"`. Increment action counter. Skip SP$ for now (handled in US2).
  4. **Triggers** (`face.getTriggers()`): For each T: string, skip if `Static$ True` or `Secondary$ True`. Extract `TriggerDescription$` → `TRIGGERED` line (no action number).
  5. **Statics** (`face.getStaticAbilities()`): For each S: string, skip if `Secondary$ True`. Extract `Description$` → `STATIC` line.
  6. **Replacements** (`face.getReplacements()`): For each R: string, extract `Description$` → `REPLACEMENT` line.
  7. **Action counter**: Sequential across all actionable items on the face, starting at 1.
  8. **Text casing**: Lowercase all text, then restore `CARDNAME`, `NICKNAME`, `ALTERNATE` to uppercase, and restore all brace-enclosed symbols (`{...}`) to uppercase. Strip reminder text (parenthesized text) from descriptions per FR-012.
  9. **Missing descriptions**: If an ability has no description parameter, log warning `[CardName] missing description for ability` and use placeholder text.

- [ ] T009 [US1] Create OutputFormatter in forge-connector/src/main/java/com/pricepredictor/connector/OutputFormatter.java

  **Details**: Static utility class. Method: `String formatCard(ConvertedCard card)`. Outputs lines in order:
  1. `name: {value}` (always present)
  2. `mana cost: {value}` (omit if null)
  3. `types: {value}` (always present)
  4. `power toughness: {value}` (omit if null)
  5. `loyalty: {value}` (omit if null)
  6. `colors: {value}` (omit if null)
  7. `text: {value}` (omit if null)
  8. Each ability line via `AbilityLine.formatLine()`

  Each line is newline-separated. No trailing newline. See data-model.md Output Format section and spec.md output examples.

- [ ] T010 [US1] Create CardScriptConverterTest in forge-connector/src/test/java/com/pricepredictor/connector/CardScriptConverterTest.java

  **Details**: Unit tests using inline card script strings parsed via `CardRules.Reader`. Test cases per US1 acceptance scenarios:
  1. Vanilla creature (Grizzly Bears-like): name, mana cost, types, P/T only, no abilities.
  2. Passive keywords: card with Flying, Trample → `keyword: flying`, `keyword: trample` (no numbers).
  3. Activatable keywords: card with Kicker → `keyword[1]: kicker {cost}` (with action number).
  4. Activated ability: card with `{T}: Add {G}` → `activated[N]: {T}: add {G}.`
  5. Triggered ability: ETB trigger → `triggered: when...` (no number).
  6. Static ability: lord effect → `static: description` (no number).
  7. Replacement effect → `replacement: description` (no number).
  8. Engine metadata excluded: AI:, DeckHints:, SVar: not in output.
  9. Action counter increments across mixed ability types on same face.
  10. Text casing: all lowercase except CARDNAME, brace symbols uppercase.
  11. `ManaCost:no cost` → mana cost line omitted.
  12. Card with `Text:` property → `text:` line in output.

- [ ] T011 [US1] Create OutputFormatterTest in forge-connector/src/test/java/com/pricepredictor/connector/OutputFormatterTest.java

  **Details**: Unit tests for output formatting. Build `ConvertedCard` objects programmatically and verify formatted output. Test cases:
  1. Vanilla creature: only property lines, no ability lines.
  2. Card with abilities: correct line ordering (properties then abilities).
  3. Null optional fields omitted (no `mana cost:`, `loyalty:`, etc. lines).
  4. Action numbers formatted correctly in brackets: `activated[1]:`, `keyword[2]:`.
  5. Non-actionable types have no brackets: `keyword:`, `triggered:`, `static:`.

**Checkpoint**: At this point, single basic card scripts can be converted and verified. Run `mvn test` in forge-connector/ — all tests should pass.

---

## Phase 4: User Story 2 — Convert Complex Card Types (Priority: P2)

**Goal**: Correctly convert planeswalkers, modal/charm spells, multi-face cards (transform, split, adventure, MDFC, flip), instant/sorcery spells, sagas, and class cards.

**Independent Test**: Convert representative cards of each complex type (Jace Beleren, Abzan Charm, Delver of Secrets, Fire // Ice, Bonecrusher Giant, Lightning Bolt) and verify output matches spec examples.

### Implementation for User Story 2

- [ ] T012 [P] [US2] Create MultiCard record in forge-connector/src/main/java/com/pricepredictor/connector/MultiCard.java

  **Details**: Immutable class with fields: `String layout` (nullable — null for single-face), `List<ConvertedCard> faces` (required, at least one entry). Layout values: `transform` (from `DoubleFaced`), `split` (from `Split`), `adventure` (from `Adventure`), `modal` (from `Modal`), `flip` (from `Flip`). Null for single-face cards. Provide static factory `singleFace(ConvertedCard)` and `multiFace(String layout, List<ConvertedCard>)`. See data-model.md MultiCard section.

- [ ] T013 [US2] Extend CardScriptConverter to handle complex card types in forge-connector/src/main/java/com/pricepredictor/connector/CardScriptConverter.java

  **Details**: Add method `MultiCard convertCard(String scriptContent)` that parses the full script via `CardRules.Reader` and handles multi-face. Extend `convertFace()` with:

  1. **Spell effects** (A: lines with `SP$` prefix): Non-Charm SP$ → `SPELL` line with `SpellDescription$`, increment action counter.
  2. **Charm/modal** (A: lines with `SP$ Charm`): Parent line → `SPELL` with "choose one —" (or "choose two —" etc.), increment counter. Follow `Choices$` SVars to resolve each choice → `OPTION` line with description from each SVar's `SpellDescription$`, each incrementing action counter.
  3. **Planeswalker** (A: lines with `AB$` + `Planeswalker$ True` + LOYALTY cost): Detect `SubCounter<N/LOYALTY>` → `[-N]:` prefix, `AddCounter<N/LOYALTY>` → `[+N]:` prefix. Use `PLANESWALKER` type with action counter. Description from `SpellDescription$`.
  4. **Multi-face** (`AlternateMode` property): Parse both faces. Map: `DoubleFaced` → `transform`, `Split` → `split`, `Adventure` → `adventure`, `Modal` → `modal`, `Flip` → `flip`. Each face gets independent action counter starting at 1.
  5. **Saga chapters** (`K:Chapter:N:SVar1,...,SvarN`): Resolve SVars to get trigger descriptions → `TRIGGERED` lines.
  6. **Class levels** (`K:Class:level:cost:effect`): Level 1 abilities are regular abilities. Level 2+ → `ACTIVATED[N]:` lines with level cost.
  7. **etbCounter** (`K:etbCounter:Type:N`): → `TRIGGERED` line ("enters with N Type counters").
  8. **Plaintext K:** (`K:CARDNAME ...`): → `STATIC` line with the text.

- [ ] T014 [US2] Extend OutputFormatter for multi-face cards in forge-connector/src/main/java/com/pricepredictor/connector/OutputFormatter.java

  **Details**: Add method `String formatMultiCard(MultiCard card)`. If `layout` is non-null, output `layout: {value}` as the first line. Format each face via `formatCard()`. Separate faces with blank line + `ALTERNATE` + blank line. Single-face cards (layout null) just format the single face with no layout line. See data-model.md multi-face output format.

- [ ] T015 [US2] Extend CardScriptConverterTest for complex card types in forge-connector/src/test/java/com/pricepredictor/connector/CardScriptConverterTest.java

  **Details**: Add test cases per US2 acceptance scenarios:
  1. Planeswalker: Jace Beleren-like script → `planeswalker[1]: [+2]: ...`, `planeswalker[2]: [-1]: ...`, `planeswalker[3]: [-10]: ...` with loyalty property.
  2. Charm/modal: Abzan Charm-like → `spell[1]: choose one —`, `option[2]: ...`, `option[3]: ...`, `option[4]: ...`.
  3. Transform: two-face card → two ConvertedCard entries with independent action counters.
  4. Split card: two halves → both faces returned.
  5. Adventure: creature + adventure → both faces returned.
  6. Spell effect: Lightning Bolt-like → `spell[1]: CARDNAME deals 3 damage to any target.`
  7. Layout detection: verify `DoubleFaced` → `transform`, `Split` → `split`, `Adventure` → `adventure`, `Modal` → `modal`, `Flip` → `flip`.

- [ ] T016 [US2] Extend OutputFormatterTest for multi-face output in forge-connector/src/test/java/com/pricepredictor/connector/OutputFormatterTest.java

  **Details**: Add test cases:
  1. Multi-face card: `layout: transform` line first, faces separated by blank line + `ALTERNATE` + blank line.
  2. Single-face card: no `layout:` line.
  3. Each face independently formatted with correct properties and abilities.

**Checkpoint**: All card types convert correctly. Run `mvn test` — all existing and new tests pass.

---

## Phase 5: User Story 3 — Batch Process Card Library (Priority: P3)

**Goal**: Process all 32,000+ card scripts in the Forge cardsfolder in a single batch run, producing mirrored output directory structure.

**Independent Test**: Run batch process on cardsfolder, verify output file count matches input, spot-check random entries.

### Implementation for User Story 3

- [ ] T017 [US3] Create BatchConverter in forge-connector/src/main/java/com/pricepredictor/connector/BatchConverter.java

  **Details**: Class with method `BatchResult convert(Path cardsPath, Path outputPath)`. Implementation:
  1. Walk `cardsPath` recursively, finding all `.txt` files.
  2. For each file: read content, call `CardScriptConverter.convertCard()`, format via `OutputFormatter.formatMultiCard()`, write to mirrored path under `outputPath`.
  3. Mirror directory structure: `cardsPath/a/abzan_charm.txt` → `outputPath/a/abzan_charm.txt`.
  4. Error handling per FR-018: catch exceptions per file, log warning `[Card Name] message`, continue processing.
  5. Use `java.util.logging` or SLF4J for warnings.
  6. Return `BatchResult` with: total files processed, files succeeded, files with warnings, list of warnings.
  7. Default output path: `./output` relative to working directory.

- [ ] T018 [US3] Create ConvertMain CLI entry point in forge-connector/src/main/java/com/pricepredictor/connector/ConvertMain.java

  **Details**: `public static void main(String[] args)`. Parse CLI arguments: `--cards-path` (required, default `../forge/forge-gui/res/cardsfolder/`), `--output-path` (optional, default `./output`). Create `BatchConverter`, call `convert()`, print summary (total/succeeded/warnings). Exit code 0 on success, 1 on fatal error. This is the `mainClass` for the shade plugin.

- [ ] T019 [US3] Create BatchConverterTest in forge-connector/src/test/java/com/pricepredictor/connector/BatchConverterTest.java

  **Details**: Integration test with fixture files in `src/test/resources/cardsfolder/`. Create 3-5 minimal card script fixtures covering: vanilla creature, creature with abilities, a malformed file. Test cases:
  1. Batch processes all fixture files, output files created in temp directory.
  2. Output directory structure mirrors input.
  3. Malformed file logs warning and continues (other files still converted).
  4. Output file count matches valid input file count.

**Checkpoint**: Full batch processing works. Can process the Forge cardsfolder end-to-end via Java CLI.

---

## Phase 6: User Story 4 — Launch Conversion from CLI (Priority: P4)

**Goal**: Provide a Python CLI `convert` subcommand that launches the Java batch converter as a subprocess.

**Independent Test**: Run `python -m price_predictor convert` and verify it produces output files.

### Implementation for User Story 4

- [ ] T020 [US4] Add 'convert' subcommand to Python CLI in src/price_predictor/infrastructure/cli.py

  **Details**: Add a `convert` subparser with arguments: `--cards-path` (default `../forge/forge-gui/res/cardsfolder/`), `--output-path` (default `./output`). Create `run_convert(args)` function that:
  1. Locates the fat JAR at `forge-connector/target/forge-connector-1.0.0-SNAPSHOT-jar-with-dependencies.jar` (relative to project root).
  2. Locates Forge JARs: `../forge/forge-game/target/forge-game-2.0.10-SNAPSHOT.jar`, `../forge/forge-core/target/forge-core-2.0.10-SNAPSHOT.jar`, and `../forge/forge-game/target/dependency/*`.
  3. Builds the classpath string (platform-aware separator: `;` on Windows, `:` on Linux).
  4. Launches `java -cp {classpath} com.pricepredictor.connector.ConvertMain --cards-path {path} --output-path {path}` via `subprocess.run()`.
  5. Returns the subprocess exit code.
  6. Prints errors to stderr if JAR not found or Java not available.

- [ ] T021 [US4] Wire convert command dispatch in src/price_predictor/infrastructure/cli.py

  **Details**: In the existing command dispatch logic (likely in a `main()` function or `__main__.py`), add `elif args.command == "convert": return run_convert(args)`. Check `__main__.py` for the dispatch location and add the routing there if needed. Follow the existing pattern used by `run_predict`, `run_train`, `run_evaluate`, `run_serve`, `run_eval`.

- [ ] T022 [US4] Create Python CLI integration test in src/tests/integration/test_convert_cli.py

  **Details**: Integration test using pytest. Test cases:
  1. `convert` subcommand appears in CLI help output.
  2. Running `convert` with `--help` shows expected arguments (cards-path, output-path).
  3. If Java/JARs are available: running convert on a small fixture produces output files.
  Mark tests that require Java/JARs with `@pytest.mark.integration` so they can be skipped in CI without Forge.

**Checkpoint**: Python CLI `convert` command works end-to-end.

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Final validation and cleanup

- [ ] T023 Run quickstart.md validation end-to-end (build, convert, verify output)
- [ ] T024 Verify existing forge-connector tests still pass (price prediction client unaffected)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion — BLOCKS all user stories
- **User Stories (Phase 3+)**: All depend on Foundational phase completion
  - US1 (Phase 3): Can start after Phase 2
  - US2 (Phase 4): Depends on US1 (extends CardScriptConverter and OutputFormatter)
  - US3 (Phase 5): Depends on US2 (BatchConverter calls the complete converter)
  - US4 (Phase 6): Depends on US3 (Python CLI launches the Java CLI built in US3)
- **Polish (Phase 7)**: Depends on all user stories complete

### User Story Dependencies

- **US1 (P1)**: Can start after Foundational (Phase 2) — No dependencies on other stories
- **US2 (P2)**: Depends on US1 — extends the converter with complex types
- **US3 (P3)**: Depends on US2 — batch converter needs the complete single-card converter
- **US4 (P4)**: Depends on US3 — Python CLI launches the Java CLI entry point

### Within Each User Story

- Tests can be written in parallel with implementation (different files)
- Data model classes before logic classes
- Core conversion before formatting
- Implementation before integration tests

### Parallel Opportunities

- Phase 2: T002, T003, T004 can all run in parallel (independent data model files)
- Phase 3: T005 and T006 (tests) can run in parallel with T007 (KeywordClassifier impl)
- Phase 3: T008 (CardScriptConverter) and T009 (OutputFormatter) are sequential (formatter depends on converter output model)
- Phase 4: T012 (MultiCard) can run in parallel with other US2 tasks

---

## Parallel Example: User Story 1

```bash
# Launch foundational data model classes in parallel:
Task: T002 "Create AbilityType.java"
Task: T003 "Create AbilityLine.java"
Task: T004 "Create ConvertedCard.java"

# Then launch US1 tests + KeywordClassifier in parallel:
Task: T005 "KeywordClassifierTest.java"
Task: T006 "CostTranslationTest.java"
Task: T007 "KeywordClassifier.java"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (pom.xml update)
2. Complete Phase 2: Foundational (3 data model classes)
3. Complete Phase 3: User Story 1 (converter + formatter + tests)
4. **STOP and VALIDATE**: Convert sample cards, verify output matches spec examples
5. Run `mvn test` — all tests pass

### Incremental Delivery

1. Setup + Foundational → Build compiles with Forge deps
2. Add US1 → Basic cards convert correctly (MVP!)
3. Add US2 → All card types convert correctly
4. Add US3 → Full library batch processing works
5. Add US4 → Python CLI integration complete
6. Each story adds capability without breaking previous stories

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- All Java files go in `forge-connector/src/main/java/com/pricepredictor/connector/`
- All Java test files go in `forge-connector/src/test/java/com/pricepredictor/connector/`
- Forge SNAPSHOTs must be installed to local Maven repo before Phase 1 build
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently

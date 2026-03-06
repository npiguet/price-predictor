# Implementation Plan: Card Script Conversion

**Branch**: `006-card-script-parsing` | **Date**: 2026-03-05 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/006-card-script-parsing/spec.md`

## Summary

Convert Forge card scripts (.txt files) into an LLM-friendly text format with English key
names, Oracle-matching ability text, numbered player-actionable abilities, and batch
processing. Implemented in the existing `forge-connector/` Java module, which leverages Forge's internal
parsing classes (`CardRules.Reader`, `Cost`, `Keyword`, `ManaCost`) for accurate script
parsing and cost translation. The Python CLI launches the Java converter as a subprocess.

**Output format highlights**:
- All text lowercased except `CARDNAME`, `NICKNAME`, `ALTERNATE`, and brace symbols
  (`{W}`, `{T}`, `{E}`, etc.) which stay uppercase
- Multi-face cards include a `layout:` line (e.g., `layout: transform`) before the first
  face; single-face cards omit it

## Technical Context

**Language/Version**: Java 17+ (converter module), Python 3.14+ (CLI integration)
**Primary Dependencies**: forge-game 2.0.10-SNAPSHOT (transitively includes forge-core), JUnit 5
**Storage**: Local text files (input: Forge card scripts, output: converted text files in `./output/`)
**Testing**: JUnit 5 (Java unit tests), pytest (Python CLI integration test)
**Target Platform**: Windows/Linux CLI
**Project Type**: CLI tool (Java fat JAR invoked by Python subcommand)
**Performance Goals**: Process 32,000+ card scripts in under 5 minutes
**Constraints**: Forge SNAPSHOTs must be installed to local Maven repo before build
**Scale/Scope**: 32,116 card script files, ~42,000 ability lines to convert

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### Pre-Research Check

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Fast Automated Tests | PASS | JUnit 5 unit tests for conversion logic (in-memory, no file I/O). Python integration test (separate marker). Java tests run via `mvn test`. |
| II. Simplicity First | PASS | Reuses Forge's existing parser and cost classes — significantly less custom code than a Python reimplementation. Custom code limited to output formatting and keyword classification. |
| III. Data Integrity | PASS | Forge's battle-tested parser ensures correct script interpretation. Description parameters cover 97.9% of abilities. Warnings logged for cards without descriptions. |
| IV. DDD & Separation | PASS | Java module is self-contained. Python CLI is a thin subprocess launcher. No domain logic leakage. |
| V. Forge Interop | PASS | Conversion code added to `forge-connector/` with forge-game as a Maven dependency. No circular dependency — forge-connector is dropped into Forge's classpath at runtime, not declared as a Forge Maven dependency. |
| VI. Documentation | PASS | quickstart.md covers build steps, CLI usage, verification. |

### Post-Design Re-Check

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Fast Automated Tests | PASS | Java unit tests: all ability types, cost translations, keyword classification, edge cases. All in-memory with inline script strings — no file I/O needed. |
| II. Simplicity First | PASS | ~8 Java classes total. Core converter reuses CardRules.Reader (parsing), Cost (cost display), Keyword.getInstance() (keyword display). Custom code: output formatting + classification + batch I/O. |
| III. Data Integrity | PASS | Forge's CardRules.Reader validates script structure. Cost class handles all 42+ cost patterns. Missing descriptions trigger warning log with card name. |
| IV. DDD & Separation | PASS | Conversion logic in Java domain classes. File I/O in BatchConverter. CLI in ConvertMain. Python integration is infrastructure-only (subprocess launch). |
| V. Forge Interop | PASS | Conversion code in forge-connector/. forge-game added as Maven dependency. No circular dependency — Forge loads forge-connector at runtime via classpath, not as a Maven dependency. |
| VI. Documentation | PASS | Build prerequisites, CLI usage, output verification all documented in quickstart.md. |

### Quality Gates

- [ ] All Java unit tests pass (`mvn test` in forge-connector/)
- [ ] All Python tests pass (`pytest` in src/)
- [ ] No new ruff warnings in Python code
- [ ] forge-connector/ builds and passes all tests (existing + new conversion tests)
- [ ] Documentation covers build prereqs, convert CLI subcommand, output format
- [ ] Existing forge-connector tests still pass (price prediction client unaffected)

## Project Structure

### Documentation (this feature)

```text
specs/006-card-script-parsing/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
└── tasks.md             # Phase 2 output (/speckit.tasks command)
```

### Source Code (repository root)

```text
forge-connector/                             # EXISTING — expanded with conversion code
├── pom.xml                                  # MODIFIED: add forge-game (provided scope), shade plugin
├── src/
│   ├── main/java/com/pricepredictor/connector/
│   │   ├── (existing price prediction client classes)
│   │   ├── CardScriptConverter.java         # NEW — Core: ICardFace → ConvertedCard
│   │   ├── ConvertedCard.java               # NEW — Data: one card face output
│   │   ├── AbilityLine.java                 # NEW — Data: single ability line
│   │   ├── AbilityType.java                 # NEW — Enum: ability type categories
│   │   ├── KeywordClassifier.java           # NEW — Logic: passive vs activatable
│   │   ├── OutputFormatter.java             # NEW — Format: ConvertedCard → text output
│   │   ├── BatchConverter.java              # NEW — I/O: directory walk + batch convert
│   │   └── ConvertMain.java                 # NEW — CLI: main class (--cards-path, --output-path)
│   └── test/java/com/pricepredictor/connector/
│       ├── (existing price prediction client tests)
│       ├── CardScriptConverterTest.java     # NEW — Unit: all ability types, all card types
│       ├── CostTranslationTest.java         # NEW — Unit: Forge Cost class display
│       ├── KeywordClassifierTest.java       # NEW — Unit: passive/active classification
│       ├── OutputFormatterTest.java         # NEW — Unit: output format correctness
│       └── BatchConverterTest.java          # NEW — Integration: batch with fixture files

src/price_predictor/
├── infrastructure/
│   └── cli.py                               # MODIFIED: add 'convert' subcommand
├── __main__.py                              # MODIFIED: wire convert command

tests/
├── integration/
│   └── test_convert_cli.py                  # NEW: Python CLI → Java subprocess test
```

**Structure Decision**: Conversion code added directly to the existing `forge-connector/`
module. forge-game is added as a Maven dependency — no circular dependency since Forge
loads forge-connector at runtime via classpath, not as a Maven dependency. All Java code
stays in one module. Python CLI integration is a thin subprocess launcher.

## Complexity Tracking

> No constitution violations to justify. This is an additive change to forge-connector that
> leaves existing code untouched.

| Aspect | Justification |
|--------|--------------|
| forge-game dependency in forge-connector | No circular dependency — Forge loads forge-connector at runtime via classpath, not as a Maven dependency. Declared as `provided` scope so Forge classes are excluded from the fat JAR (avoids duplicate classes on Forge's classpath). |
| SNAPSHOT dependencies | Required because Forge is not published to Maven Central. Build prerequisite documented in quickstart.md. |
| Two languages (Java + Python) | Java for conversion (reuses Forge classes). Python for CLI integration (existing project infrastructure). Minimal Python changes (~20 lines for subcommand + subprocess launch). |

# Implementation Plan: Card Script Conversion

**Branch**: `006-card-script-parsing` | **Date**: 2026-03-05 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/006-card-script-parsing/spec.md`

## Summary

Convert Forge card scripts (.txt files) into an LLM-friendly text format with English key
names, Oracle-matching ability text, numbered player-actionable abilities, and batch
processing. Implemented as a new Java Maven module (`forge-script-converter/`) that leverages
Forge's internal parsing classes (`CardRules.Reader`, `Cost`, `Keyword`, `ManaCost`) for
accurate script parsing and cost translation. The Python CLI launches the Java converter
as a subprocess.

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
| V. Forge Interop | PASS | New module `forge-script-converter/` consumes Forge classes. Existing `forge-connector/` (stub library) remains unchanged — no circular dependency. |
| VI. Documentation | PASS | quickstart.md covers build steps, CLI usage, verification. |

### Post-Design Re-Check

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Fast Automated Tests | PASS | Java unit tests: all ability types, cost translations, keyword classification, edge cases. All in-memory with inline script strings — no file I/O needed. |
| II. Simplicity First | PASS | ~8 Java classes total. Core converter reuses CardRules.Reader (parsing), Cost (cost display), Keyword.getInstance() (keyword display). Custom code: output formatting + classification + batch I/O. |
| III. Data Integrity | PASS | Forge's CardRules.Reader validates script structure. Cost class handles all 42+ cost patterns. Missing descriptions trigger warning log with card name. |
| IV. DDD & Separation | PASS | Conversion logic in Java domain classes. File I/O in BatchConverter. CLI in ConvertMain. Python integration is infrastructure-only (subprocess launch). |
| V. Forge Interop | PASS | forge-connector/ unchanged. forge-script-converter/ is a separate artifact with its own pom.xml. |
| VI. Documentation | PASS | Build prerequisites, CLI usage, output verification all documented in quickstart.md. |

### Quality Gates

- [ ] All Java unit tests pass (`mvn test` in forge-script-converter/)
- [ ] All Python tests pass (`pytest` in src/)
- [ ] No new ruff warnings in Python code
- [ ] forge-connector/ unchanged (no new dependencies, no code changes)
- [ ] Documentation covers build prereqs, convert CLI subcommand, output format
- [ ] Java stub library (forge-connector/) still compiles and passes tests on Java 17+

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
forge-script-converter/                      # NEW Maven module
├── pom.xml                                  # Java 17, forge-game dep, shade plugin
├── src/
│   ├── main/java/com/pricepredictor/converter/
│   │   ├── CardScriptConverter.java         # Core: ICardFace → ConvertedCard
│   │   ├── ConvertedCard.java               # Data: one card face output
│   │   ├── AbilityLine.java                 # Data: single ability line
│   │   ├── AbilityType.java                 # Enum: ability type categories
│   │   ├── KeywordClassifier.java           # Logic: passive vs activatable
│   │   ├── OutputFormatter.java             # Format: ConvertedCard → text output
│   │   ├── BatchConverter.java              # I/O: directory walk + batch convert
│   │   └── ConvertMain.java                 # CLI: main class (--cards-path, --output-path)
│   └── test/java/com/pricepredictor/converter/
│       ├── CardScriptConverterTest.java     # Unit: all ability types, all card types
│       ├── CostTranslationTest.java         # Unit: Forge Cost class display
│       ├── KeywordClassifierTest.java       # Unit: passive/active classification
│       ├── OutputFormatterTest.java          # Unit: output format correctness
│       └── BatchConverterTest.java          # Integration: batch with fixture files

forge-connector/                             # EXISTING — unchanged
├── pom.xml
├── src/...

src/price_predictor/
├── infrastructure/
│   └── cli.py                               # MODIFIED: add 'convert' subcommand
├── __main__.py                              # MODIFIED: wire convert command

tests/
├── integration/
│   └── test_convert_cli.py                  # NEW: Python CLI → Java subprocess test
```

**Structure Decision**: New `forge-script-converter/` module alongside the existing
`forge-connector/`. This keeps the Forge integration stub library (forge-connector/)
lightweight and avoids circular dependencies. The converter module consumes Forge classes;
the stub library is consumed by Forge. Python CLI integration is a thin subprocess launcher.

## Complexity Tracking

> No constitution violations to justify. The new module is an additive change that leaves
> existing code untouched.

| Aspect | Justification |
|--------|--------------|
| New Maven module | Avoids circular dependency with Forge. Constitution V requires forge-connector to remain a lightweight stub. |
| SNAPSHOT dependencies | Required because Forge is not published to Maven Central. Build prerequisite documented in quickstart.md. |
| Two languages (Java + Python) | Java for conversion (reuses Forge classes). Python for CLI integration (existing project infrastructure). Minimal Python changes (~20 lines for subcommand + subprocess launch). |

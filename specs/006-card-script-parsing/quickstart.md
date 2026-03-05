# Quickstart: Card Script Conversion

**Feature Branch**: `006-card-script-parsing` | **Date**: 2026-03-05

## Prerequisites

- Java 17+ (JDK)
- Maven 3.8+
- Python 3.14+ with the project installed (`pip install -e .`)
- Forge source at `../forge/` (relative to project root)
- Forge cardsfolder at `../forge/forge-gui/res/cardsfolder/`

## Build Steps

### 1. Install Forge dependencies to local Maven repo

```bash
cd ../forge
mvn install -DskipTests -pl forge-core,forge-game -am
```

This installs `forge-core` and `forge-game` (2.0.10-SNAPSHOT) plus their transitive
dependencies to your local `~/.m2/repository`.

### 2. Build the converter

```bash
cd forge-script-converter
mvn package -DskipTests
```

This produces `forge-script-converter/target/forge-script-converter-1.0.0-SNAPSHOT-jar-with-dependencies.jar`.

### 3. Verify the build

```bash
cd forge-script-converter
mvn test
```

## Usage

### Batch Convert via Python CLI

Process the full Forge cardsfolder:

```bash
python -m price_predictor convert
```

With custom paths:

```bash
python -m price_predictor convert \
    --cards-path ../forge/forge-gui/res/cardsfolder/ \
    --output-path ./output
```

### Batch Convert via Java directly

```bash
java -jar forge-script-converter/target/forge-script-converter-1.0.0-SNAPSHOT-jar-with-dependencies.jar \
    --cards-path ../forge/forge-gui/res/cardsfolder/ \
    --output-path ./output
```

### Output Format Notes

- All text is lowercased except: `CARDNAME`, `NICKNAME`, `ALTERNATE`, and brace symbols
  (`{W}`, `{U}`, `{B}`, `{R}`, `{G}`, `{T}`, `{E}`, `{X}`, etc.)
- Multi-face cards include a `layout:` line (e.g., `layout: transform`) before the first
  face. Single-face cards omit it.

### Output Structure

Output mirrors the source directory structure:
```
output/
├── a/
│   ├── abzan_charm.txt
│   ├── ...
├── b/
│   └── ...
└── ...
```

### Verify Output

Spot-check a converted card:

```bash
cat output/l/lightning_bolt.txt
```

Expected:
```
name: lightning bolt
mana cost: {R}
types: instant
spell[1]: CARDNAME deals 3 damage to any target.
```

## Run Tests

### Java unit tests

```bash
cd forge-script-converter
mvn test
```

### Python CLI integration test

```bash
cd src
pytest tests/integration/test_convert_cli.py -v
```

## Project Layout (new files for this feature)

```
forge-script-converter/              # NEW Maven module
├── pom.xml                          # Depends on forge-game (2.0.10-SNAPSHOT)
├── src/
│   ├── main/java/com/pricepredictor/converter/
│   │   ├── CardScriptConverter.java     # Core conversion logic
│   │   ├── ConvertedCard.java           # Output data model (one face)
│   │   ├── AbilityLine.java             # Single ability line
│   │   ├── AbilityType.java             # Enum: KEYWORD_PASSIVE, ACTIVATED, etc.
│   │   ├── KeywordClassifier.java       # Passive vs activatable classification
│   │   ├── OutputFormatter.java         # Formats ConvertedCard → text output
│   │   ├── BatchConverter.java          # Directory traversal + batch conversion
│   │   └── ConvertMain.java             # CLI entry point (main class)
│   └── test/java/com/pricepredictor/converter/
│       ├── CardScriptConverterTest.java # Unit tests: all ability types
│       ├── CostTranslationTest.java     # Tests: Forge Cost class integration
│       ├── KeywordClassifierTest.java   # Tests: keyword classification
│       ├── OutputFormatterTest.java     # Tests: output format correctness
│       └── BatchConverterTest.java      # Integration: batch processing

src/price_predictor/
├── infrastructure/
│   └── cli.py                           # MODIFIED: add 'convert' subcommand
├── application/
│   └── (no new Python files — conversion is in Java)

tests/
├── integration/
│   └── test_convert_cli.py              # NEW: Python CLI → Java process test
```

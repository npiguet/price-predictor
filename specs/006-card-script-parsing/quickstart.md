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
cd forge-connector
mvn package -DskipTests
```

This produces `forge-connector/target/forge-connector-1.0.0-SNAPSHOT-jar-with-dependencies.jar`
(fat JAR with non-Forge dependencies included; Forge classes excluded via `provided` scope).

### 3. Verify the build

```bash
cd forge-connector
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

Requires Forge JARs on the classpath (since they are excluded from the fat JAR):

```bash
java -cp "forge-connector/target/forge-connector-1.0.0-SNAPSHOT-jar-with-dependencies.jar:../forge/forge-game/target/forge-game-2.0.10-SNAPSHOT.jar:../forge/forge-core/target/forge-core-2.0.10-SNAPSHOT.jar:../forge/forge-game/target/dependency/*" \
    com.pricepredictor.connector.ConvertMain \
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
cd forge-connector
mvn test
```

### Python CLI integration test

```bash
cd src
pytest tests/integration/test_convert_cli.py -v
```

## Project Layout (new files for this feature)

```
forge-connector/                             # EXISTING — expanded with conversion code
├── pom.xml                                  # MODIFIED: add forge-game (provided scope), shade plugin
├── src/
│   ├── main/java/com/pricepredictor/connector/
│   │   ├── (existing price prediction client classes)
│   │   ├── CardScriptConverter.java         # NEW — Core conversion logic
│   │   ├── ConvertedCard.java               # NEW — Output data model (one face)
│   │   ├── AbilityLine.java                 # NEW — Single ability line
│   │   ├── AbilityType.java                 # NEW — Enum: KEYWORD_PASSIVE, ACTIVATED, etc.
│   │   ├── KeywordClassifier.java           # NEW — Passive vs activatable classification
│   │   ├── OutputFormatter.java             # NEW — Formats ConvertedCard → text output
│   │   ├── BatchConverter.java              # NEW — Directory traversal + batch conversion
│   │   └── ConvertMain.java                 # NEW — CLI entry point (main class)
│   └── test/java/com/pricepredictor/connector/
│       ├── (existing price prediction client tests)
│       ├── CardScriptConverterTest.java     # NEW — Unit tests: all ability types
│       ├── CostTranslationTest.java         # NEW — Tests: Forge Cost class integration
│       ├── KeywordClassifierTest.java       # NEW — Tests: keyword classification
│       ├── OutputFormatterTest.java         # NEW — Tests: output format correctness
│       └── BatchConverterTest.java          # NEW — Integration: batch processing

src/price_predictor/
├── infrastructure/
│   └── cli.py                           # MODIFIED: add 'convert' subcommand
├── application/
│   └── (no new Python files — conversion is in Java)

tests/
├── integration/
│   └── test_convert_cli.py              # NEW: Python CLI → Java process test
```

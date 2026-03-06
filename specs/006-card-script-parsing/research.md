# Research: Card Script Conversion

**Feature Branch**: `006-card-script-parsing` | **Date**: 2026-03-05

## R-001: Implementation Language — Java with Forge Classes

**Decision**: Java module using Forge's internal classes for parsing and text generation

**Rationale**:
- Cost translation is the hardest part of the conversion. Forge's `Cost` class and 42+
  `CostPart` subclasses handle every cost pattern in the card library (tap, sacrifice,
  discard, exile, remove counters, pay life, pay energy, etc.) with correct human-readable
  `toString()` output. Reimplementing this in Python would be massive, error-prone, and
  duplicate tested logic.
- Forge's `CardRules.Reader` parses card scripts into structured `CardFace` objects — the
  exact parser for the format, maintained by the Forge project.
- Forge's `Keyword.getInstance()` correctly parses all keyword variants and provides
  display text via `getTitle()`.
- This approach results in significantly less custom code: we reuse Forge's parsing and
  cost formatting, writing only the output formatting layer.
- Enables future feature reuse (any feature needing card data can use the same Java bridge).

**Alternatives Considered**:
- **Pure Python text parsing**: Initially planned. Viable for ~98% of description text
  (extracted from SpellDescription$/TriggerDescription$), but cost translation would
  require reimplementing 42+ cost types with all their edge cases. The user confirmed
  this is harder than it appears — costs "can be pretty much anything and never appear
  in human readable format on card scripts."

## R-002: Module Structure — Conversion Code in forge-connector

**Decision**: Add conversion code directly to the existing `forge-connector/` module

**Rationale**:
- There is no circular dependency. Forge does not declare forge-connector as a Maven
  dependency — the connector JAR is dropped into Forge's classpath at runtime.
  forge-connector can freely depend on forge-core/forge-game as build-time Maven
  dependencies without creating any cycle.
- Keeping all Java code in one module avoids unnecessary project fragmentation.
- The connector module already serves as the Java bridge between the price predictor
  project and Forge. Card script conversion is another facet of that bridge.

**Alternatives Considered**:
- **Separate `forge-script-converter/` module**: Unnecessary complexity. No circular
  dependency exists to justify splitting the code.

## R-003: Forge Dependency Management

**Decision**: SNAPSHOT dependencies on forge-core and forge-game, installed to local Maven repo

**Rationale**:
- Forge modules are at version `2.0.10-SNAPSHOT` (not published to Maven Central).
- Build prerequisite: `cd ../forge && mvn install -DskipTests -pl forge-core,forge-game`
  installs the SNAPSHOTs to the local `.m2` repository.
- Only `forge-game` is needed as a direct dependency (it transitively includes forge-core).
- Maven shade plugin creates a fat JAR in `forge-connector/` for Python CLI invocation.
  Forge dependencies (forge-core, forge-game, and their transitive deps) are declared with
  `<scope>provided</scope>` so they are available at compile time but excluded from the
  fat JAR. At runtime inside Forge, these classes are already on the classpath. For
  standalone CLI invocation (Python subprocess), the Java command must include both the
  connector JAR and the Forge classpath.

**Alternatives Considered**:
- **Copying Forge source directly**: Maintenance nightmare, diverges from upstream.
- **Git submodule**: Over-complex for two dependency artifacts.

## R-004: Forge Classes Needed (Standalone Capability)

**Decision**: Use Forge's parser and cost classes standalone — no game initialization required

**Rationale**: Research confirmed these Forge classes work without `Game`/`Player` objects:

| Class | Package | Purpose | Standalone? |
|-------|---------|---------|-------------|
| `CardRules.Reader` | `forge.card` | Parse `.txt` script into `CardFace[]` | Yes — pure string parsing |
| `CardFace` (via `ICardFace`) | `forge.card` | Holds parsed properties, raw ability strings | Yes — data container |
| `Cost` | `forge.game.cost` | Parse `Cost$` syntax into `CostPart` objects | Yes — string parsing only |
| `CostPart` subclasses | `forge.game.cost` | `toString()` for each cost type | Yes — string formatting |
| `Keyword.getInstance()` | `forge.game.keyword` | Parse `K:` lines into `KeywordInstance` | Yes — string parsing |
| `KeywordInstance` subclasses | `forge.game.keyword` | `getTitle()` for display text | Yes — string formatting |
| `ManaCost` / `ManaCostParser` | `forge.card.mana` | Parse `2 W W` → `{2}{W}{W}` | Yes — string parsing |
| `FileSection.parseToMap()` | `forge.util` | Parse `AB$ X \| Key$ Value` → Map | Yes — string parsing |

**Key methods for text extraction**:
- `Cost.abilityToString()` → formats ability cost as display text (e.g., `"{T}, Sacrifice CARDNAME"`)
- `CostPutCounter.toString()` → planeswalker `+N` format when counter is LOYALTY
- `CostRemoveCounter.toString()` → planeswalker `-N` format when counter is LOYALTY
- `KeywordInstance.getTitle()` → keyword display text (e.g., `"Flashback {2}{R}"`)
- `ManaCost.getSimpleString()` → mana cost display (e.g., `"{2}{W}{W}"`)

## R-005: Description Parameter Coverage

**Decision**: Use per-ability description parameters (SpellDescription$, TriggerDescription$,
Description$) combined with Forge's Cost formatting for the cost prefix

**Rationale**: Measured across 32,116 card scripts:

| Line Type | Total | With Description | Coverage |
|-----------|-------|-----------------|----------|
| A: (Abilities) | 17,625 | 17,058 (SpellDescription$) | 96.8% |
| T: (Triggers) | 15,929 | 15,760 (TriggerDescription$) | 98.9% |
| S: (Statics) | 6,807 | 6,651 (Description$) | 97.7% |
| R: (Replacements) | 1,614 | 1,609 (Description$) | 99.7% |
| **Total** | **41,975** | **41,078** | **97.9%** |

Missing descriptions fall into two categories (both intentional):
- **Charm/modal parent lines** (`SP$ Charm`): Choices$ SVars carry the descriptions.
- **Engine-internal triggers** (`Static$ True`, `Secondary$ True`): Hidden from card text,
  excluded per FR-013.

The conversion formula for activated abilities is:
`Cost.abilityToString(Cost$) + ": " + SpellDescription$`

## R-006: Keyword Classification

**Decision**: Hardcoded set of activatable keywords, matching Forge's keyword taxonomy

**Rationale**: Forge's `Keyword` enum and `KeywordInstance` subclass hierarchy provides
a natural classification:
- `KeywordWithCost` / `KeywordWithCostAndAmount` → activatable (player pays a cost)
- `SimpleKeyword` / `KeywordWithAmount` / `KeywordWithType` → passive (automatic)

**Activatable** (get `keyword[N]:` with action counter):
kicker, cycling, equip, ninjutsu, evoke, emerge, channel, flashback, morph, megamorph,
bestow, dash, unearth, replicate, madness, foretell, suspend, mutate, level up,
escape, encore, overload, retrace, disturb, boast, craft, prototype, prowl, spectacle,
surge, entwine, buyback, crew, reconfigure, adapt, monstrosity, scavenge, embalm,
eternalize, outlast, transfigure, transmute, forecast, fortify, reinforce, bloodrush

**Passive** (get `keyword:` without counter):
flying, trample, deathtouch, vigilance, first strike, double strike, haste, lifelink,
reach, flash, indestructible, menace, hexproof, protection, defender, shroud, fear,
intimidate, skulk, prowess, wither, infect, toxic, cascade, convoke, devoid, split second,
undying, persist, affinity, storm, changeling, exalted, flanking, phasing, bushido,
graft, modular, dredge, fabricate, renown, partner, companion, enchant, ward

**Special K: line handling**:
- `K:etbCounter:Type:N` → `triggered:` line ("enters with N counters")
- `K:CARDNAME ...` (plaintext) → `static:` line
- `K:Chapter:N:SVars` → saga chapter `triggered:` lines
- `K:Class:level:cost:effect` → class level `activated[N]:` lines

## R-007: Edge Case Card Types

**Decision**: Handle all documented multi-face layouts and special types

**Findings from reading actual card scripts**:

| Card Type | Script Pattern | Output Approach |
|-----------|---------------|-----------------|
| **Saga** | `K:Chapter:N:SVar1,...,SvarN` | Each chapter → `triggered:` line |
| **Class** | `K:Class:level:cost:effect` | Base abilities direct, levels 2+ → `activated[N]:` |
| **Room** | `AlternateMode:Split`, `Types:Enchantment Room` | `layout: split` + two faces via ALTERNATE |
| **Transform** | `AlternateMode:DoubleFaced` | `layout: transform` + two faces via ALTERNATE |
| **Split** | `AlternateMode:Split` | `layout: split` + two faces via ALTERNATE |
| **Adventure** | `AlternateMode:Adventure` | `layout: adventure` + two faces via ALTERNATE |
| **MDFC** | `AlternateMode:Modal` | `layout: modal` + two faces via ALTERNATE |
| **Flip** | `AlternateMode:Flip` | `layout: flip` + two faces via ALTERNATE |
| **Plaintext K:** | `K:CARDNAME can't...` | `static:` line |
| **etbCounter** | `K:etbCounter:Type:N` | `triggered:` line |
| **Text:** | `Text:free text` | `text:` line |

All types present in the Forge cardsfolder are handled.

## R-008: Text Casing Rules

**Decision**: Lowercase all text except placeholders, markers, and brace symbols

**Rationale**: The spec (FR-015) requires lowercase output with uppercase CARDNAME/NICKNAME/
ALTERNATE. Additionally, brace-enclosed symbols (`{W}`, `{U}`, `{B}`, `{R}`, `{G}`, `{C}`,
`{S}`, `{X}`, `{T}`, `{Q}`, `{E}`, hybrid `{W/U}`, `{G/P}`, generic `{1}`, `{2}`, etc.)
must remain uppercase because they represent standardized game symbols, not natural language
text. Forge's `ManaCost.getSimpleString()` and `CostPart.toString()` already produce these
in uppercase — the converter preserves them as-is during the lowercase pass.

**Implementation**: Lowercase the entire output string, then restore brace-enclosed tokens
to uppercase (or protect them before lowercasing).

**Alternatives Considered**: None — uppercase symbols are the standard MTG notation.

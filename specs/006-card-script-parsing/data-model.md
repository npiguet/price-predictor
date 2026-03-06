# Data Model: Card Script Conversion

**Feature Branch**: `006-card-script-parsing` | **Date**: 2026-03-05

## Java Entities (forge-connector module)

### ConvertedCard

Represents a single card face after conversion from Forge script format to LLM-friendly
output. Constructed by the converter from Forge's `ICardFace` + parsed abilities.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| name | String | Yes | Card name, lowercased |
| manaCost | String | No | Mana cost in `{N}{C}` format via `ManaCost.getSimpleString()`. Null for `no cost` cards |
| types | String | Yes | All supertypes + card types + subtypes, space-separated, lowercased |
| powerToughness | String | No | `P/T` format (e.g., `2/2`), lowercased |
| loyalty | String | No | Starting loyalty for planeswalkers |
| colors | String | No | Override colors, lowercased. Only when color differs from mana cost |
| text | String | No | Non-ability rules text from `Text:` / `getNonAbilityText()` |
| abilities | List\<AbilityLine\> | Yes | Ordered list of ability lines (may be empty for vanilla creatures) |

**Validation rules**:
- `name` must not be null or empty
- `types` must not be null or empty
- CARDNAME, NICKNAME, and ALTERNATE remain uppercase in all text fields

### AbilityLine

Represents a single ability line in the converted output.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| type | AbilityType | Yes | The category of this ability |
| description | String | Yes | Human-readable description text, lowercased |
| actionNumber | Integer | No | Global action counter (only for player-actionable abilities) |

**Validation rules**:
- `description` must not be null or empty (use placeholder if unavailable)
- `actionNumber` must be >= 1 when present
- `actionNumber` must be null for non-actionable types

### AbilityType (Enum)

| Value | Output Prefix | Actionable | Source |
|-------|--------------|------------|--------|
| KEYWORD_PASSIVE | `keyword:` | No | K: line, passive keyword |
| KEYWORD_ACTIVE | `keyword[N]:` | Yes | K: line, activatable keyword |
| ACTIVATED | `activated[N]:` | Yes | A:AB$ line |
| TRIGGERED | `triggered:` | No | T: line (non-Secondary, non-Static) |
| STATIC | `static:` | No | S: line or plaintext K: line |
| REPLACEMENT | `replacement:` | No | R: line |
| SPELL | `spell[N]:` | Yes | A:SP$ line (non-Charm) |
| PLANESWALKER | `planeswalker[N]:` | Yes | A:AB$ with LOYALTY cost + Planeswalker$ True |
| OPTION | `option[N]:` | Yes | Charm choices, modal options |
| TEXT | `text:` | No | Text: property |

### MultiCard

Represents a complete card with one or more faces.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| layout | String | No | Multi-face layout type: `transform`, `split`, `adventure`, `modal`, `flip`. Derived from `AlternateMode` in the script (e.g., `DoubleFaced` → `transform`). Null for single-face cards. Output as `layout:` line before first face; omitted when null. |
| faces | List\<ConvertedCard\> | Yes | One or more card faces. Each face has independent action counter |

**Validation rules**:
- `faces` must have at least one entry
- `layout` must be one of the recognized values

## Relationships

```
MultiCard 1──* ConvertedCard (faces)
ConvertedCard 1──* AbilityLine (abilities)
AbilityLine *──1 AbilityType
```

## Conversion Pipeline (per face)

```
ICardFace (from CardRules.Reader)
    │
    ├─ Top-level properties:
    │   getName() → name (lowercase)
    │   getManaCost() → ManaCost.getSimpleString() (symbols stay uppercase)
    │   getType() → types (lowercase)
    │   getIntPower()/getIntToughness() → power toughness
    │   getInitialLoyalty() → loyalty
    │   getColor() → colors (only if explicit override)
    │   getNonAbilityText() → text (lowercase)
    │
    ├─ Build SVar dictionary:
    │   getVariables() → Map<String, String>
    │
    ├─ Keywords (getKeywords()):
    │   For each K: string:
    │   ├─ "Chapter:N:..." → saga chapters (resolve SVars → triggered lines)
    │   ├─ "Class:level:..." → class levels (activated[N] lines)
    │   ├─ "etbCounter:..." → triggered line
    │   ├─ "CARDNAME ..." → static line
    │   ├─ Keyword.getInstance(kw) → classify:
    │   │   ├─ KeywordWithCost → KEYWORD_ACTIVE [N]
    │   │   └─ SimpleKeyword etc → KEYWORD_PASSIVE
    │   └─ Use KeywordInstance.getTitle() for display text
    │
    ├─ Abilities (getAbilities()):
    │   For each A: string:
    │   ├─ Parse to Map via FileSection.parseToMap()
    │   ├─ Detect type prefix (AB$, SP$)
    │   ├─ If SP$ Charm → follow Choices$ SVars → OPTION lines
    │   ├─ If AB$ with LOYALTY cost + Planeswalker$ → PLANESWALKER [N]
    │   ├─ If AB$ → ACTIVATED [N]
    │   │   Cost: new Cost(costParam, true).abilityToString()
    │   │   Description: SpellDescription$ param
    │   │   Output: "cost: description"
    │   ├─ If SP$ → SPELL [N]
    │   │   Description: SpellDescription$ param
    │   └─ Extract description, apply action counter
    │
    ├─ Triggers (getTriggers()):
    │   For each T: string:
    │   ├─ Skip if Static$ True or Secondary$ True
    │   └─ Extract TriggerDescription$ → TRIGGERED line
    │
    ├─ Statics (getStaticAbilities()):
    │   For each S: string:
    │   ├─ Skip if Secondary$ True
    │   └─ Extract Description$ → STATIC line
    │
    ├─ Replacements (getReplacements()):
    │   For each R: string:
    │   └─ Extract Description$ → REPLACEMENT line
    │
    └─ Assign action numbers (sequential across all actionable items on face)
```

## Cost Translation (via Forge's Cost class)

The `Cost` constructor parses `Cost$` strings into `CostPart` objects. `abilityToString()`
formats them as human-readable text. Key translations:

| Script Cost | Forge Class | Display Output |
|-------------|------------|----------------|
| `T` | `CostTap` | `{T}` |
| `Q` | `CostUntap` | `{Q}` |
| `2 W W` | `CostPartMana` | `{2}{W}{W}` |
| `Sac<1/CARDNAME>` | `CostSacrifice` | `Sacrifice CARDNAME` |
| `SubCounter<2/LOYALTY>` | `CostRemoveCounter` | `-2` |
| `AddCounter<1/LOYALTY>` | `CostPutCounter` | `+1` |
| `Discard<1/Card>` | `CostDiscard` | `Discard a card` |
| `PayLife<3>` | `CostPayLife` | `Pay 3 life` |
| `PayEnergy<2>` | `CostPayEnergy` | `{E}{E}` |
| `tapXType<2/Creature>` | `CostTapType` | `Tap two untapped creatures you control` |

Multiple cost parts joined with `, `. Planeswalker costs use `[+N]:` / `[-N]:` prefix format.

## Output Format

### Single-face cards

Lines in order:
1. Top-level properties (name, mana cost, types, power toughness, loyalty, colors, text)
2. Ability lines (keywords first, then activated, triggered, static, replacement, spell)

### Multi-face cards

A `layout:` property line appears first, before any face content. Faces are separated by
a blank line, `ALTERNATE`, blank line.

```
layout: transform
name: daring sleuth
mana cost: {1}{U}
types: creature human rogue
...

ALTERNATE

name: bearer of overwhelming truths
...
```

Layout values: `transform`, `split`, `adventure`, `modal`, `flip`.
Single-face cards omit the `layout:` line entirely (no `layout: normal`).

### Text casing

All text is lowercased except:
- `CARDNAME`, `NICKNAME`, `ALTERNATE` — uppercase placeholders/markers
- Brace symbols — mana `{W}`, `{U}`, `{B}`, `{R}`, `{G}`, `{C}`, `{S}`, `{X}`,
  generic `{1}`, `{2}`, etc., hybrid `{W/U}`, `{W/P}`, energy `{E}`,
  tap `{T}`, untap `{Q}`, and any other `{SYMBOL}` tokens remain uppercase

# Data Model: Card Script Conversion

**Feature Branch**: `006-card-script-parsing` | **Date**: 2026-03-12

## Java Entities (forge-connector module)

### CardFace

Represents a single card face after conversion from Forge script format to LLM-friendly
output. Constructed by `RulesParser` from Forge's `Card` + `ICardFace` objects.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| name | String | Yes | Card name, lowercased (placeholders uppercase) |
| manaCost | String | No | Mana cost in `{N}{C}` format via `ManaCost.getSimpleString()`. Null for no-cost cards |
| types | String | Yes | All supertypes + card types + subtypes, space-separated, lowercased (dash separator removed) |
| powerToughness | String | No | `P/T` format (e.g., `2/2`) |
| loyalty | String | No | Starting loyalty for planeswalkers |
| defense | String | No | Starting defense for battles |
| colors | String | No | Override colors, lowercased. Only when color differs from mana cost |
| text | String | No | Non-ability rules text from `getNonAbilityText()`, lowercased |
| abilities | List\<Ability\> | Yes | Ordered list of abilities (may be empty for vanilla creatures) |

**Validation rules**:
- `name` must not be null or empty
- `types` must not be null or empty
- `abilities` must not be null (immutable copy via `List.copyOf`)
- CARDNAME, NICKNAME, and ALTERNATE remain uppercase in all text fields
- Standalone variable X remains uppercase

### Ability (Interface)

Central abstraction for all card abilities. Implementations hold Forge data directly
and defer formatting to output time. 15 variant implementations in the `ability/` sub-package.

| Method | Return Type | Description |
|--------|------------|-------------|
| `type()` | AbilityType | The category of this ability |
| `descriptionText()` | String | Human-readable description text, lowercased |
| `subAbilities()` | List\<Ability\> | Nested abilities (e.g., charm options). Default: empty list |
| `ordinal()` | int | Fixed display number (e.g., class level). Default: 0 |
| `formatLine(Integer)` | String | Format as `prefix[N]: text` or `prefix: text` |
| `formatLine()` | String | Format without explicit action number |
| `formatBlock(ActionCounter)` | String | Format self + sub-abilities, auto-assigning action numbers |

**Action numbering**: Deferred to output time. `formatBlock(ActionCounter)` assigns sequential
action numbers to actionable abilities. Abilities with `ordinal() > 0` use that as their
fixed number (class levels). Non-actionable abilities get no number.

#### Variant Implementations

| Class | Package | Forge Source | Type |
|-------|---------|-------------|------|
| `TextAbility` | `ability/` | Pre-computed text | Configurable (any AbilityType) |
| `StandardKeyword` | `ability/` | `KeywordInterface` + `Keyword` | Classified by `AbilityType.classifyKeyword()` |
| `GiftKeyword` | `ability/` | `KeywordInterface` + `SpellAbility` | STATIC |
| `CompanionKeyword` | `ability/` | `KeywordInterface` + `Companion` | STATIC |
| `ChapterAbility` | `ability/` | `KeywordInterface` (Chapter:) | CHAPTER |
| `ClassLevelAbility` | `ability/` | `KeywordInterface` (Class:) | LEVEL |
| `EtbReplacementAbility` | `ability/` | `KeywordInterface` (etbCounter:/ETBReplacement:) | REPLACEMENT |
| `AlternateCostSpell` | `ability/` | `SpellAbility` (NonBasicSpell) | ALTERNATE_COST |
| `ActivatedAbilityEntry` | `ability/` | `SpellAbility` (activated) | ACTIVATED or PLANESWALKER |
| `SpellAdditionalCost` | `ability/` | `SpellAbility` (spell cost) | ADDITIONAL_COST |
| `SpellEffect` | `ability/` | `SpellAbility` (spell chain) | SPELL |
| `CharmAbility` | `ability/` | `SpellAbility` (Charm) | SPELL (with OPTION sub-abilities) |
| `TriggeredAbilityEntry` | `ability/` | `Trigger` | TRIGGERED or REPLACEMENT (if `isStatic()`) |
| `StaticAbilityEntry` | `ability/` | `StaticAbility` | STATIC or ADDITIONAL_COST (if RaiseCost/OptionalCost on self) |
| `ReplacementAbilityEntry` | `ability/` | `ReplacementEffect` | REPLACEMENT |

Each variant has a static factory method (`of()`, `fromKeyword()`, `fromChain()`, `fromSpellAbility()`)
that returns null (or empty list) when the source data is insufficient.

### AbilityType (Enum)

Behavior-based classification. Keywords are classified by game behavior rather than
having separate KEYWORD_PASSIVE / KEYWORD_ACTIVE categories.

| Value | Output Prefix | Actionable | Source |
|-------|--------------|------------|--------|
| ACTIVATED | `activated` | Yes | Activated abilities (A:AB$) |
| TRIGGERED | `triggered` | No | Triggered abilities (T:) |
| STATIC | `static` | No | Static abilities (S:), passive keywords |
| REPLACEMENT | `replacement` | No | Replacement effects (R:), static triggers |
| CHAPTER | `chapter` | No | Saga chapter triggers (Roman prefix uppercased) |
| LEVEL | `level` | Yes | Class level abilities (ordinal = level number) |
| ALTERNATE_COST | `alternate cost` | No | Alternate casting cost keywords (flashback, morph, etc.) |
| COST_REDUCTION | `cost reduction` | No | Cost reduction keywords (affinity, convoke, delve, etc.) |
| ADDITIONAL_COST | `additional cost` | No | Additional cost keywords (kicker, buyback, etc.) or RaiseCost statics on self |
| SPELL | `spell` | Yes | Spell effects (A:SP$) |
| PLANESWALKER | `planeswalker` | Yes | Planeswalker abilities (loyalty cost formatted as `[+N]:` / `[-N]:`) |
| OPTION | `option` | Yes | Charm/modal choices (nested under CharmAbility) |
| TEXT | `text` | No | Non-ability text |

**Cost type grouping**: `isCostType()` returns true for ALTERNATE_COST, ADDITIONAL_COST, COST_REDUCTION.
Cost-type abilities are sorted before non-cost abilities in output.

**Keyword classification** (`classifyKeyword()`): Maps keywords to types based on game behavior:
- Explicit cost-type keywords (e.g., flashback → ALTERNATE_COST, kicker → ADDITIONAL_COST, affinity → COST_REDUCTION)
- Keywords with activated abilities → ACTIVATED
- Keywords with triggers → TRIGGERED
- All others → STATIC

**Internal keywords**: MAYFLASHCOST, MAYFLASHSAC are internal (suppressed from output).

### MultiCard

Represents a complete card with one or more faces.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| layout | String | No | Multi-face layout type derived from `CardSplitType`: `transform`, `split`, `adventure`, `modal`, `flip`, `meld`, `specialize`. Null for single-face cards. Output as `layout:` line before first face; omitted when null. |
| faces | List\<CardFace\> | Yes | One or more card faces. Each face has an independent action counter |

**Validation rules**:
- `faces` must have at least one entry (immutable copy via `List.copyOf`)
- Factory methods: `singleFace(CardFace)` and `multiFace(String layout, List<CardFace>)`

### ActionCounter

Mutable counter for assigning sequential action numbers during output formatting.

| Method | Description |
|--------|-------------|
| `ActionCounter(int start)` | Initialize counter at given start value |
| `next()` | Increment and return next value |

Created per-face in `CardFace.formatText()` starting at 0, so first actionable ability gets `[1]`.

### AbilityDescription (Utility Class)

Static utility methods for normalizing ability description text.

| Method | Description |
|--------|-------------|
| `normalize(String)` | Full normalization: strip reminder text + apply casing. Returns null for empty input |
| `stripReminderText(String)` | Remove parenthesized reminder text |
| `applyCasing(String)` | Lowercase text, preserving brace symbols `{W}`, placeholders `CARDNAME`/`NICKNAME`/`ALTERNATE`, and standalone variable `X` |

### SpellAbilityUtils (Package-private Utility)

Package-private utility in `ability/` sub-package for walking SpellAbility chains.

| Method | Description |
|--------|-------------|
| `collectParamInChain(SpellAbility, String)` | Collect a parameter from all sub-abilities in a chain |
| `findParamInChain(SpellAbility, String)` | Find first occurrence of a parameter in a chain |

## Relationships

```
MultiCard 1──* CardFace (faces)
CardFace 1──* Ability (abilities)
Ability *──1 AbilityType
Ability 0──* Ability (subAbilities, e.g., CharmAbility → OPTION sub-abilities)
```

## Conversion Pipeline

### Card-level: `RulesParser.parseRules(CardRules)`

```
CardRules (from CardRules.Reader)
    │
    ├─ Build full Card object via CardFactory.getCard()
    │   (instantiates all keywords, spell abilities, triggers, statics, replacements)
    │
    ├─ Determine CardSplitType:
    │   ├─ None → singleFace(parseFace(card, mainPart))
    │   ├─ Split → faces: [LeftSplit, RightSplit]
    │   ├─ Specialize → faces: [Original, ...specializeParts]
    │   └─ Others → faces: [Original, changedStateName]
    │
    └─ MultiCard.multiFace(layout, faces) or MultiCard.singleFace(face)
```

### Face-level: `RulesParser.parseFace(Card, ICardFace)`

RulesParser acts as a router, delegating to variant Ability implementations.

```
Card (in correct state) + ICardFace
    │
    ├─ Keywords (card.getKeywords()):
    │   For each KeywordInterface:
    │   ├─ UNDEFINED keyword → routeUndefinedKeyword():
    │   │   ├─ "CARDNAME ..." / "NICKNAME ..." → TextAbility(STATIC, text)
    │   │   ├─ "Chapter:..." → ChapterAbility.fromKeyword(ki)
    │   │   ├─ "Class:..." → ClassLevelAbility.of(ki)
    │   │   ├─ "etbCounter:..." / "ETBReplacement:..." → EtbReplacementAbility.fromKeyword(ki)
    │   │   ├─ "AlternateAdditionalCost:..." → TextAbility(ADDITIONAL_COST, text)
    │   │   └─ fallback → StandardKeyword.of(ki, UNDEFINED)
    │   ├─ GIFT → GiftKeyword.of(ki, card)
    │   ├─ COMPANION → CompanionKeyword.of(ki, comp)
    │   └─ others → StandardKeyword.of(ki, kw)
    │
    ├─ Spell abilities (card.getSpellAbilities()):
    │   For each SpellAbility (skip keyword-generated):
    │   ├─ ApiType.Charm → CharmAbility.fromSpellAbility(sa)
    │   ├─ NonBasicSpell → AlternateCostSpell.of(sa)
    │   ├─ isActivatedAbility() → ActivatedAbilityEntry.of(sa)
    │   └─ isSpell() → SpellAdditionalCost.of(sa) + SpellEffect.fromChain(sa)
    │
    ├─ Triggers (card.getTriggers()):
    │   → TriggeredAbilityEntry.of(t) (null if keyword-generated or no description)
    │
    ├─ Static abilities (card.getStaticAbilities()):
    │   → StaticAbilityEntry.of(s) (dynamic STATIC or ADDITIONAL_COST via D3)
    │
    ├─ Replacement effects (card.getReplacementEffects()):
    │   → ReplacementAbilityEntry.of(r) (null if keyword-generated or no description)
    │
    ├─ Synthetic land mana (from face type line):
    │   → TextAbility(ACTIVATED, "{T}: add {W} or {U}") for dual lands etc.
    │
    ├─ Post-processing:
    │   ├─ Class cards: deduplicate level inner descriptions, reclassify to LEVEL, sort by ordinal
    │   └─ Sort cost-type abilities before non-cost abilities
    │
    └─ Build CardFace(name, manaCost, types, pt, loyalty, defense, colors, text, abilities)
```

### Output formatting: `CardFace.formatText()`

```
CardFace
    │
    ├─ Emit property lines (name, mana cost, types, power toughness, loyalty, defense, colors, text)
    │
    └─ Create ActionCounter(0), for each ability:
        └─ ability.formatBlock(counter):
            ├─ ordinal > 0 → use ordinal as action number (class levels)
            ├─ type.isActionable() → use counter.next() as action number
            └─ otherwise → no action number
            Format: "prefix[N]: text" or "prefix: text"
            Recurse into subAbilities (charm options)
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
1. Top-level properties: name, mana cost, types, power toughness, loyalty, defense, colors, text
2. Ability lines: cost-type abilities first (alternate cost, additional cost, cost reduction),
   then remaining abilities in parse order

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

Layout values: `transform`, `split`, `adventure`, `modal`, `flip`, `meld`, `specialize`.
Single-face cards omit the `layout:` line entirely (no `layout: normal`).

### Text casing

All text is lowercased except:
- `CARDNAME`, `NICKNAME`, `ALTERNATE` — uppercase placeholders/markers
- Standalone variable `X` (not inside words like "exile")
- Brace symbols — mana `{W}`, `{U}`, `{B}`, `{R}`, `{G}`, `{C}`, `{S}`, `{X}`,
  generic `{1}`, `{2}`, etc., hybrid `{W/U}`, `{W/P}`, energy `{E}`,
  tap `{T}`, untap `{Q}`, and any other `{SYMBOL}` tokens remain uppercase

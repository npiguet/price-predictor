# Forge Domain Class Hierarchies

Reference document describing the two major type hierarchies in the Forge game engine
that are relevant to card script parsing: **KeywordInterface** and **CardTraitBase**.

## 1. KeywordInterface Hierarchy

Keywords represent named mechanic abilities on cards (Flying, Kicker, Affinity, etc.).
Each keyword instance holds both metadata about the keyword and the collection of
game traits (triggers, replacements, spell abilities, static abilities) that implement it.

### Hierarchy

```
KeywordInterface (interface)
  extends Cloneable, IHasSVars, ICardTraitChanges
│
└── KeywordInstance<T> (abstract base class)
    │
    ├── SimpleKeyword ── no parameters (Flying, Haste, Deathtouch)
    │   ├── Companion ── deck restrictions, special rules
    │   ├── Compleated ── Phyrexian mana effects
    │   └── Partner ── "partner with X"
    │
    ├── KeywordWithAmount ── numeric parameter (Absorb N, Annihilator N)
    │   ├── Amplify ── creature-type aware
    │   ├── Devour ── also implements KeywordWithTypeInterface
    │   ├── Firebending ── red mana symbol formatting
    │   ├── Modular ── Sunburst variant
    │   └── Vanishing ── time counters
    │
    ├── KeywordWithCost ── cost parameter (Cycling, Equip)
    │   │   implements KeywordWithCostInterface
    │   ├── Craft ── separate mana + exile costs
    │   ├── Equip ── equipment validity type
    │   ├── Kicker ── private `cost2` for double kicker
    │   ├── Mayhem ── optional cost
    │   └── Ninjutsu ── Commander zone variant
    │
    ├── KeywordWithType ── type parameter (Affinity, Hexproof from X)
    │   │   implements KeywordWithTypeInterface
    │   ├── Hexproof ── generic vs "from X" variants
    │   └── Trample ── planeswalker variant
    │
    ├── KeywordWithCostAndAmount ── cost + number (Adapt, Awaken, Monstrosity)
    │   │   implements KeywordWithCostInterface
    │   ├── Emerge ── also implements KeywordWithTypeInterface
    │   └── Suspend ── handles empty details
    │
    └── KeywordWithCostAndType ── cost + type (Splice, Typecycling)
            implements KeywordWithCostInterface, KeywordWithTypeInterface
```

### KeywordInterface Methods

| Category | Method | Description |
|----------|--------|-------------|
| **Identity** | `getKeyword()` | `Keyword` enum value |
| | `getOriginal()` | Raw text from card definition |
| | `getTitle()` | Formatted display title |
| | `getReminderText()` | Formatted reminder text |
| **Host** | `getHostCard()` | Owning card |
| | `isIntrinsic()` | Whether keyword is built into card vs. granted |
| **Numeric** | `getAmount()` | Numeric value (KeywordWithAmount and subclasses) |
| | `getAmountString()` | "X" or number string |
| **Traits** | `getTriggers()` | All trigger traits created by this keyword |
| | `getReplacements()` | All replacement effect traits |
| | `getAbilities()` | All spell ability traits |
| | `getStaticAbilities()` | All static ability traits |
| | `hasTraits()` | Whether any traits exist |
| | `createTraits(Card, boolean)` | Build abilities from keyword definition |
| **Copy** | `copy(Card, boolean)` | Deep copy with new host |
| | `redundant(Collection)` | Deduplication check |

### Mixin Interfaces

**KeywordWithCostInterface** — implemented by KeywordWithCost, KeywordWithCostAndAmount, KeywordWithCostAndType:

| Method | Returns | Description |
|--------|---------|-------------|
| `getCost()` | `Cost` | Parsed cost object |
| `getCostString()` | `String` | Raw cost string |
| `getTitleWithoutCost()` | `String` | Title excluding cost portion |

**KeywordWithTypeInterface** — implemented by KeywordWithType, KeywordWithCostAndType, and some leaf classes (Devour, Emerge):

| Method | Returns | Description |
|--------|---------|-------------|
| `getValidType()` | `String` | Machine-readable type for validation |
| `getTypeDescription()` | `String` | Human-readable type description |

### Keyword Enum Mapping (Selected)

| Keyword | Mapped Class | Available Typed Accessors |
|---------|-------------|--------------------------|
| AFFINITY | KeywordWithType | `getValidType()`, `getTypeDescription()` |
| KICKER | Kicker | `getCost()` (cost2 is private) |
| COMPANION | Companion | `getDescription()`, `getDeckRestriction()`, `getSpecialRules()` |
| GIFT | SimpleKeyword | title only |
| EQUIP | Equip | `getCost()`, equipment type |
| CYCLING | KeywordWithCost | `getCost()`, `getCostString()` |
| ADAPT / MONSTROSITY | KeywordWithCostAndAmount | `getCost()`, `getAmount()` |
| ENCHANT | KeywordWithType | `getValidType()`, `getTypeDescription()` |
| SPLICE | KeywordWithCostAndType | `getCost()`, `getValidType()`, `getTypeDescription()` |

**Important:** Keywords like `Class:`, `Chapter:`, `etbCounter:`, `ETBReplacement:`,
and `AlternateAdditionalCost:` are NOT in the Keyword enum. They resolve to
`Keyword.UNDEFINED` and can only be identified by inspecting `getOriginal()`.

### KeywordCollection

`KeywordCollection` manages all keyword instances on a card state:
- `Multimap<Keyword, KeywordInterface>` internally
- `getValues(Keyword)` — all instances of a keyword
- `contains(Keyword)` / `getAmount(Keyword)` — existence and numeric total
- Implements `Iterable<KeywordInterface>`

---

## 2. CardTraitBase Hierarchy

CardTraitBase is the base for all "traits" — the executable game objects that
implement abilities on cards: triggers, replacement effects, static abilities,
and spell abilities.

### Hierarchy

```
CardTraitBase (abstract)
│   Core: mapParams, sVars, hostCard, keyword, intrinsic, suppressed
│
├── TriggerReplacementBase (abstract)
│   │   Adds: validHostZones, overridingAbility
│   │
│   ├── Trigger (abstract → many concrete TriggerXxx subclasses)
│   │       Adds: mode (TriggerType), triggerRemembered, validPhases
│   │       Key: isStatic(), isChapter(), getChapter(), ensureAbility()
│   │
│   └── ReplacementEffect (abstract → many concrete ReplacementXxx subclasses)
│           Adds: mode (ReplacementType), layer, hasRun, otherChoices
│           Key: canReplace(), ensureAbility(), setReplacingObjects()
│
├── StaticAbility (concrete)
│       Adds: modes (Set<StaticAbilityMode>), layers, ignoreEffects
│       Key: checkMode(StaticAbilityMode), checkConditions(), generateLayer()
│
└── SpellAbility (abstract)
    │   Adds: subAbility chain, additionalAbilities map, api (ApiType),
    │         payCosts, targets, triggeringObjects, replacingObjects,
    │         activatingPlayer, manaPart, optionalCosts, description
    │
    ├── AbilitySub (concrete) ── sub-abilities in a chain
    │       Adds: parent (SpellAbility), resolve() delegates to effect
    │
    ├── Spell (abstract) ── castable spells
    │       canPlay() checks zone (Hand), mana cost, restrictions
    │
    └── Ability (abstract) ── activated/static abilities
        │   canPlay() checks host in play, not face down
        │
        ├── AbilityActivated ── activated abilities with costs
        └── AbilityStatic ── morph/disguise/manifest face-up actions
```

### CardTraitBase Methods

| Category | Method | Description |
|----------|--------|-------------|
| **Parameters** | `getParam(key)` | Get ability parameter by string key |
| | `hasParam(key)` | Check if parameter exists |
| | `getParamOrDefault(key, default)` | Get with fallback |
| | `putParam(key, value)` | Set parameter |
| | `getMapParams()` | Get all parameters |
| **SVars** | `getSVar(name)` | Get string variable |
| | `hasSVar(name)` | Check SVar existence |
| **Host** | `getHostCard()` | Owning card |
| | `getCardState()` | Card state this trait belongs to |
| **Keyword** | `getKeyword()` | Associated KeywordInterface (null if standalone) |
| | `isKeyword(Keyword)` | Check keyword type |
| **Conditions** | `meetsCommonRequirements(params)` | Checks Metalcraft, Delirium, Threshold, etc. |
| | `matchesValid(obj, valids, card)` | Validate targeting/conditions |

### Trigger-Specific Methods

| Method | Description |
|--------|-------------|
| `getMode()` | `TriggerType` enum |
| `isStatic()` | Whether trigger has `Static` parameter (static trigger replacement) |
| `isChapter()` | Whether trigger is a chapter trigger |
| `getChapter()` | Chapter number (Integer, null if not chapter) |
| `isLastChapter()` | Whether this is the final chapter |
| `isManaAbility()` | Whether this is a mana trigger |
| `ensureAbility()` | Get or create the executing SpellAbility |
| `getOverridingAbility()` | The SpellAbility that executes when triggered |
| `getTriggerRemembered()` | Objects remembered by trigger |

### StaticAbility-Specific Methods

| Method | Description |
|--------|-------------|
| `checkMode(StaticAbilityMode)` | Check if ability has a specific mode |
| `checkConditions()` | Evaluate all conditions (Threshold, Metalcraft, etc.) |
| `getMode()` | Set of `StaticAbilityMode` values |
| `isCharacteristicDefining()` | Whether this defines card characteristics |
| `getPayingTrigSA()` | Cached SpellAbility from "Trigger" parameter |
| `generateLayer()` | Determine applicable static ability layers |

**StaticAbilityMode** includes (among many others): `Continuous`, `RaiseCost`,
`ReduceCost`, `SetCost`, `OptionalCost`, `AlternativeCost`, `CantBeCast`,
`CantAttack`, `CantBlock`, `MustAttack`, `MustBlock`.

### SpellAbility-Specific Methods

| Category | Method | Description |
|----------|--------|-------------|
| **Chain** | `getSubAbility()` | Next AbilitySub in chain |
| | `getTailAbility()` | Last ability in chain (walks subs) |
| | `getParent()` | Parent ability (null for root, overridden in AbilitySub) |
| | `getRootAbility()` | Walk up to topmost ability |
| | `appendSubAbility(AbilitySub)` | Append to tail |
| | `findSubAbilityByType(ApiType)` | Find sub by API type |
| **Additional** | `getAdditionalAbility(name)` | Named additional ability (e.g., "Execute", "GiftAbility") |
| | `getAdditionalAbilityList(name)` | Named list (e.g., "Choices" for charms) |
| **Type** | `getApi()` | `ApiType` enum (Charm, Draw, Pump, etc.) |
| | `isSpell()` / `isAbility()` | Fundamental type check |
| | `isActivatedAbility()` | Whether it's an activated ability |
| | `isPwAbility()` | Whether it's a planeswalker ability |
| | `isTrigger()` | Whether backed by a trigger |
| **Costs** | `getPayCosts()` | Parsed `Cost` object |
| | `getCostDescription()` | Human-readable cost string |
| | `costHasX()` | Whether cost contains X |
| **Targets** | `getTargetRestrictions()` | Target requirements |
| | `getTargets()` | Chosen targets |
| | `findTargetedCards()` | All targeted cards in chain |
| **Description** | `getDescription()` | Ability description |
| | `getStackDescription()` | Description when on stack |
| **Mechanics** | `isCycling()`, `isKicked()`, etc. | Boolean checks for many mechanics |

### Sub-Ability Chain Model

```
SpellAbility (root)
├── subAbility → AbilitySub
│   ├── subAbility → AbilitySub
│   │   └── subAbility → AbilitySub (tail)
│   └── parent ← points back up
└── additionalAbilities (Map<String, SpellAbility>)
    ├── "Execute" → SpellAbility
    ├── "GiftAbility" → SpellAbility
    └── "Choices" → List<AbilitySub>  (via additionalAbilityLists)
```

**No built-in chain traversal utility exists.** Walking the sub-ability chain
requires manual iteration:
```java
SpellAbility sub = sa.getSubAbility();
while (sub != null) {
    // process sub
    sub = sub.getSubAbility();
}
```

---

## 3. How Cards Expose Their Traits

`Card` provides typed accessors that return strongly-typed collections:

| Method | Returns | Actual Type |
|--------|---------|-------------|
| `getKeywords()` | `KeywordCollectionView` | Iterable of `KeywordInterface` |
| `getSpellAbilities()` | `FCollectionView<SpellAbility>` | Typed `SpellAbility` |
| `getTriggers()` | `FCollectionView<Trigger>` | Typed `Trigger` |
| `getStaticAbilities()` | `FCollectionView<StaticAbility>` | Typed `StaticAbility` |
| `getReplacementEffects()` | `FCollectionView<ReplacementEffect>` | Typed `ReplacementEffect` |

**Each keyword also holds its own sub-collections** of these same types,
accessible via `ki.getTriggers()`, `ki.getAbilities()`, `ki.getStaticAbilities()`,
`ki.getReplacements()`.

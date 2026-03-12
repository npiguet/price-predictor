# Feature Specification: Card Script Conversion

**Feature Branch**: `006-card-script-parsing`
**Created**: 2026-03-04
**Updated**: 2026-03-12
**Status**: Implemented
**Input**: User description: "Convert Forge card scripts into an LLM-friendly format with English key names, Oracle-matching ability text, numbered player-actionable abilities, batch processing, and Python CLI integration."

## Clarifications

### Session 2026-03-05

- Q: Should CARDNAME/NICKNAME be substituted with the actual card name or kept as literal uppercase placeholders? → A: Keep as literal uppercase placeholders. The card name is already in the `name:` property; CARDNAME provides a consistent self-reference token for the AI.
- Q: Should non-standard card types (tokens, emblems, dungeons, planes, schemes, etc.) be included or skipped during batch processing? → A: Include all cards. Non-standard types (dungeons, planes, schemes, etc.) already have distinguishing card types for downstream filtering. No special marker needed.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Convert Single Card Script (Priority: P1)

A data scientist wants to convert a single Forge card script into a human-readable, LLM-friendly format to verify the output reads like annotated Oracle text before processing the full card library.

**Why this priority**: Foundational capability. Correct single-card conversion must work before batch processing makes sense. Enables rapid iteration and debugging during development.

**Independent Test**: Provide any card script file and compare the converted output against the card's Oracle text and visible card properties.

**Acceptance Scenarios**:

1. **Given** a vanilla creature script (e.g., Grizzly Bears: Name, ManaCost, Types, PT only), **When** converted, **Then** the output contains lowercase English key names (name, mana cost, types, power toughness) with correct values and no ability lines.
2. **Given** a card with passive keywords (e.g., Flying, Trample), **When** converted, **Then** each passive keyword appears with its classified type prefix (e.g., `static: flying`) without a number label.
3. **Given** a card with cost-modifying keywords (e.g., Kicker, Flashback, Convoke), **When** converted, **Then** each appears with its cost type prefix (e.g., `additional cost: kicker {cost}`, `alternate cost: flashback {cost}`, `cost reduction: convoke`) and cost keyword lines are sorted before other abilities.
4. **Given** a card with activated abilities (e.g., "{T}: Add {G}"), **When** converted, **Then** each ability appears as "activated[N]: {T}: add {G}." with cost and description matching Oracle text.
5. **Given** a card with triggered abilities (e.g., an ETB trigger), **When** converted, **Then** the trigger appears as "triggered: when..." matching Oracle text, without a number label.
6. **Given** a card with static abilities (e.g., "Other creatures you control get +1/+1"), **When** converted, **Then** each appears as "static: description" matching Oracle text, without a number label.
7. **Given** a card with replacement effects (e.g., "CARDNAME enters tapped"), **When** converted, **Then** each appears as "replacement: description" matching Oracle text.
7b. **Given** a spell with an additional casting cost (e.g., "As an additional cost to cast this spell, discard X cards."), **When** converted, **Then** the cost appears on its own line as `additional cost: discard X cards.` (without the "As an additional cost to cast this spell" prefix) and the spell effect appears on a separate `spell[N]:` line.
8. **Given** a card with reminder text in parentheses, **When** converted, **Then** the reminder text is omitted from the output.
9. **Given** any converted card, **When** the text portions of all ability lines are concatenated, **Then** the result matches the card's Oracle text (whitespace, case, and formatting symbol differences acceptable).
10. **Given** a card script with engine metadata (AI:, DeckHints:, SVar:, etc.), **When** converted, **Then** none of these appear in the output and no original script syntax remains.

---

### User Story 2 - Convert Complex Card Types (Priority: P2)

A data scientist wants to correctly convert planeswalkers, modal/charm spells, multi-face cards, and instant/sorcery spells so the full diversity of Magic card types is represented in the dataset.

**Why this priority**: These card types represent a significant portion of the library and require special formatting rules. Without them the dataset is incomplete.

**Independent Test**: Convert representative cards of each complex type and verify output format matches the specified conventions.

**Acceptance Scenarios**:

1. **Given** a planeswalker script (e.g., Jace Beleren with +2, -1, -10 abilities), **When** converted, **Then** each loyalty ability appears as "planeswalker[N]: [+2]: each player draws a card." with the loyalty cost enclosed in square brackets to match Oracle text format.
2. **Given** a charm/modal spell (e.g., Abzan Charm: "Choose one —"), **When** converted, **Then** the charm description appears as a `spell[N]:` line and each mode appears as a nested `option[N]: description` line. If the charm has no description, the options appear as top-level `option[N]:` lines.
3. **Given** a transform card (e.g., Delver of Secrets // Insectile Aberration), **When** converted, **Then** both faces appear separated by ALTERNATE, each with its own properties, abilities, and independent action counter.
4. **Given** a split card (e.g., Fire // Ice), **When** converted, **Then** both halves appear separated by ALTERNATE.
5. **Given** an adventure card (e.g., Bonecrusher Giant // Stomp), **When** converted, **Then** both the creature and adventure spell appear separated by ALTERNATE.
6. **Given** an instant or sorcery with a single effect (e.g., Lightning Bolt), **When** converted, **Then** the main effect appears as "spell[N]: description".
7. **Given** a battle card with a defense value, **When** converted, **Then** the output includes a `defense:` property line.

---

### User Story 3 - Batch Process Card Library (Priority: P3)

A data scientist wants to process all card scripts in the Forge card folder at once to build a complete dataset for model training.

**Why this priority**: End-goal usage. Depends on single-card conversion being correct first.

**Independent Test**: Run batch process on the full cardsfolder directory, verify output file count and spot-check random entries.

**Acceptance Scenarios**:

1. **Given** the full Forge cardsfolder directory (32,000+ scripts in alphabetical subdirectories), **When** batch processing runs, **Then** the output directory mirrors the source directory structure with one converted file per card script.
2. **Given** some malformed or unusual scripts, **When** batch processing encounters them, **Then** the system logs a warning prefixed with the card name (format: `[Card Name] message`) and continues processing.
3. **Given** the process completes, **When** output location is checked, **Then** it defaults to `./output` relative to the project root.

---

### User Story 4 - Launch Conversion from CLI (Priority: P4)

A data scientist wants to trigger the conversion from the existing Python CLI so it integrates seamlessly with the rest of the price prediction pipeline.

**Why this priority**: Integration convenience. The core conversion must work first.

**Independent Test**: Run the Python CLI subcommand and verify it launches the conversion process and produces output files.

**Acceptance Scenarios**:

1. **Given** the Python CLI is available, **When** the user runs the conversion subcommand, **Then** the conversion process launches and produces the expected output files.
2. **Given** no arguments are provided, **When** defaults apply, **Then** it processes the standard cardsfolder location and writes to `./output`.

---

### Edge Cases

- What happens when a card script has no abilities (vanilla creature like Grizzly Bears)? The output contains only top-level properties with no ability lines.
- What happens when a card has `ManaCost:no cost` (e.g., lands, transform back faces)? The mana cost line is omitted from the output.
- What happens with sub-abilities chained via SubAbility$ or Execute$? They are part of their parent ability's description, NOT separate entries.
- What happens when an ability has no description parameter (no SpellDescription$, TriggerDescription$, or Description$)? The ability is silently skipped — no output line is produced.
- What happens when a card has a `Text:` property containing non-ability rules text? The text is preserved as a "text:" line in the output.
- What happens when the global action counter spans multiple ability types on one face? It increments across all player-actionable items on the same face (e.g., activated[1], spell[2], planeswalker[3]). Action numbers are assigned at output time, not during parsing.
- What happens with charm abilities that have a charm description? The charm description appears as `spell[N]:` with each choice as a nested `option[N]:` sub-ability. Without a charm description, choices become top-level `option[N]:` lines.
- What happens with saga, class, or room cards? Saga chapter abilities use `chapter:` prefix with the description following the Forge/Oracle convention (e.g., `chapter: I — draw a card.`). Class enchantments use `level[N]:` prefix: level 1 has no cost prefix, levels 2+ include the level-up cost (e.g., `level[2]: {1}{U}: effect description`). The bracket number is the level number, not the global action counter.
- What happens with cost-modifying keywords? Keywords are classified into `alternate cost:` (flashback, ninjutsu, etc.), `additional cost:` (kicker, bargain, etc.), or `cost reduction:` (convoke, delve, etc.). All cost lines are sorted before non-cost abilities on the card.
- What happens with battle cards? A `defense:` property line is included in the output.
- What happens with meld cards? Only the front half is emitted (with `layout: meld`), since the back face is shared between two different cards.
- What happens with self-targeting RaiseCost/OptionalCost statics? They are reclassified from `static:` to `additional cost:` (e.g., "you may collect evidence 8").
- What happens with basic lands that have no explicit mana ability in their script? An implicit `activated[N]: {T}: add {G}` (or appropriate color) is generated from the land's subtypes.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST extract the following top-level properties from each card script and present them with lowercase English key names: `name`, `mana cost`, `types`, `power toughness`, `loyalty`, `defense`, `colors`, `text`. Properties absent or empty in the source script MUST be omitted from output.
- **FR-002**: System MUST convert mana costs from Forge internal syntax (e.g., `2 W W`) to human-readable paper-card format (e.g., `{2}{W}{W}`). Ability costs MUST likewise be converted (e.g., `T Sac<1/CARDNAME>` becomes `{T}, sacrifice CARDNAME:`). Cards with `ManaCost:no cost` MUST omit the mana cost line entirely.
- **FR-003**: System MUST extract keyword abilities and classify each by its game behavior. Passive keywords (flying, trample, deathtouch, etc.) appear as `static: name`. Keywords with activated abilities (cycling, equip, etc.) appear as `activated[N]: name cost`. Keywords representing alternate casting costs (flashback, ninjutsu, escape, etc.) appear as `alternate cost: name cost`. Keywords representing additional casting costs (kicker, bargain, spree, etc.) appear as `additional cost: name cost`. Keywords representing cost reductions (convoke, delve, affinity, etc.) appear as `cost reduction: name`. Keyword parameters (costs, amounts) MUST be included. Special keywords (companion, gift) include their descriptions.
- **FR-004**: System MUST extract activated abilities and present each as `activated[N]: cost: description` where cost and description match Oracle text.
- **FR-005**: System MUST extract triggered abilities and present each as `triggered: description` matching Oracle text. Triggered abilities are automatic and do NOT receive a number label. Triggers with `Static$ True` (enters-the-battlefield replacement effects) are emitted as `replacement:` instead.
- **FR-006**: System MUST extract static abilities and present each as `static: description` matching Oracle text. Static abilities are passive and do NOT receive a number label. Self-targeting RaiseCost/OptionalCost statics are reclassified as `additional cost:`.
- **FR-007**: System MUST extract replacement effects and present each as `replacement: description` matching Oracle text.
- **FR-008**: System MUST extract instant and sorcery spell effects and present each as `spell[N]: description` with a number from the global action counter. Spells with additional casting costs MUST emit the cost on a separate `additional cost: description` line (without the "As an additional cost to cast this spell" prefix, since the label already conveys that). The spell effect appears on its own `spell[N]:` line. Creatures with additional casting costs (e.g., exile cards from graveyard) MUST also emit the `additional cost:` line. All cost-type lines (alternate cost, additional cost, cost reduction) MUST be sorted before non-cost abilities.
- **FR-009**: All player-actionable abilities (activated abilities, spell effects, planeswalker abilities, modal options, level abilities) MUST be labeled with a unique incrementing number per card face. The counter starts at 1 for each face and increments across all actionable ability types on that face. Action numbers are assigned at formatting time, not during parsing — this allows post-processing (cost sorting, class reclassification) without invalidating numbers.
- **FR-010**: Planeswalker loyalty abilities MUST be formatted as `planeswalker[N]: [+X]: description` with the loyalty cost enclosed in square brackets to match Oracle text format. Forge's `CostPutCounter`/`CostRemoveCounter` produce `+N`/`-N` natively; the converter wraps them in brackets.
- **FR-011**: Modal spells and charms MUST present the charm description as a `spell[N]:` line with each choice as a nested `option[N]: description` sub-ability line. If the charm has no description, choices appear as top-level `option[N]:` lines. Paw print (Season) spells MUST prepend the paw print cost to each option (e.g., `option[2]: {P} — create a 1/1 white rabbit creature token.`, `option[3]: {P}{P} — exile target nonland permanent.`).
- **FR-012**: All ability descriptions MUST read like the equivalent section of Oracle text. Reminder text (text within parentheses) MUST be omitted from the output.
- **FR-013**: Engine-internal metadata (AI:, DeckHints:, DeckHas:, DeckNeeds:, SVar:, Oracle:, and AI-specific parameters like AILogic$, AITgts$) MUST be excluded from output. No remnants of the original script syntax may appear beyond the `keyName: descriptionText` format.
- **FR-014**: Multi-face cards (transform, split, adventure, modal double-faced, flip, meld, specialize) MUST include a `layout:` line before the first face (e.g., `layout: transform`) and all faces in the output, separated by the ALTERNATE marker. Each face carries its own properties, abilities, and independent action counter. Single-face cards omit the `layout:` line. Meld cards emit only the front half since the back face is shared.
- **FR-015**: All generated text and property names MUST be lowercase, except for CARDNAME, NICKNAME, ALTERNATE (uppercase placeholders/markers), standalone variable X, and brace-enclosed symbols (`{W}`, `{U}`, `{B}`, `{R}`, `{G}`, `{C}`, `{S}`, `{X}`, `{T}`, `{Q}`, `{E}`, hybrid `{W/U}`, generic `{1}`, `{2}`, etc.) which MUST remain uppercase.
- **FR-016**: The Oracle text property is NOT included in the output. The Oracle text MUST be reconstructable by concatenating the text descriptions of all ability lines on the card (whitespace, case, and bullet/formatting differences are acceptable).
- **FR-017**: System MUST support batch processing of a directory tree of card scripts, converting ALL scripts regardless of card type (including tokens, emblems, dungeons, planes, schemes, etc.). The output directory structure MUST mirror the source directory structure. Default output location: `./output`.
- **FR-018**: Malformed or unparseable card scripts MUST be logged with a warning including the card name in the format `[Card Name] message`. Processing MUST continue for remaining files.
- **FR-019**: The existing Python CLI MUST provide a subcommand that launches the conversion process.
- **FR-020**: Basic lands with subtypes (Plains, Island, Swamp, Mountain, Forest) MUST generate an implicit mana ability (`activated[N]: {T}: add {symbol}`) even when no explicit mana ability exists in the script.

### Key Entities

- **Converted Card** (`CardFace`): A single card face in the output format. Contains top-level properties (name, mana cost, types, power/toughness, loyalty, defense, colors, text) followed by a sequence of ability entries. Each ability uses a type prefix (activated, triggered, static, replacement, spell, planeswalker, option, chapter, level, alternate cost, additional cost, cost reduction, text) with an optional action number in brackets. Multi-face cards produce multiple CardFace entries separated by ALTERNATE.
- **Ability** (interface): Represents a single ability on a card face. Implemented as a Java interface with 15 variant implementations (TextAbility, StandardKeyword, GiftKeyword, CompanionKeyword, ChapterAbility, ClassLevelAbility, EtbReplacementAbility, AlternateCostSpell, ActivatedAbilityEntry, SpellAdditionalCost, SpellEffect, CharmAbility, TriggeredAbilityEntry, StaticAbilityEntry, ReplacementAbilityEntry). Each variant encapsulates its own extraction logic in a static factory method, taking Forge objects as input. Formatting (including action number assignment) is deferred to output time via `formatBlock(ActionCounter)`.
- **Action Counter**: A per-face incrementing integer starting at 1. Assigned at output time to all player-actionable abilities (activated, spell effects, planeswalker abilities, modal options, level abilities). Provides a unique identifier the AI can reference to communicate desired actions in future features. Class levels use fixed ordinal numbers (the level number) instead of the action counter.

### Output Format Examples

**Vanilla creature** (Grizzly Bears):
```
name: grizzly bears
mana cost: {1}{G}
types: creature bear
power toughness: 2/2
```

**Creature with keywords and activated ability** (Llanowar Elves-like):
```
name: llanowar elves
mana cost: {G}
types: creature elf druid
power toughness: 1/1
activated[1]: {T}: add {G}.
```

**Creature with kicker and triggered/static abilities** (Kangee-like):
```
name: kangee, aerie keeper
mana cost: {2}{W}{U}
types: legendary creature bird wizard
power toughness: 2/2
additional cost: kicker {X}{2}
static: flying
triggered: when CARDNAME enters, if it was kicked, put X feather counters on it.
static: other bird creatures get +1/+1 for each feather counter on CARDNAME.
```

**Planeswalker** (Jace Beleren):
```
name: jace beleren
mana cost: {1}{U}{U}
types: legendary planeswalker jace
loyalty: 3
planeswalker[1]: [+2]: each player draws a card.
planeswalker[2]: [-1]: target player draws a card.
planeswalker[3]: [-10]: target player mills twenty cards.
```

**Modal spell / Charm** (Abzan Charm):
```
name: abzan charm
mana cost: {W}{B}{G}
types: instant
spell[1]: choose one —
option[2]: exile target creature with power 3 or greater.
option[3]: you draw two cards and you lose 2 life.
option[4]: distribute two +1/+1 counters among one or two target creatures.
```

**Class enchantment** (Artificer Class):
```
name: artificer class
mana cost: {1}{U}
types: enchantment class
level[1]: the first artifact spell you cast each turn costs {1} less to cast.
level[2]: {1}{U}: when this class becomes level 2, reveal cards from the top of your library until you reveal an artifact card. put that card into your hand and the rest on the bottom of your library in a random order.
level[3]: {5}{U}: at the beginning of your end step, create a token that's a copy of target artifact you control.
```

**Saga** (The Eldest Reborn-like):
```
name: the eldest reborn
mana cost: {4}{B}
types: enchantment saga
chapter: I — each opponent sacrifices a creature or planeswalker.
chapter: II — each opponent discards a card.
chapter: III — put target creature or planeswalker card from a graveyard onto the battlefield under your control.
```

**Transform card** (Daring Sleuth // Bearer of Overwhelming Truths):
```
layout: transform
name: daring sleuth
mana cost: {1}{U}
types: creature human rogue
power toughness: 2/1
triggered: when you sacrifice a clue, transform CARDNAME.

ALTERNATE

name: bearer of overwhelming truths
colors: blue
types: creature human wizard
power toughness: 3/2
static: prowess
triggered: whenever CARDNAME deals combat damage to a player, investigate.
```

**Battle card** (Invasion of Kamigawa):
```
layout: transform
name: invasion of kamigawa
mana cost: {1}{U}
types: battle siege
defense: 4
triggered: when CARDNAME enters, tap target artifact or creature an opponent controls.

ALTERNATE

name: rooftop saboteurs
types: creature moonfolk rogue
power toughness: 2/3
...
```

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: System successfully converts 99% or more of all card scripts in the Forge card folder without errors.
- **SC-002**: For any converted card, the ability description lines when concatenated match the card's Oracle text (verified by spot-checking a sample of 50 diverse cards across all major card types).
- **SC-003**: No engine-internal scripting details (SVars, API types, targeting parameters, AI hints, ability script syntax) appear in any converted output.
- **SC-004**: Multi-face cards produce correctly separated face entries, each with independently correct properties and abilities.
- **SC-005**: The full card library (32,000+ card scripts) processes in a single batch run to completion.
- **SC-006**: Every player-actionable ability across a sample of 50 cards has a unique, correctly incrementing number within its card face.

## Assumptions

- Card script files are located in the Forge repository at `../forge/forge-gui/res/cardsfolder/`, organized in alphabetical subdirectories.
- The card scripting language follows the patterns documented in `resources/CARD_SCRIPTING_REFERENCE.md`, with additional details available in the Forge Java source at `../forge/`.
- The conversion process will be implemented in the existing Java connector module (`forge-connector/`), leveraging the Forge parser and internal classes (Card, SpellAbility, Keyword, AbilityFactory, CardRules, etc.) for script parsing and text generation. Required Forge modules (forge-core, forge-game) will be added as Maven SNAPSHOT dependencies.
- The Python CLI subcommand will launch the Java conversion process as an external command.
- All conversion logs must include the card name in the format `[Card Name] message`.
- Performance does not need to be real-time; processing the entire library within minutes is acceptable.
- Sub-abilities (DB$ prefix) referenced via SubAbility$ or Execute$ are folded into their parent ability's description and never appear as separate lines.

# Feature Specification: Card Script Conversion

**Feature Branch**: `006-card-script-parsing`
**Created**: 2026-03-04
**Status**: Draft
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
2. **Given** a card with passive keywords (e.g., Flying, Trample), **When** converted, **Then** each passive keyword appears as "keyword: flying" without a number label.
3. **Given** a card with activatable keywords (e.g., Kicker, Cycling, Equip), **When** converted, **Then** each appears as "keyword[N]: kicker {cost}" with a number from the global action counter, including cost parameters.
4. **Given** a card with activated abilities (e.g., "{T}: Add {G}"), **When** converted, **Then** each ability appears as "activated[N]: {T}: add {G}." with cost and description matching Oracle text.
5. **Given** a card with triggered abilities (e.g., an ETB trigger), **When** converted, **Then** the trigger appears as "triggered: when..." matching Oracle text, without a number label.
6. **Given** a card with static abilities (e.g., "Other creatures you control get +1/+1"), **When** converted, **Then** each appears as "static: description" matching Oracle text, without a number label.
7. **Given** a card with replacement effects (e.g., "CARDNAME enters tapped"), **When** converted, **Then** each appears as "replacement: description" matching Oracle text.
8. **Given** a card with reminder text in parentheses, **When** converted, **Then** the reminder text is omitted from the output.
9. **Given** any converted card, **When** the text portions of all ability lines are concatenated, **Then** the result matches the card's Oracle text (whitespace, case, and formatting symbol differences acceptable).
10. **Given** a card script with engine metadata (AI:, DeckHints:, SVar:, etc.), **When** converted, **Then** none of these appear in the output and no original script syntax remains.

---

### User Story 2 - Convert Complex Card Types (Priority: P2)

A data scientist wants to correctly convert planeswalkers, modal/charm spells, multi-face cards, and instant/sorcery spells so the full diversity of Magic card types is represented in the dataset.

**Why this priority**: These card types represent a significant portion of the library and require special formatting rules. Without them the dataset is incomplete.

**Independent Test**: Convert representative cards of each complex type and verify output format matches the specified conventions.

**Acceptance Scenarios**:

1. **Given** a planeswalker script (e.g., Jace Beleren with +2, -1, -10 abilities), **When** converted, **Then** each loyalty ability appears as "planeswalker[N]: +2: each player draws a card." using the Forge/Oracle loyalty cost prefix (no square brackets).
2. **Given** a charm/modal spell (e.g., Abzan Charm: "Choose one —"), **When** converted, **Then** each mode appears on its own line as "option[N]: description".
3. **Given** a transform card (e.g., Delver of Secrets // Insectile Aberration), **When** converted, **Then** both faces appear separated by ALTERNATE, each with its own properties, abilities, and independent action counter.
4. **Given** a split card (e.g., Fire // Ice), **When** converted, **Then** both halves appear separated by ALTERNATE.
5. **Given** an adventure card (e.g., Bonecrusher Giant // Stomp), **When** converted, **Then** both the creature and adventure spell appear separated by ALTERNATE.
6. **Given** an instant or sorcery with a single effect (e.g., Lightning Bolt), **When** converted, **Then** the main effect appears as "spell[N]: description".

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
- What happens when an ability has no description parameter (no SpellDescription$, TriggerDescription$, or Description$)? The system logs a warning with the card name and outputs the ability with a description-unavailable marker.
- What happens when a card has a `Text:` property containing non-ability rules text? The text is preserved as a "text:" line in the output.
- What happens when the global action counter spans multiple ability types on one face? It increments across all player-actionable items on the same face (e.g., keyword[1], activated[2], activated[3]).
- What happens with triggered abilities that involve a player choice (e.g., "choose one" on an ETB)? The trigger line itself has no number, but its options each receive "option[N]:" with numbers from the global counter.
- What happens with saga, class, or room cards? Saga chapter abilities use `chapter:` prefix with the description following the Forge/Oracle convention (e.g., `chapter: I — draw a card.`). Class level and room entries follow their respective Oracle text format.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST extract the following top-level properties from each card script and present them with lowercase English key names: `name`, `mana cost`, `types`, `power toughness`, `loyalty`, `colors`, `text`. Properties absent or empty in the source script MUST be omitted from output.
- **FR-002**: System MUST convert mana costs from Forge internal syntax (e.g., `2 W W`) to human-readable paper-card format (e.g., `{2}{W}{W}`). Ability costs MUST likewise be converted (e.g., `T Sac<1/CARDNAME>` becomes `{T}, sacrifice CARDNAME:`). Cards with `ManaCost:no cost` MUST omit the mana cost line entirely.
- **FR-003**: System MUST extract keyword abilities. Passive keywords (e.g., flying, trample, deathtouch) appear as `keyword: name`. Keywords representing a player choice or action (e.g., kicker, cycling, equip, ninjutsu, evoke) appear as `keyword[N]: name cost` with a number from the global action counter. Keyword parameters (costs, amounts) MUST be included.
- **FR-004**: System MUST extract activated abilities and present each as `activated[N]: cost: description` where cost and description match Oracle text.
- **FR-005**: System MUST extract triggered abilities and present each as `triggered: description` matching Oracle text. Triggered abilities are automatic and do NOT receive a number label.
- **FR-006**: System MUST extract static abilities and present each as `static: description` matching Oracle text. Static abilities are passive and do NOT receive a number label.
- **FR-007**: System MUST extract replacement effects and present each as `replacement: description` matching Oracle text.
- **FR-008**: System MUST extract instant and sorcery spell effects and present each as `spell[N]: description` with a number from the global action counter.
- **FR-009**: All player-actionable abilities (activated abilities, activatable keywords, spell effects, planeswalker abilities, modal options) MUST be labeled with a unique incrementing number per card face. The counter starts at 1 for each face and increments across all actionable ability types on that face.
- **FR-010**: Planeswalker loyalty abilities MUST be formatted as `planeswalker[N]: +X: description` using the Forge/Oracle text loyalty cost prefix (no square brackets around the loyalty cost). Forge's `CostPutCounter`/`CostRemoveCounter` already produce `+N`/`-N` natively.
- **FR-011**: Modal spells and charms MUST present each choice on its own line as `option[N]: description`.
- **FR-012**: All ability descriptions MUST read like the equivalent section of Oracle text. Reminder text (text within parentheses) MUST be omitted from the output.
- **FR-013**: Engine-internal metadata (AI:, DeckHints:, DeckHas:, DeckNeeds:, SVar:, Oracle:, and AI-specific parameters like AILogic$, AITgts$) MUST be excluded from output. No remnants of the original script syntax may appear beyond the `keyName: descriptionText` format.
- **FR-014**: Multi-face cards (transform, split, adventure, modal double-faced, flip, room) MUST include a `layout:` line before the first face (e.g., `layout: transform`) and all faces in the output, separated by the ALTERNATE marker. Each face carries its own properties, abilities, and independent action counter. Single-face cards omit the `layout:` line.
- **FR-015**: All generated text and property names MUST be lowercase, except for CARDNAME, NICKNAME, ALTERNATE (uppercase placeholders/markers), and brace-enclosed symbols (`{W}`, `{U}`, `{B}`, `{R}`, `{G}`, `{C}`, `{S}`, `{X}`, `{T}`, `{Q}`, `{E}`, hybrid `{W/U}`, generic `{1}`, `{2}`, etc.) which MUST remain uppercase.
- **FR-016**: The Oracle text property is NOT included in the output. The Oracle text MUST be reconstructable by concatenating the text descriptions of all keyword and ability lines on the card (whitespace, case, and bullet/formatting differences are acceptable).
- **FR-017**: System MUST support batch processing of a directory tree of card scripts, converting ALL scripts regardless of card type (including tokens, emblems, dungeons, planes, schemes, etc.). The output directory structure MUST mirror the source directory structure. Default output location: `./output`.
- **FR-018**: Malformed or unparseable card scripts MUST be logged with a warning including the card name in the format `[Card Name] message`. Processing MUST continue for remaining files.
- **FR-019**: The existing Python CLI MUST provide a subcommand that launches the conversion process.

### Key Entities

- **Converted Card**: A single card face in the output format. Contains top-level properties (name, mana cost, types, power/toughness, loyalty, colors, text) followed by a sequence of ability lines. Each ability line uses a type prefix (keyword, activated, triggered, static, replacement, spell, planeswalker, option) with an optional action number in brackets. Multi-face cards produce multiple Converted Card entries separated by ALTERNATE.
- **Action Counter**: A per-face incrementing integer starting at 1. Assigned to all player-actionable abilities (activated, activatable keywords, spell effects, planeswalker abilities, modal options). Provides a unique identifier the AI can reference to communicate desired actions in future features.

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
keyword[1]: kicker {X}{2}
keyword: flying
triggered: when CARDNAME enters, if it was kicked, put x feather counters on it.
static: other bird creatures get +1/+1 for each feather counter on CARDNAME.
```

**Planeswalker** (Jace Beleren):
```
name: jace beleren
mana cost: {1}{U}{U}
types: legendary planeswalker jace
loyalty: 3
planeswalker[1]: +2: each player draws a card.
planeswalker[2]: -1: target player draws a card.
planeswalker[3]: -10: target player mills twenty cards.
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
keyword: prowess
triggered: whenever CARDNAME deals combat damage to a player, investigate.
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
- "Activatable keywords" are those where the player makes a decision (pay an additional cost, choose to activate): kicker, cycling, equip, ninjutsu, evoke, emerge, channel, and similar. "Passive keywords" are abilities that apply automatically: flying, trample, deathtouch, vigilance, first strike, and similar.
- Sub-abilities (DB$ prefix) referenced via SubAbility$ or Execute$ are folded into their parent ability's description and never appear as separate lines.

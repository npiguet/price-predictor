# Feature Specification: Card Script Data Extraction

**Feature Branch**: `006-card-script-parsing`
**Created**: 2026-02-28
**Status**: Draft
**Input**: User description: "Extract card data from Forge card scripts for ML model input, including all visible card properties, keyword abilities, ability costs and descriptions, triggered abilities, and replacement effects."

## Clarifications

### Session 2026-03-01

- Q: Should mana costs and ability costs use Forge internal syntax or human-readable paper-card format? → A: Convert to human-readable paper-card format (e.g., `{2}{W}{W}`, `{T}, Sacrifice CARDNAME:`).
- Q: Should abilities carry a type tag in the output? → A: Yes, include a type tag using official rules terminology: "activated", "triggered", "static", "replacement" — the same words used in MTG rules text and on cards (e.g., "counter target activated ability").
- Q: Should Oracle text be included alongside individual ability descriptions? → A: No, drop Oracle entirely. Use individually extracted ability descriptions as the sole source. This reduces token count and shortens model input.
- Q: How should multi-face cards be structured in the output? → A: Follow the Forge script convention: front face appears first (including AlternateMode), followed by a blank line, then "ALTERNATE", another blank line, then the back face. Same format for split cards, adventures, rooms, etc.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Parse Single Card Script (Priority: P1)

A data scientist wants to extract structured card data from a single Forge card script file so they can inspect the output and verify correctness before processing the full card library.

**Why this priority**: This is the foundational capability. Without being able to parse a single card correctly, nothing else works. It also enables rapid iteration and debugging during development.

**Independent Test**: Can be fully tested by providing a card script file and verifying the extracted data matches what appears on the physical card.

**Acceptance Scenarios**:

1. **Given** a simple creature card script (e.g., Grizzly Bears with Name, ManaCost, Types, PT), **When** the script is parsed, **Then** the output contains all top-level properties with their correct values.
2. **Given** a card script with keyword abilities (e.g., a creature with Flying, Kicker:2 R), **When** the script is parsed, **Then** all keywords appear in the output, including their parameters.
3. **Given** a card script with activated abilities (e.g., Llanowar Elves), **When** the script is parsed, **Then** the output includes each ability's cost and textual description, but not the internal scripting details.
4. **Given** a card script with triggered abilities (e.g., Aarakocra Sneak with an ETB trigger), **When** the script is parsed, **Then** the trigger description appears as it would read on the physical card.
5. **Given** a card script with replacement effects (e.g., Solitary Confinement), **When** the script is parsed, **Then** the replacement effect descriptions appear as they would read on the physical card.
6. **Given** a card script with sub-abilities chained via SubAbility$ or Execute$, **When** the script is parsed, **Then** sub-abilities are NOT included as separate entries because they are part of their parent ability's textual description.

---

### User Story 2 - Parse Multi-Face Cards (Priority: P2)

A data scientist wants to extract data from multi-face card scripts (transform, split, adventure, modal double-faced) so that each face is represented as a separate card entry with its own properties and abilities.

**Why this priority**: Multi-face cards are a significant portion of the card library. Without this, the dataset would be incomplete or incorrectly structured.

**Independent Test**: Can be tested by providing multi-face card scripts and verifying each face is extracted independently with correct properties.

**Acceptance Scenarios**:

1. **Given** a transform card script (e.g., Delver of Secrets // Insectile Aberration with AlternateMode:DoubleFaced), **When** the script is parsed, **Then** the output contains separate entries for the front and back faces, each with their own properties and abilities.
2. **Given** a split card script (e.g., Fire // Ice with AlternateMode:Split), **When** the script is parsed, **Then** both halves are extracted as separate entries linked by the AlternateMode.
3. **Given** an adventure card script (e.g., Bonecrusher Giant // Stomp), **When** the script is parsed, **Then** both the creature and the adventure spell are extracted separately.

---

### User Story 3 - Batch Process Card Library (Priority: P3)

A data scientist wants to process all card scripts in the Forge card folder at once so they can build a complete dataset for training the price prediction model.

**Why this priority**: This is the end-goal usage, but it depends on correct single-card parsing (P1) and multi-face handling (P2) being solid first.

**Independent Test**: Can be tested by pointing the system at the full cardsfolder directory and verifying the output count and spot-checking random entries.

**Acceptance Scenarios**:

1. **Given** a directory containing thousands of card script files organized in alphabetical subdirectories, **When** batch processing is run, **Then** all card scripts are parsed and the output contains one entry per card face.
2. **Given** some malformed or unusual card scripts in the directory, **When** batch processing encounters them, **Then** the system logs a warning with the file path and continues processing remaining files without crashing.

---

### Edge Cases

- What happens when a card script has no abilities at all (vanilla creature like Grizzly Bears)? The output should contain only top-level properties with an empty abilities list.
- What happens when a card script has a `Text:` property containing rules text not generated by abilities? The text should be preserved as-is in the output.
- What happens when a keyword line contains a plaintext keyword (e.g., `K:CARDNAME can't attack or block alone.`)? It should be extracted as a plaintext keyword, distinct from parameterized keywords.
- What happens when a card has `ManaCost:no cost` (e.g., back faces of transform cards)? The mana cost field should reflect this value.
- What happens when a card script contains `AI:`, `DeckHints:`, `DeckHas:`, or `DeckNeeds:` lines? These are game-engine metadata and should be excluded from the output since they are not visible on the physical card.
- What happens when an ability has no `SpellDescription$` or `TriggerDescription$`? The system should mark the description as unavailable (Oracle text is not used as a fallback since it is excluded from the output).
- What happens with static abilities (`S:` lines)? The `Description$` parameter contains the card-visible text and should be extracted.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST extract the following top-level properties from each card script: `Name`, `ManaCost`, `Types`, `PT`, `Loyalty`, `Colors`, `Text`, `AlternateMode`. The `Oracle` property is NOT extracted as output — individual ability descriptions are used instead to avoid redundancy and reduce token count. Mana costs MUST be converted from Forge internal syntax (e.g., `2 W W`) to human-readable paper-card format (e.g., `{2}{W}{W}`).
- **FR-002**: System MUST extract all keyword abilities (`K:` lines), preserving keyword names and their parameters (e.g., `Kicker:2 R` yields keyword "Kicker" with parameter "2 R").
- **FR-003**: System MUST extract ETB counter keywords (e.g., `K:etbCounter:P1P1:X`) as a distinct keyword type with counter type and amount.
- **FR-004**: System MUST extract plaintext keywords (e.g., `K:CARDNAME can't attack or block alone.`) as verbatim text, substituting `CARDNAME` with the card's actual name.
- **FR-005**: System MUST extract activated abilities (`A:` lines with `AB$` or `SP$` prefix) and include only the cost (from `Cost$`) and the textual description (from `SpellDescription$`). Ability costs MUST be converted from Forge internal syntax (e.g., `T Sac<1/CARDNAME>`) to human-readable paper-card format (e.g., `{T}, Sacrifice CARDNAME:`).
- **FR-006**: System MUST extract triggered abilities (`T:` lines) and include only the textual description (from `TriggerDescription$`), which covers both the trigger condition and the effect as it would appear on the card.
- **FR-007**: System MUST extract replacement effects (`R:` lines) and include only the textual description (from `Description$`).
- **FR-008**: System MUST extract static abilities (`S:` lines) and include only the textual description (from `Description$`).
- **FR-009**: Each extracted ability MUST carry a type tag using official MTG rules terminology: "activated" (from `A:` lines), "triggered" (from `T:` lines), "static" (from `S:` lines), or "replacement" (from `R:` lines).
- **FR-010**: System MUST NOT include sub-abilities (`DB$` prefix lines defined in SVars) as separate ability entries, since they are part of their parent ability's description.
- **FR-011**: System MUST handle multi-face cards by preserving the Forge script convention: the front face (including its `AlternateMode` property) appears first, followed by a blank line, an `ALTERNATE` delimiter line, another blank line, and then the back/second face. This same structure applies to all alternate types (transform, split, adventure, room, modal, flip).
- **FR-012**: System MUST exclude engine-internal metadata from output, including `AI:`, `DeckHints:`, `DeckHas:`, `DeckNeeds:`, `SVar:`, and AI-specific parameters like `AILogic$`, `AITgts$`.
- **FR-013**: System MUST support batch processing of a directory tree of card scripts, producing one output entry per card face.
- **FR-014**: System MUST gracefully handle malformed card scripts by logging a warning with the file path and continuing to process remaining files.
- **FR-015**: System MUST substitute `CARDNAME` and `NICKNAME` placeholders with the actual card name in all extracted text descriptions.

### Key Entities

- **Card**: A single card face. Has top-level properties (name, mana cost, types, power/toughness, loyalty, colors, text), a list of keywords, and a list of abilities. Oracle text is not included — individual ability descriptions serve as the structured text data instead. Multi-face cards are represented using the Forge script convention: front face first (with AlternateMode), then ALTERNATE delimiter, then subsequent faces.
- **Keyword**: A keyword ability on a card. Can be a simple keyword (e.g., "Flying"), a parameterized keyword (e.g., "Kicker" with cost "2 R"), an ETB counter keyword (counter type and amount), or a plaintext keyword (verbatim rules text).
- **Ability**: A non-keyword ability on a card. Contains a type tag, an optional cost, and a textual description. The type tag uses official MTG rules terminology: "activated", "triggered", "static", or "replacement". These are the same terms used in rules text and on cards (e.g., "counter target activated ability").

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: System successfully extracts structured data from 99% or more of all card scripts in the Forge card folder without errors.
- **SC-002**: For any parsed card, the extracted top-level properties, keywords, and ability descriptions match what appears on the physical card (verified by spot-checking a sample of 50 diverse cards).
- **SC-003**: No engine-internal scripting details (SVars, API types, targeting parameters, AI hints) appear in the extracted output.
- **SC-004**: Multi-face cards produce correctly separated entries for each face, with each face's properties and abilities independently correct.
- **SC-005**: The full card library (20,000+ card scripts) can be processed in a single batch run to completion.

## Assumptions

- Card script files are located in the Forge repository at `../forge/forge-gui/res/cardsfolder/`, organized in alphabetical subdirectories.
- The card scripting language follows the patterns documented in `resources/CARD_SCRIPTING_REFERENCE.md`, with additional details available in the `../forge/forge-game` source code.
- `SpellDescription$` for abilities, `TriggerDescription$` for triggers, and `Description$` for static abilities and replacement effects consistently contain the human-readable card text.
- The `Oracle` property exists in card scripts but is intentionally excluded from the output. Individual ability descriptions (`SpellDescription$`, `TriggerDescription$`, `Description$`) are the sole source of text data, avoiding redundancy and reducing model input size.
- Performance does not need to be real-time; batch processing the entire library within minutes is acceptable.

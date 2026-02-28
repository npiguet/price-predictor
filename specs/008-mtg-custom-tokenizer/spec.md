# Feature Specification: MTG Custom Tokenizer

**Feature Branch**: `008-mtg-custom-tokenizer`
**Created**: 2026-02-28
**Status**: Draft
**Input**: User description: "The AI model must use a custom tokenizer. MTG uses a relatively small and precise language. Domain-specific terms (card types, subtypes, keyword abilities, game zones, colors) should each be single tokens. The goal is a vocabulary smaller than a general-purpose LLM's, keeping memory requirements low."
**Depends on**: `006-card-script-parsing` (card data extraction), `007-pipeline-cli` (pre-training stage produces the token list)

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Build Domain Vocabulary from Card Corpus (Priority: P1)

A model developer wants to build the tokenizer's vocabulary automatically from the card corpus so that all MTG-specific terms used across real cards are captured as single tokens. The system scans all card data in the prepared datasets, identifies domain-specific terms (card types, subtypes, supertypes, keyword abilities, game zones, colors, and other recurring MTG terminology), and includes each as a single token in the vocabulary.

**Why this priority**: The vocabulary is the foundation of the tokenizer. Without a correct and complete domain vocabulary, card text cannot be tokenized effectively, and the downstream model will either waste capacity on fragmented tokens or miss important game concepts.

**Independent Test**: Can be fully tested by running vocabulary extraction against the prepared card datasets and verifying that known MTG terms (e.g., "Enchantment", "Flying", "Battlefield", "Legendary") each appear as single entries in the resulting vocabulary.

**Acceptance Scenarios**:

1. **Given** a prepared card dataset containing diverse cards, **When** the vocabulary is built, **Then** all card types that appear in the dataset (e.g., Creature, Enchantment, Artifact, Instant, Sorcery, Land, Planeswalker) are each a single token in the vocabulary.
2. **Given** a prepared card dataset, **When** the vocabulary is built, **Then** all supertypes and subtypes that appear (e.g., Legendary, Basic, Snow, Goblin, Elf, Angel, Sliver) are each a single token.
3. **Given** a prepared card dataset, **When** the vocabulary is built, **Then** all keyword abilities that appear (e.g., Flying, Haste, Trample, Scry, Deathtouch, Lifelink) are each a single token.
4. **Given** a prepared card dataset, **When** the vocabulary is built, **Then** game zone names (e.g., hand, library, battlefield, graveyard, exile, stack) are each a single token.
5. **Given** a prepared card dataset, **When** the vocabulary is built, **Then** color names (White, Blue, Black, Red, Green, Colorless) are each a single token.

---

### User Story 2 - Tokenize Card Text (Priority: P2)

A model developer wants to convert a card's text fields into a sequence of tokens using the custom tokenizer. When a card's oracle text, ability descriptions, or other text fields are tokenized, MTG domain terms are kept as single tokens while remaining words are broken down into a compact representation. The resulting token sequence is what gets fed into the model during training and prediction.

**Why this priority**: Tokenization is the bridge between raw card text and the model's input. Without it, the vocabulary (P1) has no practical use and the model cannot process cards.

**Independent Test**: Can be tested by tokenizing sample card texts and verifying that domain terms are single tokens in the output sequence, and that the full text can be reconstructed from the token sequence.

**Acceptance Scenarios**:

1. **Given** a card with oracle text "Flying, vigilance", **When** the text is tokenized, **Then** "Flying" and "vigilance" are each a single token in the output.
2. **Given** a card with oracle text "When this creature enters the battlefield, draw a card", **When** the text is tokenized, **Then** "creature", "battlefield", and "draw" appear as single tokens (recognized domain terms), while common English words like "when", "this", "enters", "the", "a", "card" are also tokenized (either as single tokens if in vocabulary, or via the fallback mechanism).
3. **Given** a card with type line "Legendary Creature — Human Wizard", **When** the type line is tokenized, **Then** "Legendary", "Creature", "Human", and "Wizard" are each single tokens.
4. **Given** a token sequence produced from a card's text, **When** the tokens are decoded back to text, **Then** the original text is recoverable (round-trip integrity).

---

### User Story 3 - Verify Compact Vocabulary Size (Priority: P3)

A model developer wants to confirm that the custom tokenizer's vocabulary is significantly smaller than a general-purpose language model tokenizer, ensuring that memory requirements for embeddings and model parameters stay low. They compare the vocabulary size and inspect the token distribution to confirm the domain focus is effective.

**Why this priority**: Vocabulary compactness is the stated goal motivating this entire feature. While P1 and P2 deliver the functional tokenizer, P3 validates that the design goal is actually achieved.

**Independent Test**: Can be tested by comparing the custom vocabulary size against typical general-purpose tokenizer sizes and verifying the custom one is substantially smaller.

**Acceptance Scenarios**:

1. **Given** the vocabulary has been built from the card corpus, **When** the total token count is measured, **Then** the vocabulary is smaller than 10,000 tokens (compared to 30,000–100,000+ for general-purpose tokenizers).
2. **Given** the vocabulary, **When** all card texts in the dataset are tokenized, **Then** at least 95% of all token occurrences in the corpus are covered by the vocabulary (less than 5% fall through to the unknown/fallback mechanism).

---

### Edge Cases

- What happens when a card contains a word never seen in the training corpus (e.g., a new keyword from a future set)? The tokenizer uses a fallback mechanism (e.g., character-level or subword decomposition) so the word is still representable, just not as a single token.
- What happens when a domain term is also a common English word (e.g., "Flash", "Reach", "Menace")? The term is included in the vocabulary as a single token regardless — in the MTG context, these words carry domain-specific meaning.
- What happens with multi-word keyword abilities (e.g., "First Strike", "Double Strike", "Split second")? Each multi-word keyword is treated as a single token because it represents one indivisible game concept.
- What happens with mana cost symbols (e.g., "{W}", "{2}", "{U/B}")? Mana symbols are distinct domain tokens and each unique symbol is a single token in the vocabulary.
- What happens with card names that appear in oracle text (e.g., "CARDNAME" or the card's actual name)? Card names are replaced with a generic placeholder token (e.g., a "CARDNAME" token) rather than tokenizing each unique card name separately, since individual card names are not generalizable vocabulary.
- What happens with numbers in card text (e.g., "deals 3 damage", "draw 2 cards")? Numbers are tokenized as individual number tokens (each distinct number value is one token), keeping the vocabulary small.
- What happens with punctuation and formatting in oracle text (commas, periods, colons, newlines)? Common punctuation marks are individual tokens in the vocabulary.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST build a domain-specific vocabulary by scanning prepared card datasets and extracting all distinct terms from the following categories: card types, subtypes, supertypes, keyword abilities (including multi-word keywords), game zone names, and color names.
- **FR-002**: Each identified MTG domain term MUST be represented as a single, indivisible token in the vocabulary, regardless of how many words it contains (e.g., "First Strike" is one token, not two).
- **FR-003**: The system MUST include mana cost symbols as single tokens in the vocabulary (e.g., each unique mana symbol is one token).
- **FR-004**: The system MUST include common structural tokens in the vocabulary: punctuation marks, number values that appear in card text, and a placeholder token for card self-references.
- **FR-005**: The system MUST provide a fallback mechanism for words not in the vocabulary, so that any text can be tokenized even if it contains unknown terms. No input may cause the tokenizer to fail.
- **FR-006**: The system MUST support tokenizing all card text fields: oracle text, ability descriptions, type lines, and any other text extracted from card data.
- **FR-007**: The system MUST support decoding a token sequence back to the original text (round-trip conversion).
- **FR-008**: The vocabulary MUST be saveable to disk and loadable from disk, so it can be produced during pre-training and reused during training and prediction.
- **FR-009**: The same vocabulary and tokenization logic MUST be used consistently across all pipeline stages (pre-training, training, and prediction) to ensure tokens have the same meaning everywhere.
- **FR-010**: The system MUST report vocabulary statistics after building: total token count, number of domain-specific tokens, and corpus coverage percentage (what fraction of token occurrences in the dataset are covered by vocabulary tokens vs. fallback).

### Key Entities

- **Vocabulary**: The complete set of tokens recognized by the tokenizer. Contains domain-specific tokens (MTG terms), structural tokens (punctuation, numbers, mana symbols), and optionally a small set of common English words. Has a fixed size after building. Saved to and loaded from disk.
- **Domain Token**: A token representing an MTG-specific concept. Belongs to a category (card type, subtype/supertype, keyword ability, game zone, color, mana symbol). Always kept as a single indivisible unit during tokenization.
- **Token Sequence**: An ordered list of token identifiers produced by tokenizing a piece of card text. Each identifier maps to exactly one entry in the vocabulary. This is the model's input format.
- **Fallback Token**: A special token (or decomposition mechanism) used when a word is not found in the vocabulary. Ensures all text is representable even with an incomplete vocabulary.

## Assumptions

- The domain vocabulary is derived from the prepared card datasets (feature 007, dataset preparation stage). The tokenizer does not require manually curated lists — it discovers terms from the actual card corpus.
- Multi-word keywords (e.g., "First Strike", "Split second") can be identified from the card data because they appear as complete keyword entries in the structured card data (feature 006 extracts keywords as whole units).
- Game zone names (hand, library, battlefield, graveyard, exile, stack, command zone) are a small, well-known set that can be seeded as a fixed list and supplemented by corpus scanning.
- Color names are a fixed set of six: White, Blue, Black, Red, Green, Colorless.
- Card names appearing in oracle text are normalized to a placeholder token rather than added to the vocabulary, since there are 20,000+ unique card names and adding them would defeat the compact vocabulary goal.
- The vocabulary is built once during the pre-training stage and remains fixed for a given model version. It does not change during training or prediction.
- A vocabulary size under 10,000 tokens is the target, but the exact size depends on the diversity of the card corpus. The key constraint is that it be substantially smaller than general-purpose tokenizers (which typically have 30,000–100,000+ tokens).

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The custom vocabulary contains fewer than 10,000 tokens after building from the full card corpus.
- **SC-002**: At least 95% of all token occurrences across the card corpus are covered by vocabulary tokens (less than 5% require fallback).
- **SC-003**: All MTG keyword abilities, card types, supertypes, and subtypes present in the card corpus are represented as single tokens in the vocabulary (100% domain term coverage).
- **SC-004**: Tokenizing and then decoding any card text from the dataset produces the original text (100% round-trip fidelity).
- **SC-005**: The vocabulary can be built from the full card corpus (20,000+ cards) within 1 minute.

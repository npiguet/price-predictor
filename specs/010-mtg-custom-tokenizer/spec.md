# Feature Specification: MTG Custom Tokenizer

**Feature Branch**: `008-mtg-custom-tokenizer`
**Created**: 2026-02-28
**Status**: Draft
**Input**: User description: "The AI model must use a custom tokenizer. MTG uses a relatively small and precise language. Domain-specific terms (card types, subtypes, keyword abilities, game zones, colors) should each be single tokens. The goal is a vocabulary smaller than a general-purpose LLM's, keeping memory requirements low."
**Depends on**: `006-card-script-parsing` (card data extraction), `007-pipeline-cli` (pre-training stage produces the token list)

## Clarifications

### Session 2026-03-01

- Q: What fallback mechanism should the tokenizer use for words not in the vocabulary? → A: ~~Byte-Pair Encoding (BPE) subword fallback~~ — revised in session 2026-03-16, see below.
- Q: Should the tokenizer be case-sensitive or case-insensitive? → A: Normalize all text to lowercase before tokenization (lossy — original casing not recoverable)
- Q: What file format should be used for persisting the vocabulary to disk? → A: ~~Two plain text files: `vocab.txt` + `merges.txt`~~ — revised in session 2026-03-16, see below.
- Q: Should high-frequency non-domain English words from oracle text be included as whole tokens? → A: Yes — many of these (target, damage, draw, destroy, etc.) are game action keywords defined in the MTG comprehensive rules and should be treated as domain terms. Include words above a frequency threshold from the corpus as whole tokens.

### Session 2026-03-16

- Q: Is BPE actually needed given MTG's constrained language? → A: No. Corpus analysis across 32,117 converted card files shows 5,319 tokens appear 5+ times, covering 98.1% of all token occurrences. The remaining 1.9% is almost entirely card-specific proper nouns (character names, named cards) that do not generalize. A word-level tokenizer with `[UNK]` for out-of-vocabulary words is sufficient. BPE adds implementation complexity for no meaningful benefit.
- Q: What is the vocabulary source — corpus scanning or structured parser output? → A: Both, combined. Domain terms (card types, subtypes, supertypes, keyword abilities) come from the structured fields already extracted by the Forge parser (feature 006), which is more reliable than frequency-based discovery for rare terms. Corpus frequency scanning supplements this with high-frequency game-action words. This avoids missing rare but valid subtypes that appear only once or twice in the corpus.
- Q: What file format for persisting the vocabulary? → A: A single plain text file `vocab.txt` (one token per line, line number = token ID). No `merges.txt` needed since there are no BPE merge rules.
- Q: Does this feature include replacing `BertTokenizer` in the transformer training pipeline, or does it deliver the tokenizer as a standalone artifact? → A: Full end-to-end replacement. This feature replaces `BertTokenizer` in `transformer_dataset.py`, updates `TransformerConfig.vocab_size` to the custom vocabulary size, and includes retraining the transformer model from scratch on the new token IDs.
- Q: What special tokens does the vocabulary need beyond `[UNK]`? → A: `[UNK]` and `[PAD]` only. Mean pooling (feature 009) masks padding positions during pooling, so no `[CLS]`, `[SEP]`, or `[MASK]` tokens are needed.
- Q: Is there a cap on number tokens (e.g., numbers above N map to `[UNK]`)? → A: No cap. All distinct number values found in the corpus are included as tokens. Numbers above 20 are rare enough that the vocabulary cost is negligible.
- Q: How is vocabulary building triggered — new CLI subcommand or automatic inside `train transformer`? → A: New dedicated `vocabulary` subcommand (`python -m price_predictor vocabulary`). Keeps vocabulary building separate from training so developers can inspect and validate `vocab.txt` before committing to a full training run.
- Q: Where does `train transformer` look for `vocab.txt` by default? → A: `models/transformer/vocab.txt`, overridable via `--vocab-path`. Co-locates the vocabulary with the model artifacts it was built for.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Build Domain Vocabulary from Card Corpus (Priority: P1)

A model developer wants to build the tokenizer's vocabulary automatically from the card corpus so that all MTG-specific terms used across real cards are captured as single tokens. The system combines structured domain terms from the Forge parser output with high-frequency words from the corpus, and includes each as a single token in the vocabulary.

**Why this priority**: The vocabulary is the foundation of the tokenizer. Without a correct and complete domain vocabulary, card text cannot be tokenized effectively, and the downstream model will either waste capacity on fragmented tokens or miss important game concepts.

**Independent Test**: Can be fully tested by running `python -m price_predictor vocabulary` against the prepared card datasets and verifying that known MTG terms (e.g., "Enchantment", "Flying", "Battlefield", "Legendary") each appear as single entries in the resulting `vocab.txt`.

**Acceptance Scenarios**:

1. **Given** a prepared card dataset containing diverse cards, **When** the vocabulary is built, **Then** all card types that appear in the dataset (e.g., Creature, Enchantment, Artifact, Instant, Sorcery, Land, Planeswalker) are each a single token in the vocabulary.
2. **Given** a prepared card dataset, **When** the vocabulary is built, **Then** all supertypes and subtypes that appear (e.g., Legendary, Basic, Snow, Goblin, Elf, Angel, Sliver) are each a single token.
3. **Given** a prepared card dataset, **When** the vocabulary is built, **Then** all keyword abilities that appear (e.g., Flying, Haste, Trample, Scry, Deathtouch, Lifelink) are each a single token.
4. **Given** a prepared card dataset, **When** the vocabulary is built, **Then** game zone names (e.g., hand, library, battlefield, graveyard, exile, stack) are each a single token.
5. **Given** a prepared card dataset, **When** the vocabulary is built, **Then** color names (White, Blue, Black, Red, Green, Colorless) are each a single token.

---

### User Story 2 - Tokenize Card Text (Priority: P2)

A model developer wants to convert a card's text fields into a sequence of tokens using the custom tokenizer. When a card's oracle text, ability descriptions, or other text fields are tokenized, MTG domain terms are kept as single tokens while words not in the vocabulary are mapped to a special `[UNK]` token. The resulting token sequence is what gets fed into the model during training and prediction.

**Why this priority**: Tokenization is the bridge between raw card text and the model's input. Without it, the vocabulary (P1) has no practical use and the model cannot process cards.

**Independent Test**: Can be tested by tokenizing sample card texts and verifying that domain terms are single tokens in the output sequence.

**Acceptance Scenarios**:

1. **Given** a card with oracle text "Flying, vigilance", **When** the text is tokenized, **Then** "Flying" and "vigilance" are each a single token in the output.
2. **Given** a card with oracle text "When this creature enters the battlefield, draw a card", **When** the text is tokenized, **Then** "creature", "battlefield", and "draw" appear as single tokens (recognized domain terms), while common English words like "when", "this", "enters", "the", "a", "card" are also single tokens if they appear above the frequency threshold, or `[UNK]` otherwise.
3. **Given** a card with type line "Legendary Creature — Human Wizard", **When** the type line is tokenized, **Then** "Legendary", "Creature", "Human", and "Wizard" are each single tokens.
4. **Given** a card whose text contains only in-vocabulary words, **When** the tokens are decoded back to text, **Then** the lowercase-normalized version of the original text is recoverable (round-trip integrity modulo casing). Words mapped to `[UNK]` are not recoverable on decode.

---

### User Story 3 - Integrate Custom Tokenizer into Transformer Pipeline (Priority: P3)

A model developer wants to retrain the transformer model using the custom tokenizer instead of `BertTokenizer`, so that the embedding table shrinks from 30,522 entries to the custom vocabulary size, reducing memory requirements and ensuring the model processes domain-appropriate tokens.

**Why this priority**: A tokenizer that isn't wired into the pipeline has no effect on the model. This story closes the loop between vocabulary building (P1/P2) and actual model improvement.

**Independent Test**: Can be tested by running `train transformer` end-to-end and verifying the resulting model's embedding table has `vocab_size` equal to the custom vocabulary size (not 30,522).

**Acceptance Scenarios**:

1. **Given** the custom vocabulary has been built, **When** `train transformer` is run, **Then** `transformer_dataset.py` encodes card texts using the custom tokenizer, not `BertTokenizer`.
2. **Given** the custom vocabulary has been built, **When** a transformer model is initialized, **Then** `TransformerConfig.vocab_size` equals the custom vocabulary size.
3. **Given** the custom tokenizer is wired in, **When** the transformer is trained to completion, **Then** the model artifact saves correctly and `evaluate transformer` runs without error.

---

### User Story 4 - Verify Compact Vocabulary Size (Priority: P4)

A model developer wants to confirm that the custom tokenizer's vocabulary is significantly smaller than a general-purpose language model tokenizer, ensuring that memory requirements for embeddings and model parameters stay low. They compare the vocabulary size and inspect the token distribution to confirm the domain focus is effective.

**Why this priority**: Vocabulary compactness is the stated goal motivating this entire feature. While P1–P3 deliver and wire in the tokenizer, P4 validates that the design goal is actually achieved.

**Independent Test**: Can be tested by comparing the custom vocabulary size against typical general-purpose tokenizer sizes and verifying the custom one is substantially smaller.

**Acceptance Scenarios**:

1. **Given** the vocabulary has been built from the card corpus, **When** the total token count is measured, **Then** the vocabulary is smaller than 10,000 tokens (compared to 30,000–100,000+ for general-purpose tokenizers).
2. **Given** the vocabulary, **When** all card texts in the dataset are tokenized, **Then** at least 95% of all token occurrences in the corpus are covered by vocabulary tokens (less than 5% map to `[UNK]`).

---

### Edge Cases

- What happens when a card contains a word never seen in the training corpus (e.g., a new keyword from a future set)? The word maps to the `[UNK]` token. The model will treat it as unknown and rely on context from surrounding tokens to infer meaning. Future keywords from MTG sets are almost always single common English words that may already be in the vocabulary as high-frequency terms.
- What happens when a domain term is also a common English word (e.g., "Flash", "Reach", "Menace")? The term is included in the vocabulary as a single token regardless — in the MTG context, these words carry domain-specific meaning.
- What happens with multi-word keyword abilities (e.g., "First Strike", "Double Strike", "Split second")? Before word splitting, the tokenizer runs a string replacement pass that substitutes each known multi-word keyword with its underscore form (e.g., "first strike" → `first_strike`). The list of multi-word keywords comes from `forge.game.keyword.Keyword` enum entries whose display name contains a space. The word-splitting regex matches `[a-z_]+` (underscores included), so `first_strike` is kept as one token.
- What happens with mana cost symbols (e.g., "{W}", "{2}", "{U/B}")? Mana symbols are distinct domain tokens and each unique symbol is a single token in the vocabulary.
- What happens with card names that appear in oracle text (e.g., "CARDNAME" or the card's actual name)? Card names are replaced with a generic placeholder token (`CARDNAME`) rather than tokenizing each unique card name separately, since individual card names are not generalizable vocabulary.
- What happens with generic mana cost symbols (e.g., `{2}`, `{3}`, `{12}`)? Each distinct value is a separate token in the vocabulary (`{0}` through `{16}` cover all practical cases, ~17 tokens total). An alternative of decomposing `{3}` → `{1}{1}{1}` (analogous to how colored pips are repeated) was considered and rejected: it causes sequence length explosion for high-CMC cards (Emrakul's `{15}` would consume 15 tokens for generic mana alone), overloads the `{1}` token with dual meaning, and provides no meaningful benefit since the model can learn ordinal relationships between `{0}`–`{16}` embeddings from training data. This decision should be revisited if the model shows poor sensitivity to CMC differences.
- What happens with numbers in card text (e.g., "deals 3 damage", "draw 2 cards")? All distinct number values found in the corpus are included as tokens with no cap. Numbers above 20 are rare in practice (very few distinct values appear), so the vocabulary cost is negligible and no `[UNK]` fallback for numbers is needed.
- What happens with punctuation and formatting in oracle text (commas, periods, colons, newlines)? Common punctuation marks are individual tokens in the vocabulary.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST build a domain-specific vocabulary from two sources combined: (a) structured fields already extracted by the Forge parser (feature 006) — card types, subtypes, supertypes, keyword abilities including multi-word keywords; (b) corpus frequency scanning — all words appearing at or above a frequency threshold of 5 occurrences across the card corpus. Game zone names, color names, mana symbols, and structural tokens are seeded as fixed lists. The combination ensures rare but valid domain terms (e.g., obscure subtypes) are not missed.
- **FR-002**: Each identified MTG domain term MUST be represented as a single, indivisible token in the vocabulary, regardless of how many words it contains. Multi-word keywords (e.g., "First Strike", "Double Strike", "Split second") are normalized by replacing spaces with underscores, producing tokens like `first_strike`, `double_strike`, `split_second`. This normalization is applied both when building the vocabulary and when tokenizing input text (via a string replacement pass before word splitting). The list of multi-word keywords is derived from `forge.game.keyword.Keyword` — all enum entries whose `toString()` display name contains a space.
- **FR-003**: The system MUST include mana cost symbols as single tokens in the vocabulary (e.g., each unique mana symbol is one token).
- **FR-004**: The system MUST include common structural tokens in the vocabulary: punctuation marks, number values that appear in card text, and a placeholder token for card self-references (`cardname`, the lowercased form of `CARDNAME` as it appears in card text). The vocabulary MUST also include exactly two special tokens: `[PAD]` (token ID 0, used for sequence padding) and `[UNK]` (used for out-of-vocabulary words). No other special tokens (`[CLS]`, `[SEP]`, `[MASK]`) are required — mean pooling makes `[CLS]` unnecessary.
- **FR-005**: The system MUST use a word-level tokenizer. Words not found in the vocabulary are mapped to a single special `[UNK]` token. The tokenizer MUST NOT fail on any input — unknown words produce `[UNK]`, not an error.
- **FR-006**: The system MUST normalize input text before tokenization using selective lowercasing: text inside `{}`-enclosed tokens (mana symbols) is preserved as-is; all other text is lowercased. Mana symbols are stored in the vocabulary in their original uppercase form (e.g., `{W}`, `{R}`, `{T}`). All other vocabulary entries are stored in lowercase. Original casing outside mana symbols is not preserved or recoverable.
- **FR-006a**: The system MUST support tokenizing all card text fields: oracle text, ability descriptions, type lines, and any other text extracted from card data.
- **FR-007**: The system MUST support decoding a token sequence back to text. Decoding is lossless for in-vocabulary tokens. `[UNK]` tokens decode to the literal string `[UNK]` — the original word is not recoverable.
- **FR-008**: The vocabulary MUST be persisted as a single plain text file `vocab.txt` (one token per line, line number = token ID). This file is produced during vocabulary building and reused unchanged during training and prediction.
- **FR-009**: The same vocabulary and tokenization logic MUST be used consistently across all pipeline stages (vocabulary building, training, and prediction) to ensure tokens have the same meaning everywhere.
- **FR-010**: The system MUST report vocabulary statistics after building: total token count, number of domain-specific tokens (from parser structured fields), number of frequency-threshold tokens, and corpus coverage percentage (fraction of token occurrences mapped to `[UNK]`).
- **FR-014**: Vocabulary building MUST be exposed as a dedicated `vocabulary` subcommand (`python -m price_predictor vocabulary`). It MUST accept `--output-dir` (default: `models/transformer/`) for the output `vocab.txt` path, and `--output-dir` determines where `vocab.txt` is written. It MUST NOT run automatically inside `train transformer` — vocabulary building is an explicit prerequisite step.
- **FR-015**: `train transformer`, `evaluate transformer`, `predict transformer`, and `serve` MUST each accept a `--vocab-path` argument (default: `models/transformer/vocab.txt`) pointing to the `vocab.txt` produced by the `vocabulary` command. If the file does not exist at the resolved path, the command MUST fail with a clear error message instructing the user to run `python -m price_predictor vocabulary` first.
- **FR-011**: `transformer_dataset.py` MUST be updated to replace `BertTokenizer.from_pretrained("bert-base-uncased")` with the custom tokenizer loaded from `vocab.txt`. The encoding interface (producing `input_ids` and `attention_mask` tensors) MUST remain unchanged so no other training code requires modification.
- **FR-012**: `TransformerConfig.vocab_size` MUST be set to the custom vocabulary size when initializing a new transformer model. The hardcoded BERT vocabulary size (30,522) MUST NOT be used for models trained with the custom tokenizer.
- **FR-013**: The transformer model MUST be retrained from scratch after the tokenizer replacement. Existing model weights trained with `BertTokenizer` are incompatible with the new embedding table and MUST NOT be reused.

### Key Entities

- **Vocabulary**: The complete set of tokens recognized by the tokenizer. Contains domain-specific tokens (sourced from Forge parser structured fields), high-frequency corpus words (above frequency threshold of 5), structural tokens (punctuation, numbers, mana symbols, `cardname` placeholder), and the special `[UNK]` token. Has a fixed size after building. Persisted as a single `vocab.txt` file.
- **Domain Token**: A token representing an MTG-specific concept. Sourced from the structured fields extracted by the Forge parser (feature 006): card types, subtypes, supertypes, keyword abilities, game zones, colors, mana symbols. Always kept as a single indivisible unit during tokenization.
- **Token Sequence**: An ordered list of token identifiers produced by tokenizing a piece of card text. Each identifier maps to exactly one entry in the vocabulary. This is the model's input format.
- **[PAD] Token**: Special token with ID 0 used to pad token sequences to `max_seq_len`. Padding positions receive attention mask value 0, which causes mean pooling to exclude them from the sequence representation. Always assigned token ID 0 so the embedding table initializes padding embeddings at the zero index.
- **[UNK] Token**: Special token produced when a word is not found in the vocabulary. Represents proper nouns, card-specific names, and any future terms not seen at vocabulary-building time. Does not carry word-specific information — the model treats all unknown words identically.

## Assumptions

- Domain vocabulary terms (card types, subtypes, supertypes, keyword abilities) are sourced from the structured fields already extracted by the Forge parser (feature 006). Corpus frequency scanning supplements this but is not the primary source for domain terms.
- Multi-word keywords are derived from `forge.game.keyword.Keyword` (Java enum in `forge-game`). All entries whose `toString()` display name contains a space are multi-word keywords. This gives a programmatically complete list without manual curation. At tokenization time, these are normalized to underscore form (e.g., `first_strike`) via a string replacement pass before word splitting.
- Game zone names (hand, library, battlefield, graveyard, exile, stack, command zone) are a small, well-known set seeded as a fixed list.
- Color names are a fixed set of six: White, Blue, Black, Red, Green, Colorless.
- Card names appearing in oracle text appear as `CARDNAME` in the converted format. After lowercasing this becomes the vocabulary token `cardname`. It is not added as `CARDNAME` to the vocabulary, since there are 20,000+ unique card names and adding them would defeat the compact vocabulary goal.
- The vocabulary is built once and remains fixed for a given model version. It does not change during training or prediction.
- Replacing the tokenizer is not backward-compatible with existing transformer model weights. Any model trained with `BertTokenizer` must be discarded and retrained after this feature is applied.
- `TransformerConfig.vocab_size` will change from 30,522 (BERT) to the custom vocabulary size (~5,000–8,000). This reduces the embedding table from ~3.9M parameters to ~640K–1M parameters.
- A frequency threshold of 5 occurrences is used to distinguish reusable game vocabulary from one-off proper nouns. Corpus analysis shows this threshold yields ~5,319 tokens covering 98.1% of all token occurrences. The threshold may be adjusted based on desired vocabulary size and coverage trade-off.
- The 1.9% of token occurrences that fall to `[UNK]` are almost entirely card-specific proper nouns (character names, place names from card lore) that do not generalize across cards. This is an acceptable loss.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The custom vocabulary contains fewer than 10,000 tokens after building from the full card corpus.
- **SC-002**: At least 95% of all token occurrences across the card corpus are covered by vocabulary tokens (less than 5% map to `[UNK]`). Corpus analysis projects ~98% coverage at a frequency threshold of 5.
- **SC-003**: All MTG keyword abilities, card types, supertypes, and subtypes present in the card corpus are represented as single tokens in the vocabulary (100% domain term coverage).
- **SC-004**: Tokenizing and then decoding any card text from the dataset produces the lowercase-normalized version of the original text for all in-vocabulary words. `[UNK]` tokens are expected for out-of-vocabulary words and do not constitute a round-trip failure.
- **SC-005**: The vocabulary can be built from the full card corpus (20,000+ cards) within 1 minute.
- **SC-006**: The transformer model trains to completion using the custom tokenizer with no errors. The resulting model's `TransformerConfig.vocab_size` equals the custom vocabulary size (not 30,522).

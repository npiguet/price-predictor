# Data Model: MTG Custom Tokenizer

## Entities

### MtgTokenizer (domain/tokenizer.py)

Pure domain class. Constructed from an in-memory vocabulary — no file I/O.

| Field | Type | Description |
|-------|------|-------------|
| `_vocab` | `dict[str, int]` | Token string → token ID mapping |
| `_reverse_vocab` | `dict[int, str]` | Token ID → token string mapping |
| `_multi_word_keywords` | `tuple[str, ...]` | Underscore-normalized multi-word keywords, sorted longest-first |

**Class constants:**
| Constant | Value | Description |
|----------|-------|-------------|
| `PAD` | `"[PAD]"` | Padding token string |
| `UNK` | `"[UNK]"` | Unknown token string |
| `PAD_ID` | `0` | Padding token ID (always 0) |
| `UNK_ID` | `1` | Unknown token ID (always 1) |

**Properties:**
- `vocab_size: int` — total number of tokens in vocabulary

**Methods:**
- `encode(text: str, max_length: int) -> tuple[list[int], list[int]]` — returns `(input_ids, attention_mask)`. Normalizes text → replaces multi-word keywords → splits → maps to IDs → truncates to `max_length` → pads with `PAD_ID` to `max_length`. `attention_mask` is 1 for real tokens, 0 for padding.
- `decode(token_ids: list[int]) -> str` — maps IDs back to tokens, stops at `PAD_ID`, joins with spaces. `UNK_ID` decodes to `"[UNK]"`.
- `_tokenize(text: str) -> list[str]` — internal: selective normalize (lowercase non-`{}`-enclosed text only, keep mana symbols uppercase) → multi-word replacement → regex split on `[a-z_]+|\{[^}]+\}|\d+|[^\s\w]`

**Invariants:**
- `_vocab["[PAD]"] == 0` always
- `_vocab["[UNK]"] == 1` always
- `len(_vocab) == len(_reverse_vocab)` (bijective)
- All tokens in `_multi_word_keywords` have a corresponding entry in `_vocab`

---

### VocabBuildResult (application/build_vocabulary.py)

Frozen dataclass returned by `VocabularyBuilder.build()`.

| Field | Type | Description |
|-------|------|-------------|
| `vocab` | `dict[str, int]` | Complete token → ID mapping |
| `vocab_size` | `int` | Total token count |
| `domain_token_count` | `int` | Tokens from fixed domain sources (zones, colors, mana symbols, multi-word keywords) |
| `freq_threshold_token_count` | `int` | Tokens added via corpus frequency threshold |
| `coverage_pct` | `float` | Fraction of corpus token occurrences covered (0.0–1.0) |
| `unk_pct` | `float` | Fraction of corpus token occurrences mapping to `[UNK]` (= 1 - coverage_pct) |

---

### vocab.txt (persisted artifact)

Plain text file. One token per line. **Line number = token ID** (0-indexed).

```
[PAD]          ← line 0, ID 0
[UNK]          ← line 1, ID 1
cardname       ← line 2, ID 2
creature       ← line 3, ID 3
...
first_strike   ← underscore = was multi-word keyword
{w}            ← mana symbol
1              ← number
,              ← punctuation
```

**Encoding**: UTF-8. No header line. No blank lines. No trailing whitespace.

**Multi-word keyword identification at load time**: Any token matching `re.search(r'_', token)` is a multi-word keyword. Its display form for text replacement is `token.replace('_', ' ')`.

---

## Vocabulary Construction Order

Token IDs are assigned in this priority order to ensure stable, predictable IDs:

1. Special tokens: `[PAD]` (0), `[UNK]` (1)
2. Structural placeholder: `cardname` (2)
3. Fixed domain terms (deterministic sets, sorted alphabetically within each group):
   - Game zones: `battlefield`, `command zone`, `exile`, `graveyard`, `hand`, `library`, `stack`
   - Color names: `black`, `blue`, `colorless`, `green`, `red`, `white`
   - Multi-word keywords (underscore form, 24 keywords)
4. Corpus-frequency tokens: all tokens with ≥5 occurrences, sorted by descending frequency (ties broken alphabetically) — **excludes tokens already added above**
5. Mana symbols not yet added (some high-frequency symbols will already be in step 4)

This ordering ensures the most common tokens get low IDs (slightly better cache behavior) and the vocabulary is fully deterministic given the same corpus.

---

## Layer Placement

| Class / File | Layer | Rationale |
|---|---|---|
| `MtgTokenizer` | `domain/tokenizer.py` | Pure encode/decode — no I/O, no framework deps |
| `VocabularyBuilder`, `VocabBuildResult` | `application/build_vocabulary.py` | Orchestration: scans corpus, applies thresholds |
| `MULTI_WORD_KEYWORDS` constant | `application/build_vocabulary.py` | Derived from Forge Keyword.java, stable list |
| `load_tokenizer(path) -> MtgTokenizer` | `infrastructure/tokenizer_store.py` | File I/O |
| `save_vocabulary(vocab, path)` | `infrastructure/tokenizer_store.py` | File I/O |
| `TransformerTrainingDataset` | `infrastructure/transformer_dataset.py` | Updated: accepts `MtgTokenizer` parameter |

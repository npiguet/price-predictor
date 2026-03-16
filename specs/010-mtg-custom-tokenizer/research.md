# Research: MTG Custom Tokenizer

## BertTokenizer Replacement Scope

**Decision**: Replace `BertTokenizer` in all 5 locations — `transformer_dataset.py`, `train_transformer.py` (analysis), `predict_transformer.py`, `server.py`, and implicitly `evaluate_transformer.py` (via dataset).

**Finding**: `vocab_size=30522` is hardcoded at `train_transformer.py:280`. It is not derived from the tokenizer — it must be replaced with the custom vocabulary size passed in from the loaded `MtgTokenizer`.

**Finding**: `analyze_sequence_lengths()` in `train_transformer.py:107` also instantiates `BertTokenizer` to estimate token lengths. This must be updated to use `MtgTokenizer`. Sequence lengths will likely decrease (word-level tokenizer = fewer tokens per card vs. BERT subwords), so `max_seq_len` may shrink beneficially.

## Multi-Word Keywords

**Decision**: Hard-code the 24 multi-word keywords as a constant derived from `forge.game.keyword.Keyword` enum (all entries whose display name contains a space). Loading from the Java source at runtime adds complexity for no benefit — the list is small and stable.

**Complete list (24 keywords, normalized to underscore form):**
```
aura_swap, bands_with_other, battle_cry, choose_a_background,
cumulative_upkeep, doctors_companion, double_agenda, double_strike,
double_team, first_strike, for_mirrodin, hidden_agenda, job_select,
level_up, living_metal, living_weapon, more_than_meets_the_eye,
partner_with, read_ahead, space_sculptor, split_second,
start_your_engines, starting_intensity, umbra_armor
```

**Note**: `doctor's_companion` contains an apostrophe. The word-splitting regex must handle apostrophes in this token specifically, or the keyword must be normalized without it (e.g., `doctors_companion`). Apostrophes are not matched by `[a-z_]+`, so the replacement `"doctor's companion"` → `"doctor's_companion"` would be split into `doctor`, `s_companion` by the regex. **Resolved**: normalize apostrophes out during replacement → `doctors_companion`.

**Keyword detection in corpus**: Keywords appear as `static: first strike` in the converted card format. They are NOT in a dedicated `keyword:` field. The tokenizer's pre-processing pass (replace known multi-word phrases before splitting) handles this correctly since it operates on the full text string.

## Mana Symbols

**Finding**: Mana symbols appear as uppercase in the converted card data (e.g., `{R}`, `{W}`, `{U/B}`) and are **exempt from lowercasing** — they are stored in the vocabulary as-is. Only non-`{}`-enclosed text is lowercased. `CARDNAME` is also uppercase in card text but is NOT in braces, so it lowercases to `cardname`.

**Normalization rule**: `re.split(r'(\{[^}]+\})', text)` splits the text on brace-enclosed tokens; brace parts are kept as-is, non-brace parts are `.lower()`-ed, then reassembled before splitting into tokens.

**Finding**: 62 distinct mana symbols appear in the corpus (confirmed with selective normalization). All are included as tokens. Notable categories:
- Generic: `{0}`–`{16}`, `{20}` (17 values — `{17}`, `{18}`, `{19}` absent)
- Colored: `{W}`, `{U}`, `{B}`, `{R}`, `{G}`, `{C}`
- Hybrid: `{W/U}`, `{U/B}`, `{B/R}`, `{R/G}`, `{G/W}`, `{W/B}`, `{U/R}`, `{B/G}`, `{R/W}`, `{G/U}` + 2/x variants
- Phyrexian: `{W/P}`, `{U/P}`, `{B/P}`, `{R/P}`, `{G/P}`
- Special: `{T}` (tap), `{Q}` (untap), `{E}` (energy), `{S}` (snow), `{X}`, `{P}`, `{H}`, `{CHAOS}`, `{TK}`
- Very rare (1 occurrence each): `{C/B}`, `{C/G}`, `{C/R}`, `{C/U}`, `{C/W}`, `{G/U/P}`, `{G/W/P}`, `{R/G/P}`, `{R/W/P}`

All 62 are included — the vocabulary cost is negligible.

## Vocabulary Size Projection

At frequency threshold of 5: **5,319 tokens** covering **98.1%** of corpus.
Adding:
- 62 mana symbols (most already in the freq-5 set for common ones)
- 24 multi-word keywords (underscore form)
- 2 special tokens `[PAD]`, `[UNK]`
- `cardname` placeholder (lowercased from `CARDNAME` in card text)

Estimated final vocabulary: **~5,500–6,000 tokens**. Well within the <10,000 target.

## Token ID Conventions

- `[PAD]` = ID 0 (always first, enables zero-padding checks)
- `[UNK]` = ID 1

Placing `[PAD]` at ID 0 is conventional and allows fast equality checks (`token_id == 0` → padding).

## Architecture Decisions

**Decision**: `MtgTokenizer` lives in `domain/` as a pure Python class (no I/O).
**Rationale**: Encode/decode is pure domain logic with no external dependencies. Keeps it testable in isolation.

**Decision**: File I/O (load/save `vocab.txt`) in `infrastructure/tokenizer_store.py`.
**Rationale**: Principle IV — infrastructure handles file I/O, domain stays clean.

**Decision**: `VocabularyBuilder` in `application/build_vocabulary.py`.
**Rationale**: Orchestration logic (scan corpus, combine sources, apply threshold) belongs in application layer.

**Decision**: Multi-word keyword list hard-coded as a constant in `build_vocabulary.py`.
**Rationale**: 24 keywords is a stable, finite list. Runtime parsing of Keyword.java adds complexity for zero benefit. Comment references the source (`forge.game.keyword.Keyword`) for traceability.

**Decision**: Any token in `vocab.txt` containing `_` is treated as a multi-word keyword by the loader.
**Rationale**: Natural MTG card text contains no underscores. All `_`-containing tokens in the vocabulary originate from multi-word keyword normalization. The loader can reconstruct the multi-word keyword list from `vocab.txt` without a separate file.

**Decision**: Apostrophes in keyword names are stripped during normalization (`doctor's companion` → `doctors_companion`).
**Rationale**: The word-splitting regex `[a-z_]+` does not match apostrophes. Stripping is simpler than extending the regex.

# Implementation Plan: MTG Custom Tokenizer

**Branch**: `010-mtg-custom-tokenizer` | **Date**: 2026-03-16 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/010-mtg-custom-tokenizer/spec.md`

## Summary

Build a compact, domain-specific word-level tokenizer for MTG card text. The tokenizer combines domain terms (from fixed lists and the Forge keyword enum) with high-frequency corpus words (threshold ≥ 5), producing a ~5,500–6,000 token vocabulary stored in `vocab.txt`. Multi-word keywords (24 total) are normalized to underscore form before splitting. `BertTokenizer` is replaced in all 5 locations it currently appears, and `TransformerConfig.vocab_size` is set dynamically from the loaded vocabulary.

## Technical Context

**Language/Version**: Python 3.14+
**Primary Dependencies**: Standard library only — no new Python packages required
**Storage**: `models/transformer/vocab.txt` (plain text, UTF-8, one token per line)
**Testing**: pytest (existing)
**Target Platform**: Local workstation (same as existing project)
**Project Type**: CLI tool + ML pipeline
**Performance Goals**: Vocabulary build < 1 minute (SC-005); tokenization at training speed unchanged
**Constraints**: vocab_size < 10,000; ≥95% corpus coverage; no new pip dependencies
**Scale/Scope**: ~32,000 card files, ~5,500 vocabulary tokens, 4 transformer pipeline touch points

## Constitution Check

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Fast Automated Tests | ✅ Pass | Unit tests for MtgTokenizer encode/decode, VocabularyBuilder, tokenizer_store; integration test for full vocabulary → train pipeline |
| II. Simplicity First | ✅ Pass | Pure Python word-level tokenizer; no new dependencies; 24 multi-word keywords hard-coded as a constant |
| III. Data Integrity | ✅ Pass | vocab.txt is deterministic given same corpus; validated on load (PAD=0, UNK=1 checked); tokenizer tested with known inputs |
| IV. DDD & Separation | ✅ Pass | MtgTokenizer in domain (pure), VocabularyBuilder in application (orchestration), file I/O in infrastructure |
| V. Forge Interoperability | ✅ Pass | Server API contract unchanged; serve endpoint updated internally |
| VI. Documentation | ✅ Pass | README update required: new `vocabulary` command, updated transformer workflow |

**Gate result: PASS. No violations.**

## Project Structure

### Documentation (this feature)

```text
specs/010-mtg-custom-tokenizer/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/
│   └── cli.md           # CLI contract
└── tasks.md             # Phase 2 output (not yet created)
```

### Source Code Changes

```text
# New files
src/price_predictor/domain/tokenizer.py          # MtgTokenizer domain class
src/price_predictor/application/build_vocabulary.py   # VocabularyBuilder + MULTI_WORD_KEYWORDS
src/price_predictor/infrastructure/tokenizer_store.py # load_tokenizer / save_vocabulary

# Modified files
src/price_predictor/infrastructure/transformer_dataset.py  # Replace BertTokenizer
src/price_predictor/application/train_transformer.py       # Replace BertTokenizer, dynamic vocab_size
src/price_predictor/application/predict_transformer.py     # Replace BertTokenizer
src/price_predictor/application/evaluate_transformer.py    # Accept vocab_path, pass tokenizer to dataset
src/price_predictor/infrastructure/server.py               # Replace BertTokenizer
src/price_predictor/infrastructure/cli.py                  # Add vocabulary command + --vocab-path

# Test files (new)
tests/unit/domain/test_tokenizer.py
tests/unit/application/test_build_vocabulary.py
tests/unit/infrastructure/test_tokenizer_store.py
tests/integration/test_vocabulary_pipeline.py
```

**Structure Decision**: Single project layout (Option 1). All new code follows the existing `domain/application/infrastructure` layering.

## Design

### MtgTokenizer (`domain/tokenizer.py`)

Pure Python class. No dataclass — needs mutable initialization followed by frozen use.

```python
class MtgTokenizer:
    PAD = "[PAD]"  ; PAD_ID = 0
    UNK = "[UNK]"  ; UNK_ID = 1

    def __init__(self, vocab: dict[str, int]) -> None:
        # vocab must have [PAD]=0 and [UNK]=1
        self._vocab = vocab
        self._reverse_vocab = {v: k for k, v in vocab.items()}
        # Multi-word keywords = any token containing '_', sorted longest-first
        self._multi_word_keywords: list[str] = sorted(
            [t for t in vocab if "_" in t], key=len, reverse=True
        )

    @property
    def vocab_size(self) -> int: ...

    def encode(self, text: str, max_length: int) -> tuple[list[int], list[int]]:
        # Returns (input_ids, attention_mask)

    def decode(self, token_ids: list[int]) -> str:
        # Stops at PAD_ID; UNK_ID → "[UNK]"

    def _tokenize(self, text: str) -> list[str]:
        # 1. Selective normalize: lowercase non-{}-enclosed text; keep mana symbols as-is
        #    parts = re.split(r'(\{[^}]+\})', text)
        #    text = ''.join(p if p.startswith('{') else p.lower() for p in parts)
        # 2. Replace multi-word keywords: "first strike" → "first_strike"
        #    (multi_word_keywords sorted longest-first to handle overlaps)
        # 3. re.findall(r"[a-z_]+|\{[^}]+\}|\d+|[^\s\w]", text)
```

### VocabularyBuilder (`application/build_vocabulary.py`)

```python
MULTI_WORD_KEYWORDS: tuple[str, ...] = (
    # Derived from forge.game.keyword.Keyword — entries with spaces in display name
    # Normalized: lowercase, apostrophes removed, spaces → underscores
    "aura_swap", "bands_with_other", "battle_cry", "choose_a_background",
    "cumulative_upkeep", "doctors_companion", "double_agenda", "double_strike",
    "double_team", "first_strike", "for_mirrodin", "hidden_agenda", "job_select",
    "level_up", "living_metal", "living_weapon", "more_than_meets_the_eye",
    "partner_with", "read_ahead", "space_sculptor", "split_second",
    "start_your_engines", "starting_intensity", "umbra_armor",
)

def build_vocabulary(cards_path: Path, freq_threshold: int = 5) -> VocabBuildResult:
    # 1. Seed: [PAD]=0, [UNK]=1, cardname=2
    # 2. Seed fixed domain terms (game zones, colors, multi-word keywords)
    # 3. Scan corpus: normalize text, tokenize, count all tokens
    # 4. Add corpus tokens with count >= freq_threshold not already in vocab
    # 5. Ensure all mana symbols from corpus are included (regardless of threshold)
    # 6. Compute coverage stats
    # 7. Return VocabBuildResult
```

### tokenizer_store (`infrastructure/tokenizer_store.py`)

```python
def save_vocabulary(vocab: dict[str, int], path: Path) -> None:
    # Write vocab.txt: tokens sorted by ID, one per line

def load_tokenizer(vocab_path: Path) -> MtgTokenizer:
    # Read vocab.txt line by line → {token: line_number}
    # Validate: line 0 == "[PAD]", line 1 == "[UNK]"
    # Return MtgTokenizer(vocab)
```

### TransformerTrainingDataset (`infrastructure/transformer_dataset.py`)

Replace `BertTokenizer.from_pretrained(...)` with `MtgTokenizer` passed as constructor parameter:

```python
class TransformerTrainingDataset(Dataset):
    def __init__(
        self,
        card_tuples: list[tuple[str, str, float]],
        max_seq_len: int,
        tokenizer: MtgTokenizer,   # NEW — replaces internal BertTokenizer
    ) -> None:
```

### train_transformer.py changes

1. `analyze_sequence_lengths()`: replace `BertTokenizer` with `MtgTokenizer` parameter
2. `train_transformer()`: accept `vocab_path: Path` parameter; load `MtgTokenizer` via `load_tokenizer`; set `vocab_size=tokenizer.vocab_size` in `TransformerConfig`; pass tokenizer to dataset and `analyze_sequence_lengths`

### predict_transformer.py changes

`PredictTransformerUseCase.execute()`: accept `tokenizer: MtgTokenizer` parameter; replace BertTokenizer encoding with `tokenizer.encode(text, config.max_seq_len)`

### server.py changes

Load `MtgTokenizer` in `create_app()` from `vocab_path` stored in `app.state`; pass to predict endpoint.

### cli.py changes

1. Add `vocabulary` subparser with `--output-dir`, `--cards-path`, `--freq-threshold`
2. Add `--vocab-path` argument to `train transformer`, `evaluate transformer`, `predict transformer`, `serve` subparsers
3. Load tokenizer in each command's run function via `load_tokenizer(args.vocab_path)`; fail with clear message if file missing

## Deliverables

| Artifact | Description |
|----------|-------------|
| `domain/tokenizer.py` | `MtgTokenizer` class |
| `application/build_vocabulary.py` | `VocabularyBuilder`, `VocabBuildResult`, `MULTI_WORD_KEYWORDS` |
| `infrastructure/tokenizer_store.py` | `load_tokenizer`, `save_vocabulary` |
| Updated pipeline files | 6 files with `BertTokenizer` replaced or updated for `vocab_path` |
| Updated `cli.py` | `vocabulary` subcommand + `--vocab-path` on 4 commands |
| Unit + integration tests | Full test coverage per Constitution Principle I |
| Updated `README.md` | New `vocabulary` workflow, updated transformer workflow |

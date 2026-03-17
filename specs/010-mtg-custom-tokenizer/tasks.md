# Tasks: MTG Custom Tokenizer

**Input**: Design documents from `/specs/010-mtg-custom-tokenizer/`
**Prerequisites**: plan.md (required), spec.md (required), research.md, data-model.md, contracts/cli.md, quickstart.md

**Tests**: Per the project constitution (Principle I: Fast Automated Tests), all features MUST include automated tests. Test tasks are MANDATORY in every task list.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

---

## Phase 1: Setup

**Purpose**: Test fixtures and shared test data

- [X] T001 Create sample `vocab.txt` fixture in `tests/fixtures/` — include `[PAD]` (line 0), `[UNK]` (line 1), `cardname`, a handful of domain terms (`flying`, `creature`, `first_strike`), and mana symbols (`{W}`, `{R}`, `{2}`)
- [X] T002 [P] Create 2–3 sample converted card `.txt` fixtures in `tests/fixtures/converted_cards_training/` that include multi-word keywords (e.g., `first strike`, `double strike`), mana symbols (e.g., `{R}`, `{2}{W}`), and `CARDNAME` references

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: `MtgTokenizer` domain class and `tokenizer_store` infrastructure — MUST be complete before any user story

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T003 [P] Add unit tests for `MtgTokenizer` in `tests/unit/domain/test_tokenizer.py` — test construction from vocab dict, `PAD_ID=0`, `UNK_ID=1`, `vocab_size` property; test `_tokenize` selective normalization (mana symbols stay uppercase, text lowercased); test `encode` produces correct `input_ids` and `attention_mask` shapes; test `decode` stops at PAD, maps unknown IDs to `[UNK]`
- [X] T004 [P] Add unit tests for `tokenizer_store` in `tests/unit/infrastructure/test_tokenizer_store.py` — test `save_vocabulary` writes correct `vocab.txt` format (line number = token ID); test `load_tokenizer` returns `MtgTokenizer` with correct vocab; test load validates `[PAD]` at line 0 and `[UNK]` at line 1; test load raises `FileNotFoundError` for missing file
- [X] T005 Create `MtgTokenizer` class in `src/price_predictor/domain/tokenizer.py` — implement `__init__(vocab: dict[str, int])`, `encode(text, max_length) -> tuple[list[int], list[int]]`, `decode(token_ids) -> str`, `vocab_size` property, `pad_token_id`, `unk_token_id`; implement `_tokenize` with selective normalization (`re.split(r'(\{[^}]+\})', text)` for brace-exempt lowercasing), multi-word keyword replacement (tokens containing `_` from vocab, sorted longest-first), and `re.findall(r"[a-z_]+|\{[^}]+\}|\d+|[^\s\w]", text)` splitting
- [X] T006 Create `src/price_predictor/infrastructure/tokenizer_store.py` with `save_vocabulary(vocab: dict[str, int], path: Path) -> None` (writes `vocab.txt`, tokens sorted by ID, UTF-8) and `load_tokenizer(vocab_path: Path) -> MtgTokenizer` (reads `vocab.txt`, builds `{token: line_number}` dict, validates PAD at 0 and UNK at 1, returns `MtgTokenizer(vocab)`)

**Checkpoint**: `MtgTokenizer` encodes/decodes correctly, `tokenizer_store` saves/loads `vocab.txt`. All Foundational tests pass.

---

## Phase 3: User Story 1 — Build Domain Vocabulary from Card Corpus (Priority: P1) 🎯 MVP

**Goal**: `python -m price_predictor vocabulary` scans the converted card corpus, builds a compact domain vocabulary, and writes `vocab.txt`.

**Independent Test**: Run `python -m price_predictor vocabulary --output-dir /tmp/test-vocab/` against `./output/`; verify `vocab.txt` exists, `[PAD]` is line 0, `first_strike` is present, `{W}` is present, vocab size < 10,000, and the JSON output reports coverage ≥ 95%.

### Tests for User Story 1 (MANDATORY per Constitution)

- [X] T007 [P] [US1] Add unit tests for `VocabularyBuilder` in `tests/unit/application/test_build_vocabulary.py` — test that `[PAD]`=0 and `[UNK]`=1 in output vocab; test `MULTI_WORD_KEYWORDS` constant has 24 entries; test that all 24 multi-word keywords appear as underscore tokens; test that `doctor's companion` normalizes to `doctors_companion`; test frequency threshold filtering (words below threshold excluded); test mana symbols always included regardless of threshold; test `VocabBuildResult` fields (coverage_pct, domain_token_count, etc.)
- [X] T008 [P] [US1] Add integration test for vocabulary build in `tests/integration/test_vocabulary_pipeline.py` — run `build_vocabulary()` against the fixture corpus in `tests/fixtures/converted_cards_training/`; assert `vocab.txt` can be round-tripped through `save_vocabulary` / `load_tokenizer`; assert known tokens from fixtures appear in vocab

### Implementation for User Story 1

- [X] T009 [US1] Create `src/price_predictor/application/build_vocabulary.py` with: `MULTI_WORD_KEYWORDS: tuple[str, ...]` constant (24 entries from Forge Keyword.java, lowercase underscore form, apostrophes removed); `VocabBuildResult` frozen dataclass; `build_vocabulary(cards_path: Path, freq_threshold: int = 5) -> VocabBuildResult` — seed special tokens (`[PAD]`=0, `[UNK]`=1, `cardname`=2), seed fixed domain terms (game zones, color names, multi-word keywords), scan corpus with selective normalization, add freq-threshold tokens, add all mana symbols, compute coverage stats
- [X] T010 [US1] Add `vocabulary` subcommand to `src/price_predictor/infrastructure/cli.py` — arguments: `--output-dir` (default `models/transformer/`), `--cards-path` (default `./output`), `--freq-threshold` (default `5`); call `build_vocabulary()`, call `save_vocabulary(result.vocab, output_dir / "vocab.txt")`; print JSON to stdout with `vocab_path`, `vocab_size`, `domain_token_count`, `freq_threshold_token_count`, `coverage_pct`, `unk_pct`; exit 1 if cards path not found

**Checkpoint**: `python -m price_predictor vocabulary` runs end-to-end, produces valid `vocab.txt`, prints stats. US1 acceptance scenarios 1–5 testable.

---

## Phase 4: User Story 2 — Tokenize Card Text (Priority: P2)

**Goal**: `MtgTokenizer.encode()` correctly handles all MTG card text patterns: multi-word keywords as single tokens, mana symbols as single tokens, domain terms as single tokens, `[UNK]` for unknown words, round-trip decode for in-vocab text.

**Independent Test**: Load the fixture `vocab.txt`, call `tokenizer.encode("first strike", max_length=10)` and verify `first_strike` resolves to a single token ID; call `tokenizer.encode("{W}{W}: Flying", max_length=10)` and verify `{W}` appears as uppercase token IDs; verify `decode(encode(text, max_length=100)[0])` recovers the normalized text for a fully in-vocab card.

### Tests for User Story 2 (MANDATORY per Constitution)

- [X] T011 [P] [US2] Add acceptance scenario tests to `tests/unit/domain/test_tokenizer.py` — test "flying, vigilance" produces two tokens; test "first strike" encodes as single `first_strike` token; test "{W}{W}" encodes as two `{W}` tokens (uppercase, not lowercased); test CARDNAME appears as `cardname` token; test unknown word maps to `UNK_ID`; test round-trip decode for all-in-vocab text; test padding fills to `max_length` with `PAD_ID`; test `attention_mask` is 1 for real tokens and 0 for padding
- [X] T012 [P] [US2] Add edge case tests to `tests/unit/domain/test_tokenizer.py` — test truncation at `max_length`; test empty string input; test text with only mana symbols; test multi-word keyword that overlaps with single keyword (e.g. verify "double strike" encodes as `double_strike` not `double` + `strike`)

### Implementation for User Story 2

- [X] T013 [US2] Verify `MtgTokenizer._tokenize` handles all US2 acceptance scenarios by running the new tests (T011, T012); fix any edge cases found (no new code expected if T005 was correct — this task is verification)

**Checkpoint**: All US2 acceptance scenarios pass. The tokenizer correctly handles the full MTG text domain.

---

## Phase 5: User Story 3 — Integrate Custom Tokenizer into Transformer Pipeline (Priority: P3)

**Goal**: `BertTokenizer` is removed from all 5 locations in the transformer pipeline. `TransformerConfig.vocab_size` is set from the loaded tokenizer. `train transformer`, `evaluate transformer`, `predict transformer`, and `serve` all accept `--vocab-path` and fail clearly if the file is missing.

**Independent Test**: Run `python -m price_predictor train transformer --epochs 1` after building vocabulary; verify training completes; load the saved model and assert `config["vocab_size"] != 30522`.

### Tests for User Story 3 (MANDATORY per Constitution)

- [X] T014 [P] [US3] Update `tests/unit/infrastructure/test_transformer_dataset.py` — replace BertTokenizer mock with `MtgTokenizer` constructed from fixture vocab; verify `TransformerTrainingDataset.__init__` accepts `tokenizer: MtgTokenizer` parameter; verify output `input_ids` shape is `(batch, max_seq_len)` and `attention_mask` matches padding positions
- [X] T015 [P] [US3] Update `tests/unit/application/test_train_transformer.py` — mock `load_tokenizer` to return a fixture `MtgTokenizer`; verify `TransformerConfig.vocab_size` equals `tokenizer.vocab_size` (not 30522); verify `train_transformer()` accepts `vocab_path` parameter
- [X] T016 [P] [US3] Update `tests/unit/application/test_predict_transformer.py` — mock `load_tokenizer`; verify `PredictTransformerUseCase.execute()` accepts `tokenizer` parameter and uses it to encode card text
- [X] T017 [P] [US3] Update `tests/unit/application/test_evaluate_transformer.py` — mock `load_tokenizer`; verify `evaluate_transformer()` accepts `vocab_path` parameter and passes tokenizer to dataset
- [X] T018 [P] [US3] Add integration test to `tests/integration/test_vocabulary_pipeline.py` — full pipeline: build vocabulary → save → load tokenizer → construct `TransformerTrainingDataset` → verify batch shapes

### Implementation for User Story 3

- [X] T019 [US3] Update `TransformerTrainingDataset.__init__` in `src/price_predictor/infrastructure/transformer_dataset.py` — replace `BertTokenizer.from_pretrained("bert-base-uncased")` with `tokenizer: MtgTokenizer` constructor parameter; replace `tokenizer(text, ...)` call with `tokenizer.encode(text, max_seq_len)`
- [X] T020 [US3] Update `src/price_predictor/application/train_transformer.py` — (a) `analyze_sequence_lengths(texts, tokenizer: MtgTokenizer)`: replace BertTokenizer with tokenizer parameter; (b) `train_transformer(..., vocab_path: Path)`: call `load_tokenizer(vocab_path)`, set `vocab_size=tokenizer.vocab_size` in `TransformerConfig`, pass tokenizer to `TransformerTrainingDataset` and `analyze_sequence_lengths`; fail with clear error if `vocab_path` not found
- [X] T021 [US3] Update `PredictTransformerUseCase.execute()` in `src/price_predictor/application/predict_transformer.py` — replace `BertTokenizer.from_pretrained(...)` with `tokenizer: MtgTokenizer` parameter; replace encoding call with `tokenizer.encode(text, config.max_seq_len)`
- [X] T022 [US3] Update `evaluate_transformer()` in `src/price_predictor/application/evaluate_transformer.py` — accept `vocab_path: Path` parameter; call `load_tokenizer(vocab_path)`; pass tokenizer to `TransformerTrainingDataset`
- [X] T023 [US3] Update predict endpoint in `src/price_predictor/infrastructure/server.py` — load `MtgTokenizer` via `load_tokenizer(vocab_path)` in `create_app()`; store in `app.state.tokenizer`; replace `BertTokenizer.from_pretrained(...)` in the predict handler with `app.state.tokenizer`
- [X] T024 [US3] Add `--vocab-path` (default `models/transformer/vocab.txt`) to `train transformer`, `evaluate transformer`, `predict transformer`, and `serve` subparsers in `src/price_predictor/infrastructure/cli.py`; load tokenizer in each run function via `load_tokenizer(args.vocab_path)`; print clear error and exit 1 if `vocab.txt` not found

**Checkpoint**: All 5 BertTokenizer references removed. `transformers` package no longer required at runtime for inference. Model trains with `vocab_size` from custom tokenizer. US3 acceptance scenarios 1–3 testable.

---

## Phase 6: User Story 4 — Verify Compact Vocabulary Size (Priority: P4)

**Goal**: Confirm SC-001 (< 10,000 tokens), SC-002 (≥ 95% coverage), SC-003 (all MTG domain terms present), SC-006 (model trains successfully with custom vocab).

**Independent Test**: Run `python -m price_predictor vocabulary` against full corpus and verify output JSON shows `vocab_size < 10000` and `coverage_pct >= 95.0`.

### Tests for User Story 4 (MANDATORY per Constitution)

- [X] T025 [P] [US4] Add SC-001/SC-002 validation test in `tests/unit/application/test_build_vocabulary.py` — build vocabulary from fixture corpus, assert `result.vocab_size < 10000`, assert `result.coverage_pct >= 95.0`, assert all color names and game zones are present as tokens
- [X] T026 [P] [US4] Add SC-003 domain coverage test in `tests/unit/application/test_build_vocabulary.py` — assert all 24 multi-word keywords (underscore form) present; assert `{W}`, `{U}`, `{B}`, `{R}`, `{G}`, `{T}`, `{X}` present; assert `creature`, `flying`, `battlefield`, `legendary` present

### Implementation for User Story 4

- [X] T027 [US4] Run `python -m price_predictor vocabulary` against full card corpus, record results (vocab_size, coverage_pct) in a comment in `build_vocabulary.py` or as a note in `research.md`; verify SC-001 and SC-002 pass; this is informational — no code change if thresholds are met

**Checkpoint**: SC-001 through SC-004 verified. SC-006 verified after retraining.

---

## Phase 7: Polish & Cross-Cutting Concerns

- [X] T028 [P] Run `ruff check .` from `src/` and fix any linting issues introduced by this feature
- [X] T029 Run full test suite (`pytest tests/unit/`) and fix any failures
- [X] T030 Run `pytest tests/integration/` and fix any failures
- [X] T031 [P] Update `README.md` — add `vocabulary` subcommand to Workflows section (inputs, outputs, options); update `train transformer` workflow to show `vocabulary` as prerequisite step; update `serve` options table with `--vocab-path`
- [X] T032 Run quickstart.md validation — manually verify the workflow from `specs/010-mtg-custom-tokenizer/quickstart.md` end-to-end; retrain transformer model; verify SC-006 (model trains, `vocab_size != 30522` in saved checkpoint)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — can start immediately
- **Foundational (Phase 2)**: Depends on Setup (T001, T002) for fixtures — BLOCKS all user stories
- **US1 (Phase 3)**: Depends on Foundational (Phase 2) — `build_vocabulary` uses `MtgTokenizer` for normalization
- **US2 (Phase 4)**: Depends on Foundational (Phase 2) — tests `MtgTokenizer` acceptance scenarios; independent of US1
- **US3 (Phase 5)**: Depends on Foundational (Phase 2) + US1 (needs `vocab.txt` at runtime); pipeline tests require working tokenizer
- **US4 (Phase 6)**: Depends on US1 (needs vocabulary built) + US3 (SC-006 requires full pipeline)
- **Polish (Phase 7)**: Depends on all user stories complete

### User Story Dependencies

- **US1 (P1)**: Can start after Foundational — delivers the `vocabulary` CLI command independently
- **US2 (P2)**: Can start after Foundational — validates tokenizer behavior; no dependency on US1
- **US3 (P3)**: Depends on US1 (vocabulary must be buildable); can proceed once US1 checkpoint is reached
- **US4 (P4)**: Depends on US1 + US3; purely validation

### Within Each User Story

- Tests MUST be written first and FAIL before implementation
- Core implementation before integration
- Story complete before moving to next priority

### Parallel Opportunities

**Phase 2 (Foundational)**:
- T003 + T004 (tests) can run in parallel — different files
- T005 + T006 (implementation) can run in parallel — different files

**Phase 3 (US1)**:
- T007 + T008 (tests) can run in parallel
- T009 + T010 are sequential (CLI depends on `build_vocabulary`)

**Phase 4 (US2)**:
- T011 + T012 (tests) can run in parallel
- T013 (verify) can run after T011/T012

**Phase 5 (US3)**:
- T014, T015, T016, T017 (tests) can all run in parallel — different files
- T019, T020, T021, T022, T023 can run in parallel — all different files
- T024 (CLI) runs after T019–T023 (needs functions to wire up)

---

## Parallel Example: User Story 3

```bash
# Launch all US3 tests together (write-first, should fail):
Task: "T014 — test_transformer_dataset.py MtgTokenizer acceptance"
Task: "T015 — test_train_transformer.py vocab_path + vocab_size"
Task: "T016 — test_predict_transformer.py tokenizer parameter"
Task: "T017 — test_evaluate_transformer.py vocab_path parameter"
Task: "T018 — integration test vocabulary → dataset pipeline"

# Launch implementation in parallel (all different files):
Task: "T019 — transformer_dataset.py replace BertTokenizer"
Task: "T020 — train_transformer.py vocab_path + vocab_size"
Task: "T021 — predict_transformer.py tokenizer parameter"
Task: "T022 — evaluate_transformer.py vocab_path parameter"
Task: "T023 — server.py replace BertTokenizer"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (fixtures)
2. Complete Phase 2: Foundational (MtgTokenizer + tokenizer_store)
3. Complete Phase 3: User Story 1 (vocabulary builder + CLI command)
4. **STOP and VALIDATE**: Run `python -m price_predictor vocabulary`, inspect `vocab.txt`
5. Commit — the vocabulary artifact is independently valuable

### Incremental Delivery

1. Setup + Foundational → MtgTokenizer working
2. Add US1 → `vocabulary` command → produces `vocab.txt` (MVP!)
3. Add US2 → tokenizer acceptance scenarios verified
4. Add US3 → BertTokenizer replaced → retrain transformer
5. Add US4 + Polish → verify size constraints + docs

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- `transformers` package (huggingface) is still a dependency until US3 is complete — BertTokenizer is used in 5 places; do not remove the dependency until all 5 are replaced
- `vocab_size=30522` hardcode in `train_transformer.py:280` is the critical value to replace in T020
- Multi-word keywords in vocab are identified at load time by checking for `_` in token string — no separate keyword file needed
- Card entity is not changed — this feature touches only the transformer pipeline (dataset, train, predict, evaluate, server, cli)

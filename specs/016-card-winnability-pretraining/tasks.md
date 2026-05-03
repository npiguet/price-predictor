---

description: "Task list for spec 016 (card winnability pretraining for sealed encoder)"
---

# Tasks: Card Winnability Pretraining for Sealed Encoder

**Input**: Design documents from `/specs/016-card-winnability-pretraining/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/cli.md, contracts/files.md, quickstart.md

**Tests**: Per Constitution Principle I (Fast Automated Tests), tests are MANDATORY and are written alongside (or before) the implementation tasks they cover.

**Organization**: Tasks are grouped by user story (US1–US5). Phase ordering follows priority (P1 → P2 → P3). US2 and US3 implementation work uses fixtures, so it does not block on US1 finishing match collection or on US4 finishing vocab build.

## Format

`- [ ] [TaskID] [P?] [Story?] Description with file path`

- **[P]** = different files, no dependency on other incomplete tasks → parallelizable.
- **[Story]** = US1 / US2 / US3 / US4 / US5. Setup, Foundational, and Polish phases carry no story label.

## Path Conventions

- Python source: `src/sealed/...`, `src/price_predictor/...`
- Python tests: `tests/unit/sealed/...`, `tests/integration/...`
- Java source: `forge-connector/src/main/java/com/pricepredictor/connector/...`
- Java tests: `forge-connector/src/test/java/com/pricepredictor/connector/...`
- Specs: `specs/016-card-winnability-pretraining/...`

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Cheap one-time bootstrap. The repository, build, and test stacks already exist, so this phase is intentionally small.

- [ ] T001 Confirm `models/sealed/encoder/` and `output/sealed/` parent directories are auto-created by their writers (no static checked-in scaffolding); document the auto-create expectation in `specs/016-card-winnability-pretraining/quickstart.md` if not already explicit.
- [ ] T002 [P] Verify `forge-connector` builds clean from main (`cd forge-connector; mvn -DskipTests package`) so subsequent Java tasks have a known-good baseline.
- [ ] T003 [P] Verify the Python fast suite is green on the current branch (`pytest -m "not integration"`) so subsequent Python tasks have a known-good baseline.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Shared domain entities and storage adapters consumed by US2 (creator) AND US3 (loader). Must complete before either P1 Python story can finish.

**⚠️ CRITICAL**: No US2 / US3 work should land before this phase is done.

- [ ] T004 [P] Add `SealedEncoderConfig` dataclass in `src/sealed/domain/encoder_model.py` mirroring the field set in `data-model.md` §SealedEncoderConfig (vocab_size, d_model, n_layers, n_heads, ff_dim, max_seq_len, dropout, n_pool_queries) with the validation rules (`d_model % n_pool_queries == 0`, `n_heads` divides `d_model`, etc.). Prior art: `TransformerConfig` in `src/price_predictor/domain/entities.py:103`; parallel concept justified in research.md §"Overlapping domain vocabulary".
- [ ] T005 Implement `SealedEncoderModel` in the same file (`src/sealed/domain/encoder_model.py`) with three child modules — `token_encoder` (token+positional embedding), `card_encoder` (`nn.TransformerEncoder` stack + dual pool), `regression_head` (`Linear(2*d_model, 1)` + `Sigmoid`). Implement the multi-query attention pool concatenated with element-wise max pool to produce `(B, 2*d_model)` per data-model.md §"Multi-query attention pool". Expose `encode(input_ids, attention_mask) -> (B, 2*d_model)` (no-grad-friendly) and `forward(...) -> (B,)` (full path through head + sigmoid). Prior art: `CardPriceTransformerModel._encode_and_pool` in `src/price_predictor/infrastructure/transformer_model.py:11` (masking + positional pattern copied per research.md D-3).
- [ ] T006 [P] Unit tests for `SealedEncoderModel` in `tests/unit/sealed/domain/test_encoder_model.py`: forward output shape `(B,)`, encode output shape `(B, 2*d_model)`, regression head presence, padding mask honored, `d_model % n_pool_queries != 0` raises in config, AND a constructor smoke test that two fresh models built from the same config but different `torch.manual_seed` values produce *different* weight tensors (FR-016: encoder is randomly initialized, never seeded from a price-side checkpoint).
- [ ] T007 Implement `SealedEncoderStore` in `src/sealed/infrastructure/encoder_store.py` exposing `save_encoder(model, config, output_dir) -> Path` (writes `{ISO_TIMESTAMP}.pt` then byte-copies to `latest.pt`; filters `model_state_dict` to keys with prefix `token_encoder.` or `card_encoder.`) and `load_encoder(path) -> (SealedEncoderModel, SealedEncoderConfig)` (instantiates fresh model from saved config dict, calls `load_state_dict(strict=True)`). Prior art: `ScorerStore.save_checkpoint` in `src/sealed/infrastructure/scorer_store.py:42` (pattern-mirror per research.md §"Adjacent prior art"). Spec refs: FR-020, files.md §`models/sealed/encoder/*.pt`.
- [ ] T008 [P] Unit tests for `SealedEncoderStore` in `tests/unit/sealed/infrastructure/test_encoder_store.py`: round-trip save→load reconstructs an identically-behaving model, regression-head keys are stripped from the saved state-dict, `latest.pt` is a byte-for-byte copy of the chosen timestamped file, `strict=True` load fails if a regression-head key sneaks in.

**Checkpoint**: Domain model and store exist. US2 (creator) and US3 (consumer) can now proceed in parallel.

---

## Phase 3: User Story 1 - Per-game card-play data accumulates during self-play (Priority: P1)

**Goal**: Every Forge match worker emits one line per played game to `output/sealed/cards-played.txt` with the eleven-field schema in `contracts/files.md`.

**Independent Test**: Run `python -m sealed match-outcomes` for a handful of matches; verify that `output/sealed/cards-played.txt` line count equals the sum of game counts across the `match-outcomes.txt` lines written during the same run, that game lines for a given match share the parent's `run_id` and appear contiguously and in game order, and that basic lands never appear in any column.

### Tests for User Story 1 (write first; let them fail; then implement)

- [ ] T009 [P] [US1] Java unit test `forge-connector/src/test/java/com/pricepredictor/connector/CardsPlayedRowTest.java`: line-format round-trip for the eleven fields (per files.md §`cards-played.txt`), pipe-separated multiplicities preserved, empty card lists round-trip as empty strings between two `;`, ISO 8601 UTC timestamp format matches `MatchResultWriter` output style.
- [ ] T010 [P] [US1] Java unit test `forge-connector/src/test/java/com/pricepredictor/connector/CardsPlayedWriterTest.java`: open-write-close per line, two writers from concurrent threads do not interleave bytes within a line, trailing partial line is acceptable on truncation (writer must flush at line granularity).
- [ ] T011 [P] [US1] Java unit test `forge-connector/src/test/java/com/pricepredictor/connector/PlayedCardCollectorTest.java`: zone-change to Battlefield records, `GameEventSpellAbilityCast` records, basic land filter (Plains, Island, Swamp, Mountain, Forest, Wastes, snow basics) drops at observation time, `card.getController() != card.getOwner()` filter drops stolen cards, `card.isToken()` drops tokens, copy effects record `getPaperCard().getName()` not the copied permanent's name. Per research.md D-3.
- [ ] T012 [US1] Java integration-tagged test `forge-connector/src/test/java/com/pricepredictor/connector/MatchGeneratorCardsPlayedIntegrationTest.java` (`@Tag("integration")`): drive a single match through `MatchGenerator.generateMatch`, assert that the returned `List<CardsPlayedRow>` length equals the played-game count, every row's `run_id` matches the parent match, every row's set_code/method_A/method_B equals the parent's, basic lands absent, and the union of `cardsPlayedA` and `cardsNotPlayedA` equals deck A minus basics (multiplicities preserved).

### Implementation for User Story 1

- [ ] T013 [P] [US1] Create `forge-connector/src/main/java/com/pricepredictor/connector/CardsPlayedRow.java` Java record with the eleven fields (`Instant timestamp, String runId, String setCode, String methodA, String methodB, List<String> cardsPlayedA, List<String> cardsPlayedB, List<String> cardsNotPlayedA, List<String> cardsNotPlayedB, char winner, char starter`) plus a `toLine()` formatter that emits the schema in files.md (semicolon-separated, no trailing `;`, pipes inside list columns, ISO_INSTANT timestamp). Prior art: `MatchResult` / `MatchResultWriter`.
- [ ] T014 [P] [US1] Create `forge-connector/src/main/java/com/pricepredictor/connector/CardsPlayedWriter.java` mirroring `MatchResultWriter`'s open-append-close-per-line strategy with target path `output/sealed/cards-played.txt`. Method: `void write(CardsPlayedRow row)`. Prior art: `MatchResultWriter` (research.md §"Adjacent prior art").
- [ ] T015 [P] [US1] Create `forge-connector/src/main/java/com/pricepredictor/connector/PlayedCardCollector.java` as an `IGameEventVisitor.Base<Void>` that subscribes to `GameEventCardChangeZone` (filtered to `event.to().getZoneType() == ZoneType.Battlefield`) and `GameEventSpellAbilityCast`, applying the four filters from research.md D-3 (`controller==owner`, `!isToken`, `!isBasicLand`, record `paperCard.name`). Buckets cards by `card.getOwner().getName()` and resolves to `A` / `B` via `LOBBY_NAME_A` / `LOBBY_NAME_B` from `GamePlayer`. Mirrors `../jumpstart-tierlist/.../JumpstartMatch.java#CardCollector` (research.md D-3 cites this as the prior-art template).
- [ ] T016 [US1] Extend `GameOutcome` in `forge-connector/src/main/java/com/pricepredictor/connector/GamePlayer.java` to add `Set<String> cardsPlayedA` and `Set<String> cardsPlayedB` fields (per data-model.md §`GameOutcome`). Update `GamePlayer.playMatch()` to construct one fresh `PlayedCardCollector` per game, call `game.subscribeToEvents(collector)` before play, and attach the collector's two sets to the outgoing `GameOutcome` after each game.
- [ ] T017 [US1] Update `forge-connector/src/main/java/com/pricepredictor/connector/MatchGenerator.java` so `generateMatch(...)` returns a tuple `(MatchResult, List<CardsPlayedRow>)` (introduce a tiny `MatchGenerationResult` record if a tuple type is needed). Compose each `CardsPlayedRow` from the per-game `GameOutcome.cardsPlayed{A,B}` plus the deck card lists (deck minus basics minus played = `cardsNotPlayedX`), winner/starter from `GameOutcome`, and the parent match's timestamp/run_id/set_code/method_A/method_B.
- [ ] T018 [US1] Update `forge-connector/src/main/java/com/pricepredictor/connector/MatchWorkerMain.java` so the worker loop writes the match-line via `MatchResultWriter` and the per-game lines via a `CardsPlayedWriter` instance pointing at `output/sealed/cards-played.txt`. The writes happen in the same iteration as the existing match write. Per research.md D-7: Python supervisor untouched.
- [ ] T019 [US1] Add a Python integration test in `tests/integration/test_cards_played_collection.py` (`@pytest.mark.integration`) that invokes `python -m sealed match-outcomes` for a handful of matches (or seeds via the worker entry point directly), then verifies:
  - line count of `output/sealed/cards-played.txt` equals the sum of game counts (`match-outcomes.txt` field 8 length) for the matching `run_id`,
  - `run_id` and `(set_code, method_A, method_B)` tuples line up between the two files,
  - basic-land names never appear in any of the four list columns,
  - game lines for one match appear contiguously and in game order.

**Checkpoint**: `cards-played.txt` is written by every match-outcomes run. US1 is independently demoable.

---

## Phase 4: User Story 2 - Train a sealed encoder from scratch on the winnability target (Priority: P1)

**Goal**: `python -m sealed train-encoder` reads `cards-played.txt`, aggregates labels inline (with shrinkage), trains the dual-pool encoder + regression head from random init on MSE, and saves only encoder weights to `models/sealed/encoder/{timestamp}.pt` + `latest.pt`. Discards the regression head.

**Independent Test**: Against a fixture `cards-played.txt` and fixture `vocab.txt`, run `train-encoder`; verify (a) `latest.pt` round-trips through `SealedEncoderStore.load_encoder` with `strict=True`, (b) the saved state-dict contains only `token_encoder.*` and `card_encoder.*` keys, (c) the train/val split is card-disjoint, (d) the saved file corresponds to the best-by-val-loss epoch within `--patience`, (e) `output/sealed/cards-win-rates.txt` is populated and sorted by raw ratio descending.

### Tests for User Story 2 (write first; let them fail; then implement)

- [ ] T020 [P] [US2] Unit test `tests/unit/sealed/infrastructure/test_cards_played_reader.py`: parses 11-field lines, raises on mid-stream malformed line, tolerates a trailing partial (non-newline-terminated) line silently (Edge Case: JVM crash mid-write), decodes the four pipe-separated card-list columns into Python lists with multiplicities preserved, recovers `winner` / `starter` characters. ALSO add a fixture where one match's game block is shorter than the parent `match-outcomes.txt` row's game count, and assert the reader yields the available game lines without aborting (Edge Case: line-count disagreement is tolerated by aggregation, FR-005).
- [ ] T021 [P] [US2] Unit test `tests/unit/sealed/application/test_label_aggregation.py`: given a synthetic in-memory cards-played stream, computes `wins_when_played[c]` and `wins_when_in_deck[c]` per FR-010 (only winning side counts), excludes cards with `wins_when_in_deck == 0` (FR-012), and returns the per-card map keyed by canonical name.
- [ ] T022 [P] [US2] Unit test `tests/unit/sealed/application/test_shrinkage.py`: with `k = 0`, shrunk label equals raw ratio. With `k = 20` and `wins_when_in_deck = 2, wins_when_played = 2`, shrunk label is meaningfully below 1.0 (pulled toward 0.5). With `k = 20` and `wins_when_in_deck = 1000`, shrunk label is within a few thousandths of the raw ratio (FR-011, US5 acceptance).
- [ ] T023 [P] [US2] Unit test `tests/unit/sealed/application/test_train_val_split.py`: stratified-by-quartile, card-level disjoint (no overlap), val_fraction == 0.2 ± rounding, deterministic with seed=42, falls back gracefully when fewer than four distinct quantile bins exist (Edge Case in spec).
- [ ] T024 [P] [US2] Unit test `tests/unit/sealed/application/test_cards_win_rates_writer.py`: header row first, rows sorted by `raw_ratio` descending (FR-013a), semicolon-separated, five-decimal float formatting, file is overwritten each call.
- [ ] T025 [P] [US2] Unit test `tests/unit/sealed/application/test_corpus_consistency.py`: passes when every label-map card has a matching `output/cardsfolder/{name}.txt` (after `card_name_corrections`), raises a custom error naming up to 20 missing cards plus a total count when one or more cards are absent (FR-023d).
- [ ] T026 [P] [US2] Unit test `tests/unit/sealed/infrastructure/test_cli_train_encoder_argparse.py`: each FR-021 flag parses with the documented default, hardcoded constants (d_model, ff_dim, val_fraction, random_seed) are absent from the CLI surface (FR-022), `--n-pool-queries` that does not divide `d_model` exits code 6, and `python -m sealed --help` does NOT advertise an `aggregate-labels` subcommand (FR-013: aggregation is inline, no auxiliary command).
- [ ] T027 [P] [US2] Integration smoke test `tests/integration/test_train_encoder_smoke.py` (`@pytest.mark.integration`): point at `tests/fixtures/sealed/cards-played.sample.txt` + a tiny vocab fixture + a tiny corpus fixture; run `train_encoder(config)` for `--epochs 1 --patience 1`; assert that `models/sealed/encoder/latest.pt` round-trips through `load_encoder` and that `cards-win-rates.txt` has been written.

### Implementation for User Story 2

- [ ] T028 [P] [US2] Create `tests/fixtures/sealed/cards-played.sample.txt` with a synthetic ~50-line `cards-played.txt` covering: high-winnability cards (>= 5 wins-in-deck), low-winnability cards (1–2 wins-in-deck for shrinkage tests), cards never on a winning side, basic lands absent (sanity), at least two distinct `(run_id, set_code, method_A, method_B)` groups. Reuse a small `output/cardsfolder/` subset already present under `tests/fixtures/` if the layout fits.
- [ ] T029 [P] [US2] Implement `src/sealed/infrastructure/cards_played_reader.py` exposing `iter_rows(path) -> Iterator[CardsPlayedRow]` (Python record). Streams lines, tolerates a trailing partial line, raises a clear `ValueError` on mid-file malformed lines.
- [ ] T030 [P] [US2] Implement label aggregation in `src/sealed/application/train_encoder.py` (or a sibling `_aggregation.py` private helper): single pass over `cards_played_reader`, accumulates `wins_when_played` and `wins_when_in_deck` for the winning side only, returns `dict[str, (wins_when_played, wins_when_in_deck)]`. FR-010, FR-012.
- [ ] T031 [US2] Add the shrinkage transformation `_shrink(wins_when_played, wins_when_in_deck, k) -> float` in the same module, implementing FR-011: `(wins_when_played + k/2) / (wins_when_in_deck + k)`. Build the `WinnabilityMap = dict[str, CardLabel]` (`CardLabel` dataclass per data-model.md).
- [ ] T032 [US2] Implement the `cards-win-rates.txt` writer in the same module: opens `output/sealed/cards-win-rates.txt` (fixed path, FR-013a), writes header row + N data rows sorted by raw ratio descending with 5-decimal float formatting; overwrites on each call.
- [ ] T033 [US2] Implement the corpus consistency check (FR-023d) using `ConvertedCardLocator` (`src/sealed/infrastructure/converted_card_locator.py`) and `CardNameCorrections` (`src/sealed/infrastructure/card_name_corrections.py`); if any label-map card cannot be resolved on disk, raise a `CorpusInconsistencyError` naming up to 20 cards plus the total count, pointing the user at `python -m price_predictor convert`. The CLI handler in T039 maps this to exit code 5 (per cli.md §"Exit codes"). The check MUST run AFTER aggregation so the error can enumerate offending names. Per research.md §"Adjacent prior art".
- [ ] T034 [P] [US2] Implement the stratified card-level train/val split helper `_split_cards(label_map, val_fraction=0.2, seed=42) -> (train_names, val_names)`: bucket cards into 4 quantiles by shrunk label, sample 20% of each, fall back to fewer strata if quantile bins collapse (Edge Cases). Card-disjoint (FR-018).
- [ ] T035 [US2] Implement the training dataset/dataloader: tokenizes each card's converted text with `MtgTokenizer` after stripping the `name:` line via `ConvertedCardText.without_name_line()` (FR-014a; reuse from `src/sealed/domain/card_encoder.py:34`); pads to per-batch `max_seq_len`; supplies `(input_ids, attention_mask, label)` triples. `max_seq_len` for the encoder is the corpus-longest length rounded up to a multiple of 8 (FR-022).
- [ ] T036 [US2] Define `TrainEncoderConfig` dataclass in `src/sealed/application/train_encoder.py` (per data-model.md §`TrainEncoderConfig`) with the FR-021 flag fields plus the FR-022 hardcoded class constants. Top-level `run(config: TrainEncoderConfig) -> None` mirrors `train_scorer.train_scorer(config)` shape.
- [ ] T037 [US2] Implement the training loop in `src/sealed/application/train_encoder.py` consuming `TrainEncoderConfig` from T036: AdamW + LambdaLR (linear warmup → constant), MSE on sigmoid output vs. shrunk label, per-epoch progress lines (`_log()` style from `train_scorer.py:38`), `_BestCheckpoint` early-stopping pattern copied per research.md §"Adjacent prior art" (mark a follow-up to extract on the next training feature; do NOT extract now). On train end, snapshot best weights, hand to `SealedEncoderStore.save_encoder`. Print final summary `Best epoch: E, val_loss: V. Saved <path>`.
- [ ] T038 [US2] Implement pre-flight checks at the top of `run(config)` per contracts/cli.md §`train-encoder` "Pre-flight checks": missing vocab → exit 2 (point at `build-vocab`), missing/empty cards-played → exit 3 (point at `match-outcomes`), empty cards-folder → exit 4 (point at `convert`), config validation failure → exit 6. The corpus-consistency check (FR-023d, T033) is NOT a pre-flight — it runs after aggregation so the error can enumerate cards — and surfaces as exit code 5 via the CLI handler in T039.
- [ ] T039 [US2] Wire the `train-encoder` subcommand into `src/sealed/infrastructure/cli.py`: add `_build_train_encoder_parser(subparsers)` and `run_train_encoder(args)` following the established pattern (`build_parser`, line ~60). Register the FR-021 flags with documented defaults; map exit codes from raised exceptions to the contracts/cli.md table (0/2/3/4/5/6).

**Checkpoint**: `train-encoder` produces a usable `models/sealed/encoder/latest.pt`. US2 is independently demoable using fixture data.

---

## Phase 5: User Story 3 - Sealed scorer consumes the sealed-trained encoder (Priority: P1)

**Goal**: `encode-cards` and `train-scorer` default `--encoder-checkpoint` to `models/sealed/encoder/latest.pt`. Missing-default-file is a hard error pointing at `train-encoder`. Explicit `--encoder-checkpoint` overrides remain functional.

**Independent Test**: With a sealed encoder at `models/sealed/encoder/latest.pt`, run `python -m sealed encode-cards` (no flag) and `python -m sealed train-scorer` (no flag); assert the resulting `.npz` files come from the sealed encoder and the scorer checkpoint's `train_config['encoder_checkpoint']` records the sealed encoder path. Then run with `--encoder-checkpoint models/price-predictor/transformer/latest.pt` and assert the older encoder is used (SC-006).

### Tests for User Story 3 (write first; let them fail; then implement)

- [ ] T040 [P] [US3] Unit test `tests/unit/sealed/infrastructure/test_cli_encode_cards_default.py`: with no flag, the resolved `--encoder-checkpoint` is `models/sealed/encoder/latest.pt` and `--vocab-path` is `models/sealed/encoder/vocab.txt` (per cli.md §encode-cards "Vocab path coupling"). With explicit `--encoder-checkpoint <other>`, the explicit value wins.
- [ ] T041 [P] [US3] Unit test `tests/unit/sealed/infrastructure/test_cli_train_scorer_default.py`: with no flag, the resolved `encoder_checkpoint` in the constructed `TrainScorerConfig` is `models/sealed/encoder/latest.pt`. With `--encoder-checkpoint models/price-predictor/transformer/latest.pt`, the resolved value is the price-side path.
- [ ] T042 [P] [US3] Unit test `tests/unit/sealed/application/test_encoder_default_missing.py`: when the resolved default sealed-encoder file does not exist AND no explicit `--encoder-checkpoint` was passed, both `encode-cards` and `train-scorer` exit with code 2 and the error message names the missing file plus `python -m sealed train-encoder`. When explicit `--encoder-checkpoint` is passed at a missing path, the existing error path (whatever it currently is) is preserved (no new behavior).
- [ ] T043 [P] [US3] Integration test `tests/integration/test_sealed_encoder_default_flow.py` (`@pytest.mark.integration`): create a tiny sealed encoder via `SealedEncoderStore.save_encoder` (no full training); run `encode-cards` against a tiny corpus; assert one `.npz` shape is `(2*d_model + FEATURE_COUNT,)` and that the encoder weights used were the sealed ones (e.g., compare a known fingerprint vector).

### Implementation for User Story 3

- [ ] T044 [US3] In `src/sealed/application/train_scorer.py:45` flip `TrainScorerConfig.encoder_checkpoint` default from `Path("models/price-predictor/transformer/latest.pt")` to `Path("models/sealed/encoder/latest.pt")` per research.md D-6 / FR-024.
- [ ] T045 [US3] In `src/sealed/infrastructure/cli.py` flip the `_ENCODE_CARDS_DEFAULT_ENCODER` constant (currently around line 523) from the price-predictor path to `Path("models/sealed/encoder/latest.pt")` per FR-025. Also flip the `encode-cards` `--vocab-path` default to `Path("models/sealed/encoder/vocab.txt")` per cli.md §"Vocab path coupling".
- [ ] T046 [US3] Add a missing-default-file guard in both call paths (`run_encode_cards`, `run_train_scorer`): if the user did NOT pass `--encoder-checkpoint` AND the resolved default path does not exist, raise/exit with code 2 and the message:
  ```
  Sealed encoder not found at models/sealed/encoder/latest.pt.
  Run python -m sealed train-encoder, or pass --encoder-checkpoint <path> explicitly.
  ```
  (FR-026, cli.md §"New error condition"). Whether the user passed the flag explicitly is detected via argparse (e.g., a sentinel default + `args.encoder_checkpoint is _NOT_PASSED`).
- [ ] T047 [US3] Verify Phase B `--resume <phaseB>.pt` and `--scorer-checkpoint <phaseA>.pt` paths are unaffected (cli.md §"Resume semantics"): the resume path already loads encoder weights from the resumed checkpoint, and the encoder-checkpoint default flip applies only to fresh kickoffs. Add a unit test in `tests/unit/sealed/infrastructure/test_cli_train_scorer_default.py` covering this.

**Checkpoint**: The sealed pipeline picks up the new encoder by default. US3 is independently demoable. P1 stories are complete (MVP achieved).

---

## Phase 6: User Story 4 - Build a sealed-specific vocabulary (Priority: P2)

**Goal**: `python -m sealed build-vocab` writes `models/sealed/encoder/vocab.txt` from the converted card corpus, delegating to the existing price-side `build_vocabulary` utility. Independent of the price-predictor vocab file.

**Independent Test**: Run `python -m sealed build-vocab` against `output/cardsfolder/`. Verify `models/sealed/encoder/vocab.txt` exists with one token per line, the first three tokens are `[PAD] [UNK] cardname` (seeded specials per data-model.md), known MTG terms (`creature`, `enchantment`, `flying`) are each a single line, and `models/price-predictor/transformer/vocab.txt` is unchanged.

### Tests for User Story 4 (write first; let them fail; then implement)

- [ ] T048 [P] [US4] Unit test `tests/unit/sealed/application/test_build_vocab.py`: given a tiny fixture corpus, the wrapper calls the price-side `build_vocabulary` and writes the output to the sealed path. The price-side vocab file (if present) is byte-unchanged after the call. With `--target-size N` smaller than the seed-token count, exits code 2 with a `ValueError`-derived message (cli.md §"Exit codes", D-1).
- [ ] T049 [P] [US4] Unit test `tests/unit/sealed/infrastructure/test_cli_build_vocab_argparse.py`: each flag parses with the documented default (`--cards-folder=output/cardsfolder/`, `--vocab-path=models/sealed/encoder/vocab.txt`, `--target-size=5000`).

### Implementation for User Story 4

- [ ] T050 [US4] Implement `src/sealed/application/build_vocab.py`: define `BuildVocabConfig` dataclass (per data-model.md §`BuildVocabConfig`) and `run(config: BuildVocabConfig) -> None`. Body: empty-folder pre-check → exit 1, call `price_predictor.application.build_vocabulary.build_vocabulary(cards_path=config.cards_folder, freq_threshold=2, printings_path=config.printings_path if exists else None)`, truncate the resulting vocab dict to `--target-size` entries by frequency (always preserving seeded specials per Decision D-1), call `tokenizer_store.save_vocabulary(vocab, config.vocab_path)`, print `"Wrote N tokens to <vocab-path> (corpus coverage: P%)"`.
- [ ] T051 [US4] Wire `build-vocab` into `src/sealed/infrastructure/cli.py`: `_build_build_vocab_parser(subparsers)` and `run_build_vocab(args)` following the existing pattern. Register flags `--cards-folder`, `--vocab-path`, `--target-size`. Map exits 0/1/2 per cli.md §"Exit codes".

**Checkpoint**: Vocab build is one-step. The full quickstart Step 2 works. The price-side vocab file is independent (FR-008).

---

## Phase 7: User Story 5 - Tune low-n shrinkage for noisy labels (Priority: P3)

**Goal**: `--shrinkage-k` is observable through the inspection file. Cards with few in-deck observations shift visibly between `k=0` and `k=20`; cards with many observations stay nearly identical (SC-005).

**Independent Test**: Run `train-encoder --shrinkage-k 0` and `train-encoder --shrinkage-k 20` on the same corpus; diff the two `output/sealed/cards-win-rates.txt` snapshots. Low-observation cards (e.g., `wins_when_in_deck = 2`) shift by more than 0.05 between runs; high-observation cards (e.g., `wins_when_in_deck = 1000`) shift by less than 0.005.

Most of US5's implementation lands inside US2's `train-encoder` flag surface (FR-021) and aggregation math (T031 / T024). This phase only adds the verification tasks that prove the flag does what SC-005 claims.

### Tests for User Story 5

- [ ] T052 [P] [US5] Unit test `tests/unit/sealed/application/test_shrinkage_endpoints.py`: tightening assertions on the existing shrinkage helper at the SC-005 boundary values (low-n shifts visibly with `k=20`; high-n labels are within `0.005` of the raw ratio). May extend `test_shrinkage.py` rather than introducing a new file if cleaner.
- [ ] T053 [US5] Integration test `tests/integration/test_shrinkage_diff_snapshot.py` (`@pytest.mark.integration`): runs `train-encoder` twice with `--shrinkage-k 0` then `--shrinkage-k 20` on the same fixture corpus (`--epochs 1`), captures the two `cards-win-rates.txt` snapshots before they get overwritten (e.g., copy after each run), and asserts the SC-005 expected shifts hold.

**Checkpoint**: SC-005 is verifiable from a clean checkout in two commands.

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Documentation, end-to-end validation, and final cleanup. Does not introduce new behavior.

- [ ] T054 [P] Update `CLAUDE.md` (`## Architecture` → `### `sealed`` subsection) to describe the new `build-vocab` and `train-encoder` subcommands, the new `output/sealed/cards-played.txt` artifact, and the flipped `--encoder-checkpoint` defaults for `encode-cards` and `train-scorer`. Match the existing tone (timeless present tense — no "now supports", no benchmark numbers).
- [ ] T055 [P] Update the price-predictor entry in `CLAUDE.md` if needed to clarify that the price-predictor encoder is no longer the default `--encoder-checkpoint` for the sealed pipeline (still a valid explicit flag).
- [ ] T056 Run `specs/016-card-winnability-pretraining/quickstart.md` end to end against a real (non-fixture) checkout. Address any drift between quickstart and current behavior; if any drift surfaces it goes back as a bug into the appropriate user story.
- [ ] T057 [P] `ruff check src/ tests/` clean.
- [ ] T058 [P] `pytest -m "not integration"` clean.
- [ ] T059 [P] `pytest -m integration` clean (runs Java integration tests indirectly via the smoke tests in T019, T027, T043, T053).
- [ ] T060 [P] `cd forge-connector; mvn test` clean (runs JUnit tests added in T009–T012).

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No upstream deps. T001–T003 are confirmation tasks.
- **Foundational (Phase 2)**: Depends on Setup. **Blocks** US2 and US3.
- **US1 (Phase 3)**: Depends on Setup only. Independent of Foundational. Can start immediately and run in parallel with Foundational.
- **US2 (Phase 4)**: Depends on Foundational (needs `SealedEncoderModel`, `SealedEncoderStore`). Uses fixtures, so does NOT depend on US1 finishing match collection or on US4 finishing vocab build.
- **US3 (Phase 5)**: Depends on Foundational (`SealedEncoderStore.load_encoder`). Tests use fixture encoder files and do not require a real `train-encoder` run.
- **US4 (Phase 6)**: Depends on Setup only. No downstream Python deps inside this feature.
- **US5 (Phase 7)**: Depends on US2 (needs `train-encoder` end-to-end + `cards-win-rates.txt` writer).
- **Polish (Phase 8)**: Depends on US1–US4 being complete. T056 (quickstart end-to-end) additionally needs US1 having produced a real `cards-played.txt`.

### User Story Dependencies (informational)

- **US1, US4** are independent of every other story.
- **US2** is independent of US1 / US4 at implementation time (uses fixtures); end-to-end use depends on both.
- **US3** is independent of US2 at implementation time (uses fixture encoder); end-to-end use depends on US2.
- **US5** is a thin verification layer over US2.

### Within Each User Story

- Tests written first (per Constitution Principle I); let them fail; then implement.
- Models / records / writers before the consuming application code.
- Application code before CLI wiring.
- CLI wiring before integration tests that invoke `python -m sealed ...`.

### Parallel Opportunities

- Phase 1: T002, T003 in parallel.
- Phase 2: T004, T006, T008 in parallel; T005 then T007 sequentially after.
- Phase 3 (US1): all four Java unit-test stubs (T009–T012) in parallel; record/writer/collector implementations (T013–T015) in parallel after the tests fail.
- Phase 4 (US2): the seven Python test files (T020–T026) in parallel; fixture creation (T028) in parallel with reader/aggregation/writer modules (T029–T032) once the test stubs exist.
- Phase 5 (US3): three test files (T040–T042) in parallel; default flips (T044, T045) in parallel.
- Phase 6 (US4): T048 and T049 in parallel.
- Phase 8 (Polish): T054, T055, T057, T058, T059, T060 all in parallel.

---

## Parallel Example: Phase 3 (US1)

```text
# Drop the four failing Java unit tests in parallel:
- forge-connector/src/test/java/com/pricepredictor/connector/CardsPlayedRowTest.java        (T009)
- forge-connector/src/test/java/com/pricepredictor/connector/CardsPlayedWriterTest.java     (T010)
- forge-connector/src/test/java/com/pricepredictor/connector/PlayedCardCollectorTest.java   (T011)
- forge-connector/src/test/java/com/pricepredictor/connector/MatchGeneratorCardsPlayedIntegrationTest.java (T012)

# Then implement the three new sources in parallel:
- forge-connector/src/main/java/com/pricepredictor/connector/CardsPlayedRow.java       (T013)
- forge-connector/src/main/java/com/pricepredictor/connector/CardsPlayedWriter.java    (T014)
- forge-connector/src/main/java/com/pricepredictor/connector/PlayedCardCollector.java  (T015)
```

## Parallel Example: Phase 4 (US2) test scaffolding

```text
- tests/unit/sealed/infrastructure/test_cards_played_reader.py       (T020)
- tests/unit/sealed/application/test_label_aggregation.py            (T021)
- tests/unit/sealed/application/test_shrinkage.py                    (T022)
- tests/unit/sealed/application/test_train_val_split.py              (T023)
- tests/unit/sealed/application/test_cards_win_rates_writer.py       (T024)
- tests/unit/sealed/application/test_corpus_consistency.py           (T025)
- tests/unit/sealed/infrastructure/test_cli_train_encoder_argparse.py (T026)
```

---

## Implementation Strategy

### MVP First (P1 stories: US1 + US2 + US3)

1. Phase 1 + Phase 2 land first (one developer, ~one sitting).
2. Then in parallel:
   - Developer A picks up US1 (Phase 3, Java side).
   - Developer B picks up US2 (Phase 4, Python `train-encoder`).
   - Developer C picks up US3 (Phase 5, default flips).
3. Validate the MVP with `tests/integration/test_sealed_encoder_default_flow.py` + a real `match-outcomes` smoke run.
4. Demo: a sealed encoder is trained from real `cards-played.txt` and the existing scorer pipeline picks it up by default.

### Incremental Delivery After MVP

1. Add US4 (build-vocab). Quickstart Step 2 works directly without manually copying the price-side vocab.
2. Add US5 (shrinkage verification). SC-005 becomes one-line reproducible.

### Solo Strategy

If working alone, the natural sequential order is:
1. Phase 1 → Phase 2 → US4 → US1 → US2 → US3 → US5 → Polish.
2. US4 ahead of US1 lets the developer build a real vocab before walking away to wait on the long-running `match-outcomes` collection; the priority "P2" reflects feature importance, not implementation order.

---

## Notes

- Tests are MANDATORY per Constitution Principle I and are listed before implementation tasks within each user-story phase. Verify each test fails before implementing.
- Every new entity/service in this task list cites its prior art in plan.md's Codebase Survey (Principle VII): `SealedEncoderConfig` parallels `TransformerConfig`, `SealedEncoderModel` parallels `CardPriceTransformerModel`, `CardsPlayedWriter` mirrors `MatchResultWriter`, `PlayedCardCollector` mirrors `../jumpstart-tierlist/.../JumpstartMatch.java#CardCollector`, `SealedEncoderStore` mirrors `ScorerStore`, `_BestCheckpoint` early-stop is copied from `train_transformer.py:146` (follow-up flagged in research.md to extract on next training feature).
- Commit after each task or logical group; a clean commit per phase boundary is a reasonable default for this feature.
- All hot paths in `train-encoder` (Phase 4) are covered by fast unit tests; the only `@pytest.mark.integration` tests are the end-to-end smokes (T019, T027, T043, T053).
- File-format changes (`cards-played.txt` schema, `cards-win-rates.txt` schema, encoder `.pt` payload shape) are pinned by `contracts/files.md`; downstream readers may rely on those guarantees.
